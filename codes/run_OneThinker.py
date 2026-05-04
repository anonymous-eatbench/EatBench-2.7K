import os
import json
import ast
import re
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# ================== TASK DEFINITION ==================
FINE_DEFS = (
    "Identify fine-grained eating micro-actions in the video. "
    "Categories:\n"
    "1) contacting_food — direct contact with food using hand/utensil (pick/grab/cut/scoop/pour/stir/serve).\n"
    "2) food_approaching_mouth — transporting/aligning food toward the lips/teeth/tongue.\n"
    "3) food_in_mouth — from first contact with lips until food crosses the lip line."
)

FORMAT_INSTRUCTION = (
    "Please provide only the action localization results as a JSON dictionary within the <answer>...</answer> tags. "
    "Example:\n<answer>{{\"contacting_food\": [[0.0, 1.2]], \"food_approaching_mouth\": [[2.3, 3.1]], \"food_in_mouth\": [[3.2, 5.0]]}}</answer>"
)

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.\n"
    + FORMAT_INSTRUCTION
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run OneThinker on EatBench-2.7K.")
    parser.add_argument("--checkpoint", type=str, default="OneThink/OneThinker-8B",
                        help="Model checkpoint path or HuggingFace model ID.")
    parser.add_argument("--annotation_json", type=str, required=True,
                        help="Path to EatBench annotation JSON.")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files.")
    parser.add_argument("--manifest_json", type=str, required=True,
                        help="Path to frame manifest produced by run_SAFR.py.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to save prediction results.")
    return parser.parse_args()


# ================== UTILS ==================

def get_video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0, 0, 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    return duration, frames, fps


def try_parse_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            return ast.literal_eval(content)
        except Exception:
            clean = re.sub(r'```json|```', '', content).strip()
            try:
                return json.loads(clean)
            except Exception:
                pass
    return {}


def normalize(obj):
    keys = ["contacting_food", "food_approaching_mouth", "food_in_mouth"]
    out = {k: [] for k in keys}
    if not isinstance(obj, dict):
        return out
    for k in keys:
        for it in obj.get(k, []):
            if isinstance(it, (list, tuple)) and len(it) == 2:
                out[k].append([round(float(it[0]), 1), round(float(it[1]), 1)])
    return out


def build_timeline_header(frames_list):
    """Build a compact frame timestamp header to prepend to the prompt."""
    frames_list = sorted(frames_list, key=lambda x: int(x.get("idx", 0)))
    lines = ["Selected frames (time in seconds):"]
    for fr in frames_list:
        lines.append(f"#{int(fr.get('k', 0))} t={float(fr.get('t', 0.0)):.2f}s")
    return "\n".join(lines)


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
        if video_kwargs is not None and "video_grid_thw" in video_kwargs:
            mm_data["video_metadata"] = {"video_grid_thw": video_kwargs["video_grid_thw"]}
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


# ================== MAIN ==================

def main():
    args = parse_args()

    with open(args.manifest_json) as f:
        manifest = json.load(f)

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    llm = LLM(
        model=args.checkpoint,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=1,
        max_model_len=24576,
        gpu_memory_utilization=0.5,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    with open(args.annotation_json) as f:
        test_list = json.load(f)

    all_inputs = []
    video_names = []

    print("Pre-processing video inputs...")
    for entry in tqdm(test_list):
        videoname = entry.get("Video Name")
        if not videoname:
            continue
        video_path = os.path.join(args.video_dir, videoname)
        if not os.path.exists(video_path):
            continue

        info = manifest.get(videoname)
        if info and isinstance(info.get("frames"), list) and info["frames"]:
            frames_list = sorted(info["frames"], key=lambda x: int(x.get("idx", 0)))
            frame_paths = [fr["path"] for fr in frames_list if os.path.exists(fr.get("path", ""))]
            video_field = frame_paths if frame_paths else video_path
        else:
            frames_list = None
            video_field = video_path

        duration, _, _ = get_video_meta(video_path)

        if frames_list and isinstance(video_field, list):
            timeline = build_timeline_header(frames_list)
            question_text = f"{timeline}\nThe video lasts {duration:.1f}s. {FINE_DEFS}"
        else:
            question_text = f"The video lasts {duration:.1f}s. {FINE_DEFS}"

        full_text = QUESTION_TEMPLATE.format(Question=question_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_field, "max_pixels": 256 * 32 * 32},
                    {"type": "text", "text": full_text},
                ],
            }
        ]

        all_inputs.append(prepare_inputs_for_vllm(messages, processor))
        video_names.append(videoname)

    print(f"Running inference on {len(all_inputs)} videos...")
    outputs = llm.generate(all_inputs, sampling_params=sampling_params)

    results = {}
    for videoname, output in zip(video_names, outputs):
        parsed = try_parse_answer(output.outputs[0].text)
        results[videoname] = normalize(parsed)

    os.makedirs(str(Path(args.output_json).parent), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
