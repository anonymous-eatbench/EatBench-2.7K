import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ================== CONFIG ==================
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
ACTION_TEXTS = [
    "contacting food with hand or utensil (pick/grab/cut/scoop/pour/stir/serve)",
    "food approaching mouth (transporting food toward lips/teeth/tongue)",
    "food in mouth (food crosses the lip line, chewing or inside mouth)",
]
FPS_SAMPLE = 2.0
MAX_FRAMES = 16
# ============================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames with uniform sampling or SAFR (Semantic-Anchored Frame Relocation)."
    )
    parser.add_argument("--mode", type=str, default="safr", choices=["uniform", "safr"],
                        help="Frame selection mode: 'uniform' or 'safr'.")
    parser.add_argument("--smooth_w", type=int, default=3,
                        help="Temporal smoothing window size for CLIP similarity scores (paper default: 3).")
    parser.add_argument("--annotation_json", type=str, required=True,
                        help="Path to EatBench annotation JSON.")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory for cached frames and manifest.")
    return parser.parse_args()


# ================== VIDEO UTILS ==================

def get_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0, 0, 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return duration, total_frames, fps


def desired_num_frames(duration_s: float) -> int:
    """Compute frame budget: duration * FPS_SAMPLE, capped at MAX_FRAMES."""
    n = int(round(duration_s * FPS_SAMPLE))
    return max(1, min(n, MAX_FRAMES))


def uniform_sample_frames(total_frames: int, num_frames: int):
    """Tick/center uniform sampling. Returns frame indices in [0, total_frames-1]."""
    if num_frames <= 0 or total_frames <= 0:
        return []
    if num_frames >= total_frames:
        return list(range(total_frames))
    tick = total_frames / num_frames
    idx = [int(tick / 2.0 + tick * x) for x in range(num_frames)]
    idx = [min(max(i, 0), total_frames - 1) for i in idx]
    out, last = [], None
    for i in idx:
        if last is None or i != last:
            out.append(i)
        last = i
    return out


def extract_frames_by_indices(video_path: str, out_dir: Path, indices, skip_if_exists=True):
    """Extract frames at given indices and save as JPEG. Returns list of (k, idx, path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"f_{k:03d}_idx{idx:06d}.jpg" for k, idx in enumerate(indices)]
    if skip_if_exists and expected and all(p.exists() for p in expected):
        return [(k, indices[k], str(expected[k])) for k in range(len(indices))]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = []
    for k, idx in enumerate(indices):
        idx = int(min(max(idx, 0), total - 1))
        save_path = out_dir / f"f_{k:03d}_idx{idx:06d}.jpg"
        if skip_if_exists and save_path.exists():
            saved.append((k, idx, str(save_path)))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(save_path), frame)
        saved.append((k, idx, str(save_path)))
    cap.release()
    return saved


# ================== SAFR ==================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Symmetric moving average with reflect padding."""
    if w <= 1:
        return x.astype(np.float64)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def build_windows(total_frames: int, anchors: list):
    """
    Build K disjoint windows W_i = [L_i, R_i] around uniform anchors,
    bounded by midpoints between adjacent anchors (Algorithm 1, SAFR).
    """
    n = len(anchors)
    if n == 0:
        return []
    a = [int(x) for x in anchors]
    bounds = [0]
    for i in range(n - 1):
        bounds.append((a[i] + a[i + 1]) // 2)
    bounds.append(total_frames - 1)

    segs = []
    for i in range(n):
        L = bounds[i]
        R = bounds[i + 1]
        if i > 0:
            L = max(L, segs[-1][1] + 1)
        segs.append((L, max(L, R)))
    segs[-1] = (segs[-1][0], total_frames - 1)
    return segs


def select_indices_safr(s_mat: np.ndarray, anchors: list, smooth_w: int) -> list:
    """
    SAFR frame selection (Algorithm 1 in the paper).

    Args:
        s_mat:    (A, T) array of per-action CLIP similarity scores.
        anchors:  K uniform anchor frame indices.
        smooth_w: Temporal smoothing window size.

    Returns:
        List of K selected frame indices, one per window.
    """
    T = s_mat.shape[1]

    # Smooth each action's similarity sequence independently, then aggregate (Eq. 4)
    s_mat_sm = np.stack([moving_average(s_mat[a], smooth_w) for a in range(s_mat.shape[0])])
    S = s_mat_sm.max(axis=0)  # S(t) = max_a s̃_a(t)

    # Select argmax within each local window (Eq. 5)
    windows = build_windows(T, anchors)
    chosen = []
    for (L, R) in windows:
        sub = S[L:R + 1]
        t = L if sub.size == 0 else int(L + np.argmax(sub))
        chosen.append(t)

    # Enforce strictly increasing indices
    out, last = [], -1
    for t in chosen:
        if t <= last:
            t = min(last + 1, T - 1)
        out.append(t)
        last = t
    return out[:len(anchors)]


# ================== CLIP SCORING ==================

def clip_scores_all_frames(video_path: str, total_frames: int, model, processor, text_emb, device):
    """
    Compute per-action CLIP similarity for every frame in the video.
    Returns s_mat of shape (A, total_frames).
    """
    from PIL import Image
    import torch

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    A = text_emb.shape[0]
    s_mat = np.zeros((A, total_frames), dtype=np.float32)
    imgs, idxs = [], []
    t = 0

    def flush(imgs, idxs):
        with torch.no_grad():
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            img_emb = model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = (text_emb @ img_emb.T).float().cpu().numpy()
        for b, fr_idx in enumerate(idxs):
            s_mat[:, fr_idx] = sim[:, b]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idxs.append(t)
        t += 1
        if len(imgs) >= BATCH_SIZE:
            flush(imgs, idxs)
            imgs, idxs = [], []

    if imgs:
        flush(imgs, idxs)
    cap.release()
    return s_mat


# ================== MAIN ==================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir) / f"safr_{args.mode}_fps{FPS_SAMPLE}_max{MAX_FRAMES}"
    if args.mode == "safr":
        output_dir = Path(str(output_dir) + f"_smooth{args.smooth_w}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_with_time.json"

    with open(args.annotation_json, "r") as f:
        test_list = json.load(f)

    # Load CLIP only when needed
    if args.mode == "safr":
        import torch
        from transformers import CLIPProcessor, CLIPModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        with torch.no_grad():
            text_inputs = clip_proc(text=ACTION_TEXTS, return_tensors="pt", padding=True).to(device)
            text_emb = clip_model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    else:
        clip_model = clip_proc = text_emb = device = None

    manifest = {}
    missing = 0

    for entry in tqdm(test_list, desc=f"Frame extraction [{args.mode}]"):
        videoname = entry.get("Video Name")
        if not videoname:
            continue
        video_path = os.path.join(args.video_dir, videoname)
        if not os.path.exists(video_path):
            missing += 1
            continue

        duration, total_frames, fps = get_video_meta(video_path)
        if total_frames <= 0 or fps <= 0:
            continue

        n = desired_num_frames(duration)
        anchors = uniform_sample_frames(total_frames, n)

        if args.mode == "uniform":
            indices = anchors
        else:
            s_mat = clip_scores_all_frames(video_path, total_frames, clip_model, clip_proc, text_emb, device)
            if s_mat is None:
                continue
            indices = select_indices_safr(s_mat, anchors, smooth_w=args.smooth_w)

        saved = extract_frames_by_indices(video_path, output_dir / videoname, indices)
        if not saved:
            continue

        frames = [{"k": k, "idx": idx, "t": round(idx / fps, 3), "path": p} for k, idx, p in saved]
        manifest[videoname] = {
            "video_path": video_path,
            "duration": round(duration, 3),
            "fps": round(fps, 6),
            "total_frames": int(total_frames),
            "nframes": len(frames),
            "frames": frames,
            "mode": args.mode,
            "smooth_w": args.smooth_w if args.mode == "safr" else None,
            "clip_model": CLIP_MODEL_ID if args.mode == "safr" else None,
        }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Processed: {len(manifest)} videos")
    if missing:
        print(f"Missing video files: {missing}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
