# EatBench-2.7K: Evaluation Code

This directory contains the evaluation code for EatBench-2.7K, including the SAFR frame selection strategy, the OneThinker inference pipeline, and the evaluation script.

## Overview

The full pipeline runs in three steps:

1. **`run_SAFR.py`** — Extract frames from each video using uniform sampling or SAFR (Semantic-Anchored Frame Relocation). Saves frames to disk and produces a manifest JSON with frame paths and timestamps.
2. **`run_OneThinker.py`** — Run OneThinker inference using the cached frames from the manifest, and save predicted action segments to a results JSON.
3. **`Evaluation/evaluate.py`** — Evaluate predictions against ground truth using Hungarian matching and report per-class Precision, Recall, F1, mIoU, and Macro-F1.

## Requirements

```bash
pip install torch transformers opencv-python numpy tqdm vllm qwen-vl-utils
```

CLIP model (`openai/clip-vit-base-patch32`) and OneThinker checkpoint (`OneThink/OneThinker-8B`) will be downloaded automatically from HuggingFace on first run.

## Step 1: Frame Extraction

### Uniform Sampling (baseline)

```bash
python run_SAFR.py \
    --mode uniform \
    --annotation_json /path/to/eatbench_annotation_full.json \
    --video_dir /path/to/videos/ \
    --output_dir /path/to/frames/
```

### SAFR (Semantic-Anchored Frame Relocation)

```bash
python run_SAFR.py \
    --mode safr \
    --smooth_w 3 \
    --annotation_json /path/to/eatbench_annotation_full.json \
    --video_dir /path/to/videos/ \
    --output_dir /path/to/frames/
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--mode` | `safr` | Frame selection mode: `uniform` or `safr` |
| `--smooth_w` | `3` | Temporal smoothing window size for CLIP similarity (SAFR only) |
| `--annotation_json` | required | Path to `eatbench_annotation_full.json` |
| `--video_dir` | required | Directory containing video `.mp4` files |
| `--output_dir` | required | Root directory for cached frames and manifest |

**Output:** A subdirectory is created under `--output_dir` containing:
- Extracted frame images (`*.jpg`) organized per video
- `manifest_with_time.json` — maps each video to its selected frame paths and timestamps

## Step 2: OneThinker Inference

```bash
python run_OneThinker.py \
    --annotation_json /path/to/eatbench_annotation_full.json \
    --video_dir /path/to/videos/ \
    --manifest_json /path/to/frames/safr_safr_fps2.0_max16_smooth3/manifest_with_time.json \
    --output_json /path/to/results/onethinker_safr.json
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `OneThink/OneThinker-8B` | Model checkpoint path or HuggingFace model ID |
| `--annotation_json` | required | Path to `eatbench_annotation_full.json` |
| `--video_dir` | required | Directory containing video `.mp4` files |
| `--manifest_json` | required | Path to `manifest_with_time.json` from Step 1 |
| `--output_json` | required | Path to save prediction results |

**Output:** A JSON file mapping each video name to predicted action segments:

```json
{
  "video_name.mp4": {
    "contacting_food": [[0.0, 1.2], [8.3, 9.1]],
    "food_approaching_mouth": [[1.2, 2.0], [9.1, 9.8]],
    "food_in_mouth": [[2.0, 5.5], [9.8, 12.3]]
  },
  ...
}
```

## Full Pipeline Example

```bash
# Step 1: extract frames with SAFR
python run_SAFR.py \
    --mode safr \
    --annotation_json eatbench_annotation_full.json \
    --video_dir videos/ \
    --output_dir frames/

# Step 2: run OneThinker inference
python run_OneThinker.py \
    --annotation_json eatbench_annotation_full.json \
    --video_dir videos/ \
    --manifest_json frames/safr_safr_fps2.0_max16_smooth3/manifest_with_time.json \
    --output_json results/onethinker_safr.json

# Step 3: evaluate
python Evaluation/evaluate.py \
    --annotation_json eatbench_annotation_full.json \
    --pred_json results/onethinker_safr.json \
    --output_json results/onethinker_safr_metrics.json
```

## Step 3: Evaluation

```bash
python Evaluation/evaluate.py \
    --annotation_json /path/to/eatbench_annotation_full.json \
    --pred_json /path/to/results/onethinker_safr.json \
    --output_json /path/to/results/onethinker_safr_metrics.json
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--annotation_json` | required | Path to `eatbench_annotation_full.json` (ground truth) |
| `--pred_json` | required | Path to model prediction JSON from Step 2 |
| `--output_json` | `None` | Optional path to save evaluation results as JSON |
| `--thresholds` | `0.1 0.3 0.5` | tIoU thresholds to evaluate at |

**Console output:**
```
Videos evaluated: 525

==================================================
  tIoU@0.1   Macro-F1: 0.3330
==================================================
  Class                           P        R       F1     mIoU
  ------------------------------------------------------
  Contacting Food            0.2880   0.3460   0.3150   0.1892
  Food Approaching Mouth     0.3620   0.1830   0.2430   0.1421
  Foodin Mouth               0.7330   0.3150   0.4410   0.2105
```

**Output JSON format:**
```json
{
  "summary": {"num_videos": 525, "thresholds": [0.1, 0.3, 0.5]},
  "per_threshold": {
    "tIoU@0.1": {
      "Macro_F1": 0.333,
      "per_class": {
        "Contacting Food":        {"Precision": 0.288, "Recall": 0.346, "F1": 0.315, "mIoU": 0.189, ...},
        "Food Approaching Mouth": {"Precision": 0.362, "Recall": 0.183, "F1": 0.243, "mIoU": 0.142, ...},
        "Food in Mouth":          {"Precision": 0.733, "Recall": 0.315, "F1": 0.441, "mIoU": 0.211, ...}
      }
    }
  }
}
```

The evaluator accepts predictions in two formats:
- `{video_name: {snake_key: [[s, e], ...], ...}}` — output of `run_OneThinker.py`
- `{video_name: {"prediction": {snake_key: [{"segment": [s, e], "score": ...}, ...]}}}` — score-based format (scores ignored)

## SAFR Algorithm

SAFR partitions the video timeline into K equal windows around uniform anchors and relocates each anchor to the frame with the highest semantic similarity to the eating-action prompts (Algorithm 1 in the paper).

Given T video frames and frame budget K:
1. Compute per-frame, per-action CLIP similarity scores s_a(t)
2. Smooth each s_a(t) with a temporal window, then aggregate: S(t) = max_a s̃_a(t)
3. Place uniform anchors at u_i = floor((T/K)(i − 0.5))
4. For each window W_i around u_i, select y_i = argmax_{t ∈ W_i} S(t)

SAFR adds O(T) overhead over uniform sampling and requires no training or model modification.
