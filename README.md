# EatBench-2.7K: A Benchmark for Fine-Grained Eating Action Grounding in Videos

[[Paper]](#) | [[Dataset (HuggingFace)]](https://huggingface.co/datasets/anonymous-eatbench-2026/EatBench-2.7K)

## Overview

**EatBench-2.7K** is the first video benchmark for fine-grained eating-action grounding. It contains **525 video clips** with **2,690 temporally annotated micro-action instances** spanning **10 food categories** and **3 eating stages**:

| Stage | Abbreviation | Description |
|---|---|---|
| Contacting Food | CF | Hand/utensil contacts food (reach, grasp, scoop) |
| Food Approaching Mouth | FAM | Food transported toward lips/teeth/tongue |
| Food in Mouth | FIM | Food crosses lip line; chewing and swallowing |

EatBench-2.7K exposes key challenges for fine-grained temporal grounding:
- **600× duration range** (0.042s – 25.2s)
- **Sub-second FAM** events (median 0.79s; 67.4% under 1s)
- **Dense action cycles** with near-deterministic CF→FAM→FIM transitions
- **Naturalistic recording**: 69.3% of videos begin mid-eating-cycle

## Dataset

Videos and annotations are hosted on HuggingFace:
**[https://huggingface.co/datasets/anonymous-eatbench-2026/EatBench-2.7K](https://huggingface.co/datasets/anonymous-eatbench-2026/EatBench-2.7K)**

The annotation file (`eatbench_annotation_full.json`) is also included in this repository.

### Download Videos

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download the full dataset (videos + annotations)
huggingface-cli download anonymous-eatbench-2026/EatBench-2.7K \
    --repo-type dataset \
    --local-dir ./EatBench-2.7K-data
```

## Repository Structure

```
EatBench-2.7K/
├── README.md
├── eatbench_annotation_full.json     # Annotations for all 525 videos
└── codes/
    ├── README.md                     # Detailed usage instructions
    ├── requirements.txt
    ├── run_SAFR.py                   # Frame extraction (uniform / SAFR)
    ├── run_OneThinker.py             # OneThinker inference
    └── Evaluation/
        └── evaluate.py               # Hungarian-matching evaluation
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r codes/requirements.txt
```

### 2. Extract frames (SAFR)

```bash
python codes/run_SAFR.py \
    --mode safr \
    --annotation_json eatbench_annotation_full.json \
    --video_dir /path/to/videos/ \
    --output_dir frames/
```

### 3. Run inference (OneThinker)

```bash
python codes/run_OneThinker.py \
    --annotation_json eatbench_annotation_full.json \
    --video_dir /path/to/videos/ \
    --manifest_json frames/safr_safr_fps2.0_max16_smooth3/manifest_with_time.json \
    --output_json results/onethinker_safr.json
```

### 4. Evaluate

```bash
python codes/Evaluation/evaluate.py \
    --annotation_json eatbench_annotation_full.json \
    --pred_json results/onethinker_safr.json \
    --output_json results/onethinker_safr_metrics.json
```

See [`codes/README.md`](codes/README.md) for full documentation.

## Benchmark Results

Macro-F1 at tIoU = 0.1 (zero-shot, no fine-tuning):

| Model | Frame Selection | CF F1 | FAM F1 | FIM F1 | Macro-F1 |
|---|---|---|---|---|---|
| VideoChat-Flash | Uniform | 27.6 | 17.0 | 37.5 | 27.4 |
| OneThinker | Uniform | 30.4 | 22.2 | 43.9 | 32.2 |
| OneThinker | **SAFR** | **31.5** | **24.3** | **44.1** | **33.3** |
| VAPO-Thinker-7B | Uniform | 18.8 | 8.1 | 24.8 | 17.3 |
| VAPO-Thinker-7B | **SAFR** | 19.4 | 12.1 | 31.1 | 20.9 |
| Qwen2.5-VL-7B | Uniform | 19.5 | 11.6 | 25.0 | 18.7 |
| Qwen2.5-VL-7B | **SAFR** | 20.9 | 12.6 | 26.4 | 20.0 |
| InternVL3-8B | Uniform | 14.9 | 17.4 | 28.4 | 20.2 |
| InternVL3-8B | **SAFR** | 14.1 | 19.7 | 32.4 | 22.1 |

All models suffer >74% relative drop from tIoU=0.1 to tIoU=0.5, confirming EatBench-2.7K as an open challenge.

## License

Annotations and code are released under [CC BY 4.0](LICENSE). Videos are sourced from Kinetics-400 under their original Creative Commons license.

## Citation

```bibtex
@article{eatbench2025,
  title     = {EatBench-2.7K: A Benchmark for Fine-Grained Eating Action Grounding in Videos},
  author    = {Anonymous},
  journal   = {Advances in Neural Information Processing Systems},
  year      = {2025},
}
```
