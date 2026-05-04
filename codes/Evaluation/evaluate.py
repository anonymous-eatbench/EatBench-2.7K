import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scipy.optimize import linear_sum_assignment

CLASSES = ["Contacting Food", "Food Approaching Mouth", "Food in Mouth"]

# Map prediction keys (snake_case) to canonical class names
PRED_LABEL_MAP = {
    "contacting_food": "Contacting Food",
    "food_approaching_mouth": "Food Approaching Mouth",
    "food_in_mouth": "Food in Mouth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal action localization on EatBench-2.7K."
    )
    parser.add_argument("--annotation_json", type=str, required=True,
                        help="Path to eatbench_annotation_full.json (ground truth).")
    parser.add_argument("--pred_json", type=str, required=True,
                        help="Path to model prediction JSON.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save evaluation results.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                        help="tIoU thresholds to evaluate at (default: 0.1 0.3 0.5).")
    return parser.parse_args()


# ================== UTILS ==================

def tiou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Temporal Intersection over Union."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 1e-9 else 0.0


def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


# ================== LOADERS ==================

def load_gt(gt_path: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Load ground-truth annotations from eatbench_annotation_full.json."""
    raw = json.loads(Path(gt_path).read_text())
    out = {}
    for entry in raw:
        vid = entry.get("Video Name")
        if not vid:
            continue
        per = {c: [] for c in CLASSES}
        for act in entry.get("Actions", []):
            cls = act.get("Label")
            if cls not in CLASSES:
                continue
            s, e = safe_float(act.get("Start")), safe_float(act.get("End"))
            if s > e:
                s, e = e, s
            if e - s > 1e-9:
                per[cls].append((s, e))
        out[vid] = per
    return out


def load_pred(pred_path: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    Load model predictions. Supports two formats:
      - {vid: {snake_key: [[s, e], ...], ...}}
      - {vid: {"prediction": {snake_key: [{"segment": [s, e]}, ...]}}}
    Scores are ignored (score-free evaluation).
    """
    raw = json.loads(Path(pred_path).read_text())
    out = {}
    for vid, item in raw.items():
        block = item.get("prediction", item)
        per = {c: [] for c in CLASSES}
        if not isinstance(block, dict):
            out[vid] = per
            continue
        for pred_lbl, segs in block.items():
            cls = PRED_LABEL_MAP.get(pred_lbl)
            if cls is None or not isinstance(segs, list):
                continue
            for it in segs:
                seg = it.get("segment") if isinstance(it, dict) else it
                if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
                    continue
                s, e = safe_float(seg[0]), safe_float(seg[1])
                if s > e:
                    s, e = e, s
                if e - s > 1e-9:
                    per[cls].append((s, e))
        out[vid] = per
    return out


# ================== MATCHING ==================

def hungarian_match(
    pred: List[Tuple[float, float]],
    gt: List[Tuple[float, float]],
    thr: float,
) -> List[Tuple[int, int, float]]:
    """
    One-to-one Hungarian matching maximizing total tIoU (Eq. 1 in paper).
    Returns matched pairs (pred_i, gt_j, iou) with iou >= thr.
    """
    if not pred or not gt:
        return []

    BIG = 1e6
    iou_mat = [[tiou(p, g) for g in gt] for p in pred]
    cost = [[BIG if iou_mat[i][j] < thr else 1.0 - iou_mat[i][j]
             for j in range(len(gt))]
            for i in range(len(pred))]

    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (i, j, iou_mat[i][j])
        for i, j in zip(row_ind.tolist(), col_ind.tolist())
        if iou_mat[i][j] >= thr and cost[i][j] < BIG
    ]


# ================== EVALUATION ==================

def evaluate(
    gt_path: str,
    pred_path: str,
    thresholds: Tuple[float, ...] = (0.1, 0.3, 0.5),
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth using Hungarian matching.
    Reports per-class Precision, Recall, F1, matched mIoU, and Macro-F1
    at each tIoU threshold (Section 3.6 in paper).
    """
    gt = load_gt(gt_path)
    pred = load_pred(pred_path)
    videos = sorted(set(gt.keys()) & set(pred.keys()))

    results: Dict[str, Any] = {
        "summary": {
            "num_videos": len(videos),
            "thresholds": list(thresholds),
        },
        "per_threshold": {},
    }

    for thr in thresholds:
        per_class = {}
        for cls in CLASSES:
            TP = FP = FN = 0
            iou_sum = 0.0
            iou_cnt = 0
            gt_total = pred_total = 0

            for v in videos:
                G = gt[v].get(cls, [])
                P = pred[v].get(cls, [])
                gt_total += len(G)
                pred_total += len(P)

                matches = hungarian_match(P, G, thr)
                tp = len(matches)
                TP += tp
                FP += len(P) - tp
                FN += len(G) - tp
                for _, _, iou in matches:
                    iou_sum += iou
                    iou_cnt += 1

            prec = TP / (TP + FP) if TP + FP > 0 else 0.0
            rec  = TP / (TP + FN) if TP + FN > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            miou = iou_sum / iou_cnt if iou_cnt > 0 else 0.0

            per_class[cls] = {
                "Precision": round(prec, 4),
                "Recall":    round(rec,  4),
                "F1":        round(f1,   4),
                "mIoU":      round(miou, 4),
                "TP": TP, "FP": FP, "FN": FN,
                "GT": gt_total, "Pred": pred_total,
            }

        macro_f1 = sum(per_class[c]["F1"] for c in CLASSES) / len(CLASSES)
        results["per_threshold"][f"tIoU@{thr}"] = {
            "per_class": per_class,
            "Macro_F1": round(macro_f1, 4),
        }

    return results


def print_results(results: Dict[str, Any]):
    print(f"\nVideos evaluated: {results['summary']['num_videos']}")
    for thr_key, data in results["per_threshold"].items():
        print(f"\n{'='*50}")
        print(f"  {thr_key}   Macro-F1: {data['Macro_F1']:.4f}")
        print(f"{'='*50}")
        print(f"  {'Class':<28} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6}")
        print(f"  {'-'*54}")
        for cls, m in data["per_class"].items():
            print(f"  {cls:<28} {m['Precision']:>6.4f} {m['Recall']:>6.4f} {m['F1']:>6.4f} {m['mIoU']:>6.4f}")


# ================== MAIN ==================

if __name__ == "__main__":
    args = parse_args()

    results = evaluate(
        gt_path=args.annotation_json,
        pred_path=args.pred_json,
        thresholds=tuple(args.thresholds),
    )

    print_results(results)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.output_json}")