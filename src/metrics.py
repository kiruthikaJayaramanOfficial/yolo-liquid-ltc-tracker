"""src/metrics.py — shared metrics used by all trackers and MLflow runs."""
from __future__ import annotations
import numpy as np
from typing import List, Dict

FOG_BUCKETS = [
    ("Clear",        0.00, 0.15),
    ("Light Haze",   0.15, 0.35),
    ("Moderate Fog", 0.35, 0.55),
    ("Dense Fog",    0.55, 0.75),
    ("Extreme Fog",  0.75, 1.01),
]

def compute_recovery_pct(results_log: List[Dict]) -> Dict[str, float]:
    """
    Recovery % = objects maintained relative to clear-frame YOLO baseline.
    clear_baseline = avg YOLO detections when fog < 15%
    recovery_pct   = (avg_yolo + avg_ghost) / clear_baseline * 100
    YOLO-only in clear = 100% reference. LNN in fog stays higher.
    """
    clear_rows = [r for r in results_log if r["fog"] < 0.15]
    if clear_rows:
        clear_baseline = float(np.mean([r["n_yolo"] for r in clear_rows]))
    else:
        clear_baseline = float(max(r["n_yolo"] for r in results_log)) if results_log else 1.0
    clear_baseline = max(clear_baseline, 1.0)

    bucket_metrics: Dict[str, Dict] = {}
    for bucket_name, fog_lo, fog_hi in FOG_BUCKETS:
        rows = [r for r in results_log if fog_lo <= r["fog"] < fog_hi]
        if not rows:
            continue
        avg_yolo   = float(np.mean([r["n_yolo"]  for r in rows]))
        avg_ghost  = float(np.mean([r["n_ghost"] for r in rows]))
        total_objs = avg_yolo + avg_ghost
        recovery   = (total_objs / clear_baseline) * 100
        bucket_metrics[bucket_name] = {
            "avg_yolo"      : round(avg_yolo,       2),
            "avg_ghost"     : round(avg_ghost,      2),
            "avg_total"     : round(total_objs,     2),
            "clear_baseline": round(clear_baseline, 2),
            "recovery"      : round(recovery,       2),
            "n_frames"      : len(rows),
        }
    return bucket_metrics

def flat_mlflow_metrics(bucket_metrics: Dict[str, Dict]) -> Dict[str, float]:
    flat = {}
    name_map = {
        "Clear":"clear", "Light Haze":"light_haze",
        "Moderate Fog":"moderate_fog", "Dense Fog":"dense_fog", "Extreme Fog":"extreme_fog",
    }
    for bucket, data in bucket_metrics.items():
        key = name_map.get(bucket, bucket.lower().replace(" ","_"))
        flat[f"recovery_pct_{key}"] = data["recovery"]
        flat[f"avg_yolo_{key}"]     = data["avg_yolo"]
        flat[f"avg_ghost_{key}"]    = data["avg_ghost"]
        flat[f"avg_total_{key}"]    = data.get("avg_total", data["avg_yolo"])
    return flat

def summary_table(all_runs: List[Dict]) -> str:
    header = f"{'Run name':<30} {'Recovery%':>10} {'MemStr':>8} {'Type':>12}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for r in sorted(all_runs, key=lambda x: x.get("recovery_extreme", 0), reverse=True):
        rows.append(
            f"{r['name']:<30} {r.get('recovery_extreme',0):>9.1f}%"
            f"  {r.get('mem_str','N/A'):>8}  {r.get('type',''):>12}"
        )
    return "\n".join(rows)