"""
mlops/run_experiments.py
Sprint 2 — Run all 15 MLflow experiments:
  3 baselines + 12 LTC configs → Champion selection

Usage:
    python mlops/run_experiments.py --frames_dir data/demo_frames --runs 15
    python mlops/run_experiments.py --frames_dir data/demo_frames --baselines_only
    python mlops/run_experiments.py --frames_dir data/demo_frames --ltc_only
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import cv2
import mlflow
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fog_utils   import apply_fog, build_fog_schedule, fog_category
from src.ltc_memory  import LNNMemoryBank
from src.baselines   import YOLOOnlyBaseline, SORTBaseline, VelocityBaseline
from src.metrics     import compute_recovery_pct, flat_mlflow_metrics

try:
    from ultralytics import YOLO as UltralyticsYOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — using dummy detections for demo")

EXPERIMENT_NAME = "YOLO-Liquid-BaselineVsLTC"
CONF_THRESH     = 0.15
NMS_IOU         = 0.45
COCO_TO_VD      = {0:1,1:10,2:4,3:10,5:9,6:6,7:6}


# ── 12 LTC configurations to sweep ────────────────────────────────────────────
LTC_CONFIGS = [
    # name,              hidden, steps, iou_thr, max_lost
    ("LTC_base_h32",        32,     6,    0.25,      15),
    ("LTC_wide_h64",        64,     6,    0.25,      15),
    ("LTC_narrow_h16",      16,     6,    0.25,      15),
    ("LTC_deep_h32_s10",    32,    10,    0.25,      15),
    ("LTC_base_iou35",      32,     6,    0.35,      15),
    ("LTC_base_iou15",      32,     6,    0.15,      15),
    ("LTC_slow_decay_20",   32,     6,    0.25,      20),
    ("LTC_fast_decay_8",    32,     6,    0.25,       8),
    ("LTC_wide_deep",       64,    10,    0.25,      15),
    ("LTC_wide_iou35",      64,     6,    0.35,      15),
    ("LTC_wide_slow",       64,     6,    0.25,      20),
    ("LTC_wide_iou15",      64,     6,    0.15,      15),
]

# 3 baseline configs
BASELINE_CONFIGS = [
    ("YOLO-only",  "baseline", {}),
    ("SORT",       "baseline", {"max_age":15,"min_hits":3,"iou_threshold":0.25}),
    ("Velocity",   "baseline", {"max_frames_lost":10}),
]


def load_frames(frames_dir: Path, max_frames: int = 40):
    imgs = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    return imgs[:max_frames] if max_frames else imgs


def load_yolo_model():
    if not _YOLO_AVAILABLE:
        return None
    model = UltralyticsYOLO("yolov8n.pt")
    for p in model.model.parameters():
        p.requires_grad = False
    return model


def get_yolo_dets(model, frame_fog, H, W):
    if model is None:
        # Dummy detections for dry-run testing
        return [(np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9),
                 0.05, 0.08, 0.8, 4) for _ in range(np.random.randint(3,12))]
    raw = model(frame_fog, verbose=False, conf=CONF_THRESH, iou=NMS_IOU)[0]
    dets = []
    if raw.boxes is not None:
        for box in raw.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0]); coco_cls = int(box.cls[0])
            vd_cls = COCO_TO_VD.get(coco_cls, 99)
            if vd_cls == 99: continue
            cx = ((x1+x2)/2)/W; cy = ((y1+y2)/2)/H
            bw = (x2-x1)/W;     bh = (y2-y1)/H
            dets.append((cx,cy,bw,bh,conf,vd_cls))
    return dets


def run_single_experiment(
    run_name: str,
    tracker,
    frames: list,
    fog_schedule: np.ndarray,
    yolo_model,
    extra_params: dict,
    run_type: str = "ltc",
):
    """Run one tracker on all frames, log to MLflow, return metrics dict."""
    results_log = []
    tracker.reset()

    for fi, img_path in enumerate(frames):
        fog   = float(fog_schedule[fi])
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        H, W  = frame.shape[:2]
        frame_fog = apply_fog(frame, fog)

        yolo_dets = get_yolo_dets(yolo_model, frame_fog, H, W)
        confirmed, ghosts = tracker.process(yolo_dets, fog, fi)

        avg_mem = float(np.mean([g[4] for g in ghosts])) if ghosts else 0.0
        avg_tau = float(np.mean([g[6] for g in ghosts])) if ghosts else 0.0

        results_log.append({
            "frame"  : fi,
            "fog"    : fog,
            "n_yolo" : len(confirmed),
            "n_ghost": len(ghosts),
            "avg_mem": avg_mem,
            "avg_tau": avg_tau,
        })

    bucket_metrics = compute_recovery_pct(results_log)
    flat           = flat_mlflow_metrics(bucket_metrics)

    # Key headline metric
    extreme = bucket_metrics.get("Extreme Fog", {})
    recovery_extreme = extreme.get("recovery", 0.0)
    avg_mem_extreme  = float(np.mean(
        [r["avg_mem"] for r in results_log if r["fog"] >= 0.75]
    )) if any(r["fog"] >= 0.75 for r in results_log) else 0.0
    avg_tau_extreme  = float(np.mean(
        [r["avg_tau"] for r in results_log if r["fog"] >= 0.75 and r["avg_tau"] > 0]
    )) if any(r["fog"] >= 0.75 and r["avg_tau"] > 0 for r in results_log) else 0.0

    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_param("run_type",  run_type)
        mlflow.log_param("n_frames",  len(frames))
        mlflow.log_param("tracker",   getattr(tracker, "name", run_name))
        for k, v in extra_params.items():
            mlflow.log_param(k, v)

        # Log all metrics
        mlflow.log_metrics(flat)
        mlflow.log_metric("recovery_pct_extreme",  recovery_extreme)
        mlflow.log_metric("avg_mem_strength_extreme", avg_mem_extreme)
        mlflow.log_metric("avg_tau_extreme",          avg_tau_extreme)
        mlflow.log_metric("total_yolo_dets",
                          sum(r["n_yolo"]  for r in results_log))
        mlflow.log_metric("total_ghost_boxes",
                          sum(r["n_ghost"] for r in results_log))

        # Save per-frame log as artifact
        import csv, tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                        delete=False, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results_log[0].keys()))
            w.writeheader(); w.writerows(results_log)
            tmp_csv = f.name
        mlflow.log_artifact(tmp_csv, "frame_data")
        os.unlink(tmp_csv)

        run_id = mlflow.active_run().info.run_id

    print(f"  {run_name:<30} recovery_extreme={recovery_extreme:>7.1f}%  "
          f"mem={avg_mem_extreme:.3f}  run_id={run_id[:8]}")

    return {
        "name"             : run_name,
        "type"             : run_type,
        "recovery_extreme" : recovery_extreme,
        "mem_str"          : f"{avg_mem_extreme:.3f}",
        "run_id"           : run_id,
        **flat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir",     type=Path, default=Path("data/demo_frames"))
    parser.add_argument("--max_frames",     type=int,  default=40)
    parser.add_argument("--baselines_only", action="store_true")
    parser.add_argument("--ltc_only",       action="store_true")
    parser.add_argument("--mlflow_uri",     type=str,  default="mlruns")
    args = parser.parse_args()

    frames = load_frames(args.frames_dir, args.max_frames)
    if not frames:
        print(f"[ERROR] No frames found in {args.frames_dir}")
        print("  Run python mlops/save_demo_frames.py first.")
        sys.exit(1)

    fog_schedule = build_fog_schedule(len(frames))
    yolo_model   = load_yolo_model()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = []
    print(f"\nExperiment : {EXPERIMENT_NAME}")
    print(f"Frames     : {len(frames)}")
    print(f"Fog peak   : {fog_schedule.max():.0%}")
    print("=" * 70)

    # ── Baselines ─────────────────────────────────────────────────────────────
    if not args.ltc_only:
        print("\n── Baselines ─────────────────────────────────────────────────────")
        baseline_map = {
            "YOLO-only": (YOLOOnlyBaseline(), {}),
            "SORT"     : (SORTBaseline(max_age=15, min_hits=3, iou_threshold=0.25),
                          {"max_age":15,"min_hits":3,"iou_threshold":0.25}),
            "Velocity" : (VelocityBaseline(max_frames_lost=10),
                          {"max_frames_lost":10}),
        }
        for name, (tracker, params) in baseline_map.items():
            r = run_single_experiment(name, tracker, frames, fog_schedule,
                                      yolo_model, params, run_type="baseline")
            all_results.append(r)

    # ── 12 LTC configs ────────────────────────────────────────────────────────
    if not args.baselines_only:
        print("\n── LTC configurations ────────────────────────────────────────────")
        for (cfg_name, hidden, steps, iou_thr, max_lost) in LTC_CONFIGS:
            tracker = LNNMemoryBank(
                max_objects=40,
                iou_thresh=iou_thr,
                ltc_hidden=hidden,
                ltc_steps=steps,
                max_frames_lost=max_lost,
            )
            params = {
                "ltc_hidden"     : hidden,
                "ltc_steps"      : steps,
                "iou_threshold"  : iou_thr,
                "max_frames_lost": max_lost,
                "ltc_param_count": tracker.param_count,
            }
            r = run_single_experiment(cfg_name, tracker, frames, fog_schedule,
                                      yolo_model, params, run_type="ltc")
            all_results.append(r)

    # ── Champion selection ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    ltc_runs = [r for r in all_results if r["type"] == "ltc"]
    if ltc_runs:
        champion = max(ltc_runs, key=lambda x: x["recovery_extreme"])
        print(f"\nChampion : {champion['name']}")
        print(f"  recovery_extreme = {champion['recovery_extreme']:.1f}%")
        print(f"  run_id           = {champion['run_id']}")

        # Find baseline comparison
        sort_run = next((r for r in all_results if r["name"] == "SORT"), None)
        yolo_run = next((r for r in all_results if r["name"] == "YOLO-only"), None)
        if sort_run:
            ratio = champion["recovery_extreme"] / max(sort_run["recovery_extreme"], 1)
            print(f"\n  LTC vs SORT : {ratio:.1f}× better in extreme fog")
            print(f"  YOLO-only={yolo_run['recovery_extreme']:.0f}%  "
                  f"SORT={sort_run['recovery_extreme']:.0f}%  "
                  f"LTC={champion['recovery_extreme']:.0f}%")

        # Save champion config
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        champion_cfg = {
            "run_name"        : champion["name"],
            "run_id"          : champion["run_id"],
            "recovery_extreme": champion["recovery_extreme"],
        }
        # Find matching LTC config
        for (cfg_name, hidden, steps, iou_thr, max_lost) in LTC_CONFIGS:
            if cfg_name == champion["name"]:
                champion_cfg.update({
                    "ltc_hidden"     : hidden,
                    "ltc_steps"      : steps,
                    "iou_threshold"  : iou_thr,
                    "max_frames_lost": max_lost,
                })
                break
        with open(models_dir / "champion_config.json", "w") as f:
            json.dump(champion_cfg, f, indent=2)
        print(f"\n  Champion config saved → models/champion_config.json")

    # Summary table
    print("\n── All runs ranked by recovery_pct_extreme ──────────────────────")
    print(f"{'Run name':<30} {'Recovery%':>10}  {'MemStr':>8}  {'Type':>10}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: x["recovery_extreme"], reverse=True):
        tag = " ← CHAMPION" if ltc_runs and r["name"] == champion["name"] else ""
        print(f"  {r['name']:<28} {r['recovery_extreme']:>9.1f}%  "
              f"{r['mem_str']:>8}  {r['type']:>10}{tag}")

    print(f"\nMLflow UI: mlflow ui --port 5001")
    print(f"Open:      http://localhost:5001")


if __name__ == "__main__":
    main()
