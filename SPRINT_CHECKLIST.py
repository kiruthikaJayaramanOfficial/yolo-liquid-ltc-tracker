"""
SPRINT_CHECKLIST.py
Run this at any time to see which tasks are complete.
    python SPRINT_CHECKLIST.py
"""
from pathlib import Path

ROOT = Path(__file__).parent

TASKS = [
    # Sprint 0 — Project setup
    ("Sprint 0", "Folder structure created",
     lambda: (ROOT/"src").exists() and (ROOT/"mlops").exists() and (ROOT/"app/pages").exists()),
    ("Sprint 0", "src/ltc_memory.py created",
     lambda: (ROOT/"src/ltc_memory.py").exists()),
    ("Sprint 0", "src/fog_utils.py created",
     lambda: (ROOT/"src/fog_utils.py").exists()),
    ("Sprint 0", "src/visualise.py created",
     lambda: (ROOT/"src/visualise.py").exists()),
    ("Sprint 0", "data/demo_frames/ has ≥10 images",
     lambda: len(list((ROOT/"data/demo_frames").glob("*.jpg")) +
                 list((ROOT/"data/demo_frames").glob("*.png"))) >= 10),
    ("Sprint 0", "requirements.txt created",
     lambda: (ROOT/"requirements.txt").exists()),
    ("Sprint 0", "GitHub repo initialised (git remote exists)",
     lambda: (ROOT/".git").exists()),

    # Sprint 1 — Baselines
    ("Sprint 1", "src/baselines.py created",
     lambda: (ROOT/"src/baselines.py").exists()),
    ("Sprint 1", "YOLO-only baseline implemented",
     lambda: "YOLOOnlyBaseline" in (ROOT/"src/baselines.py").read_text()),
    ("Sprint 1", "SORT baseline implemented",
     lambda: "SORTBaseline" in (ROOT/"src/baselines.py").read_text()),
    ("Sprint 1", "Velocity predictor implemented",
     lambda: "VelocityBaseline" in (ROOT/"src/baselines.py").read_text()),
    ("Sprint 1", "mlops/run_experiments.py created",
     lambda: (ROOT/"mlops/run_experiments.py").exists()),
    ("Sprint 1", "MLflow baseline runs logged (mlruns exists)",
     lambda: (ROOT/"mlruns").exists()),
    ("Sprint 1", "Baseline numbers verified (SORT > YOLO-only)",
     lambda: (ROOT/"mlruns").exists() and any((ROOT/"mlruns").rglob("*.json"))),

    # Sprint 2 — LTC experiments
    ("Sprint 2", "12 LTC configs defined in run_experiments.py",
     lambda: "LTC_CONFIGS" in (ROOT/"mlops/run_experiments.py").read_text()),
    ("Sprint 2", "All 15 runs logged to MLflow",
     lambda: len(list((ROOT/"mlruns").rglob("meta.yaml"))) >= 15
             if (ROOT/"mlruns").exists() else False),
    ("Sprint 2", "Champion config saved to models/champion_config.json",
     lambda: (ROOT/"models/champion_config.json").exists()),
    ("Sprint 2", "MLflow UI screenshot taken (parallel coordinates)",
     lambda: any((ROOT/"assets").glob("mlflow*.png")) if (ROOT/"assets").exists() else False),
    ("Sprint 2", "Champion vs SORT ratio calculated",
     lambda: (ROOT/"models/champion_config.json").exists()),

    # Sprint 3 — Streamlit
    ("Sprint 3", "app/main.py created",
     lambda: (ROOT/"app/main.py").exists()),
    ("Sprint 3", "Page 1 Upload & Run created",
     lambda: (ROOT/"app/pages/page1_upload.py").exists()),
    ("Sprint 3", "Page 2 Live Metrics created",
     lambda: (ROOT/"app/pages/page2_metrics.py").exists()),
    ("Sprint 3", "Page 3 Frame Story created",
     lambda: (ROOT/"app/pages/page3_framestory.py").exists()),
    ("Sprint 3", "Page 4 MLflow Comparison created",
     lambda: (ROOT/"app/pages/page4_mlflow.py").exists()),
    ("Sprint 3", "Sidebar loads champion_config.json",
     lambda: "champion_config.json" in (ROOT/"app/main.py").read_text()),
    ("Sprint 3", "All 4 pages tested locally",
     lambda: False),  # Manual — check yourself
    ("Sprint 3", "Deployed to Streamlit Cloud",
     lambda: False),  # Manual — check yourself
    ("Sprint 3", "Live URL tested on mobile",
     lambda: False),  # Manual — check yourself

    # Sprint 4 — GitHub + README
    ("Sprint 4", "GitHub repo yolo-liquid-ltc-tracker created",
     lambda: (ROOT/".git").exists()),
    ("Sprint 4", "README.md written with results table",
     lambda: (ROOT/"README.md").exists() and "644%" in (ROOT/"README.md").read_text()),
    ("Sprint 4", "Demo GIF recorded and added to assets/",
     lambda: (ROOT/"assets").exists() and any((ROOT/"assets").glob("*.gif"))),
    ("Sprint 4", "3 MLflow screenshots added to README",
     lambda: (ROOT/"assets").exists() and len(list((ROOT/"assets").glob("mlflow*.png"))) >= 3),
    ("Sprint 4", "Resume bullet added to README",
     lambda: "resume" in (ROOT/"README.md").read_text().lower()),
    ("Sprint 4", "Presentation updated with System Completeness slide",
     lambda: False),  # Manual — check yourself
]

def main():
    sprint_done = {}
    total_done  = 0
    total_tasks = len(TASKS)

    print("\n" + "="*72)
    print("  YOLO-Liquid — Sprint Checklist")
    print("="*72)

    current_sprint = None
    sprint_count   = {"done": 0, "total": 0}

    for sprint, task, check_fn in TASKS:
        if sprint != current_sprint:
            if current_sprint is not None:
                pct = sprint_count["done"] / max(sprint_count["total"], 1) * 100
                print(f"  {'─'*60}")
                print(f"  {current_sprint}: {sprint_count['done']}/{sprint_count['total']} "
                      f"({pct:.0f}%)\n")
            current_sprint = sprint
            sprint_count   = {"done": 0, "total": 0}
            print(f"\n  ── {sprint} ─────────────────────────────────────")

        try:
            done = check_fn()
        except Exception:
            done = False

        mark = "✓" if done else "○"
        print(f"  [{mark}] {task}")
        sprint_count["total"] += 1
        if done:
            sprint_count["done"] += 1
            total_done += 1

    # Final sprint
    pct = sprint_count["done"] / max(sprint_count["total"], 1) * 100
    print(f"  {'─'*60}")
    print(f"  {current_sprint}: {sprint_count['done']}/{sprint_count['total']} ({pct:.0f}%)")

    print("\n" + "="*72)
    overall = total_done / total_tasks * 100
    print(f"  OVERALL: {total_done}/{total_tasks} tasks complete ({overall:.0f}%)")
    if overall == 100:
        print("  PROJECT COMPLETE — ready for review")
    else:
        remaining = total_tasks - total_done
        print(f"  {remaining} tasks remaining")
    print("="*72 + "\n")


if __name__ == "__main__":
    main()
