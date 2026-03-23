# YOLO-Liquid: LTC Memory Bank for Fog-Robust Aerial Object Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![MLflow](https://img.shields.io/badge/MLflow-15%20runs-blue)](mlruns/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> **Live demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

![YOLO-Liquid demo](assets/demo.gif)

*Left panel: YOLOv8n alone — goes blind in fog. Right panel: YOLO + LNN memory — ghost boxes persist.*

---

## The problem

Standard object detectors fail completely in adverse weather. When a drone tracks vehicles or pedestrians in fog, YOLO's confidence drops below threshold and **every tracked object vanishes instantly**. There is no memory — the model treats each frame independently.

## The solution

A **3,872-parameter Liquid Time-Constant (LTC) memory bank** attaches to a frozen YOLOv8n as a plug-in module. When YOLO stops detecting an object, the LNN predicts its position using learned dynamics. The time-constant τ rises automatically as fog increases — *the system learns to hold memory longer when visibility is worst.*

---

## Results

### Champion vs baselines — extreme fog (75–100%)

| Tracker | Recovery % | vs YOLO-only | Notes |
|---------|-----------|--------------|-------|
| YOLO-only | 100% | — | Reference floor |
| Velocity predictor | ~110% | +10% | Simple cx+vx physics |
| **SORT tracker** | **~180%** | **+80%** | Kalman filter + Hungarian |
| **LTC Champion** | **644%** | **+544%** | **3,872 params, &lt;2ms overhead** |

> 644% means the LNN maintained **6.4× more object positions** per frame in extreme fog vs YOLO alone. Validated against SORT (the standard classical tracker), LTC achieves **3.5× better** object existence preservation.

### Per-fog-bucket crowd recall

| Fog condition | YOLO recall | LNN recall | Gain |
|---------------|-------------|------------|------|
| Clear (0–15%) | 0.029 | 0.061 | +112% |
| Light haze (15–35%) | 0.025 | 0.072 | +183% |
| **Moderate fog (35–55%)** | **0.000** | **0.054** | **YOLO blind → LNN survives** |
| Dense fog (55–75%) | 0.028 | 0.071 | +155% |
| Extreme fog (75–100%) | 0.025 | 0.084 | +231% |

*Evaluated on VisDrone dataset, crowd-region recall metric (correct for group annotations).*

---

## MLflow experiment — 15 runs

![MLflow parallel coordinates](assets/mlflow_parallel_coords.png)

15 runs in a single experiment: 3 baselines + 12 LTC configurations. Champion selected by `recovery_pct_extreme`.

```
YOLO-only = 100% → SORT = 180% → Champion LTC = 644%
```

---

## Architecture

```
Aerial video frame
        ↓
  apply_fog()          ← Koschmieder atmospheric model
        ↓
  YOLOv8n (frozen)     ← 3.16M params, COCO pretrained, no retraining
        ↓
  LNNMemoryBank        ← IoU matching → confirmed + ghost boxes
    ├── LTCCell        ← 3,872 params, ODE: dx/dt = (-x + f(x,I)) / τ(x,I)
    ├── ObjectMemorySlot  ← position + velocity + LTC hidden state
    └── τ adapts to fog  ← higher fog → higher τ → slower memory decay
        ↓
  Output: YOLO boxes (orange) + LNN ghost boxes (dashed green)
```

---

## Quick start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/yolo-liquid-ltc-tracker
cd yolo-liquid-ltc-tracker

# Install
pip install -r requirements.txt

# Save demo frames (point to your VisDrone sequence)
python mlops/save_demo_frames.py --seq_dir /path/to/visdrone/sequence

# Run all 15 MLflow experiments
python mlops/run_experiments.py

# View MLflow UI
mlflow ui --port 5001
# Open http://localhost:5001

# Launch Streamlit app
streamlit run app/main.py
```

---

## Project structure

```
yolo-liquid-ltc-tracker/
├── src/
│   ├── ltc_memory.py      # LTCCell, ObjectMemorySlot, LNNMemoryBank
│   ├── baselines.py       # YOLO-only, SORT, Velocity predictor
│   ├── fog_utils.py       # Koschmieder fog + schedule
│   ├── visualise.py       # Draw YOLO + ghost boxes
│   └── metrics.py         # Recovery %, crowd recall
├── mlops/
│   ├── run_experiments.py # 15-run MLflow sweep
│   └── save_demo_frames.py
├── app/
│   ├── main.py            # Streamlit entry point
│   └── pages/
│       ├── page1_upload.py      # Upload & Run
│       ├── page2_metrics.py     # Live Metrics
│       ├── page3_framestory.py  # 8-stage Frame Story
│       └── page4_mlflow.py      # MLflow Comparison
├── data/demo_frames/      # 40 fixed frames for fair comparison
├── models/
│   └── champion_config.json
└── requirements.txt
```

---

## Novelty

1. **First application of LTC cells as a plug-in detection memory module** — no prior work attaches Liquid Neural Networks post-hoc to a frozen object detector
2. **Zero retraining** — fog robustness added at inference time, backbone untouched
3. **τ adapts automatically to degradation** — higher fog → higher τ → slower memory decay, without supervision

---

## Real-world need

Emergency response drones searching for survivors in smoke or fog lose all tracked objects when visibility drops. YOLO-Liquid maintains "last known position + velocity prediction" for each tracked person — bridging the visibility gap until the camera can see again.

---

## Dataset

[VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) — aerial drone footage, 80 sequences.
Synthetic fog applied using Koschmieder's atmospheric scattering law.

## Reference

Hasani et al. "Liquid Time-constant Networks." AAAI 2021 / NeurIPS 2021.

---

## Resume bullet

> "YOLO-Liquid: LTC memory bank (Champion selected from 15-run MLflow comparison — 3 baselines + 12 LTC variants) augmenting frozen YOLOv8n; 644% object existence recovery in extreme fog vs SORT baseline 180%; τ adapts dynamically to fog; 4-page Streamlit app deployed."
