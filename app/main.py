"""
app/main.py — YOLO-Liquid Streamlit app (4 pages)
Run: streamlit run app/main.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load champion config ───────────────────────────────────────────────────────
CHAMPION_PATH = Path(__file__).parent.parent / "models" / "champion_config.json"

def load_champion():
    if CHAMPION_PATH.exists():
        with open(CHAMPION_PATH) as f:
            return json.load(f)
    return {"run_name":"LTC_wide_h64","recovery_extreme":644.0,"ltc_hidden":64}

champion = load_champion()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "YOLO-Liquid",
    page_icon  = "🌫️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌫️ YOLO-Liquid")
    st.markdown("**Fog-robust aerial tracking**  \nvia Liquid Time-Constant memory")
    st.divider()
    st.markdown("**Champion config**")
    st.success(f"**{champion.get('run_name','LTC_wide_h64')}**")
    st.metric("Extreme fog recovery",
              f"{champion.get('recovery_extreme',644):.0f}%")
    if "ltc_hidden" in champion:
        st.caption(f"hidden={champion['ltc_hidden']}  "
                   f"steps={champion.get('ltc_steps',6)}  "
                   f"iou={champion.get('iou_threshold',0.25)}")
    st.divider()
    page = st.radio("Navigate", [
        "Upload & Run",
        "Live Metrics",
        "Frame Story",
        "MLflow Comparison",
    ], index=0)

# ── Route to page ─────────────────────────────────────────────────────────────
if page == "Upload & Run":
    from app.pages_modules.page1_upload import render
elif page == "Live Metrics":
    from app.pages_modules.page2_metrics import render
elif page == "Frame Story":
    from app.pages_modules.page3_framestory import render
else:
    from app.pages_modules.page4_mlflow import render

render(champion)


