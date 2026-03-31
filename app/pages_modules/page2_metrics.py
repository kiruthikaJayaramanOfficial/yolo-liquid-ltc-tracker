"""app/pages/page2_metrics.py — charts from the last run."""
import numpy as np
import streamlit as st


def render(champion: dict):
    st.title("Live Metrics")
    st.caption("Charts from the most recent Upload & Run session.")

    if "yolo_counts" not in st.session_state:
        st.info("Run the **Upload & Run** page first to populate metrics.")
        return

    yolo_counts = st.session_state["yolo_counts"]
    lnn_counts  = st.session_state["lnn_counts"]
    fog_vals    = st.session_state["fog_vals"]
    frames      = list(range(len(fog_vals)))

    # ── Metric cards ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    recovery_vals = [lnn_counts[i]/max(yolo_counts[i],1)*100 for i in range(len(frames))]
    extreme_frames = [i for i,f in enumerate(fog_vals) if f >= 0.75]

    col1.metric("Peak objects tracked (LNN)", f"{max(lnn_counts)}")
    col2.metric("Min objects (YOLO extreme fog)",
                f"{min(yolo_counts[i] for i in extreme_frames) if extreme_frames else 'N/A'}")
    col3.metric("Peak recovery %",
                f"{max(recovery_vals):.0f}%")
    col4.metric("Champion config",
                champion.get("run_name","LTC_wide_h64"))

    st.divider()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Chart 1: Detection count over frames ─────────────────────────────────
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fog_scaled = [f * max(lnn_counts) * 0.8 for f in fog_vals]
    fig1.add_trace(go.Bar(x=frames, y=yolo_counts, name="YOLO detections",
                          marker_color="rgba(224,82,82,0.85)"), secondary_y=False)
    fig1.add_trace(go.Bar(x=frames, y=lnn_counts, name="LNN total",
                          marker_color="rgba(76,175,118,0.7)"), secondary_y=False)
    fig1.add_trace(go.Scatter(x=frames, y=fog_vals, name="Fog level",
                              line=dict(color="steelblue", width=1.5, dash="dash"),
                              opacity=0.6), secondary_y=True)
    fig1.update_layout(title="Detection count — YOLO vs LNN", barmode="overlay",
                       height=320, margin=dict(t=40,b=20,l=20,r=20))
    fig1.update_yaxes(title_text="Object count", secondary_y=False)
    fig1.update_yaxes(title_text="Fog level", secondary_y=True, range=[0, 2])
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Recovery % ───────────────────────────────────────────────────
    colors = ["#4caf76" if r >= 100 else "#e09a52" if r >= 50 else "#e05252"
              for r in recovery_vals]
    fig2 = go.Figure(go.Bar(x=frames, y=recovery_vals,
                             marker_color=colors, name="Recovery %"))
    fig2.add_hline(y=100, line_dash="dash", line_color="steelblue",
                   annotation_text="100% baseline")
    fig2.update_layout(title="Object recovery % (ghost ÷ YOLO × 100)",
                       height=280, margin=dict(t=40,b=20,l=20,r=20),
                       yaxis_title="Recovery %")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Per-fog-bucket table ──────────────────────────────────────────────────
    st.markdown("#### Per-fog-bucket summary")
    from src.metrics import FOG_BUCKETS
    rows = []
    for bucket, fog_lo, fog_hi in FOG_BUCKETS:
        idx = [i for i,f in enumerate(fog_vals) if fog_lo <= f < fog_hi]
        if not idx: continue
        avg_y = np.mean([yolo_counts[i] for i in idx])
        avg_l = np.mean([lnn_counts[i]  for i in idx])
        rec   = avg_l / max(avg_y, 1) * 100
        rows.append({
            "Condition": bucket,
            "Fog range": f"{fog_lo:.0%}–{fog_hi:.0%}",
            "Avg YOLO dets": f"{avg_y:.1f}",
            "Avg LNN total": f"{avg_l:.1f}",
            "Recovery %": f"{rec:.0f}%",
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
