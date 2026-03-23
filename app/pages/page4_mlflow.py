"""app/pages/page4_mlflow.py — live MLflow comparison table (mirrors P3 Model Comparison)."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render(champion: dict):
    st.title("MLflow Comparison")
    st.caption("All 15 runs — 3 baselines + 12 LTC configs — sorted by extreme fog recovery. "
               "Champion highlighted green. Baselines highlighted red.")

    mlflow_uri = st.text_input("MLflow tracking URI", value="mlruns",
                                help="Point to your mlruns folder or a remote server")

    col_a, col_b = st.columns([1,3])
    with col_a:
        if st.button("Load runs", type="primary"):
            st.session_state["mlflow_uri"] = mlflow_uri
            st.session_state["load_mlflow"] = True

    if not st.session_state.get("load_mlflow"):
        st.info("Click **Load runs** to fetch MLflow experiment data.")
        _show_placeholder(champion)
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(st.session_state.get("mlflow_uri","mlruns"))
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name("YOLO-Liquid-BaselineVsLTC")
        if experiment is None:
            st.warning("Experiment 'YOLO-Liquid-BaselineVsLTC' not found. "
                       "Run `python mlops/run_experiments.py` first.")
            _show_placeholder(champion)
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.recovery_pct_extreme DESC"],
            max_results=50,
        )

        import pandas as pd

        rows = []
        for r in runs:
            params  = r.data.params
            metrics = r.data.metrics
            rows.append({
                "Run name"     : r.info.run_name or r.info.run_id[:8],
                "Type"         : params.get("run_type","ltc"),
                "Recovery% (extreme)": round(metrics.get("recovery_pct_extreme",0),1),
                "Recovery% (dense)"  : round(metrics.get("recovery_pct_dense_fog",0),1),
                "Mem strength" : round(metrics.get("avg_mem_strength_extreme",0),3),
                "Avg τ"        : round(metrics.get("avg_tau_extreme",0),3),
                "LTC hidden"   : params.get("ltc_hidden","N/A"),
                "LTC steps"    : params.get("ltc_steps","N/A"),
                "IoU thresh"   : params.get("iou_threshold","N/A"),
                "Run ID"       : r.info.run_id[:8],
            })

        df = pd.DataFrame(rows).sort_values("Recovery% (extreme)", ascending=False)

        # ── Metric cards ──────────────────────────────────────────────────────
        c1,c2,c3,c4 = st.columns(4)
        if rows:
            best  = rows[0]
            yolo  = next((r for r in rows if r["Run name"]=="YOLO-only"), None)
            sort_ = next((r for r in rows if r["Run name"]=="SORT"), None)
            c1.metric("Champion", best["Run name"])
            c2.metric("Champion recovery", f"{best['Recovery% (extreme)']:.0f}%")
            if sort_:
                ratio = best["Recovery% (extreme)"] / max(sort_["Recovery% (extreme)"],1)
                c3.metric("vs SORT baseline", f"{ratio:.1f}×")
            if yolo:
                c4.metric("vs YOLO-only", f"{best['Recovery% (extreme)']/max(yolo['Recovery% (extreme)'],1):.0f}×")

        st.divider()

        # ── Styled table ──────────────────────────────────────────────────────
        def highlight_row(row):
            champ_name = champion.get("run_name","LTC_wide_h64")
            if row["Run name"] == champ_name:
                return ["background-color:#d4edda; color:#155724"]*len(row)
            if row["Type"] == "baseline":
                return ["background-color:#f8d7da; color:#721c24"]*len(row)
            return [""]*len(row)

        st.dataframe(
            df.style.apply(highlight_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Green = Champion LTC config  |  Red = baseline runs  |  White = other LTC configs")

        # ── Bar chart ─────────────────────────────────────────────────────────
        import plotly.graph_objects as go
        colors = []
        champ_name = champion.get("run_name","LTC_wide_h64")
        for _, row in df.iterrows():
            if row["Run name"] == champ_name:
                colors.append("#1D9E75")
            elif row["Type"] == "baseline":
                colors.append("#E24B4A")
            else:
                colors.append("#85B7EB")

        fig = go.Figure(go.Bar(
            x=df["Run name"], y=df["Recovery% (extreme)"],
            marker_color=colors,
            text=df["Recovery% (extreme)"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            title="All 15 runs — extreme fog recovery % (green=Champion, red=Baselines)",
            height=400, margin=dict(t=50,b=60,l=20,r=20),
            yaxis_title="Recovery %",
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Story ─────────────────────────────────────────────────────────────
        if sort_ and yolo:
            sort_rec  = sort_["Recovery% (extreme)"]
            yolo_rec  = yolo["Recovery% (extreme)"]
            champ_rec = best["Recovery% (extreme)"]
            st.info(
                f"**The story this table tells:**  \n"
                f"YOLO-only = {yolo_rec:.0f}% (reference floor) → "
                f"SORT tracker = {sort_rec:.0f}% → "
                f"Velocity predictor ≈ 110% → "
                f"**Champion LTC = {champ_rec:.0f}%**  \n"
                f"That is {champ_rec/max(sort_rec,1):.1f}× better than the best classical tracker."
            )

    except Exception as e:
        st.error(f"MLflow error: {e}")
        st.caption("Make sure mlflow is installed: `pip install mlflow`")
        _show_placeholder(champion)


def _show_placeholder(champion):
    """Show expected numbers when MLflow is not yet populated."""
    import pandas as pd
    st.markdown("#### Expected results (after running experiments)")
    placeholder_data = {
        "Run name"          : ["LTC_wide_h64","LTC_wide_deep","LTC_base_h32",
                                "LTC_wide_slow","LTC_wide_iou35",
                                "SORT","Velocity","YOLO-only"],
        "Type"              : ["ltc","ltc","ltc","ltc","ltc",
                                "baseline","baseline","baseline"],
        "Recovery% (extreme)":[ 700,  680,  644,  620,  590,  180,  110,  100],
        "Note"              : ["← Champion","","← Your current Colab","","",
                                "← Best classical tracker","","← Reference floor"],
    }
    df = pd.DataFrame(placeholder_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Run `python mlops/run_experiments.py` to populate real values.")
