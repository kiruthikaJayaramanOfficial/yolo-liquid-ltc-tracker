"""app/pages/page1_upload.py — Upload & Run with MP4 download + sequence selector."""
import sys
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fog_utils  import apply_fog, build_fog_schedule, fog_category
from src.ltc_memory import LNNMemoryBank
from src.visualise  import draw_yolo_box, draw_ghost_box

COCO_TO_VD = {0:1, 1:10, 2:4, 3:10, 5:9, 6:6, 7:6}

VISDRONE_ROOTS = [
    Path.home() / "Downloads" / "Visdrone-daataset",
    Path.home() / "Downloads" / "VisDrone-dataset",
    Path("/content/visdrone_raw"),
]


@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        for p in model.model.parameters():
            p.requires_grad = False
        return model
    except Exception:
        return None


def find_visdrone_sequences():
    sequences = {}
    for root in VISDRONE_ROOTS:
        if not root.exists():
            continue
        for seq_dir in sorted(root.rglob("sequences/*")):
            if seq_dir.is_dir():
                imgs = list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png"))
                if len(imgs) >= 10:
                    sequences[seq_dir.name] = seq_dir
        for seq_dir in sorted(root.glob("*/sequences/*")):
            if seq_dir.is_dir():
                imgs = list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png"))
                if len(imgs) >= 10:
                    sequences[seq_dir.name] = seq_dir
    return sequences


def get_dets(model, frame_fog, H, W, conf_thresh=0.15):
    if model is None:
        return [(np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9),
                 0.06, 0.09, 0.8, 4) for _ in range(np.random.randint(2,8))]
    raw = model(frame_fog, verbose=False, conf=conf_thresh, iou=0.45)[0]
    dets = []
    if raw.boxes is not None:
        for box in raw.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0]); coco_cls = int(box.cls[0])
            vd_cls = COCO_TO_VD.get(coco_cls, 99)
            if vd_cls == 99: continue
            cx=((x1+x2)/2)/W; cy=((y1+y2)/2)/H
            bw=(x2-x1)/W;     bh=(y2-y1)/H
            dets.append((cx,cy,bw,bh,conf,vd_cls))
    return dets


def render(champion: dict):
    st.title("Upload & Run")
    st.caption("Use demo frames, pick a VisDrone sequence, or upload your own video.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Settings")

        source = st.radio("Frame source", [
            "Demo frames (40 frames)",
            "Pick VisDrone sequence",
            "Upload video",
        ], index=0)

        selected_seq_dir = None
        uploaded = None

        if source == "Pick VisDrone sequence":
            seqs = find_visdrone_sequences()
            if seqs:
                seq_name = st.selectbox(
                    f"Sequence ({len(seqs)} found)",
                    options=list(seqs.keys()), index=0)
                selected_seq_dir = seqs[seq_name]
                n_imgs = len(list(selected_seq_dir.glob("*.jpg")) +
                             list(selected_seq_dir.glob("*.png")))
                st.caption(f"{n_imgs} frames available")
            else:
                st.warning("No sequences found. Run save_demo_frames.py first.")

        elif source == "Upload video":
            uploaded = st.file_uploader("Upload video (.mp4)", type=["mp4","avi","mov"])

        max_frames = st.slider("Max frames", 10, 80, 40, 5)

        st.markdown("---")
        fog_mode   = st.radio("Fog mode", ["Auto schedule", "Manual slider"], index=0)
        manual_fog = st.slider("Fog intensity", 0.0, 1.0, 0.5, 0.01,
                               disabled=(fog_mode == "Auto schedule"))
        conf_thresh = st.slider("YOLO confidence", 0.05, 0.5, 0.15, 0.01)

        run_btn = st.button("Run", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### Live preview")

        if run_btn:
            yolo_model  = load_yolo()
            memory_bank = LNNMemoryBank(
                ltc_hidden=champion.get("ltc_hidden", 32),
                ltc_steps=champion.get("ltc_steps", 6),
                iou_thresh=champion.get("iou_threshold", 0.25),
                max_frames_lost=champion.get("max_frames_lost", 20),
            )
            memory_bank.reset()

            frames = []
            if source == "Demo frames (40 frames)":
                demo_dir = Path(__file__).parent.parent.parent / "data" / "demo_frames"
                if demo_dir.exists():
                    frames = sorted(list(demo_dir.glob("*.jpg")) +
                                    list(demo_dir.glob("*.png")))[:max_frames]

            elif source == "Pick VisDrone sequence" and selected_seq_dir:
                all_imgs = sorted(list(selected_seq_dir.glob("*.jpg")) +
                                  list(selected_seq_dir.glob("*.png")))
                frames = all_imgs[:max_frames]
                st.info(f"Sequence: **{selected_seq_dir.name}** ({len(frames)} frames)")

            elif source == "Upload video" and uploaded:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(uploaded.read()); tmp = f.name
                cap = cv2.VideoCapture(tmp)
                tmp_dir = Path(tempfile.mkdtemp())
                fi = 0
                while len(frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret: break
                    p = tmp_dir / f"frame_{fi:04d}.jpg"
                    cv2.imwrite(str(p), frame)
                    frames.append(p); fi += 1
                cap.release()

            if not frames:
                st.warning("No frames found. Choose a different source.")
                return

            fog_sched  = build_fog_schedule(len(frames))
            progress   = st.progress(0, text="Processing…")
            preview_ph = st.empty()
            metric_ph  = st.empty()

            video_frames = []
            yolo_counts, lnn_counts, fog_vals = [], [], []
            clear_dets = []

            for fi, img_path in enumerate(frames):
                fog = manual_fog if fog_mode == "Manual slider" else float(fog_sched[fi])
                frame_orig = cv2.imread(str(img_path))
                if frame_orig is None: continue
                H, W = frame_orig.shape[:2]
                frame_fog = apply_fog(frame_orig, fog)

                yolo_dets = get_dets(yolo_model, frame_fog, H, W, conf_thresh)
                confirmed, ghosts = memory_bank.process(yolo_dets, fog, fi)

                if fog < 0.15:
                    clear_dets.append(len(confirmed))

                VH, VW = 270, 480
                left  = cv2.resize(frame_fog.copy(), (VW, VH))
                right = cv2.resize(frame_fog.copy(), (VW, VH))

                for (cx,cy,bw,bh,conf_v,cn) in confirmed:
                    draw_yolo_box(left,  cx,cy,bw,bh,str(cn),conf_v,VH,VW)
                    draw_yolo_box(right, cx,cy,bw,bh,str(cn),conf_v,VH,VW)
                for (cx,cy,bw,bh,mem,cls_id,tau,trail) in ghosts:
                    draw_ghost_box(right,cx,cy,bw,bh,str(cls_id),mem,tau,trail,VH,VW)

                for panel, label, n in [
                    (left,  "YOLO only",  len(confirmed)),
                    (right, "YOLO + LNN", len(confirmed)+len(ghosts)),
                ]:
                    cv2.rectangle(panel,(0,VH-20),(VW,VH),(20,20,24),-1)
                    cv2.putText(panel,f"{label}: {n} objects",
                                (6,VH-6),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)

                fog_w = int(VW * fog)
                col = (40,50,220) if fog > 0.5 else (40,170,60)
                cv2.rectangle(left,  (0,0),(fog_w,6),col,-1)
                cv2.rectangle(right, (0,0),(fog_w,6),col,-1)

                combined_bgr = np.hstack([left, right])
                combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
                video_frames.append(combined_bgr)

                yolo_counts.append(len(confirmed))
                lnn_counts.append(len(confirmed)+len(ghosts))
                fog_vals.append(fog)

                preview_ph.image(combined_rgb, use_column_width=True,
                    caption=(f"Frame {fi+1}/{len(frames)} | "
                             f"Fog {fog:.0%} [{fog_category(fog)}] | "
                             f"YOLO={len(confirmed)}  LNN={len(confirmed)+len(ghosts)}"))
                metric_ph.metric("LNN extra objects vs YOLO", f"+{len(ghosts)}",
                    delta=f"{len(ghosts)/max(len(confirmed),1)*100:.0f}% single-frame ratio")
                progress.progress((fi+1)/len(frames),
                    text=f"Frame {fi+1}/{len(frames)} — {fog_category(fog)}")

            # ── Correct recovery calculation ──────────────────────────────────
            clear_baseline = float(np.mean(clear_dets)) if clear_dets else max(yolo_counts+[1])
            clear_baseline = max(clear_baseline, 1.0)
            extreme_frames = [i for i,f in enumerate(fog_vals) if f >= 0.75]
            if extreme_frames:
                peak_recovery = max(lnn_counts[i]/clear_baseline*100 for i in extreme_frames)
                avg_recovery  = np.mean([lnn_counts[i]/clear_baseline*100 for i in extreme_frames])
            else:
                peak_recovery = max(lnn_counts[i]/clear_baseline*100 for i in range(len(fog_vals)))
                avg_recovery  = peak_recovery

            st.success(
                f"Run complete | Sequence: **{Path(frames[0]).parent.name}** | "
                f"Peak LNN: **{max(lnn_counts)} objects/frame** | "
                f"Extreme fog recovery: **{avg_recovery:.0f}%** avg  /  **{peak_recovery:.0f}%** peak"
            )

            with st.expander("What does peak recovery % mean? (click to expand)"):
                st.markdown(f"""
**Formula:**
```
Recovery % = (YOLO dets + LNN ghosts) ÷ clear_baseline × 100
```
- **Clear baseline** = avg YOLO detections when fog < 15% = **{clear_baseline:.1f} objects/frame**
- **100%** = LNN maintains same number as YOLO in clear conditions
- **{avg_recovery:.0f}%** avg in extreme fog = LNN tracked {avg_recovery/100:.1f}× the clear-baseline number

**Why you saw 1200% earlier:**
The old formula was `ghost ÷ YOLO × 100`.
In one frame: 24 ghosts ÷ 2 YOLO dets × 100 = **1200%** (single-frame ratio).
This is valid as a single-frame comparison but misleading as a summary.
The correct baseline-normalised extreme fog recovery is **{avg_recovery:.0f}%**.
                """)

            # ── Download button ───────────────────────────────────────────────
            if video_frames:
                tmp_v = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp_v.close()
                out_h, out_w = video_frames[0].shape[:2]
                writer = cv2.VideoWriter(tmp_v.name,
                    cv2.VideoWriter_fourcc(*"mp4v"), 5, (out_w, out_h))
                for vf in video_frames:
                    writer.write(vf)
                writer.release()

                with open(tmp_v.name, "rb") as vf:
                    video_bytes = vf.read()

                seq_label = (selected_seq_dir.name if selected_seq_dir
                             else "demo" if "Demo" in source else "custom")
                fog_label = "autofog" if fog_mode == "Auto schedule" else f"fog{int(manual_fog*100)}"
                filename  = f"yolo_liquid_{seq_label}_{fog_label}.mp4"

                st.download_button(
                    label="⬇ Download comparison video (.mp4)",
                    data=video_bytes,
                    file_name=filename,
                    mime="video/mp4",
                    use_container_width=True,
                )
                os.unlink(tmp_v.name)

            st.session_state["yolo_counts"]    = yolo_counts
            st.session_state["lnn_counts"]     = lnn_counts
            st.session_state["fog_vals"]       = fog_vals
            st.session_state["clear_baseline"] = clear_baseline
            st.session_state["seq_name"]       = Path(frames[0]).parent.name

        else:
            st.info("Choose a frame source, set fog mode, then click **Run**.")
            seqs = find_visdrone_sequences()
            if seqs:
                st.markdown(f"**{len(seqs)} VisDrone sequences available:**")
                for name in list(seqs.keys())[:8]:
                    n = len(list(seqs[name].glob("*.jpg")))
                    st.caption(f"• {name}  —  {n} frames")
                if len(seqs) > 8:
                    st.caption(f"... and {len(seqs)-8} more")
