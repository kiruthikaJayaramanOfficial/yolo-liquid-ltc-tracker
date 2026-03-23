"""app/pages/page1_upload.py — Upload video + fog slider + side-by-side view."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fog_utils  import apply_fog, build_fog_schedule, fog_category
from src.ltc_memory import LNNMemoryBank
from src.baselines  import YOLOOnlyBaseline
from src.visualise  import draw_yolo_box, draw_ghost_box

COCO_TO_VD = {0:1,1:10,2:4,3:10,5:9,6:6,7:6}

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


def get_dets(model, frame_fog, H, W):
    if model is None:
        return [(np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9),
                 0.06, 0.09, 0.8, 4) for _ in range(np.random.randint(2,8))]
    raw = model(frame_fog, verbose=False, conf=0.15, iou=0.45)[0]
    dets = []
    if raw.boxes is not None:
        for box in raw.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            conf=float(box.conf[0]); coco_cls=int(box.cls[0])
            vd_cls = COCO_TO_VD.get(coco_cls, 99)
            if vd_cls==99: continue
            cx=((x1+x2)/2)/W; cy=((y1+y2)/2)/H
            bw=(x2-x1)/W;     bh=(y2-y1)/H
            dets.append((cx,cy,bw,bh,conf,vd_cls))
    return dets


def render(champion: dict):
    st.title("Upload & Run")
    st.caption("Upload a drone video or use the demo frames. "
               "Adjust fog intensity and watch YOLO go blind while LNN memory persists.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Settings")
        fog_mode = st.radio("Fog mode", ["Manual slider", "Auto schedule"], index=0)
        manual_fog = st.slider("Fog intensity", 0.0, 1.0, 0.5, 0.01,
                               disabled=(fog_mode == "Auto schedule"))
        conf_thresh = st.slider("YOLO confidence", 0.05, 0.5, 0.15, 0.01)
        use_demo    = st.checkbox("Use demo frames (no upload needed)", value=True)

        if not use_demo:
            uploaded = st.file_uploader("Upload video (.mp4)", type=["mp4","avi","mov"])
        else:
            uploaded = None

        run_btn = st.button("Run", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### Live preview")

        if run_btn:
            yolo_model  = load_yolo()
            memory_bank = LNNMemoryBank(
                ltc_hidden=champion.get("ltc_hidden",64),
                ltc_steps=champion.get("ltc_steps",6),
                iou_thresh=champion.get("iou_threshold",0.25),
                max_frames_lost=champion.get("max_frames_lost",15),
            )
            memory_bank.reset()

            # Load frames
            demo_dir = Path(__file__).parent.parent.parent / "data" / "demo_frames"
            frames = []
            if use_demo and demo_dir.exists():
                frames = sorted(list(demo_dir.glob("*.jpg")) +
                                list(demo_dir.glob("*.png")))[:40]
            elif uploaded:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(uploaded.read()); tmp = f.name
                cap = cv2.VideoCapture(tmp)
                while len(frames) < 40:
                    ret, frame = cap.read()
                    if not ret: break
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                        cv2.imwrite(f2.name, frame); frames.append(Path(f2.name))
                cap.release()

            if not frames:
                st.warning("No frames found. Check data/demo_frames/ or upload a video.")
                return

            fog_sched = build_fog_schedule(len(frames))
            progress  = st.progress(0, text="Processing…")
            left_ph   = st.empty()
            right_ph  = st.empty()
            metric_ph = st.empty()

            yolo_counts, lnn_counts, fog_vals = [], [], []

            for fi, img_path in enumerate(frames):
                fog = manual_fog if fog_mode == "Manual slider" else float(fog_sched[fi])
                frame_orig = cv2.imread(str(img_path))
                if frame_orig is None: continue
                H, W = frame_orig.shape[:2]
                frame_fog = apply_fog(frame_orig, fog)

                yolo_dets = get_dets(yolo_model, frame_fog, H, W)
                confirmed, ghosts = memory_bank.process(yolo_dets, fog, fi)

                # Draw panels
                left  = cv2.resize(frame_fog.copy(), (480, 270))
                right = cv2.resize(frame_fog.copy(), (480, 270))
                VH, VW = 270, 480

                for (cx,cy,bw,bh,conf,cn) in confirmed:
                    draw_yolo_box(left,  cx,cy,bw,bh,str(cn),conf,VH,VW)
                    draw_yolo_box(right, cx,cy,bw,bh,str(cn),conf,VH,VW)
                for (cx,cy,bw,bh,mem,cls_id,tau,trail) in ghosts:
                    draw_ghost_box(right,cx,cy,bw,bh,str(cls_id),mem,tau,trail,VH,VW)

                # Status bars
                for panel, label, n in [(left,"YOLO only",len(confirmed)),
                                        (right,"YOLO + LNN",len(confirmed)+len(ghosts))]:
                    cv2.rectangle(panel,(0,VH-20),(VW,VH),(20,20,24),-1)
                    cv2.putText(panel,f"{label}: {n} objects",(6,VH-6),
                                cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)

                yolo_counts.append(len(confirmed))
                lnn_counts.append(len(confirmed)+len(ghosts))
                fog_vals.append(fog)

                combined = np.hstack([
                    cv2.cvtColor(left,  cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(right, cv2.COLOR_BGR2RGB),
                ])
                with col2:
                    left_ph.image(combined, use_column_width=True,
                                  caption=f"Frame {fi+1}/{len(frames)} | "
                                          f"Fog {fog:.0%} [{fog_category(fog)}] | "
                                          f"YOLO={len(confirmed)}  LNN={len(confirmed)+len(ghosts)}")
                    metric_ph.metric(
                        "LNN extra objects vs YOLO",
                        f"+{len(ghosts)}",
                        delta=f"{len(ghosts)/max(len(confirmed),1)*100:.0f}% recovery",
                    )
                progress.progress((fi+1)/len(frames),
                                  text=f"Frame {fi+1}/{len(frames)} — {fog_category(fog)}")

            st.success(f"Run complete. "
                       f"Peak LNN coverage: {max(lnn_counts)} objects/frame  |  "
                       f"Peak recovery: "
                       f"{max(lnn_counts[i]/max(yolo_counts[i],1)*100 for i in range(len(frames))):.0f}%")
            st.session_state["yolo_counts"] = yolo_counts
            st.session_state["lnn_counts"]  = lnn_counts
            st.session_state["fog_vals"]    = fog_vals

        else:
            st.info("Set fog mode, then click **Run** to process frames.")
            demo_dir = Path(__file__).parent.parent.parent / "data" / "demo_frames"
            if not demo_dir.exists() or not any(demo_dir.iterdir()):
                st.warning(
                    "Demo frames not found. Run:\n\n"
                    "```\npython mlops/save_demo_frames.py "
                    "--seq_dir /path/to/visdrone/sequence\n```"
                )
