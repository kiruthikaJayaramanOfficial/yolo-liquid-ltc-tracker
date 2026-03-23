"""app/pages/page3_framestory.py — 8-stage fog progression visual narrative."""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fog_utils  import apply_fog, build_fog_schedule, fog_category
from src.ltc_memory import LNNMemoryBank
from src.visualise  import draw_yolo_box, draw_ghost_box

COCO_TO_VD = {0:1,1:10,2:4,3:10,5:9,6:6,7:6}
FONT = cv2.FONT_HERSHEY_SIMPLEX

STAGE_LABELS = [
    "Stage 1: Clear — YOLO active, memory building",
    "Stage 2: Clear — memory slots filling",
    "Stage 3: Fog rising — LNN first ghosts",
    "Stage 4: Dense fog — YOLO degrades",
    "Stage 5: Peak fog — LNN memory only",
    "Stage 6: Still foggy — memory decaying",
    "Stage 7: Clearing — YOLO recovering",
    "Stage 8: Clear again — memory re-anchored",
]


def render(champion: dict):
    st.title("Frame Story")
    st.caption("8-stage visual narrative showing fog progression. "
               "Orange = YOLO detections. Dashed green = LNN memory ghosts.")

    demo_dir = Path(__file__).parent.parent.parent / "data" / "demo_frames"
    frames   = sorted(list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png")))[:40]

    if not frames:
        st.warning("Demo frames not found. Run `python mlops/save_demo_frames.py` first.")
        return

    if st.button("Generate frame story", type="primary"):
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            yolo = UltralyticsYOLO("yolov8n.pt")
            for p in yolo.model.parameters(): p.requires_grad = False
        except Exception:
            yolo = None

        memory_bank = LNNMemoryBank(
            ltc_hidden=champion.get("ltc_hidden",64),
            ltc_steps=champion.get("ltc_steps",6),
            iou_thresh=champion.get("iou_threshold",0.25),
            max_frames_lost=champion.get("max_frames_lost",15),
        )
        memory_bank.reset()
        fog_sched = build_fog_schedule(len(frames))

        # Run all frames to build memory state
        all_results = []
        for fi, img_path in enumerate(frames):
            fog   = float(fog_sched[fi])
            frame = cv2.imread(str(img_path))
            if frame is None: continue
            H, W  = frame.shape[:2]
            frame_fog = apply_fog(frame, fog)

            if yolo:
                raw = yolo(frame_fog, verbose=False, conf=0.15, iou=0.45)[0]
                dets = []
                if raw.boxes is not None:
                    for box in raw.boxes:
                        x1,y1,x2,y2=box.xyxy[0].tolist()
                        conf=float(box.conf[0]); coco_cls=int(box.cls[0])
                        vd_cls=COCO_TO_VD.get(coco_cls,99)
                        if vd_cls==99: continue
                        cx=((x1+x2)/2)/W; cy=((y1+y2)/2)/H
                        bw=(x2-x1)/W; bh=(y2-y1)/H
                        dets.append((cx,cy,bw,bh,conf,vd_cls))
            else:
                dets = [(np.random.uniform(0.1,0.9),np.random.uniform(0.1,0.9),
                         0.05,0.08,0.8,4) for _ in range(max(1,8-int(fog*8)))]

            confirmed, ghosts = memory_bank.process(dets, fog, fi)
            all_results.append((fi, fog, frame_fog, confirmed, ghosts, H, W))

        # Pick 8 representative frames
        n = len(all_results)
        picks = [int(x*n) for x in [0.08,0.20,0.32,0.45,0.55,0.67,0.78,0.92]]
        picks = [min(p,n-1) for p in picks]

        # Render 2×4 grid
        cols1 = st.columns(4)
        cols2 = st.columns(4)
        all_cols = cols1 + cols2

        for ax_col, (stage_idx, pick) in enumerate(zip(STAGE_LABELS, picks)):
            fi, fog, frame_fog, confirmed, ghosts, H, W = all_results[pick]
            vis = cv2.resize(frame_fog.copy(), (480, 270))
            VH, VW = 270, 480

            for (cx,cy,bw,bh,conf,cn) in confirmed:
                x1=max(0,int((cx-bw/2)*VW)); y1=max(0,int((cy-bh/2)*VH))
                x2=min(VW,int((cx+bw/2)*VW)); y2=min(VH,int((cy+bh/2)*VH))
                cv2.rectangle(vis,(x1,y1),(x2,y2),(30,120,255),2)

            for (cx,cy,bw,bh,mem,cls_id,tau,trail) in ghosts:
                draw_ghost_box(vis,cx,cy,bw,bh,str(cls_id),mem,tau,trail,VH,VW)

            # Stage label
            cv2.rectangle(vis,(0,0),(VW,20),(18,20,28),-1)
            label_short = stage_idx.split(":")[0]
            cv2.putText(vis,label_short,(4,14),FONT,0.35,(180,180,200),1)
            cv2.putText(vis,
                        f"Fog {fog:.0%} | YOLO:{len(confirmed)} Ghost:{len(ghosts)}",
                        (4,VH-6),FONT,0.32,(200,200,200),1)

            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            with all_cols[ax_col]:
                st.image(vis_rgb, use_column_width=True,
                         caption=stage_idx)

        st.success("Frame story generated. Orange boxes = YOLO | Dashed green = LNN memory ghosts")
    else:
        st.info("Click **Generate frame story** to render the 8-stage fog progression.")
        st.markdown("""
| Stage | Fog level | What happens |
|-------|-----------|--------------|
| 1–2   | 0%        | YOLO active, LNN memory builds |
| 3     | 20–40%    | Fog rising, first LNN ghosts appear |
| 4     | 60–80%    | YOLO degrades, ghosts filling the gap |
| 5     | 92%       | Peak fog — LNN memory only |
| 6     | 92%       | Memory decaying — drift increases |
| 7     | 40%       | Fog clearing — YOLO recovering |
| 8     | 0%        | Clear — YOLO re-anchors memory positions |
        """)
