"""src/visualise.py — draw YOLO boxes and LNN ghost boxes on frames."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def mem_color(m: float):
    """Green → yellow → red as memory fades (BGR)."""
    return (0, min(255, int(m * 280 + 60)), min(255, int((1 - m) * 380)))


def draw_yolo_box(img, cx, cy, bw, bh, label, conf, H, W):
    x1 = max(0, int((cx - bw/2)*W)); y1 = max(0, int((cy - bh/2)*H))
    x2 = min(W, int((cx + bw/2)*W)); y2 = min(H, int((cy + bh/2)*H))
    if x2 <= x1 or y2 <= y1: return
    cv2.rectangle(img, (x1,y1),(x2,y2),(30,120,255),2)
    txt = f"{label[:7]} {conf:.2f}"
    (tw,th),_ = cv2.getTextSize(txt,FONT,0.40,1)
    cv2.rectangle(img,(x1,max(0,y1-th-4)),(x1+tw+3,y1),(30,120,255),-1)
    cv2.putText(img,txt,(x1+2,y1-2),FONT,0.40,(255,255,255),1)


def draw_ghost_box(img, cx, cy, bw, bh, cls_name, mem, tau, trail, H, W):
    x1 = max(0, int((cx - bw/2)*W)); y1 = max(0, int((cy - bh/2)*H))
    x2 = min(W, int((cx + bw/2)*W)); y2 = min(H, int((cy + bh/2)*H))
    if x2 <= x1 or y2 <= y1: return
    COL = mem_color(mem)
    d = 12
    for i in range(x1,x2,d*2): cv2.line(img,(i,y1),(min(i+d,x2),y1),COL,2)
    for i in range(x1,x2,d*2): cv2.line(img,(i,y2),(min(i+d,x2),y2),COL,2)
    for i in range(y1,y2,d*2): cv2.line(img,(x1,i),(x1,min(i+d,y2)),COL,2)
    for i in range(y1,y2,d*2): cv2.line(img,(x2,i),(x2,min(i+d,y2)),COL,2)
    bar_y = y1-15 if y1>19 else y2+3
    bw_px = x2-x1; fill = max(1,int(bw_px*mem))
    cv2.rectangle(img,(x1,bar_y),(x2,bar_y+7),(35,35,40),-1)
    cv2.rectangle(img,(x1,bar_y),(x1+fill,bar_y+7),COL,-1)
    lbl = f"{cls_name[:6]} {mem*100:.0f}%"
    cv2.putText(img,lbl,(x1+2,bar_y-2),FONT,0.35,COL,1)
    if len(trail) >= 2:
        pts = [(int(tcx*W),int(tcy*H)) for (_,tcx,tcy) in trail
               if 0<=int(tcx*W)<W and 0<=int(tcy*H)<H]
        for i in range(1,len(pts)):
            cv2.line(img,pts[i-1],pts[i],(0,120,70),1)
        if pts: cv2.circle(img,pts[-1],3,COL,-1)
