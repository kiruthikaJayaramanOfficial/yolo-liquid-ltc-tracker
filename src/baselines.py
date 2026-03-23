"""
src/baselines.py
Three trackers that don't use LTC — Sprint 1.
  1. YOLOOnlyBaseline   — no memory, just count detections
  2. SORTBaseline       — Kalman filter + Hungarian algorithm
  3. VelocityBaseline   — simple cx+vx physics, no neural network
"""
from __future__ import annotations
import numpy as np


# ── 1. YOLO-only ──────────────────────────────────────────────────────────────
class YOLOOnlyBaseline:
    """No memory. Just returns YOLO detections as-is."""
    name = "YOLO-only"

    def reset(self): pass

    def process(self, yolo_dets, fog_level, frame_id):
        confirmed = list(yolo_dets)
        ghosts    = []
        return confirmed, ghosts


# ── 2. SORT tracker ───────────────────────────────────────────────────────────
try:
    from sort import Sort as _Sort
    _SORT_AVAILABLE = True
except ImportError:
    _SORT_AVAILABLE = False


class SORTBaseline:
    """
    Kalman filter + Hungarian algorithm.
    Falls back to a minimal pure-Python implementation if sort-track not installed.
    Install: pip install sort-track
    """
    name = "SORT"

    def __init__(self, max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.25):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracker      = None
        self._use_lib      = _SORT_AVAILABLE

    def reset(self):
        if self._use_lib:
            from sort import Sort
            self._tracker = Sort(max_age=self.max_age, min_hits=self.min_hits,
                                 iou_threshold=self.iou_threshold)
        else:
            self._tracker = _MinimalSORT(max_age=self.max_age, iou_threshold=self.iou_threshold)

    def process(self, yolo_dets, fog_level, frame_id):
        if self._tracker is None:
            self.reset()

        if not yolo_dets:
            if self._use_lib:
                self._tracker.update(np.empty((0,5)))
            else:
                self._tracker.update([])
            return [], []

        # Convert normalised (cx,cy,w,h,conf,cls) → xyxy for SORT
        boxes = []
        for (cx,cy,bw,bh,conf,cls_id) in yolo_dets:
            x1 = cx - bw/2; y1 = cy - bh/2
            x2 = cx + bw/2; y2 = cy + bh/2
            boxes.append([x1,y1,x2,y2,conf])

        if self._use_lib:
            tracks = self._tracker.update(np.array(boxes))
        else:
            tracks = self._tracker.update(boxes)

        confirmed = []
        for t in tracks:
            x1,y1,x2,y2 = t[:4]
            cx = (x1+x2)/2; cy = (y1+y2)/2
            w  = x2-x1;     h  = y2-y1
            confirmed.append((cx,cy,w,h,0.9,0))   # conf=0.9 placeholder, cls=0

        return confirmed, []   # SORT produces no ghost boxes


# ── Minimal pure-Python SORT fallback ─────────────────────────────────────────
class _KalmanBox:
    """Minimal 1-D Kalman state for a bounding box (no external dependency)."""
    _id_counter = 0

    def __init__(self, box):
        _KalmanBox._id_counter += 1
        self.id = _KalmanBox._id_counter
        x1,y1,x2,y2,_ = box
        cx=(x1+x2)/2; cy=(y1+y2)/2; w=x2-x1; h=y2-y1
        self.state = np.array([cx,cy,w,h,0.,0.,0.,0.], dtype=float)
        self.hits       = 1
        self.no_updates = 0

    def predict(self):
        self.state[0] += self.state[4]
        self.state[1] += self.state[5]
        self.state[2] += self.state[6]
        self.state[3] += self.state[7]
        self.no_updates += 1

    def update(self, box):
        x1,y1,x2,y2,_ = box
        cx=(x1+x2)/2; cy=(y1+y2)/2; w=x2-x1; h=y2-y1
        self.state[4] = cx - self.state[0]
        self.state[5] = cy - self.state[1]
        self.state[0:4] = [cx,cy,w,h]
        self.hits      += 1
        self.no_updates = 0

    def to_xyxy(self):
        cx,cy,w,h = self.state[:4]
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def _iou(b1, b2):
    ax1,ay1,ax2,ay2 = b1; bx1,by1,bx2,by2 = b2
    ix = max(0,min(ax2,bx2)-max(ax1,bx1))
    iy = max(0,min(ay2,by2)-max(ay1,by1))
    inter = ix*iy
    return inter/max((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter,1e-6)


def _hungarian_match(cost):
    """Greedy matching (good enough for small sets)."""
    pairs = []
    used_r, used_c = set(), set()
    flat = sorted(((cost[r,c],r,c) for r in range(cost.shape[0])
                   for c in range(cost.shape[1])), reverse=True)
    for v,r,c in flat:
        if r in used_r or c in used_c: continue
        if v < 0.25: break
        pairs.append((r,c)); used_r.add(r); used_c.add(c)
    return pairs


class _MinimalSORT:
    def __init__(self, max_age=15, iou_threshold=0.25):
        self.max_age  = max_age
        self.iou_thr  = iou_threshold
        self.trackers: list[_KalmanBox] = []
        _KalmanBox._id_counter = 0

    def update(self, detections):
        for t in self.trackers: t.predict()

        if not detections:
            self.trackers = [t for t in self.trackers if t.no_updates <= self.max_age]
            return [[*t.to_xyxy(), t.id] for t in self.trackers if t.hits >= 3]

        trk_boxes = [t.to_xyxy() for t in self.trackers]
        if trk_boxes:
            cost = np.zeros((len(detections), len(trk_boxes)))
            for d,det in enumerate(detections):
                for k,trk in enumerate(trk_boxes):
                    cost[d,k] = _iou(det[:4], trk)
            pairs = _hungarian_match(cost)
        else:
            pairs = []

        matched_d, matched_t = set(), set()
        for d,k in pairs:
            self.trackers[k].update(detections[d])
            matched_d.add(d); matched_t.add(k)

        for d,det in enumerate(detections):
            if d not in matched_d:
                self.trackers.append(_KalmanBox(det))

        self.trackers = [t for i,t in enumerate(self.trackers)
                         if i in matched_t or t.no_updates <= self.max_age]
        return [[*t.to_xyxy(), t.id] for t in self.trackers if t.hits >= 3]


# ── 3. Velocity predictor ─────────────────────────────────────────────────────
class VelocityBaseline:
    """
    Simple cx+vx physics. No neural network.
    LTC must beat this to prove the ODE adds value.
    """
    name = "Velocity"

    def __init__(self, max_frames_lost: int = 10):
        self.max_frames_lost = max_frames_lost
        self.slots: dict[int, dict] = {}
        self.next_id = 0
        self._iou_thresh = 0.25

    def reset(self):
        self.slots = {}; self.next_id = 0

    @staticmethod
    def _iou(b1, b2):
        def c(b): return b[0]-b[2]/2,b[1]-b[3]/2,b[0]+b[2]/2,b[1]+b[3]/2
        ax1,ay1,ax2,ay2=c(b1); bx1,by1,bx2,by2=c(b2)
        ix=max(0,min(ax2,bx2)-max(ax1,bx1)); iy=max(0,min(ay2,by2)-max(ay1,by1))
        inter=ix*iy
        return inter/max((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter,1e-6)

    def process(self, yolo_dets, fog_level, frame_id):
        matched, confirmed = set(), []

        for (cx,cy,bw,bh,conf,cls_id) in yolo_dets:
            best_id, best_iou = None, self._iou_thresh
            for tid, s in self.slots.items():
                if s["cls"] != cls_id: continue
                iou = self._iou((cx,cy,bw,bh),(s["cx"],s["cy"],s["w"],s["h"]))
                if iou > best_iou: best_iou, best_id = iou, tid
            if best_id is not None:
                s = self.slots[best_id]
                s["vx"], s["vy"] = cx-s["cx"], cy-s["cy"]
                s["cx"],s["cy"],s["w"],s["h"] = cx,cy,bw,bh
                s["frames_lost"] = 0
                matched.add(best_id)
            else:
                tid = self.next_id; self.next_id += 1
                self.slots[tid] = dict(cls=cls_id,cx=cx,cy=cy,w=bw,h=bh,
                                       vx=0.,vy=0.,frames_lost=0,strength=1.0)
                matched.add(tid)
            confirmed.append((cx,cy,bw,bh,conf,cls_id))

        ghosts, dead = [], []
        for tid, s in self.slots.items():
            if tid in matched: continue
            s["frames_lost"] += 1
            s["vx"] *= 0.85; s["vy"] *= 0.85
            s["cx"] += s["vx"]; s["cy"] += s["vy"]
            s["strength"] *= 0.85
            if s["strength"] > 0.05:
                ghosts.append((s["cx"],s["cy"],s["w"],s["h"],s["strength"],s["cls"],0.0,[]))
            if s["frames_lost"] > self.max_frames_lost or s["strength"] < 0.03:
                dead.append(tid)
        for tid in dead: self.slots.pop(tid, None)

        return confirmed, ghosts
