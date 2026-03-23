"""
src/ltc_memory.py
LTC Cell, ObjectMemorySlot, LNNMemoryBank
Accepts ltc_hidden, ltc_steps, max_frames_lost as params for MLflow experiments.
"""
import numpy as np
import torch
import torch.nn as nn


class LTCCell(nn.Module):
    """Liquid Time-Constant Cell — Hasani et al. NeurIPS 2021."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, ode_steps: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steps = ode_steps
        self.W_in   = nn.Linear(input_dim, hidden_dim)
        self.W_rec  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.E_rev  = nn.Parameter(torch.zeros(hidden_dim))
        self.tau_net  = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Softplus())
        self.gate_net = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Sigmoid())
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.orthogonal_(self.W_rec.weight)

    def forward(self, x, state=None):
        if state is None:
            state = torch.zeros(x.size(0), self.hidden_dim)
        dt, tau_log = 1.0 / self.steps, []
        for _ in range(self.steps):
            combined = torch.cat([x, state], dim=-1)
            tau  = self.tau_net(combined) + 0.1
            gate = self.gate_net(combined)
            f    = gate * (torch.tanh(self.W_in(x)) + torch.tanh(self.W_rec(state)) + self.E_rev)
            state = state + dt * ((-state + f) / tau)
            tau_log.append(tau.mean().item())
        return state, float(np.mean(tau_log))


class ObjectMemorySlot:
    def __init__(self, cls_id, cx, cy, w, h, conf, frame_id):
        self.cls_id = cls_id
        self.cx, self.cy = cx, cy
        self.w,   self.h  = w, h
        self.vx = self.vy = 0.0
        self.conf        = conf
        self.ltc_state   = None
        self.memory_str  = 1.0
        self.frames_lost = 0
        self.tau_val     = 1.0
        self.born_frame  = frame_id
        self.trail       = [(frame_id, cx, cy)]

    def update(self, cx, cy, w, h, conf, frame_id):
        self.vx, self.vy = cx - self.cx, cy - self.cy
        self.cx, self.cy = cx, cy
        self.w,  self.h  = w, h
        self.conf        = conf
        self.memory_str  = 1.0
        self.frames_lost = 0
        self.trail.append((frame_id, cx, cy))
        if len(self.trail) > 12:
            self.trail.pop(0)

    def predict(self, fog_level, ltc, frame_id):
        self.frames_lost += 1
        self.vx *= 0.85; self.vy *= 0.85
        self.cx += self.vx; self.cy += self.vy
        x_in = torch.tensor([[self.cx, self.cy, self.w, self.h,
                               0.0, fog_level, self.frames_lost / 10.0]])
        with torch.no_grad():
            new_state, tau = ltc(x_in, self.ltc_state)
        self.ltc_state = new_state.detach()
        self.tau_val   = tau
        self.memory_str *= np.exp(-1.0 / max(tau * 3.0, 0.5) * 0.15)
        self.trail.append((frame_id, self.cx, self.cy))
        if len(self.trail) > 12:
            self.trail.pop(0)
        return self.memory_str > 0.05


class LNNMemoryBank:
    """
    IoU-based memory bank. Params exposed for MLflow sweeps:
      ltc_hidden      — hidden dim of LTCCell (default 32)
      ltc_steps       — ODE steps per forward (default 6)
      iou_thresh      — match threshold (default 0.25)
      max_frames_lost — evict after N frames lost (default 15)
    """

    def __init__(
        self,
        max_objects: int = 40,
        iou_thresh: float = 0.25,
        ltc_hidden: int = 32,
        ltc_steps: int = 6,
        max_frames_lost: int = 15,
    ):
        self.ltc = LTCCell(input_dim=7, hidden_dim=ltc_hidden, ode_steps=ltc_steps)
        self.ltc.eval()
        self.slots          = {}
        self.next_id        = 0
        self.max_obj        = max_objects
        self.iou_thresh     = iou_thresh
        self.max_frames_lost = max_frames_lost

    @staticmethod
    def _iou(b1, b2):
        def corners(b): return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
        ax1,ay1,ax2,ay2 = corners(b1); bx1,by1,bx2,by2 = corners(b2)
        ix = max(0, min(ax2,bx2) - max(ax1,bx1))
        iy = max(0, min(ay2,by2) - max(ay1,by1))
        inter = ix * iy
        return inter / max((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter, 1e-6)

    def _match(self, cx, cy, w, h, cls_id):
        best_id, best_iou = None, self.iou_thresh
        for tid, s in self.slots.items():
            if s.cls_id != cls_id: continue
            iou = self._iou((cx,cy,w,h),(s.cx,s.cy,s.w,s.h))
            if iou > best_iou: best_iou, best_id = iou, tid
        return best_id

    def process(self, yolo_dets, fog_level, frame_id):
        matched, confirmed = set(), []
        for (cx,cy,bw,bh,conf,cls_id) in yolo_dets:
            tid = self._match(cx,cy,bw,bh,cls_id)
            if tid is not None:
                self.slots[tid].update(cx,cy,bw,bh,conf,frame_id); matched.add(tid)
            elif len(self.slots) < self.max_obj:
                tid = self.next_id; self.next_id += 1
                sl  = ObjectMemorySlot(cls_id,cx,cy,bw,bh,conf,frame_id)
                x_in = torch.tensor([[cx,cy,bw,bh,conf,fog_level,0.0]])
                with torch.no_grad(): st,tau = self.ltc(x_in,None)
                sl.ltc_state = st.detach(); sl.tau_val = tau
                self.slots[tid] = sl; matched.add(tid)
            confirmed.append((cx,cy,bw,bh,conf,cls_id))

        ghosts, dead = [], []
        for tid, sl in self.slots.items():
            if tid in matched: continue
            alive = sl.predict(fog_level, self.ltc, frame_id)
            if alive:
                ghosts.append((sl.cx,sl.cy,sl.w,sl.h,sl.memory_str,sl.cls_id,sl.tau_val,sl.trail[-5:]))
            if sl.frames_lost > self.max_frames_lost or sl.memory_str < 0.03:
                dead.append(tid)
        for tid in dead: self.slots.pop(tid, None)
        return confirmed, ghosts

    def reset(self):
        self.slots = {}; self.next_id = 0

    @property
    def param_count(self):
        return sum(p.numel() for p in self.ltc.parameters())
