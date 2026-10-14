"""Performance metrics for a simulation run."""
from __future__ import annotations
import numpy as np
from .simulate import SimResult


def compute_metrics(res: SimResult, capture_thresh: float = 25.0, post_turn: float = 3000.0) -> dict:
    """Return a dict of scalar metrics (+ per-turn list).

    capture_thresh : |XTE| below which the leg is considered "captured" [m]
    post_turn      : extra path distance after a transition included in its
                     evaluation window (settling after roll-out) [m]
    """
    t, xte = res.t, res.xte
    idx = np.where(np.abs(xte) <= capture_thresh)[0]
    i_cap = int(idx[0]) if len(idx) else len(t) - 1
    t_cap = float(t[i_cap])
    after = slice(i_cap, None)

    out = {
        "label": res.label,
        "finished": res.finished,
        "t_capture_s": t_cap,
        "max_abs_xte_m": float(np.max(np.abs(xte[after]))),
        "rms_xte_m": float(np.sqrt(np.mean(xte[after] ** 2))),
        "max_bank_deg": float(np.degrees(np.max(np.abs(res.phi)))),
        "duration_s": float(t[-1]),
        "turns": [],
    }
    for tr in res.path.turns:
        m = (res.s >= tr.s_start) & (res.s <= tr.s_end + post_turn)
        if not np.any(m):
            continue
        wx, wy = tr.wp_xy
        miss = float(np.min(np.hypot(res.x - wx, res.y - wy)))
        out["turns"].append({
            "wp": tr.wp_index,
            "kind": tr.kind,
            "delta_deg": float(np.degrees(tr.delta)),
            "radius_m": float(tr.radius),
            "max_abs_xte_m": float(np.max(np.abs(xte[m]))),
            "max_bank_deg": float(np.degrees(np.max(np.abs(res.phi[m])))),
            "wp_miss_m": miss,
            "wp_miss_analytic_m": float(tr.analytic_miss),
            "a_start_m": float(tr.a_start),
            "a_end_m": float(tr.a_end),
        })
    return out
