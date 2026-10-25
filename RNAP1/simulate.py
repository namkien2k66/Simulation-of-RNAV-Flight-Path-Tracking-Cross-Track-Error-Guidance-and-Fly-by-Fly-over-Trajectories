"""Closed-loop simulation: aircraft + guidance + reference path."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import G, wrap_pi
from .path import ReferencePath
from .aircraft import Aircraft


@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray        # heading [rad]
    track: np.ndarray      # ground track [rad]
    phi: np.ndarray        # bank [rad]
    phi_cmd: np.ndarray    # commanded bank [rad]
    xte: np.ndarray        # cross-track error [m], + = right
    chi_err: np.ndarray    # track angle error [rad]
    s: np.ndarray          # path distance of closest point [m]
    path: ReferencePath
    label: str = ""
    finished: bool = True  # False if t_max reached before the last waypoint


def run_simulation(path: ReferencePath, aircraft: Aircraft, guidance,
                   dt: float = 0.1, t_max: float = 7200.0, label: str = "") -> SimResult:
    log = {k: [] for k in ["t", "x", "y", "psi", "track", "phi", "phi_cmd", "xte", "chi_err", "s"]}
    t, hint, finished = 0.0, None, True
    while True:
        proj = path.project(aircraft.pos, hint)
        hint = proj.j
        a_cmd = guidance.lateral_accel(aircraft, path, proj)
        phi_cmd = float(np.arctan(a_cmd / G))
        phi_cmd = float(np.clip(phi_cmd, -np.radians(aircraft.p.bank_max_deg),
                                np.radians(aircraft.p.bank_max_deg)))
        # log state BEFORE stepping
        log["t"].append(t)
        log["x"].append(aircraft.state[0]); log["y"].append(aircraft.state[1])
        log["psi"].append(aircraft.psi); log["track"].append(aircraft.track)
        log["phi"].append(aircraft.phi); log["phi_cmd"].append(phi_cmd)
        log["xte"].append(proj.xte)
        log["chi_err"].append(float(wrap_pi(aircraft.track - proj.chi)))
        log["s"].append(proj.s)
        if proj.s >= path.s_last_wp:
            break
        if t >= t_max:
            finished = False
            break
        aircraft.step(phi_cmd, dt)
        t += dt
    arr = {k: np.array(v) for k, v in log.items()}
    return SimResult(path=path, label=label, finished=finished, **arr)
