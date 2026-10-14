"""Reference path generation: waypoints -> legs + fly-by / fly-over transitions.

The ideal (reference) path is built geometrically and stored as a dense
polyline (spacing ``ds`` metres).  For every sample we keep the course
``chi`` and the signed curvature ``kappa`` (+ = right turn).  This makes the
guidance code independent of the path type (line or arc): it only needs

    * project(position)  -> closest point, signed XTE, path course, curvature
    * point_at(s)        -> look-ahead point at path distance s
    * kappa_at(s)        -> curvature preview

Transition geometry
-------------------
Turn radius is designed with a nominal bank angle:  R = V^2 / (g tan(phi_d)).

FLY-BY  (turn anticipation):
    d  = R tan(|Delta|/2)
    The turn starts d metres BEFORE the waypoint, is a single circular arc
    tangent to both legs, and the waypoint is never overflown.
    Closest approach to WP:  R (1/cos(|Delta|/2) - 1).

FLY-OVER:
    1. fly exactly over the waypoint,
    2. arc #1: turn (same direction) through |Delta| + beta  (overshoots the
       new course by the intercept angle beta),
    3. straight segment of length  Ls = R (2 cos(beta) - 1 - cos|Delta|)/sin(beta)
       (flown on heading chi_leg2 +/- beta),
    4. arc #2: turn back (opposite direction) through beta and roll out
       tangent on the next leg.
    beta = min(beta_nominal, 0.9 * acos(cos^2(|Delta|/2)))  guarantees Ls >= 0.
    (Derivation is in the README / report appendix.)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from .geometry import wrap_pi, unit, right_normal, course, turn_radius


# --------------------------------------------------------------------------
# data classes
# --------------------------------------------------------------------------
@dataclass
class Waypoint:
    x: float                 # East  [m]
    y: float                 # North [m]
    kind: str = "flyby"      # "flyby" | "flyover"  (ignored for first/last WP)
    name: str = ""


@dataclass
class TurnInfo:
    wp_index: int
    kind: str
    delta: float             # signed turn angle [rad], + = right
    radius: float            # design turn radius [m]
    a_start: float           # distance used BEFORE the WP on the incoming leg [m]
    a_end: float             # distance used AFTER  the WP on the outgoing leg [m]
    analytic_miss: float     # analytic closest approach of ref. path to WP [m]
    beta: float = 0.0        # intercept angle (fly-over) [rad]
    straight_len: float = 0.0
    s_start: float = 0.0     # path distance at start of transition
    s_end: float = 0.0       # path distance at end of transition
    wp_xy: Tuple[float, float] = (0.0, 0.0)


@dataclass
class Projection:
    j: int                   # index of the polyline segment
    q: np.ndarray            # closest point on path (E, N)
    xte: float               # signed cross-track error [m], + = right of track
    chi: float               # path course at closest point [rad]
    kappa: float             # path curvature at closest point [1/m], + = right
    s: float                 # path distance of closest point [m]


# --------------------------------------------------------------------------
# sampling primitives
# --------------------------------------------------------------------------
def _sample_line(p0, p1, ds):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    L = float(np.hypot(*(p1 - p0)))
    n = max(2, int(np.ceil(L / ds)) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    chi = np.full(n, course(p0, p1))
    kap = np.zeros(n)
    return pts, chi, kap


def _sample_arc(center, R, chi0, dtheta, ds):
    """Arc of radius R around ``center`` starting at heading chi0, turning by
    dtheta (signed, + = right).  Position: C - s*R*n_r(heading)."""
    s = 1.0 if dtheta >= 0 else -1.0
    n = max(2, int(np.ceil(R * abs(dtheta) / ds)) + 1)
    th = np.linspace(0.0, dtheta, n)
    h = chi0 + th
    pts = np.asarray(center, float)[None, :] - s * R * np.stack([np.cos(h), -np.sin(h)], axis=1)
    kap = np.full(n, s / R)
    return pts, h, kap


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------
def _flyby(wp, chi0, chi1, R, ds):
    delta = float(wrap_pi(chi1 - chi0))
    s = 1.0 if delta >= 0 else -1.0
    d = R * np.tan(abs(delta) / 2.0)
    T1 = wp - d * unit(chi0)                       # turn start
    C = T1 + s * R * right_normal(chi0)            # arc centre
    arcs = [_sample_arc(C, R, chi0, delta, ds)]
    info = TurnInfo(
        wp_index=-1, kind="flyby", delta=delta, radius=R,
        a_start=d, a_end=d,
        analytic_miss=R * (1.0 / np.cos(abs(delta) / 2.0) - 1.0),
    )
    return T1, arcs, info


def _flyover(wp, chi0, chi1, R, beta_nom, ds):
    delta = float(wrap_pi(chi1 - chi0))
    s = 1.0 if delta >= 0 else -1.0
    D = abs(delta)
    beta_max = np.arccos(np.cos(D / 2.0) ** 2)
    beta = float(min(beta_nom, 0.9 * beta_max))
    th1 = D + beta
    chi_int = chi0 + s * th1                       # intercept heading
    C1 = wp + s * R * right_normal(chi0)
    arc1 = _sample_arc(C1, R, chi0, s * th1, ds)
    E1 = arc1[0][-1]
    Ls = R * (2.0 * np.cos(beta) - 1.0 - np.cos(D)) / np.sin(beta)
    Ls = max(Ls, 0.0)
    P2 = E1 + Ls * unit(chi_int)
    C2 = P2 - s * R * right_normal(chi_int)        # centre of the opposite turn
    arc2 = _sample_arc(C2, R, chi_int, -s * beta, ds)
    parts = [arc1]
    if Ls > 1e-3:
        parts.append(_sample_line(E1, P2, ds))
    parts.append(arc2)
    E2 = arc2[0][-1]
    info = TurnInfo(
        wp_index=-1, kind="flyover", delta=delta, radius=R,
        a_start=0.0, a_end=float((E2 - wp) @ unit(chi1)),
        analytic_miss=0.0, beta=beta, straight_len=float(Ls),
    )
    return np.asarray(wp, float), parts, info


# --------------------------------------------------------------------------
# reference path
# --------------------------------------------------------------------------
class ReferencePath:
    def __init__(self, x, y, chi, kappa, turns: List[TurnInfo],
                 waypoints: Sequence[Waypoint], idx_last_wp: int, R: float):
        self.P = np.column_stack([x, y])
        self.x, self.y = self.P[:, 0], self.P[:, 1]
        self.chi = wrap_pi(chi)
        self.kappa = np.asarray(kappa, float)
        seg = np.diff(self.P, axis=0)
        self.s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        self.turns = turns
        self.waypoints = list(waypoints)
        self.R = R
        self.idx_last_wp = idx_last_wp
        self.s_last_wp = float(self.s[idx_last_wp])
        for t in self.turns:                      # convert stored indices -> s
            t.s_start = float(self.s[int(t.s_start)])
            t.s_end = float(self.s[int(t.s_end)])

    @property
    def s_end(self) -> float:
        return float(self.s[-1])

    # -- projection ---------------------------------------------------------
    def project(self, p, hint: Optional[int] = None, back: int = 40, fwd: int = 800) -> Projection:
        p = np.asarray(p, float)
        nseg = len(self.P) - 1
        if hint is None:
            j0, j1 = 0, nseg
        else:
            j0, j1 = max(0, hint - back), min(nseg, hint + fwd)
        A = self.P[j0:j1]
        B = self.P[j0 + 1:j1 + 1]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
        t = np.clip(((p - A) * AB).sum(axis=1) / L2, 0.0, 1.0)
        Q = A + t[:, None] * AB
        d2 = ((p - Q) ** 2).sum(axis=1)
        k = int(np.argmin(d2))
        j = j0 + k
        tk = float(t[k])
        q = Q[k]
        # signed distance: right of the (segment) direction is positive
        Lk = np.sqrt(L2[k])
        n_r = np.array([AB[k, 1], -AB[k, 0]]) / Lk
        xte = float(np.copysign(np.sqrt(d2[k]), (p - A[k]) @ n_r))
        chi = float(self.chi[j] + tk * wrap_pi(self.chi[j + 1] - self.chi[j]))
        s = float(self.s[j] + tk * (self.s[j + 1] - self.s[j]))
        return Projection(j=j, q=q, xte=xte, chi=chi, kappa=float(self.kappa[j]), s=s)

    # -- look-ahead helpers ---------------------------------------------------
    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.s_end))
        return np.array([np.interp(s, self.s, self.x), np.interp(s, self.s, self.y)])

    def kappa_at(self, s: float) -> float:
        i = int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))
        return float(self.kappa[i])


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------
def build_path(waypoints: Sequence[Waypoint], V: float, bank_design_deg: float = 20.0,
               ds: float = 5.0, intercept_deg: float = 30.0,
               extension: float = 6000.0) -> ReferencePath:
    """Build the reference path through ``waypoints``.

    V               : design true airspeed [m/s] (sets the turn radius)
    bank_design_deg : nominal bank angle used to size the turns
    ds              : polyline sample spacing [m]
    intercept_deg   : nominal intercept angle for fly-over transitions
    extension       : straight extension after the last WP (look-ahead room) [m]
    """
    pts = np.array([[w.x, w.y] for w in waypoints], float)
    n = len(pts)
    if n < 2:
        raise ValueError("need at least 2 waypoints")
    R = turn_radius(V, np.radians(bank_design_deg))
    chi_leg = [course(pts[i], pts[i + 1]) for i in range(n - 1)]
    len_leg = [float(np.hypot(*(pts[i + 1] - pts[i]))) for i in range(n - 1)]

    # 1) transitions at interior waypoints
    trans = {}   # wp index -> (T_in, parts, info)
    for i in range(1, n - 1):
        delta = float(wrap_pi(chi_leg[i] - chi_leg[i - 1]))
        if abs(delta) < np.radians(0.5):
            continue                                   # (almost) straight: no turn
        kind = waypoints[i].kind
        if kind == "flyby":
            T_in, parts, info = _flyby(pts[i], chi_leg[i - 1], chi_leg[i], R, ds)
        elif kind == "flyover":
            T_in, parts, info = _flyover(pts[i], chi_leg[i - 1], chi_leg[i], R,
                                         np.radians(intercept_deg), ds)
        else:
            raise ValueError(f"unknown waypoint kind {kind!r}")
        info.wp_index = i
        info.wp_xy = (float(pts[i, 0]), float(pts[i, 1]))
        trans[i] = (T_in, parts, info)

    # 2) feasibility: transitions must fit on the legs
    for i in range(n - 1):
        used_start = trans[i][2].a_end if i in trans else 0.0
        used_end = trans[i + 1][2].a_start if (i + 1) in trans else 0.0
        if used_start + used_end > len_leg[i] + 1e-6:
            raise ValueError(
                f"leg {i}->{i+1} is too short ({len_leg[i]:.0f} m) for the transitions "
                f"({used_start:.0f} + {used_end:.0f} m). Lengthen the leg, lower V or "
                f"increase the design bank angle.")

    # 3) sample the whole path
    X: List[np.ndarray] = []; CH: List[np.ndarray] = []; KA: List[np.ndarray] = []
    count = 0

    def add(pc):
        nonlocal count
        p_, c_, k_ = pc
        if count > 0:                       # drop duplicated joint sample
            p_, c_, k_ = p_[1:], c_[1:], k_[1:]
        X.append(p_); CH.append(c_); KA.append(k_)
        count += len(p_)

    cur = pts[0].copy()
    turns: List[TurnInfo] = []
    for i in range(n - 1):
        if (i + 1) in trans:
            T_in, parts, info = trans[i + 1]
            add(_sample_line(cur, T_in, ds))
            info.s_start = count - 1            # (index for now, converted later)
            for part in parts:
                add(part)
            info.s_end = count - 1
            turns.append(info)
            cur = X[-1][-1].copy()
        else:
            add(_sample_line(cur, pts[i + 1], ds))
            cur = pts[i + 1].copy()
    idx_last_wp = count - 1
    add(_sample_line(cur, cur + extension * unit(chi_leg[-1]), ds))

    return ReferencePath(np.vstack(X)[:, 0], np.vstack(X)[:, 1],
                         np.concatenate(CH), np.concatenate(KA),
                         turns, waypoints, idx_last_wp, R)


# --------------------------------------------------------------------------
# convenience: build a route from leg lengths + turn angles
# --------------------------------------------------------------------------
def make_route(leg_len: Union[float, Sequence[float]],
               turns_deg: Sequence[float],
               kinds: Union[str, Sequence[str]] = "flyby",
               start=(0.0, 0.0), start_course_deg: float = 0.0) -> List[Waypoint]:
    """Create waypoints: start heading ``start_course_deg``, then after each leg
    turn by ``turns_deg[k]`` (+ right / - left).  len(turns)+1 legs."""
    m = len(turns_deg)
    legs = [leg_len] * (m + 1) if np.isscalar(leg_len) else list(leg_len)
    if len(legs) != m + 1:
        raise ValueError("need len(turns_deg)+1 leg lengths")
    kinds_l = [kinds] * m if isinstance(kinds, str) else list(kinds)
    wps = [Waypoint(float(start[0]), float(start[1]), name="WP0")]
    chi = np.radians(start_course_deg)
    p = np.array(start, float)
    for k in range(m + 1):
        p = p + legs[k] * unit(chi)
        kind = kinds_l[k] if k < m else "flyby"
        wps.append(Waypoint(float(p[0]), float(p[1]), kind=kind, name=f"WP{k+1}"))
        if k < m:
            chi = chi + np.radians(turns_deg[k])
    return wps
