"""Lateral guidance laws.  Both return a commanded LATERAL ACCELERATION
(+ = to the right, m/s^2) which the simulator converts to a bank command
    phi_cmd = atan(a_cmd / g).

1) L1Guidance  - "carrot" / pure-pursuit type law
       a = 2 Vg^2 / L * sin(eta)
   eta = angle between ground-velocity vector and the line of sight to a
   reference point located ``L`` metres ahead of the closest path point.
   Look-ahead:  L = max(L_min, T_L * Vg).
   Linearised, it behaves as a 2nd-order system with  w_n = sqrt(2) Vg / L,
   zeta = 0.707.  Weakness: it "cuts" curves (steady-state offset on arcs).

2) PDFFGuidance - cascaded cross-track controller + curvature feed-forward
       v_des = clip(-k_e * XTE, +/- Vg sin(theta_max))     (limited intercept angle)
       a     = Vg^2 * kappa_ref + k_v * (v_des - XTE_dot)
   with XTE_dot = Vg sin(chi - chi_path).
   Gains from desired natural frequency / damping:
       k_v = 2 zeta w_n ,   k_e = w_n / (2 zeta)   =>   XTE'' + 2 zeta w_n XTE' + w_n^2 XTE = 0
   ``preview_time`` shifts the curvature feed-forward ahead along the path so
   that the roll-in starts early enough to compensate roll lag.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import wrap_pi


@dataclass
class L1Guidance:
    T_L: float = 10.0        # look-ahead time [s]
    L_min: float = 300.0     # minimum look-ahead distance [m]
    name: str = "L1"

    def lateral_accel(self, ac, path, proj):
        v = ac.ground_velocity
        Vg = float(np.hypot(*v))
        L = max(self.L_min, self.T_L * Vg)
        target = path.point_at(proj.s + L)
        los = target - ac.pos
        dist = max(float(np.hypot(*los)), 1.0)
        chi_g = np.arctan2(v[0], v[1])
        chi_los = np.arctan2(los[0], los[1])
        eta = float(wrap_pi(chi_los - chi_g))
        return 2.0 * Vg ** 2 / dist * np.sin(eta)


@dataclass
class PDFFGuidance:
    omega_n: float = 0.06        # natural frequency of the XTE loop [rad/s]
    zeta: float = 0.9            # damping ratio
    theta_max_deg: float = 45.0  # max intercept angle
    preview_time: float = 3.0    # curvature preview [s]
    name: str = "PD+FF"

    def lateral_accel(self, ac, path, proj):
        v = ac.ground_velocity
        Vg = float(np.hypot(*v))
        chi_g = np.arctan2(v[0], v[1])
        chi_err = float(wrap_pi(chi_g - proj.chi))
        xte_dot = Vg * np.sin(chi_err)
        k_v = 2.0 * self.zeta * self.omega_n
        k_e = self.omega_n / (2.0 * self.zeta)
        vmax = Vg * np.sin(np.radians(self.theta_max_deg))
        v_des = float(np.clip(-k_e * proj.xte, -vmax, vmax))
        kappa_ff = path.kappa_at(proj.s + Vg * self.preview_time)
        return Vg ** 2 * kappa_ff + k_v * (v_des - xte_dot)
