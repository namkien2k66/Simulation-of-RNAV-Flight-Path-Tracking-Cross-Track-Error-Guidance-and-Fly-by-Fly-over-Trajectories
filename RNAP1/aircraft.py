"""Point-mass 2-D aircraft with a bank-limited, first-order-roll turn model.

State  : x (E), y (N), psi (heading), phi (bank)
Inputs : phi_cmd (commanded bank angle)

    x'   = V sin(psi) + w_E
    y'   = V cos(psi) + w_N
    psi' = g tan(phi) / V               (coordinated turn, V = true airspeed)
    phi' = clip((phi_cmd - phi)/tau, +/- roll_rate_max)

``wind`` is the vector the air mass moves TOWARD, (w_E, w_N) in m/s.
Integration: classic RK4, with phi_cmd held constant over a step.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import G


@dataclass
class AircraftParams:
    V: float = 120.0                 # true airspeed [m/s]  (~233 kt)
    bank_max_deg: float = 25.0       # bank limit [deg]
    roll_tau: float = 1.5            # roll-response time constant [s]
    roll_rate_max_deg: float = 5.0   # roll-rate limit [deg/s]


class Aircraft:
    def __init__(self, params: AircraftParams, x: float, y: float, psi: float,
                 phi: float = 0.0, wind=(0.0, 0.0)):
        self.p = params
        self.state = np.array([x, y, psi, phi], float)
        self.wind = np.asarray(wind, float)

    # -- derivatives -----------------------------------------------------------
    def _deriv(self, s, phi_cmd):
        p = self.p
        _, _, psi, phi = s
        phi_dot = np.clip((phi_cmd - phi) / p.roll_tau,
                          -np.radians(p.roll_rate_max_deg), np.radians(p.roll_rate_max_deg))
        return np.array([
            p.V * np.sin(psi) + self.wind[0],
            p.V * np.cos(psi) + self.wind[1],
            G * np.tan(phi) / p.V,
            phi_dot,
        ])

    def step(self, phi_cmd: float, dt: float) -> None:
        bmax = np.radians(self.p.bank_max_deg)
        u = float(np.clip(phi_cmd, -bmax, bmax))
        s = self.state
        k1 = self._deriv(s, u)
        k2 = self._deriv(s + 0.5 * dt * k1, u)
        k3 = self._deriv(s + 0.5 * dt * k2, u)
        k4 = self._deriv(s + dt * k3, u)
        self.state = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # -- convenience -------------------------------------------------------------
    @property
    def pos(self):
        return self.state[:2].copy()

    @property
    def psi(self):
        return float(self.state[2])

    @property
    def phi(self):
        return float(self.state[3])

    @property
    def ground_velocity(self):
        psi = self.state[2]
        return np.array([self.p.V * np.sin(psi), self.p.V * np.cos(psi)]) + self.wind

    @property
    def ground_speed(self):
        return float(np.hypot(*self.ground_velocity))

    @property
    def track(self):
        v = self.ground_velocity
        return float(np.arctan2(v[0], v[1]))
