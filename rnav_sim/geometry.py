"""Basic geometry helpers and conventions.

Conventions (used everywhere in this project)
---------------------------------------------
* Local flat-earth frame, units in metres: x = East, y = North.
* Heading / course chi is measured from North, clockwise positive (aviation
  convention).  Unit vector of a course:  u = (sin chi, cos chi).
* "Right" normal of a course:            n_r = (cos chi, -sin chi).
* Cross-track error (XTE) is positive when the aircraft is to the RIGHT of
  the desired track.
* Curvature kappa and lateral acceleration are positive for RIGHT turns.
"""
from __future__ import annotations
import numpy as np

G = 9.80665          # m/s^2
NM = 1852.0          # metres per nautical mile
KT = 0.514444        # m/s per knot


def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def unit(chi):
    """Unit vector (East, North) for course chi [rad]."""
    return np.array([np.sin(chi), np.cos(chi)])


def right_normal(chi):
    """Unit vector pointing to the right of course chi."""
    return np.array([np.cos(chi), -np.sin(chi)])


def course(p0, p1):
    """Course [rad] of the straight line from p0 to p1."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    return float(np.arctan2(d[0], d[1]))


def turn_radius(V, bank_rad):
    """Steady coordinated-turn radius  R = V^2 / (g tan(phi))."""
    return V ** 2 / (G * np.tan(bank_rad))
