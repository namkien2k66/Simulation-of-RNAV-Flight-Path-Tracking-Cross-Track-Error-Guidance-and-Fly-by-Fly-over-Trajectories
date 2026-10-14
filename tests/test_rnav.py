"""Sanity tests.  Run:  python tests/test_rnav.py   (or pytest, if installed)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from rnav_sim import *
from rnav_sim.geometry import wrap_pi, unit, right_normal

V, BANK = 120.0, 20.0


def test_path_is_continuous_and_tangent():
    for kind in ["flyby", "flyover"]:
        for ang in [30, 90, 150, -60, -120]:
            p = build_path(make_route(45_000, [ang], kind), V, BANK)
            assert np.max(np.hypot(*np.diff(p.P, axis=0).T)) < 6.0          # no jumps
            assert np.max(np.abs(wrap_pi(np.diff(p.chi)))) < 0.01            # no course kinks


def test_flyby_geometry_matches_theory():
    for ang in [30, 60, 90, 120]:
        p = build_path(make_route(45_000, [ang], "flyby"), V, BANK)
        t = p.turns[0]
        w = np.array(t.wp_xy)
        miss = np.min(np.hypot(*(p.P - w).T))
        assert abs(miss - t.analytic_miss) < 3.0, (ang, miss, t.analytic_miss)
        assert abs(t.a_start - t.radius * np.tan(np.radians(ang) / 2)) < 1e-6


def test_flyover_passes_wp_and_rejoins_leg():
    for ang in [30, 60, 90, 120, -90]:
        wps = make_route(45_000, [ang], "flyover")
        p = build_path(wps, V, BANK)
        t = p.turns[0]
        w = np.array(t.wp_xy)
        assert np.min(np.hypot(*(p.P - w).T)) < 3.0
        # after the transition every sample must lie on the 2nd leg line
        chi1 = np.arctan2(*(np.array([wps[2].x, wps[2].y]) - w)[[0, 1]])
        after = p.P[(p.s > t.s_end + 1)][:2000]
        lat = (after - w) @ right_normal(chi1)
        assert np.max(np.abs(lat)) < 0.05, (ang, np.max(np.abs(lat)))


def test_xte_sign_right_positive():
    p = build_path(make_route(30_000, [], "flyby"), V, BANK)       # north-bound leg
    assert p.project([+100.0, 5000.0]).xte > 99.0                  # east = right
    assert p.project([-100.0, 5000.0]).xte < -99.0


def test_leg_capture():
    p = build_path(make_route(60_000, [], "flyby"), V, BANK)
    for g in [PDFFGuidance(), L1Guidance()]:
        ac = Aircraft(AircraftParams(), 800.0, 0.0, 0.0)
        r = run_simulation(p, ac, g, 0.1)
        assert r.finished and abs(r.xte[-1]) < 1.0, (g.name, r.xte[-1])


def test_turn_tracking_accuracy():
    for kind in ["flyby", "flyover"]:
        p = build_path(make_route(45_000, [90], kind), V, BANK)
        r = run_simulation(p, Aircraft(AircraftParams(), 0, 0, 0.0), PDFFGuidance(), 0.1)
        assert compute_metrics(r)["max_abs_xte_m"] < 60.0
        assert np.degrees(np.max(np.abs(r.phi))) <= 25.0 + 1e-6


def test_short_leg_is_rejected():
    try:
        build_path(make_route(5_000, [90, 90], "flyby"), V, BANK)
    except ValueError:
        return
    raise AssertionError("expected ValueError for too-short leg")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
