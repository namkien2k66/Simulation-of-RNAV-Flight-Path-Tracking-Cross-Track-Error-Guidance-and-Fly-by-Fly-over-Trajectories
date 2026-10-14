"""The five experiments used in the report.  Each writes figures + CSV to ``out``."""
from __future__ import annotations
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .path import build_path, make_route
from .aircraft import Aircraft, AircraftParams
from .guidance import L1Guidance, PDFFGuidance
from .simulate import run_simulation
from .metrics import compute_metrics
from .plotting import plot_trajectory, plot_timeseries
from .geometry import NM

# ---- global defaults (change here, rerun, discuss in the report) --------------
V = 120.0              # m/s  (~233 kt)
BANK_DESIGN = 20.0     # deg  used to size turns
BANK_MAX = 25.0        # deg  aircraft limit
LEG = 45_000.0         # m    default leg length
DT = 0.1               # s


def _ac(x=0.0, y=0.0, psi=0.0, wind=(0.0, 0.0), **kw):
    p = AircraftParams(V=kw.pop("V", V), bank_max_deg=kw.pop("bank_max", BANK_MAX), **kw)
    return Aircraft(p, x, y, psi, 0.0, wind)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        names = []
        for r in rows:                      # union of keys, keep first-seen order
            for k in r:
                if k not in names:
                    names.append(k)
        w = csv.DictWriter(f, fieldnames=names, restval="")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------
def exp1_capture(out):
    """Leg capture from an initial cross-track offset, two guidance laws."""
    wps = make_route(60_000, [], "flyby")
    path = build_path(wps, V, BANK_DESIGN)
    rows, fig_res = [], {}
    for off in [200, 500, 1000, 2000]:
        for g in [PDFFGuidance(), L1Guidance()]:
            res = run_simulation(path, _ac(x=off), g, DT, label=f"{g.name}, offset {off} m")
            m = compute_metrics(res, capture_thresh=0.05 * off)
            overshoot = max(0.0, -float(np.min(res.xte * np.sign(res.xte[0]))))
            rows.append({"guidance": g.name, "offset_m": off,
                         "t_capture_s": round(m["t_capture_s"], 1),
                         "overshoot_m": round(overshoot, 1),
                         "max_bank_deg": round(m["max_bank_deg"], 1)})
            fig_res.setdefault(off, []).append(res)
    _write_csv(os.path.join(out, "exp1_capture.csv"), rows)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, name in zip(axs, ["PD+FF", "L1"]):
        for off, rs in fig_res.items():
            r = [x for x in rs if x.label.startswith(name)][0]
            ax.plot(r.t, r.xte, label=f"offset {off} m")
        ax.set_title(f"Leg capture - {name}"); ax.set_xlabel("Time [s]"); ax.grid(alpha=.3)
    axs[0].set_ylabel("XTE [m]"); axs[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "exp1_capture.png"), dpi=150); plt.close(fig)
    return rows


def exp2_flyby_vs_flyover(out):
    """Fly-by vs fly-over for several turn angles (PD+FF guidance)."""
    angles = [30, 60, 90, 120]
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, ang in zip(axs.ravel(), angles):
        res_list = []
        for kind in ["flyby", "flyover"]:
            wps = make_route(LEG, [ang], kind)
            path = build_path(wps, V, BANK_DESIGN)
            res = run_simulation(path, _ac(), PDFFGuidance(), DT, label=kind)
            res_list.append(res)
            m = compute_metrics(res)
            t = m["turns"][0]
            rows.append({"delta_deg": ang, "kind": kind,
                         "radius_m": round(t["radius_m"]),
                         "turn_xte_max_m": round(t["max_abs_xte_m"], 1),
                         "wp_miss_m": round(t["wp_miss_m"], 1),
                         "wp_miss_analytic_m": round(t["wp_miss_analytic_m"], 1),
                         "max_bank_deg": round(t["max_bank_deg"], 1),
                         "flight_time_s": round(m["duration_s"], 1)})
            if kind == "flyby":
                ref_by = path
        # reference of fly-over drawn too
        ax.plot(ref_by.x * 1e-3, ref_by.y * 1e-3, color="tab:blue", lw=4, alpha=.2, label="ref fly-by")
        ax.plot(path.x * 1e-3, path.y * 1e-3, color="tab:orange", lw=4, alpha=.2, label="ref fly-over")
        for r, c in zip(res_list, ["tab:blue", "tab:orange"]):
            ax.plot(r.x * 1e-3, r.y * 1e-3, color=c, lw=1.2, label=f"flown {r.label}")
        w = ref_by.waypoints[1]
        ax.plot(w.x * 1e-3, w.y * 1e-3, "k^", ms=8, label="WP")
        ax.set_xlim(w.x * 1e-3 - 10, w.x * 1e-3 + 30)
        ax.set_ylim(w.y * 1e-3 - 15, w.y * 1e-3 + 10)
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.3)
        ax.set_title(f"Turn angle {ang} deg"); ax.set_xlabel("East [km]"); ax.set_ylabel("North [km]")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "exp2_flyby_vs_flyover.png"), dpi=150); plt.close(fig)
    _write_csv(os.path.join(out, "exp2_flyby_vs_flyover.csv"), rows)

    # XTE time series for the 90 deg case
    rs = []
    for kind in ["flyby", "flyover"]:
        path = build_path(make_route(LEG, [90], kind), V, BANK_DESIGN)
        rs.append(run_simulation(path, _ac(), PDFFGuidance(), DT, label=f"{kind} 90 deg"))
    plot_timeseries(rs, "Fly-by vs fly-over, 90 deg turn", os.path.join(out, "exp2_timeseries_90deg.png"))
    return rows


def exp3_guidance_comparison(out):
    """Multi-turn route (mixed fly-by / fly-over): L1 vs PD+FF vs PD without preview."""
    wps = make_route([40_000, 45_000, 45_000, 50_000], [70, -110, 60],
                     ["flyby", "flyover", "flyby"])
    path = build_path(wps, V, BANK_DESIGN)
    guidances = [PDFFGuidance(), PDFFGuidance(preview_time=0.0, name="PD+FF (no preview)"),
                 L1Guidance()]
    rs, rows = [], []
    for g in guidances:
        r = run_simulation(path, _ac(x=300.0), g, DT, label=g.name)
        rs.append(r)
        m = compute_metrics(r, capture_thresh=15.0)
        rows.append({"guidance": g.name, "t_capture_s": round(m["t_capture_s"], 1),
                     "max_abs_xte_m": round(m["max_abs_xte_m"], 1),
                     "rms_xte_m": round(m["rms_xte_m"], 1),
                     "max_bank_deg": round(m["max_bank_deg"], 1)})
    _write_csv(os.path.join(out, "exp3_guidance.csv"), rows)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_trajectory(ax, path, rs)
    ax.legend(fontsize=8); ax.set_title("Mixed route: fly-by / fly-over / fly-by")
    fig.tight_layout(); fig.savefig(os.path.join(out, "exp3_route.png"), dpi=150); plt.close(fig)
    plot_timeseries(rs, "Guidance comparison on mixed route", os.path.join(out, "exp3_timeseries.png"))
    return rows


def exp4_parameter_sweeps(out):
    """Effect of airspeed, design bank and guidance bandwidth on a 90 deg fly-by."""
    rows = []
    # (a) airspeed
    for v in [80, 120, 160, 200]:
        path = build_path(make_route(LEG * 1.5, [90], "flyby"), v, BANK_DESIGN)
        r = run_simulation(path, _ac(V=v), PDFFGuidance(), DT, label=f"V={v}")
        t = compute_metrics(r)["turns"][0]
        rows.append({"sweep": "airspeed", "value": v, "radius_m": round(t["radius_m"]),
                     "turn_xte_max_m": round(t["max_abs_xte_m"], 1),
                     "max_bank_deg": round(t["max_bank_deg"], 1)})
    # (b) design bank
    for b in [10, 15, 20, 24]:
        path = build_path(make_route(LEG * 1.5, [90], "flyby"), V, b)
        r = run_simulation(path, _ac(), PDFFGuidance(), DT, label=f"bank={b}")
        t = compute_metrics(r)["turns"][0]
        rows.append({"sweep": "design_bank_deg", "value": b, "radius_m": round(t["radius_m"]),
                     "turn_xte_max_m": round(t["max_abs_xte_m"], 1),
                     "max_bank_deg": round(t["max_bank_deg"], 1)})
    # (c) guidance bandwidth: capture of a 1 km offset
    path = build_path(make_route(80_000, [], "flyby"), V, BANK_DESIGN)
    caps = []
    for w in [0.02, 0.04, 0.06, 0.10, 0.15]:
        r = run_simulation(path, _ac(x=1000.0), PDFFGuidance(omega_n=w), DT, label=f"w={w}")
        m = compute_metrics(r, capture_thresh=50.0)
        ov = max(0.0, -float(np.min(r.xte * np.sign(r.xte[0]))))
        rows.append({"sweep": "omega_n", "value": w, "t_capture_s": round(m["t_capture_s"], 1),
                     "overshoot_m": round(ov, 1), "max_bank_deg": round(m["max_bank_deg"], 1)})
        caps.append(r)
    _write_csv(os.path.join(out, "exp4_sweeps.csv"), rows)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    a = [r for r in rows if r["sweep"] == "airspeed"]
    axs[0].plot([r["value"] for r in a], [r["turn_xte_max_m"] for r in a], "o-")
    axs[0].set_xlabel("Airspeed [m/s]"); axs[0].set_ylabel("Max |XTE| in turn [m]")
    b = [r for r in rows if r["sweep"] == "design_bank_deg"]
    axs[1].plot([r["value"] for r in b], [r["turn_xte_max_m"] for r in b], "o-")
    axs[1].set_xlabel("Design bank angle [deg]"); axs[1].set_ylabel("Max |XTE| in turn [m]")
    for r in caps:
        axs[2].plot(r.t, r.xte, label=r.label)
    axs[2].set_xlabel("Time [s]"); axs[2].set_ylabel("XTE [m]"); axs[2].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "exp4_sweeps.png"), dpi=150); plt.close(fig)
    return rows


def exp5_wind(out):
    """Constant crosswind (wind blows toward East) on a 90 deg fly-by route."""
    rows, rs = [], []
    path = build_path(make_route(LEG, [90], "flyby"), V, BANK_DESIGN)
    for w in [0, 10, 20, 30]:
        for g in [PDFFGuidance(), L1Guidance()]:
            r = run_simulation(path, _ac(wind=(w, 0.0)), g, DT, label=f"{g.name}, wind {w} m/s")
            m = compute_metrics(r)
            rows.append({"guidance": g.name, "wind_ms": w, "wind_kt": round(w / 0.514444),
                         "max_abs_xte_m": round(m["max_abs_xte_m"], 1),
                         "max_xte_nm": round(m["max_abs_xte_m"] / NM, 3),
                         "max_bank_deg": round(m["max_bank_deg"], 1)})
            if g.name == "PD+FF":
                rs.append(r)
    _write_csv(os.path.join(out, "exp5_wind.csv"), rows)
    plot_timeseries(rs, "Crosswind effect (PD+FF)", os.path.join(out, "exp5_wind.png"))
    return rows


ALL = {1: exp1_capture, 2: exp2_flyby_vs_flyover, 3: exp3_guidance_comparison,
       4: exp4_parameter_sweeps, 5: exp5_wind}
