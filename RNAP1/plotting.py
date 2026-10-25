"""Matplotlib helpers (Agg backend, figures are saved to files)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(ax, path, results=None, km=True, zoom=None, show_wps=True):
    """Reference path (grey) + actual trajectories.  zoom = (xmin,xmax,ymin,ymax) in km."""
    f = 1e-3 if km else 1.0
    ax.plot(path.x * f, path.y * f, color="0.55", lw=3.0, alpha=0.6, label="Reference path")
    for r in (results or []):
        ax.plot(r.x * f, r.y * f, lw=1.2, label=r.label or "Flown")
    if show_wps:
        for i, w in enumerate(path.waypoints):
            ax.plot(w.x * f, w.y * f, "k^" if w.kind == "flyby" else "ks", ms=7)
            ax.annotate(w.name or f"WP{i}", (w.x * f, w.y * f), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East [km]" if km else "East [m]")
    ax.set_ylabel("North [km]" if km else "North [m]")
    ax.grid(alpha=0.3)
    if zoom:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])


def plot_timeseries(results, title="", fname=None):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for r in results:
        axs[0].plot(r.t, r.xte, lw=1.2, label=r.label)
        axs[1].plot(r.t, np.degrees(r.phi), lw=1.2, label=r.label)
        axs[2].plot(r.t, np.degrees(r.chi_err), lw=1.2, label=r.label)
    axs[0].set_ylabel("XTE [m]  (+ right)")
    axs[1].set_ylabel("Bank angle [deg]")
    axs[2].set_ylabel("Track error [deg]")
    axs[2].set_xlabel("Time [s]")
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[0].set_title(title)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fig
