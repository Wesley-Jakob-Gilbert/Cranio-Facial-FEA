#!/usr/bin/env python3
"""Build the README / docs figures from results/ (run scripts/reproduce_results.py first).

Writes to docs/figures/:
  result_threshold.png    sustained-load weighting needed before the palate responds
  result_ct_contrast.png  12-month delta-rho, mewing minus mouth breathing, three weightings
  result_ct_evolution.gif monthly delta-rho at static_equivalent_cycles = 1e7 (hypothetical)
  result_timeseries.png   regional mean delta-rho over time (CT, 1e7)
  result_shares.png       share of the daily stimulus by load case (CT, mewing, step 0)
  result_sweep.png        palatal contrast vs static_equivalent_cycles (template)
  result_morris.png       Morris screening (template)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from craniofea.config import load_config  # noqa: E402
from craniofea.model import Mesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"

# Palette: categorical slots in fixed order, recessive chrome, blue/red diverging.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = {"mewing": "#2a78d6", "mouth_breathing": "#eb6834", "reference": "#1baf7a"}
CASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DIVERGING = LinearSegmentedColormap.from_list(
    "resorb_form", ["#8f2424", "#e34948", "#f0efec", "#2a78d6", "#123f75"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#dbe8f8", "#2a78d6", "#0d2c55"])
LABEL = {"mewing": "Mewing (2 kPa sustained)", "mouth_breathing": "Mouth breathing (0 kPa)",
         "reference": "Reference habit (0.2 kPa)"}
REGION = {"region_palatal_vault": "Palatal vault",
          "region_alveolar_anterior": "Anterior alveolus",
          "region_alveolar_posterior": "Posterior alveolus",
          "region_basal": "Basal / buttress"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 150,
    })


def read_summary(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k: (v if k in ("scenario", "region") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def ct_mesh() -> Mesh:
    cfg = load_config(ROOT / "configs" / "study.yaml")
    return Mesh.load(ROOT / cfg.mesh.path)


def final_drho(run: str, scenario: str) -> np.ndarray:
    d = np.load(RES / run / f"density_{scenario}.npz")
    return d["rho"][-1] - d["rho"][0]


def fig_threshold(mesh: Mesh) -> None:
    from craniofea.viz.render import render_surface_field

    path = RES / "ct_critical_static_cycles" / "critical_static_cycles.npz"
    if not path.exists():
        return
    n = np.load(path)["n_crit"]
    # 0 = forms at any weight (bottom of the scale), inf = never (top of the scale);
    # the midpalatal suture is not remodeled and is drawn neutral gray.
    v = np.log10(np.clip(np.where(np.isfinite(n), n, 1e12), 1e4, 1e12))
    v[np.asarray(mesh.elem_sets["suture_mid"], dtype=np.int64)] = np.nan
    img = render_surface_field(mesh, v, view="inferior", clim=(4.0, 10.0), cmap=SEQUENTIAL,
                               show_scalar_bar=False, window_size=(900, 820),
                               background=SURFACE)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.imshow(img)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(4, 10))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
    cb.set_label("log$_{10}$ sustained-load weight (cycles/day) needed for apposition",
                 color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("How strongly must a 2 kPa posture count before bone responds?", loc="left")
    fig.text(0.02, -0.06, "Inferior view, CT maxilla; gray: midpalatal suture (not remodeled).\n"
             "Literature convention: 1 cycle/day; posture held all day at 1 Hz: 8.6e4.",
             color=MUTED, fontsize=8)
    fig.savefig(FIG / "result_threshold.png", bbox_inches="tight")
    plt.close(fig)


def fig_ct_contrast(mesh: Mesh) -> None:
    from craniofea.viz.render import render_surface_field

    runs = [("ct_default", "1 (literature)"), ("ct_static_1e6", "10$^6$"),
            ("ct_static_1e7", "10$^7$")]
    runs = [r for r in runs if (RES / r[0] / "density_mewing.npz").exists()]
    if not runs:
        return
    diffs = [final_drho(r, "mewing") - final_drho(r, "mouth_breathing") for r, _ in runs]
    lim = max(0.02, float(np.percentile(np.abs(np.concatenate(diffs)), 99.5)))
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.4))
    for ax, (_, label), d in zip(np.atleast_1d(axes), runs, diffs, strict=True):
        img = render_surface_field(mesh, d, view="inferior", clim=(-lim, lim), cmap=DIVERGING,
                                   show_scalar_bar=False, window_size=(800, 760),
                                   background=SURFACE)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"sustained-load weight {label}", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=np.atleast_1d(axes).tolist(), orientation="horizontal",
                      fraction=0.05, pad=0.03, shrink=0.6)
    cb.set_ticks(np.linspace(-lim, lim, 5))
    cb.set_label("Δρ after 12 months, mewing − mouth breathing  (red: less bone, "
                 "blue: more bone; gray strip: suture, not remodeled)", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(FIG / "result_ct_contrast.png", bbox_inches="tight")
    plt.close(fig)


def gif_ct_evolution(mesh: Mesh, run: str = "ct_static_1e7") -> None:
    from craniofea.viz.render import render_surface_field, side_by_side, write_gif

    if not (RES / run / "density_mewing.npz").exists():
        return
    data = {s: np.load(RES / run / f"density_{s}.npz") for s in ("mewing", "mouth_breathing")}
    d = {s: v["rho"] - v["rho"][0] for s, v in data.items()}
    lim = max(0.02, float(np.percentile(np.abs(np.concatenate([x[-1] for x in d.values()])),
                                        99.5)))
    frames = []
    for k in range(d["mewing"].shape[0]):
        month = data["mewing"]["times"][k] / (365.0 / 12)
        imgs = [render_surface_field(
            mesh, d[s][k], view="inferior", clim=(-lim, lim), cmap=DIVERGING,
            label="Δρ" if s == "mouth_breathing" else "", show_scalar_bar=s == "mouth_breathing",
            title=f"{LABEL[s]} — month {month:.0f}", window_size=(640, 600),
            background=SURFACE)
            for s in ("mewing", "mouth_breathing")]
        frames.append(side_by_side(imgs))
    write_gif(frames, FIG / "result_ct_evolution.gif", fps=2.0)


def fig_timeseries(run: str = "ct_static_1e7") -> None:
    p = RES / run / "summary.csv"
    if not p.exists():
        return
    rows = read_summary(p)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
    for ax, (reg, title) in zip(axes, REGION.items(), strict=True):
        for s in ("mewing", "mouth_breathing"):
            r = sorted((x for x in rows if x["scenario"] == s and x["region"] == reg),
                       key=lambda x: x["step"])
            if r:
                ax.plot([x["day"] / 30.42 for x in r], [x["mean_drho"] for x in r],
                        color=SERIES[s], label=LABEL[s])
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("month")
        ax.set_xticks([0, 3, 6, 9, 12])
    axes[0].set_ylabel("mean Δρ (volume-weighted)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Regional density change if sustained loads were weighted at 10$^7$ "
                 "cycles/day (hypothetical)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "result_timeseries.png", bbox_inches="tight")
    plt.close(fig)


def fig_shares(run: str = "ct_default") -> None:
    p = RES / run / "stimulus_shares.json"
    if not p.exists():
        return
    shares = json.loads(p.read_text())
    scen = "mewing" if "mewing" in shares else next(iter(shares))
    regs = [r for r in REGION if r in shares[scen]]
    cases = list(shares[scen][regs[0]])
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    left = np.zeros(len(regs))
    for i, c in enumerate(cases):
        vals = np.array([shares[scen][r][c] for r in regs])
        ax.barh(range(len(regs)), vals, left=left, color=CASE_COLORS[i % 4], height=0.62,
                edgecolor=SURFACE, linewidth=2, label=c.replace("_", " "))
        left += vals
    rest = [c for c in cases if "rest" in c]
    if rest:
        for y, r in enumerate(regs):
            ax.text(1.01, y, f"rest posture {100 * shares[scen][r][rest[0]]:.2g}%",
                    va="center", fontsize=8, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(regs)), [REGION[r] for r in regs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the daily stimulus Σ nᵢ εᵢᵐ")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(cases), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    ax.set_title(f"Where the daily stimulus comes from ({scen.replace('_', ' ')}, CT, "
                 "literature weighting)", loc="left", pad=24)
    fig.tight_layout()
    fig.savefig(FIG / "result_shares.png", bbox_inches="tight")
    plt.close(fig)


def fig_sweep() -> None:
    p = RES / "template_sensitivity" / "static_cycles_sweep.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    x = np.array([r["static_equivalent_cycles"] for r in rows])
    y = np.array([r["response"] for r in rows])
    keep = x > 0
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.plot(x[keep], y[keep], color=SERIES["mewing"], marker="o", ms=5)
    for xv, txt in ((1.0, "literature convention"), (8.64e4, "1 Hz, all day")):
        ax.axvline(xv, color=MUTED, lw=0.9, ls="--")
        ax.text(xv * 1.3, ax.get_ylim()[1] * 0.9, txt, color=MUTED, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("sustained-load weight, static_equivalent_cycles (per day)")
    ax.set_ylabel("palatal Δρ, mewing − mouth br.")
    ax.set_title("The mewing effect appears only for extreme sustained-load weighting",
                 loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "result_sweep.png", bbox_inches="tight")
    plt.close(fig)


def fig_morris() -> None:
    p = RES / "template_sensitivity" / "morris.json"
    if not p.exists():
        return
    table = sorted(json.loads(p.read_text())["table"], key=lambda r: r["mu_star"])
    names = [r["parameter"].split(".", 1)[1] if "." in r["parameter"] else r["parameter"]
             for r in table]
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(table) + 1.0))
    ax.barh(names, [r["mu_star"] for r in table], color=SERIES["mewing"], height=0.6)
    ax.set_xlabel("Morris μ* (mean |elementary effect| on palatal contrast)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2, labelsize=8)
    ax.set_title("Which assumptions matter (template mesh)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "result_morris.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_sweep()
    fig_morris()
    fig_shares()
    fig_timeseries()
    try:
        mesh = ct_mesh()
    except FileNotFoundError:
        print("CT mesh not found; skipping 3D figures")
        return
    fig_threshold(mesh)
    fig_ct_contrast(mesh)
    gif_ct_evolution(mesh)
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
