"""Render the saved design distribution and reward ablation data.

Run from the code repository:
    .venv/bin/python plots/design_distribution.py [output.png]

The default output is plots/generated/design_distribution.png in the code repository.
"""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]


def plot_design_distribution(output=None):
    output = Path(output) if output is not None else ROOT / "plots/generated/design_distribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    design = json.loads((ROOT / "runs/review_20260914/manifest.json").read_text())
    means = np.array(design["mixture_means"])
    weights = np.array(design["mixture_probabilities"])
    sigma = design["sigma_normalized"]
    crossings = sorted(design["evaluation_proposals"], key=lambda p: p["x"])
    metrics = ["total_veh_waiting_time", "total_ped_waiting_time",
               "max_wait_times_veh", "max_wait_times_ped"]
    runs = []
    for name in ["mwaq", "mwaq_linear", "mwaq_exponential"]:
        data = json.loads((ROOT / "ablation" / f"{name}.json").read_text())
        runs.append([[row[key] for key in metrics]
                     for scale in data.values() for row in scale.values()])
    values = np.array(runs)
    averages, deviations = values.mean(axis=1), values.std(axis=1)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "text.color": "#1f2937", "axes.labelcolor": "#202124",
        "axes.labelweight": "bold", "axes.labelsize": 8,
        "axes.edgecolor": "#c9ced5", "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#626b7b", "ytick.color": "#626b7b",
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.framealpha": .98,
        "legend.edgecolor": "#d1d5db", "legend.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.05))
    plot_height = .22 * 7.15 / 2.05
    for ax, left, width in zip(axes, [.062, .418, .755], [.22, .24, .24]):
        ax.set_position([left, .20, width, plot_height])
        ax.tick_params(length=2, width=.6, pad=2)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", linewidth=.4, color="#999999", alpha=.4)

    ax = axes[0]
    ax.set_box_aspect(1)
    x, y = np.meshgrid(np.linspace(-.06, 1.06, 260), np.linspace(-.08, 1.12, 260))
    grid = np.stack([x, y], axis=-1)
    density = np.sum(weights * np.exp(-np.sum((grid[:, :, None, :] - means)**2,
                     axis=-1) / (2 * sigma**2)) / (2 * np.pi * sigma**2), axis=-1)
    contour = ax.contourf(x, y, density, levels=np.linspace(0, 5, 25), cmap="coolwarm", alpha=.87)
    ax.scatter(means[:, 0], means[:, 1], s=17.6, color="#0066ff", edgecolors="#172554",
               linewidths=.55, label="Means", zorder=3)
    for i, proposal in enumerate(crossings, 1):
        px, py = proposal["normalized"]
        ax.scatter(px, py, marker="*", s=68, color="#00bf5f", edgecolors="white",
                   linewidths=.7, label="Merged" if i == 1 else None, zorder=4)
        offset = (-7, 7) if i == 4 else (7, -6) if i in (1, 3) else (7, 4)
        ax.annotate(f"MB{i}", (px, py), xytext=offset, textcoords="offset points",
                    ha="right" if i == 4 else "left", va="center", fontsize=7,
                    weight="bold", color="white", zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#283449")])
    ax.set(xlim=(-.06, 1.06), ylim=(-.08, 1.12), xlabel="Location", ylabel="Width")
    ax.set_xticks([0, .5, 1], ["0.0", "0.5", "1.0"])
    ax.set_yticks([0, .5, 1], ["0.0", "0.5", "1.0"])
    ax.legend(loc="upper right", handletextpad=.3, handlelength=1,
              borderpad=.35, labelspacing=.35, borderaxespad=.35)
    color_ax = ax.inset_axes([1.025, 0, .045, 1])
    colorbar = fig.colorbar(contour, cax=color_ax, ticks=[0, 2.5, 5])
    colorbar.outline.set_visible(False)
    color_ax.tick_params(labelsize=7, length=1.5, pad=1)
    colorbar.set_label("Density", fontsize=8, fontweight="normal", labelpad=3)

    for ax, start, unit, ylabel, top, ticks in [
        (axes[1], 0, 1000, "Total Wait (×10³ s)", 6.3, [0, 2, 4, 6]),
        (axes[2], 2, 1, "Max Wait (s)", 135, [0, 40, 80, 120]),
    ]:
        for offset, metric, label, color in [
            (-.14, start, "Vehicle", "#3771A1"),
            (.14, start + 1, "Pedestrian", "#3C9F40"),
        ]:
            ax.bar(np.arange(3) + offset, averages[:, metric] / unit, width=.28,
                   yerr=deviations[:, metric] / unit, color=color, alpha=.9,
                   edgecolor=color, linewidth=.4, label=label, capsize=2.5,
                   error_kw={"elinewidth": .9, "ecolor": "#333333", "capthick": .9})
        ax.set(xlim=(-.5, 2.5), ylim=(0, top), ylabel=ylabel, xlabel="Reward Function")
        ax.set_xticks(range(3), ["MWAQ", "LI-MWAQ", "EI-MWAQ"])
        ax.set_yticks(ticks)
        ax.legend(loc="upper right", handletextpad=.3, labelspacing=.35,
                  handlelength=1, borderpad=.35, borderaxespad=.35)

    fig.savefig(output, dpi=600, pad_inches=0,
                metadata={"Software": None})
    plt.close(fig)
    with Image.open(output) as source:
        image = source.convert("RGB")
    bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    image.crop((0, bounds[1], image.width, bounds[3])).save(output, dpi=(600, 600))


if __name__ == "__main__":
    plot_design_distribution(sys.argv[1] if len(sys.argv) > 1 else None)
