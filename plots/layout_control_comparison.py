"""Render the historical layout and control evaluation comparisons.

Run from the code repository:
    .venv/bin/python plots/layout_control_comparison.py [output.png]

The default output is plots/generated/layout_control_comparison.png in the code repository.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import get_averages


def plot_layout_control_comparison(output=None):
    output = Path(output) if output is not None else ROOT / "plots/generated/layout_control_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    base = ROOT / "runs/readout_32/May09_11-34-05/results/eval_May10_16-16-52"
    files = {
        "original": "realworld_unsignalized.json",
        "unsignalized": "policy_at_7603200_unsignalized.json",
        "fixed": "policy_at_7603200_tl.json",
        "learned": "policy_at_7603200_ppo.json",
    }
    results = {key: [get_averages(base / name, total=total) for total in [False, True]]
               for key, name in files.items()}
    colors = {"Real-world": "#E63946", "DeCoR": "#2D9334",
              "Fixed-time": "#1f77b4", "Unsignalized": "#ff7f0e"}
    control = [("Fixed-time", "fixed"), ("Unsignalized", "unsignalized"), ("DeCoR", "learned")]
    panels = [
        ("Pedestrian Arrival Time", [("Real-world", "original"), ("DeCoR", "unsignalized")],
         3, np.arange(60, 111, 10), np.arange(0, 26, 5)),
        ("Pedestrian Wait Time", control, 2, np.arange(0, 9, 2), np.arange(0, 5)),
        ("Vehicle Wait Time", control, 1, np.arange(0, 76, 15), np.arange(0, 5)),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "text.color": "#1f2937", "axes.labelcolor": "#202124",
        "axes.labelweight": "bold", "axes.labelsize": 8,
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.edgecolor": "#c9ced5", "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#626b7b", "ytick.color": "#626b7b",
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.framealpha": .98,
        "legend.edgecolor": "#d1d5db", "legend.facecolor": "white",
    })
    width, height = 7.15, 2.61
    fig = plt.figure(figsize=(width, height))
    legends = []
    for column, (title, series, metric, average_ticks, total_ticks) in enumerate(panels):
        left = [.40, 2.88, 5.15][column]
        for row, (bottom, ticks, divisor) in enumerate([
            (1.4892, average_ticks, 1), (.51, total_ticks, 1000),
        ]):
            ax = fig.add_axes([left / width, bottom / height, (1.94 if column == 0 else 1.97) / width, .8992 / height])
            ax.set_xlim(.3875, 2.8625)
            ax.axvspan(.3875, 1.0, facecolor="#e2e8f0", alpha=.6, zorder=-2)
            ax.axvspan(2.25, 2.8625, facecolor="#e2e8f0", alpha=.6, zorder=-2)
            ax.set_axisbelow(True)
            ax.grid(linestyle="--", linewidth=.4, color="#999999", alpha=.3)
            ax.tick_params(length=2, width=.6, pad=2)
            for label, source in series:
                data = results[source][row]
                scales, mean, std = data[0], data[metric] / divisor, data[metric + 3] / divisor
                ax.fill_between(scales, mean - std, mean + std, color=colors[label], alpha=.15)
                ax.plot(scales, mean, color=colors[label], linewidth=1,
                        marker="*" if label == "DeCoR" else "o",
                        markersize=5.3 if label == "DeCoR" else 3.2,
                        markeredgecolor="white", markeredgewidth=.4, label=label)
            pad = (ticks[1] - ticks[0]) * .15
            ax.set(yticks=ticks, ylim=(ticks[0] - pad, ticks[-1] + pad))
            ax.set_xticks([.5, 1, 1.5, 2, 2.5], ["0.5x", "1.0x", "1.5x", "2.0x", "2.5x"])
            ax.set_xticks(np.arange(.5, 2.76, .25), minor=True)
            if row == 0:
                ax.set_title(title, pad=5)
                if column < 2:
                    legends.append(ax.get_legend_handles_labels())
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Demand Scale", labelpad=3)
            if column < 2:
                ax.set_ylabel("Average (s)" if row == 0 else "Total (×10³ s)", labelpad=3)

    for center, (handles, labels) in zip([1.36, 5.0], legends):
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(center / width, .015 / height),
                   ncol=len(labels), borderpad=.35, handlelength=1.7,
                   handletextpad=.5, columnspacing=.9, borderaxespad=0)
    fig.savefig(output, dpi=600,
                metadata={"Software": None})
    plt.close(fig)
    with Image.open(output) as source:
        image = source.convert("RGB")
    bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    image.crop((0, bounds[1], image.width, bounds[3])).save(output, dpi=(600, 600))


if __name__ == "__main__":
    plot_layout_control_comparison(sys.argv[1] if len(sys.argv) > 1 else None)
