"""Render the saved training rewards and added-crossing transfer comparisons.

Run from the code repository:
    .venv/bin/python plots/training_transfer_comparison.py [output.png]

The default output is plots/generated/training_transfer_comparison.png in the code repository.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import get_averages


def training_curve(filename):
    data = pd.read_csv(ROOT / "runs" / filename)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["step"] = pd.to_numeric(data["step"], errors="coerce")
    columns = [column for column in data.columns if column.startswith("reward")]
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[data["step"] <= 20e6].copy()
    rewards = data[columns].mean(axis=1).rolling(200, min_periods=1)
    return (data["step"].to_numpy(dtype=float),
            rewards.mean().to_numpy(dtype=float),
            rewards.std().fillna(0).to_numpy(dtype=float))


def plot_training_transfer_comparison(output=None):
    output = Path(output) if output is not None else ROOT / "plots/generated/training_transfer_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "text.color": "#1f2937", "axes.labelcolor": "#202124",
        "axes.labelsize": 8, "axes.labelweight": "bold",
        "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.edgecolor": "#c9ced5", "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#626b7b", "ytick.color": "#626b7b",
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    width, height = 7.15, 1.75
    fig = plt.figure(figsize=(width, height))
    axes = [fig.add_axes([left / width, .36 / height, 1.87 / width, 1.20 / height])
            for left in [.49, 2.97, 5.23]]
    series = [
        ("DeCoR", "#3C9F40", "*", 5.5, "combined_rewards_codesign.csv",
         "readout_32/May09_11-34-05/results/codesign_added.json"),
        ("Sequential", "#3771A1", "o", 3.3, "combined_rewards_control_only.csv",
         "first_design_then_control/May11_10-18-09/results/separate_added.json"),
    ]
    for label, color, marker, size, csv, results in series:
        steps, mean, std = training_curve(csv)
        axes[0].fill_between(steps, (mean - std) / 100, (mean + std) / 100,
                             color=color, alpha=.2, edgecolor="none", zorder=2)
        axes[0].plot(steps, mean / 100, color=color, linewidth=1, zorder=3)
        values = get_averages(ROOT / "runs" / results, total=False)
        for ax, mean_index, std_index in [(axes[1], 2, 5), (axes[2], 1, 4)]:
            scales, mean, std = values[0], values[mean_index], values[std_index]
            ax.fill_between(scales, mean - std, mean + std,
                            color=color, alpha=.15, edgecolor="none", zorder=2)
            ax.plot(scales, mean, color=color, linewidth=1, zorder=3)
            ax.scatter(scales, mean, color=color, marker=marker, s=size ** 2,
                       edgecolors="white", linewidths=.4, zorder=4)

    axes[0].set(xlim=(-1e6, 20e6), ylim=(-18, 1.5), yticks=[-16, -12, -8, -4, 0],
                xticks=[0, 5e6, 10e6, 15e6, 20e6], xticklabels=["0", "5", "10", "15", "20"])
    axes[0].set_xlabel("Environment Step (×10⁶)", labelpad=3)
    axes[0].set_ylabel("Control Reward (×10²)", labelpad=3)
    for ax, title, limits in [(axes[1], "Pedestrian Wait Time", (-5, 100)),
                              (axes[2], "Vehicle Wait Time", (-12.5, 280))]:
        ax.set_title(title, pad=4)
        ax.set(xlim=(.375, 2.875), ylim=limits, xticks=[.5, 1, 1.5, 2, 2.5],
               xticklabels=["0.5x", "1.0x", "1.5x", "2.0x", "2.5x"])
        ax.set_xlabel("Demand Scale", labelpad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[1].set_ylabel("Average (s)", labelpad=3)
    for ax in axes:
        ax.tick_params(length=2, width=.6, pad=2)
        ax.set_axisbelow(True)
        ax.grid(linestyle="--", linewidth=.4, color="#999999", alpha=.3)

    handles = [Line2D([], [], color=color, linewidth=1, marker=marker,
                      markersize=size, markeredgecolor="white", markeredgewidth=.4,
                      label=label) for label, color, marker, size, _, _ in reversed(series)]
    for ax, location in zip(axes, ["lower right", "upper right", "upper right"]):
        ax.legend(handles=handles, loc=location, frameon=True,
                  facecolor="white", edgecolor="#e5e7eb", framealpha=.98,
                  borderpad=.4, labelspacing=.3, handlelength=1.6, handletextpad=.5)
    fig.savefig(output, dpi=600, metadata={"Software": None})
    plt.close(fig)
    with Image.open(output) as source:
        image = source.convert("RGB")
    bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    image.crop((0, bounds[1], image.width, bounds[3])).save(output, dpi=(600, 600))


if __name__ == "__main__":
    plot_training_transfer_comparison(sys.argv[1] if len(sys.argv) > 1 else None)
