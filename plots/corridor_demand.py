"""Render the original corridor demand and annotated corridor image.

Run from the code repository:
    .venv/bin/python plots/corridor_demand.py [output.png]

The default output is plots/generated/corridor_demand.png in the code repository.
"""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]


def demand(filename, tag):
    departures = np.array([float(element.attrib["depart"])
                           for element in ET.parse(ROOT / "simulation" / filename).iter(tag)
                           if "depart" in element.attrib])
    edges = np.arange(0, departures.max() + 60, 60)
    return edges[:-1] + 30, np.histogram(departures, bins=edges)[0]


def plot_corridor_demand(output=None):
    output = Path(output) if output is not None else ROOT / "plots/generated/corridor_demand.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(ROOT.parent / "ICRA writeup/Poudel2026Design_ICRA/figures/new_hero.png") as source:
        corridor = source.convert("RGB")
    bounds = ImageChops.difference(corridor, Image.new("RGB", corridor.size, "white")).getbbox()
    corridor = corridor.crop(bounds)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "text.color": "#1f2937", "axes.labelcolor": "#202124",
        "axes.labelweight": "bold", "axes.labelsize": 7,
        "axes.edgecolor": "#c9ced5", "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#626b7b", "ytick.color": "#626b7b",
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    })
    width, height = 7.15, 1.37
    fig = plt.figure(figsize=(width, height))
    for bottom, filename, tag, color, title, limits, ticks in [
        (.81, "original_pedtrips.xml", "person", "#6A5ACD", "Pedestrian", (20, 75), [20, 40, 60]),
        (.23, "original_vehtrips.xml", "trip", "#FF7F50", "Vehicle", (0, 10), [0, 4, 8]),
    ]:
        ax = fig.add_axes([5.70 / width, bottom / height, 1.34 / width, .49 / height])
        x, counts = demand(filename, tag)
        ax.plot(x, counts, color=color, linewidth=.6)
        ax.axvline(2400, color="#32862b", linestyle=(0, (3, 3)), linewidth=1.1)
        fig.text(5.41 / width, (bottom + .245) / height, title, rotation=90,
                 ha="center", va="center", weight="normal")
        ax.set(xlim=(0, 3600), ylim=limits, yticks=ticks, xticks=[0, 1200, 2400, 3600])
        ax.tick_params(length=2, width=.6, pad=2)
        ax.set_axisbelow(True)
        ax.grid(linestyle="--", linewidth=.4, color="#999999", alpha=.3)
        if tag == "person":
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (s)", labelpad=2)
    fig.text(5.26 / width, .765 / height, "Departures / min", rotation=90,
             ha="center", va="center", weight="bold")

    image_width = 5.04
    image_height = image_width * corridor.height / corridor.width
    image_ax = fig.add_axes([0, .02 / height, image_width / width, image_height / height])
    image_ax.imshow(corridor, interpolation="lanczos")
    image_ax.set_axis_off()

    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0,
                metadata={"Software": None})
    plt.close(fig)
    with Image.open(output) as source:
        image = source.convert("RGB")
    bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    image.crop(bounds).save(output, dpi=(600, 600))


if __name__ == "__main__":
    plot_corridor_demand(sys.argv[1] if len(sys.argv) > 1 else None)
