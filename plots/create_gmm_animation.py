"""Create a README-ready GMM animation.

The animation follows the visual language of ``graphs_gmm_v2.png``:
an isometric GMM density surface on the left and a top-down contour map
on the right, with component means merging into final maxima.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from mpl_toolkits.mplot3d import proj3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMM_PATH = PROJECT_ROOT / "runs/readout_32/May09_11-34-05/gmm_iterations/gmm_i_eval_final_b0_data.pkl"


def load_pickle_with_cpu_fallback(path: Path):
    """Load pickles that may contain torch storages serialized on CUDA."""
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except RuntimeError as exc:
            if "Attempting to deserialize object on a CUDA device" not in str(exc):
                raise
            f.seek(0)
            original_torch_load = torch.load

            def cpu_torch_load(*args, **kwargs):
                kwargs.setdefault("map_location", torch.device("cpu"))
                return original_torch_load(*args, **kwargs)

            torch.load = cpu_torch_load
            try:
                return pickle.load(f)
            finally:
                torch.load = original_torch_load


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def gaussian_mixture_density(x_grid: np.ndarray, y_grid: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    density = np.zeros_like(x_grid, dtype=float)
    norm = 1.0 / (2.0 * np.pi * sigma * sigma * len(means))
    for mean_x, mean_y in means:
        dx = (x_grid - mean_x) / sigma
        dy = (y_grid - mean_y) / sigma
        density += norm * np.exp(-0.5 * (dx * dx + dy * dy))
    return density


def density_at(points: np.ndarray, means: np.ndarray, sigma: float, z_min: float, z_ptp: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    vals = []
    for px, py in points:
        d = 0.0
        norm = 1.0 / (2.0 * np.pi * sigma * sigma * len(means))
        for mean_x, mean_y in means:
            dx = (px - mean_x) / sigma
            dy = (py - mean_y) / sigma
            d += norm * np.exp(-0.5 * (dx * dx + dy * dy))
        vals.append((d - z_min) / z_ptp)
    return np.asarray(vals)


def modified_means_from_gmm(gmm_path: Path) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = load_pickle_with_cpu_fallback(gmm_path)
    gmm_single = data[0]
    means = gmm_single.component_distribution.loc.detach().cpu().numpy().copy()
    means[0][1] = 0.38
    means[1][1] = 0.36
    means[5][1] = 0.32
    means[2][1] = 0.32
    means[4][1] = 0.30
    return means


def maxima_from_means(means: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    clusters = [[3], [0, 1, 5], [2, 4], [6]]
    maxima = np.array([means[idxs].mean(axis=0) for idxs in clusters])
    return maxima, clusters


def add_figure_legend(fig: plt.Figure) -> None:
    handles = [
        mlines.Line2D([], [], marker="o", markersize=13, color="#0066ff",
                      markeredgecolor="black", linewidth=0, label="Mean"),
        mlines.Line2D([], [], marker="*", markersize=19.5, color="#00c853",
                      markeredgecolor="white", markeredgewidth=1.2,
                      linewidth=0, label="Maxima"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.033),
               ncol=2, frameon=True, fancybox=True, fontsize=17.5,
               handlelength=2.2, columnspacing=2.5, labelcolor="#1f2937",
               facecolor="white", edgecolor="#111827", framealpha=1.0,
               borderpad=0.55)


def add_projected_peak_stars(
    fig: plt.Figure,
    ax,
    maxima: np.ndarray,
    density_maxima: np.ndarray,
    surface_p: float,
    mb_progress: np.ndarray,
    t: float,
) -> None:
    for i, (mx, my) in enumerate(maxima):
        peak_p = mb_progress[i]
        if peak_p <= 0.0:
            continue
        star_pulse = 1.0 + 0.12 * np.sin(2.0 * np.pi * (t * 3.0 + i * 0.18))
        star_z = min(1.16, density_maxima[i] * surface_p + 0.16 * peak_p)
        x_proj, y_proj, _ = proj3d.proj_transform(mx, my, star_z, ax.get_proj())
        fig_x, fig_y = fig.transFigure.inverted().transform(ax.transData.transform((x_proj, y_proj)))
        marker_size = 19.0 * (0.45 + 0.55 * peak_p) * star_pulse
        fig.lines.append(mlines.Line2D([fig_x], [fig_y], marker="*", markersize=marker_size + 7.0,
                                       color="#111827", markeredgecolor="white",
                                       markeredgewidth=0.8, linewidth=0,
                                       transform=fig.transFigure, alpha=0.72 * peak_p,
                                       zorder=200))
        fig.lines.append(mlines.Line2D([fig_x], [fig_y], marker="*", markersize=marker_size,
                                       color="#00e676", markeredgecolor="#ffffff",
                                       markeredgewidth=1.4, linewidth=0,
                                       transform=fig.transFigure, alpha=peak_p,
                                       zorder=201))


def draw_frame(
    frame_path: Path,
    frame_index: int,
    total_frames: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_norm: np.ndarray,
    means: np.ndarray,
    maxima: np.ndarray,
    clusters: list[list[int]],
    density_maxima: np.ndarray,
) -> None:
    t = frame_index / max(total_frames - 1, 1)
    surface_p = smoothstep((t - 0.05) / 0.35)
    contour_p = smoothstep((t - 0.20) / 0.35)
    means_p = smoothstep((t - 0.32) / 0.25)
    mb_starts = np.array([0.50, 0.62, 0.74, 0.86])
    mb_progress = np.array([smoothstep((t - start) / 0.09) for start in mb_starts])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "SF Pro Display", "Segoe UI", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "#202124",
    })

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    fig.lines.append(mlines.Line2D([0.50, 0.50], [0.14, 0.89],
                                   transform=fig.transFigure, color="#000000",
                                   linewidth=1.25, zorder=20))
    add_figure_legend(fig)

    fig.text(0.5, 0.95, "Gaussian Mixture Model", ha="center", va="center",
             fontsize=31, fontweight="bold", color="#202124")
    fig.text(0.255, 0.88, "Isometric View", ha="center", va="center",
             fontsize=18, fontweight="normal", color="#202124")
    fig.text(0.705, 0.88, "Top-Down View", ha="center", va="center",
             fontsize=18, fontweight="normal", color="#202124")

    ax_3d = fig.add_axes([0.030, 0.205, 0.435, 0.66], projection="3d")
    ax_2d = fig.add_axes([0.570, 0.225, 0.270, 0.58])

    cmap = plt.get_cmap("coolwarm")
    z_current = z_norm * surface_p
    ax_3d.plot_wireframe(x_grid, y_grid, z_current, rstride=6, cstride=6,
                         color="#94a3b8", linewidth=0.55, alpha=0.85 * (1.0 - 0.65 * surface_p))
    ax_3d.plot_surface(
        x_grid,
        y_grid,
        z_current,
        rstride=2,
        cstride=2,
        facecolors=cmap(z_norm),
        linewidth=0.04,
        edgecolor="white",
        alpha=0.15 + 0.78 * surface_p,
        antialiased=True,
        shade=False,
    )
    ax_3d.set_xlim(0.0, 1.05)
    ax_3d.set_ylim(0.0, 1.05)
    ax_3d.set_zlim(0.0, 1.18)
    ax_3d.set_xticks([0.0, 0.5, 1.0])
    ax_3d.set_yticks([0.0, 0.5, 1.0])
    ax_3d.set_zticks([0.0, 0.5, 1.0])
    ax_3d.set_xlabel("Location", fontsize=15, labelpad=12, color="#202124")
    ax_3d.set_ylabel("Width", fontsize=15, labelpad=12, color="#202124")
    ax_3d.set_zlabel("Density", fontsize=15, labelpad=10, color="#202124")
    ax_3d.tick_params(axis="both", which="major", labelsize=12, colors="#5f6368", pad=4)
    ax_3d.tick_params(axis="z", which="major", labelsize=12, colors="#5f6368", pad=4)
    ax_3d.view_init(elev=32, azim=-70 + 55 * t + 3.0 * np.sin(2 * np.pi * t))
    ax_3d.set_box_aspect((1.05, 1.0, 0.65))
    for axis_obj in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
        axis_obj._axinfo["grid"].update({"color": (0.0, 0.0, 0.0, 0.18), "linestyle": (0, (5, 5)), "linewidth": 0.55})
        axis_obj.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis_obj.pane.set_edgecolor("#d1d5db")
    add_projected_peak_stars(fig, ax_3d, maxima, density_maxima, surface_p, mb_progress, t)

    if contour_p > 0.0:
        ax_2d.contourf(x_grid, y_grid, z_norm, levels=26, cmap=cmap, alpha=0.86 * contour_p)
    for val in np.arange(0.0, 1.1, 0.2):
        ax_2d.axhline(val, color=(0.0, 0.0, 0.0, 0.32 * contour_p),
                      linestyle=(0, (5, 5)), linewidth=0.55, zorder=2)
        ax_2d.axvline(val, color=(0.0, 0.0, 0.0, 0.32 * contour_p),
                      linestyle=(0, (5, 5)), linewidth=0.55, zorder=2)

    ax_2d.scatter(means[:, 0], means[:, 1], c="#0066ff", marker="o", s=95,
                  edgecolors="black", linewidths=0.7, alpha=means_p, zorder=6)

    for target_index, (target, members) in enumerate(zip(maxima, clusters)):
        if len(members) <= 1:
            continue
        target_p = mb_progress[target_index]
        if target_p <= 0.0:
            continue
        for idx in members:
            start = means[idx]
            current = start + target_p * (target - start)
            ax_2d.plot([start[0], current[0]], [start[1], current[1]],
                       color="#f59e0b", linestyle=(0, (5, 4)), linewidth=2.2,
                       alpha=0.25 + 0.75 * target_p, zorder=7)

    pulse = 0.018 + 0.018 * (0.5 + 0.5 * np.sin(2.0 * np.pi * (t * 3.0)))
    for i, (mx, my) in enumerate(maxima):
        if mb_progress[i] <= 0.0:
            continue
        circle = patches.Circle((mx, my), radius=pulse, facecolor="none",
                                edgecolor="#00c853", linewidth=2.0,
                                alpha=0.25 * mb_progress[i], zorder=8)
        ax_2d.add_patch(circle)

    text_effects = [pe.withStroke(linewidth=4.0, foreground="black"), pe.Normal()]
    for i, (mx, my) in enumerate(maxima):
        point_p = mb_progress[i]
        if point_p <= 0.0:
            continue
        star_pulse = 1.0 + 0.10 * np.sin(2.0 * np.pi * (t * 3.0 + i * 0.18))
        ax_2d.scatter([mx], [my], c="#00c853", marker="*", s=310 * point_p * star_pulse,
                      edgecolors="white", linewidths=1.1, alpha=point_p, zorder=9)
        dx = -0.045 if i == 3 else 0.045
        ha = "right" if i == 3 else "left"
        ax_2d.text(mx + dx, my, f"MB{i + 1}", fontsize=19, fontweight="bold",
                   color="white", ha=ha, va="center", alpha=point_p,
                   path_effects=text_effects, zorder=10)

    ax_2d.set_xlim(0.0, 1.05)
    ax_2d.set_ylim(0.0, 1.05)
    ax_2d.set_aspect("equal", adjustable="box")
    ax_2d.set_xticks([0.0, 0.5, 1.0])
    ax_2d.set_yticks([0.0, 0.5, 1.0])
    ax_2d.set_xlabel("Location", fontsize=16, color="#202124")
    ax_2d.set_ylabel("Width", fontsize=16, color="#202124")
    ax_2d.tick_params(axis="both", which="major", labelsize=13, colors="#5f6368")
    ax_2d.spines["top"].set_visible(False)
    ax_2d.spines["right"].set_visible(False)
    ax_2d.spines["left"].set_color("#5f6368")
    ax_2d.spines["bottom"].set_color("#5f6368")

    cax = fig.add_axes([0.890, 0.285, 0.014, 0.46])
    gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cax.imshow(gradient, cmap=cmap, origin="lower", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 128, 255])
    cax.set_yticklabels(["0.0", "0.5", "1.0"])
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Density", fontsize=11, color="#202124", rotation=90, labelpad=8)
    cax.tick_params(axis="y", labelsize=10, colors="#5f6368", pad=2, length=2)
    for spine in cax.spines.values():
        spine.set_color("#c8ced8")
        spine.set_linewidth(0.8)

    fig.savefig(frame_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmm-path", type=Path, default=DEFAULT_GMM_PATH)
    parser.add_argument("--frames-dir", type=Path, default=Path("/tmp/decor_gmm_animation_frames"))
    parser.add_argument("--frames", type=int, default=110)
    parser.add_argument("--surface-res", type=int, default=120)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_frames:
        shutil.rmtree(args.frames_dir)
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    means = modified_means_from_gmm(args.gmm_path)
    maxima, clusters = maxima_from_means(means)
    sigma = 0.082085

    x = np.linspace(0.0, 1.05, args.surface_res)
    y = np.linspace(0.0, 1.05, args.surface_res)
    x_grid, y_grid = np.meshgrid(x, y)
    z = gaussian_mixture_density(x_grid, y_grid, means, sigma)
    z_ptp = np.ptp(z)
    z_norm = (z - z.min()) / z_ptp
    density_maxima = density_at(maxima, means, sigma, z.min(), z_ptp)

    for idx in range(args.frames):
        draw_frame(
            args.frames_dir / f"frame_{idx:04d}.png",
            idx,
            args.frames,
            x_grid,
            y_grid,
            z_norm,
            means,
            maxima,
            clusters,
            density_maxima,
        )
        if idx % 12 == 0 or idx == args.frames - 1:
            print(f"rendered {idx + 1}/{args.frames}")


if __name__ == "__main__":
    main()
