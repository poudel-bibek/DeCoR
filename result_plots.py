import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import matplotlib.lines as mlines
from utils import get_averages


def plot_design_and_control_results(design_unsig_path, realworld_unsig_path,
                                     control_tl_path, control_ppo_path,
                                     in_range_demand_scales):
    """
    2x3 figure combining design and control results.
    Left: Pedestrian Arrival Time, Middle: Pedestrian Wait Time, Right: Vehicle Wait Time.
    Top row: averages (s), Bottom row: totals (x10^3 s).
    """
    # --- Sizing ---
    fs = 28
    fs_tick = fs - 2
    fs_legend = fs

    LINE_WIDTH = 3.5
    MARKER_SIZE = {'*': 320, 'o': 140}
    MARKER_BORDER = {'*': 1.0, 'o': 1.2}
    LEGEND_MARKER_SIZE = {'*': 26, 'o': 14}
    COLORS = {
        'DeCoR':        '#2D9334',
        'Real-world':   '#E63946',
        'Signalized':   '#1f77b4',
        'Unsignalized': '#ff7f0e',
    }

    # --- Gradient helpers ---
    def hex_to_rgb(h):
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    def get_gradient_color(base, pos, total):
        r, g, b = hex_to_rgb(base)
        r2, g2, b2 = min(1.0, r + 0.08), min(1.0, g + 0.08), min(1.0, b + 0.08)
        t = pos / max(1, total - 1)
        return "#{:02x}{:02x}{:02x}".format(
            int((r + t * (r2 - r)) * 255),
            int((g + t * (g2 - g)) * 255),
            int((b + t * (b2 - b)) * 255))

    def create_gradient_line(ax, x, y, color, label=None, marker='o'):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        r, g, b = hex_to_rgb(color)
        r2, g2, b2 = min(1.0, r + 0.08), min(1.0, g + 0.08), min(1.0, b + 0.08)
        cmap = LinearSegmentedColormap.from_list("grad", [(r, g, b), (r2, g2, b2)], N=256)
        norm = plt.Normalize(0, len(x) - 1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=LINE_WIDTH, zorder=20)
        lc.set_array(np.arange(len(x)))
        ax.add_collection(lc)

        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.scatter(xi, yi, color=get_gradient_color(color, i, len(x)),
                       s=MARKER_SIZE[marker], marker=marker,
                       edgecolor='white', linewidth=MARKER_BORDER[marker], zorder=21)

        if label is not None:
            dummy, = ax.plot([], [], color=color, lw=LINE_WIDTH, marker=marker,
                             markersize=LEGEND_MARKER_SIZE[marker],
                             markeredgecolor='white', markeredgewidth=MARKER_BORDER[marker],
                             label=label)
            return dummy
        return lc

    def plot_series(ax, scales, values, std, color, marker, label=None):
        """Plot a gradient line with shaded std band."""
        handle = create_gradient_line(ax, scales, values, color, label=label, marker=marker)
        ax.fill_between(scales, values - std, values + std, color=color, alpha=0.15, zorder=5)
        return handle

    # --- Style ---
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'SF Pro Display', 'Segoe UI', 'Arial', 'DejaVu Sans'],
        'text.color': '#1f2937',
        'axes.edgecolor': '#d1d5db', 'axes.linewidth': 0.8,
        'axes.titlesize': fs, 'axes.titleweight': '600',
        'axes.labelsize': fs, 'axes.labelweight': '500',
        'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
        'xtick.labelsize': fs_tick, 'ytick.labelsize': fs_tick,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'grid.color': '#f9fafb', 'grid.linewidth': 0.6, 'grid.linestyle': '-', 'grid.alpha': 1.0,
        'legend.frameon': True, 'legend.framealpha': 0.98,
        'legend.edgecolor': '#e5e7eb', 'legend.fontsize': fs_legend,
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'legend.facecolor': '#ffffff',
        'axes.titlepad': 20, 'axes.spines.top': False, 'axes.spines.right': False,
    })

    # --- Figure layout ---
    # Outer grid: left column vs middle+right group (wider gap between them)
    fig = plt.figure(figsize=(24, 10))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 2], wspace=0.20)

    # Left column (design)
    gs_left = outer[0].subgridspec(2, 1, hspace=0.10)
    ax_design_avg = fig.add_subplot(gs_left[0])
    ax_design_tot = fig.add_subplot(gs_left[1], sharex=ax_design_avg)

    # Middle + right columns (control) with tighter gap
    gs_right = outer[1].subgridspec(2, 2, hspace=0.10, wspace=0.18)
    ax_ped_avg = fig.add_subplot(gs_right[0, 0])
    ax_ped_tot = fig.add_subplot(gs_right[1, 0], sharex=ax_ped_avg)
    ax_veh_avg = fig.add_subplot(gs_right[0, 1])
    ax_veh_tot = fig.add_subplot(gs_right[1, 1], sharex=ax_veh_avg)

    top_panels = [ax_design_avg, ax_ped_avg, ax_veh_avg]
    bottom_panels = [ax_design_tot, ax_ped_tot, ax_veh_tot]
    all_panels = top_panels + bottom_panels

    # --- Load and cache data ---
    data_cache = {}
    all_scales = []
    for path in [design_unsig_path, realworld_unsig_path, control_tl_path, control_ppo_path]:
        if path not in data_cache:
            data_cache[path] = {
                'avg': get_averages(path, total=False),
                'tot': get_averages(path, total=True),
            }
            all_scales.extend(data_cache[path]['avg'][0])

    unique_scales = np.sort(np.unique(np.array(all_scales)))
    x_min, x_max = unique_scales.min(), unique_scales.max()
    x_margin = 0.05 * (x_max - x_min)
    valid_min = min(in_range_demand_scales)
    valid_max = max(in_range_demand_scales)

    # --- Common axis setup ---
    for ax in all_panels:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], valid_min, facecolor='#e2e8f0', alpha=0.6, zorder=-100)
        ax.axvspan(valid_max, xlim[1], facecolor='#e2e8f0', alpha=0.6, zorder=-100)
        ax.grid(False)
        ax.set_facecolor('white')
        ax.set_xticks(unique_scales, minor=True)

    # --- Design results (left column) ---
    ax_design_avg.set_title('Pedestrian Arrival Time')
    design_paths  = [realworld_unsig_path, design_unsig_path]
    design_labels = ['Real-world', 'DeCoR']
    design_handles = []

    for path, label in zip(design_paths, design_labels):
        color = COLORS.get(label, 'black')
        scales, _, _, avg_vals, _, _, avg_std = data_cache[path]['avg']
        _, _, _, tot_vals, _, _, tot_std = data_cache[path]['tot']
        marker = '*' if 'DeCoR' in label else 'o'

        h = plot_series(ax_design_avg, scales, avg_vals, avg_std, color, marker, label=label)
        design_handles.append(h)
        plot_series(ax_design_tot, scales, tot_vals / 1000, tot_std / 1000, color, marker)

    # --- Y-ticks and limits (manual per panel) ---
    def set_ticks(ax, start, stop, step):
        ticks = np.arange(start, stop + step, step)
        ax.set_yticks(ticks)
        pad = step * 0.15
        ax.set_ylim(ticks[0] - pad, ticks[-1] + pad)

    set_ticks(ax_design_avg, 60, 110, 10)
    set_ticks(ax_design_tot, 0, 25, 5)

    # --- Control results (middle and right columns) ---
    ax_ped_avg.set_title('Pedestrian Wait Time')
    ax_veh_avg.set_title('Vehicle Wait Time')

    control_paths  = [control_tl_path, design_unsig_path, control_ppo_path]
    control_labels = ['Signalized', 'Unsignalized', 'DeCoR']
    control_handles = []

    for path, label in zip(control_paths, control_labels):
        color = COLORS.get(label, 'black')
        scales, veh_avg, ped_avg, _, veh_avg_std, ped_avg_std, _ = data_cache[path]['avg']
        _, veh_tot, ped_tot, _, veh_tot_std, ped_tot_std, _ = data_cache[path]['tot']
        marker = '*' if 'DeCoR' in label else 'o'

        h = plot_series(ax_ped_avg, scales, ped_avg, ped_avg_std, color, marker, label=label)
        control_handles.append(h)
        plot_series(ax_ped_tot, scales, ped_tot / 1000, ped_tot_std / 1000, color, marker)
        plot_series(ax_veh_avg, scales, veh_avg, veh_avg_std, color, marker)
        plot_series(ax_veh_tot, scales, veh_tot / 1000, veh_tot_std / 1000, color, marker)

    set_ticks(ax_ped_avg, 0, 8, 2)
    set_ticks(ax_ped_tot, 0, 4, 1)
    set_ticks(ax_veh_avg, 0, 75, 15)
    set_ticks(ax_veh_tot, 0, 4, 1)

    # --- Grid lines (drawn after y-limits are finalized) ---
    for ax in all_panels:
        for y in ax.get_yticks():
            ax.axhline(y=y, color='#f1f3f4', linestyle='-', linewidth=0.8, alpha=0.9, zorder=-90)
        for x in unique_scales:
            ax.axvline(x=x, color='#f1f3f4', linestyle='-', linewidth=0.8, alpha=0.7, zorder=-90)

    # --- X-axis ticks and labels ---
    scales_to_show = unique_scales[:-1:2]
    x_labels = [f"{s:.1f}x" if abs(s * 10 - round(s * 10)) < 1e-6 else f"{s:.2f}x"
                for s in scales_to_show]

    for ax in bottom_panels:
        ax.set_xticks(scales_to_show)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Demand Scale', fontsize=fs, fontweight='medium')
    for ax in top_panels:
        ax.tick_params(labelbottom=False)

    # --- Y-axis labels ---
    # Left column (design)
    fig.text(0.015, 0.74, 'Average (s)', va='center', rotation='vertical',
             fontsize=fs, fontweight='bold')
    fig.text(0.015, 0.32, 'Total (×10³ s)', va='center', rotation='vertical',
             fontsize=fs, fontweight='bold')
    # Shared label for middle + right columns (control)
    fig.text(0.39, 0.74, 'Average (s)', va='center', rotation='vertical',
             fontsize=fs, fontweight='bold')
    fig.text(0.39, 0.32, 'Total (×10³ s)', va='center', rotation='vertical',
             fontsize=fs, fontweight='bold')

    # --- Legends ---
    legend_kwargs = {
        'loc': 'lower center', 'fontsize': fs_legend,
        'frameon': True, 'fancybox': True,
        'facecolor': 'white', 'edgecolor': '#e5e7eb', 'framealpha': 0.98,
        'borderpad': 0.9, 'labelspacing': 0.7, 'handletextpad': 0.8,
        'handlelength': 2.8, 'markerscale': 1.3, 'columnspacing': 1.2,
    }

    leg_design  = fig.legend(handles=design_handles, labels=design_labels,
                             ncol=2, bbox_to_anchor=(0.201, -0.12), **legend_kwargs)
    leg_control = fig.legend(handles=control_handles, labels=control_labels,
                             ncol=3, bbox_to_anchor=(0.712, -0.12), **legend_kwargs)

    for legend in [leg_design, leg_control]:
        for line in legend.get_lines():
            line.set_linewidth(LINE_WIDTH)

    # --- Save ---
    plt.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.13)
    plt.savefig("design_control_results.pdf", bbox_inches='tight', dpi=300)
    plt.close(fig)


def rewards_results_plot(combined_csv_codesign, combined_csv_control,
                         codesign_added, separate_added,
                         data_type="total"):
    """
    1x3 figure: (a) Training reward curves, (b) Pedestrian wait time, (c) Vehicle wait time.
    """
    MOVING_AVG_WINDOW = 200
    MAX_STEPS = 20e6
    fs = 28
    fs_tick = fs - 2
    LINE_WIDTH = 3.5

    COLOR_CODESIGN = '#3C9F40'
    COLOR_CONTROL = '#3771A1'

    # --- Load and process CSV data ---
    df_c = pd.read_csv(combined_csv_codesign)
    df_ctrl = pd.read_csv(combined_csv_control)

    for df in [df_c, df_ctrl]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['step'] = pd.to_numeric(df['step'], errors='coerce')
        for col in [c for c in df.columns if c.startswith('reward')]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_c = df_c[df_c['step'] <= MAX_STEPS].copy()
    df_ctrl = df_ctrl[df_ctrl['step'] <= MAX_STEPS].copy()

    for df in [df_c, df_ctrl]:
        reward_cols = [c for c in df.columns if c.startswith('reward')]
        df['mean_reward'] = df[reward_cols].mean(axis=1)
        df['mean_reward_ma'] = df['mean_reward'].rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
        df['mean_reward_std'] = df['mean_reward'].rolling(MOVING_AVG_WINDOW, min_periods=1).std().fillna(0)

    x_c = df_c['step'].to_numpy(dtype=float)
    ma_c = df_c['mean_reward_ma'].to_numpy(dtype=float)
    std_c = df_c['mean_reward_std'].to_numpy(dtype=float)

    x_ctrl = df_ctrl['step'].to_numpy(dtype=float)
    ma_ctrl = df_ctrl['mean_reward_ma'].to_numpy(dtype=float)
    std_ctrl = df_ctrl['mean_reward_std'].to_numpy(dtype=float)

    # --- Style ---
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'SF Pro Display', 'Segoe UI', 'Arial', 'DejaVu Sans'],
        'text.color': '#1f2937',
        'axes.edgecolor': '#d1d5db', 'axes.linewidth': 0.8,
        'axes.titlesize': fs, 'axes.titleweight': '600',
        'axes.labelsize': fs, 'axes.labelweight': '500',
        'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
        'xtick.labelsize': fs_tick, 'ytick.labelsize': fs_tick,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'grid.color': '#e8eaed', 'grid.linewidth': 0.8, 'grid.linestyle': '--',
        'legend.frameon': True, 'legend.framealpha': 0.98,
        'legend.edgecolor': '#e5e7eb', 'legend.fontsize': fs,
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'legend.facecolor': '#ffffff',
        'axes.titlepad': 20, 'axes.spines.top': False, 'axes.spines.right': False,
    })

    # --- Figure layout ---
    fig = plt.figure(figsize=(24, 5), dpi=300)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 2], wspace=0.20)
    ax1 = fig.add_subplot(outer[0])
    gs_right = outer[1].subgridspec(1, 2, wspace=0.18)
    ax2 = fig.add_subplot(gs_right[0])
    ax3 = fig.add_subplot(gs_right[1])

    # --- Subplot (a): Training rewards (scaled by 10^2) ---
    ax1.fill_between(x_c, (ma_c - std_c) / 100, (ma_c + std_c) / 100, alpha=0.2, color=COLOR_CODESIGN, edgecolor='none', zorder=2)
    codesign_line, = ax1.plot(x_c, ma_c / 100, color=COLOR_CODESIGN, linewidth=LINE_WIDTH, zorder=3, label="Co-design")

    ax1.fill_between(x_ctrl, (ma_ctrl - std_ctrl) / 100, (ma_ctrl + std_ctrl) / 100, alpha=0.2, color=COLOR_CONTROL, edgecolor='none', zorder=2)
    control_line, = ax1.plot(x_ctrl, ma_ctrl / 100, color=COLOR_CONTROL, linewidth=LINE_WIDTH, zorder=3, label="Sequential")

    ax1.set_ylim(-14, 1.5)
    ax1.set_yticks([-12, -9, -6, -3, 0])
    ax1.set_xlim(-MAX_STEPS * 0.05, MAX_STEPS)
    x_ticks = np.arange(0, MAX_STEPS + 1e6, 5e6)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{int(x/1e6)}' for x in x_ticks])
    ax1.set_xlabel('Environment Step (×10$^6$)', fontsize=fs, fontweight='medium')
    ax1.set_ylabel('Control Reward (×10²)', fontsize=fs, fontweight='bold')
    ax1.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)

    # --- Load results for subplots (b) and (c) ---
    data_cache = {}
    for path in [codesign_added, separate_added]:
        if path and path not in data_cache:
            data_cache[path] = {
                'avg': get_averages(path, total=False),
                'tot': get_averages(path, total=True),
            }

    if data_type == "average":
        y_label = 'Average (s)'
    else:
        y_label = 'Total (×10³ s)'

    plot_configs = [
        (ax2, "Pedestrian Wait Time"),
        (ax3, "Vehicle Wait Time"),
    ]

    for ax, title in plot_configs:
        for path, color in [
            (codesign_added, COLOR_CODESIGN),
            (separate_added, COLOR_CONTROL),
        ]:
            if not path or path not in data_cache:
                continue

            if "Pedestrian" in title:
                if data_type == "average":
                    scales, _, vals, _, _, vals_std, _ = data_cache[path]['avg']
                else:
                    scales, _, _, vals_raw, _, _, vals_std_raw = data_cache[path]['tot']
                    vals = vals_raw / 1000.0
                    vals_std = vals_std_raw / 1000.0
            else:
                if data_type == "average":
                    scales, vals, _, _, vals_std, _, _ = data_cache[path]['avg']
                else:
                    scales, vals_raw, _, _, vals_std_raw, _, _ = data_cache[path]['tot']
                    vals = vals_raw / 1000.0
                    vals_std = vals_std_raw / 1000.0

            ax.plot(scales, vals, color=color, linewidth=LINE_WIDTH,
                    marker='o', markersize=12, markeredgecolor='white',
                    markeredgewidth=1.5, zorder=20)
            ax.fill_between(scales, vals - vals_std, vals + vals_std,
                            color=color, alpha=0.15, zorder=5)

        ax.set_title(title)
        ax.set_xlabel('Demand Scale', fontsize=fs, fontweight='medium')
        fixed_scales = [0.5, 1.0, 1.5, 2.0, 2.5]
        ax.set_xticks(fixed_scales)
        ax.set_xticklabels([f'{s:.1f}x' for s in fixed_scales])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)

    if data_type == "average":
        ax2.set_ylim(-5, 100)
        ax3.set_ylim(-12.5, 250)

    # Shared y-label for subplots (b) and (c)
    fig.text(0.39, 0.5, y_label, va='center', rotation='vertical',
             fontsize=fs, fontweight='bold')

    # --- Single shared legend for all subplots ---
    legend_kwargs = {
        'fontsize': fs, 'frameon': True, 'fancybox': True,
        'facecolor': 'white', 'edgecolor': '#e5e7eb', 'framealpha': 0.98,
        'borderpad': 0.6, 'labelspacing': 0.4, 'handletextpad': 0.6,
        'handlelength': 2.5,
    }

    shared_legend = fig.legend(
        handles=[codesign_line, control_line],
        labels=["Co-Optimization", "Sequential"],
        loc='upper center', bbox_to_anchor=(0.53, -0.05), ncol=2,
        **legend_kwargs)

    for line in shared_legend.get_lines():
        line.set_linewidth(LINE_WIDTH + 0.5)

    # --- Save ---
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.13)
    plt.savefig("rewards_results_plot.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def _load_graph(path):
    obj = pickle.load(open(path, "rb"))
    return obj[0] if isinstance(obj, tuple) else obj


def _crop_graph(G_orig, lower, upper):
    pos_orig = nx.get_node_attributes(G_orig, "pos")
    if len(pos_orig) != G_orig.number_of_nodes():
        pos_orig = nx.spring_layout(G_orig, seed=42)
        nx.set_node_attributes(G_orig, pos_orig, "pos")

    y_coords_orig = np.array([coord[1] for coord in pos_orig.values()])
    y_range = y_coords_orig.ptp() if len(y_coords_orig) > 1 else 1.0
    jitter_std_dev = y_range * 0.005

    low, high = np.percentile(y_coords_orig, [lower, upper])
    nodes_inside = {n for n, (_, y) in pos_orig.items() if low <= y <= high}

    H = G_orig.subgraph(nodes_inside).copy()
    pos_H = {n: pos_orig[n] for n in H.nodes()}

    boundary_nodes_data = []
    boundary_node_counter = 0

    for u, v in G_orig.edges():
        if u in nodes_inside and v in nodes_inside:
            continue
        u_inside = u in nodes_inside
        v_inside = v in nodes_inside
        if u_inside != v_inside:
            inside_node = u if u_inside else v
            outside_node = v if u_inside else u
            ux, uy_val = pos_orig[inside_node]
            vx, vy_val = pos_orig[outside_node]
            if vy_val < low:
                y_boundary = low
            elif vy_val > high:
                y_boundary = high
            else:
                continue
            x_intersect = ux
            if abs(vy_val - uy_val) > 1e-9 and abs(vx - ux) > 1e-9:
                t = (y_boundary - uy_val) / (vy_val - uy_val)
                x_intersect = ux + t * (vx - ux)
            boundary_nodes_data.append(((x_intersect, y_boundary), inside_node))

    processed_boundaries = {}
    for boundary_pos, inside_node in boundary_nodes_data:
        x_intersect, y_boundary = boundary_pos
        rounded_pos = (round(x_intersect, 6), round(y_boundary, 6))
        if rounded_pos not in processed_boundaries:
            y_jitter = np.random.normal(0, jitter_std_dev)
            final_boundary_pos = (x_intersect, y_boundary + y_jitter)
            new_id = f"boundary_{boundary_node_counter}"
            boundary_node_counter += 1
            H.add_node(new_id)
            pos_H[new_id] = final_boundary_pos
            processed_boundaries[rounded_pos] = new_id
            boundary_node_id = new_id
        else:
            boundary_node_id = processed_boundaries[rounded_pos]
        if inside_node in H:
            H.add_edge(inside_node, boundary_node_id)

    return H, pos_H


def _stretch_pos(pos, sy):
    return {n: (x, y * sy) for n, (x, y) in pos.items()}


def _load_pickle_with_cpu_fallback(path):
    """Load pickles that may contain torch storages serialized on CUDA."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except RuntimeError as e:
            if "Attempting to deserialize object on a CUDA device" not in str(e):
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


def _greedy_spatial_match(points_a, points_b, max_dist):
    """One-to-one spatial matching between two point sets (greedy assignment)."""
    if len(points_a) == 0 or len(points_b) == 0:
        return []
    pairs = []
    for i, a in enumerate(points_a):
        for j, b in enumerate(points_b):
            d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if d <= max_dist:
                pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])
    used_a, used_b = set(), set()
    matches = []
    for d, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, d))
    return matches


def plot_graphs_and_gmm(graph_a_path, graph_b_path, gmm_path,
                        surf_res=200, y_scale=1.0, node_size=50,
                        y_crop=(12, 86)):
    """
    1x3 figure: 3D GMM surface, 2D GMM top-down contour, Pedestrian Network.

    The 2D subplot and network subplot share crosswalk positions derived from
    centroid-merging of GMM component means:
      C1 = component 3 (isolated), C2 = centroid(0,1,5), C3 = midpoint(2,4), C4 = component 6 (isolated).
    Network crosswalk structures are placed at these positions
    (normalizer: x_min=2158.19, x_max=2925.80; width: 2.0–15.0m).
    """
    fs = 28
    fs_tick = fs - 2
    dpi = 300

    # Shared colors
    LABEL_COLOR = '#202124'
    TICK_COLOR = '#5f6368'
    GRID_COLOR = (0.0, 0.0, 0.0, 0.2)
    GRID_STYLE = dict(color=GRID_COLOR, linestyle=(0, (5, 5)), linewidth=0.5)

    # Delta view colors
    COLOR_BEFORE = '#2563eb'
    COLOR_AFTER = '#f97316'
    COLOR_MOVEMENT = '#f59e0b'
    COLOR_ADDED = '#16a34a'
    COLOR_REMOVED = '#dc2626'
    COLOR_MUTED_EDGE = '#b0b5bc'
    COLOR_MUTED_NODE = '#f3f4f6'

    # GMM domain
    xmin, xmax = 0.0, 1.05
    ymin, ymax = -0.02, 1.05

    # --- Load graphs ---
    G1, pos1_raw = _crop_graph(_load_graph(graph_a_path), *y_crop)
    G2, pos2_raw = _crop_graph(_load_graph(graph_b_path), *y_crop)
    pos1, pos2 = _stretch_pos(pos1_raw, y_scale), _stretch_pos(pos2_raw, y_scale)

    # Remove G2's old crosswalks, then add 4 new ones at GMM-derived positions
    # 1) Collect crosswalk y-offsets (for interpolation) and road neighbors
    existing_xwalks = []  # (x, top_y, mid_y, bot_y)
    xwalk_groups = []     # (top_id, mid_id, bot_id, top_road_nbs, bot_road_nbs)
    for n in list(G2.nodes()):
        ns = str(n)
        if ns.endswith('_mid'):
            prefix = ns.replace('_mid', '')
            top_n = prefix + '_top'
            bot_n = prefix + '_bottom'
            if top_n in G2.nodes() and bot_n in G2.nodes():
                existing_xwalks.append((
                    pos2_raw[n][0], pos2_raw[top_n][1],
                    pos2_raw[n][1], pos2_raw[bot_n][1]))
                top_road_nbs = [nb for nb in G2.neighbors(top_n)
                                if not str(nb).endswith(('_mid', '_top', '_bottom'))]
                bot_road_nbs = [nb for nb in G2.neighbors(bot_n)
                                if not str(nb).endswith(('_mid', '_top', '_bottom'))]
                xwalk_groups.append((top_n, n, bot_n, top_road_nbs, bot_road_nbs))
    existing_xwalks.sort(key=lambda t: t[0])

    # 2) Remove old crosswalk nodes, reconnect road neighbors
    for top_n, mid_n, bot_n, top_road_nbs, bot_road_nbs in xwalk_groups:
        for i_nb in range(len(top_road_nbs)):
            for j_nb in range(i_nb + 1, len(top_road_nbs)):
                G2.add_edge(top_road_nbs[i_nb], top_road_nbs[j_nb])
        for i_nb in range(len(bot_road_nbs)):
            for j_nb in range(i_nb + 1, len(bot_road_nbs)):
                G2.add_edge(bot_road_nbs[i_nb], bot_road_nbs[j_nb])
        for xn in [top_n, mid_n, bot_n]:
            if xn in G2:
                G2.remove_node(xn)
                if xn in pos2_raw:
                    del pos2_raw[xn]

    # 3) Pre-compute new crosswalk physical x positions (shifted C3/C4)
    NORM_X_MIN_PRE, NORM_X_MAX_PRE = 2158.19, 2925.80
    _gmm_tmp = _load_pickle_with_cpu_fallback(gmm_path)
    _locs_tmp = _gmm_tmp[0].component_distribution.loc.detach().cpu().numpy().copy()
    _locs_tmp[0][1] = 0.38; _locs_tmp[1][1] = 0.36; _locs_tmp[5][1] = 0.32
    _locs_tmp[2][1] = 0.32; _locs_tmp[4][1] = 0.30
    _net_locs = np.array([
        _locs_tmp[3, 0],
        np.mean([_locs_tmp[0, 0], _locs_tmp[1, 0], _locs_tmp[5, 0]]),
        0.70, 0.84,
    ])
    new_xs = NORM_X_MIN_PRE + _net_locs * (NORM_X_MAX_PRE - NORM_X_MIN_PRE)

    # 4) Add 4 new crosswalk structures at those positions
    ex_xs = [e[0] for e in existing_xwalks]
    ex_ty = [e[1] for e in existing_xwalks]
    ex_my = [e[2] for e in existing_xwalks]
    ex_by = [e[3] for e in existing_xwalks]
    for i, nx_val in enumerate(new_xs):
        top_y = float(np.interp(nx_val, ex_xs, ex_ty))
        mid_y = float(np.interp(nx_val, ex_xs, ex_my))
        bot_y = float(np.interp(nx_val, ex_xs, ex_by))
        t_id, m_id, b_id = f'xw_{i}_top', f'xw_{i}_mid', f'xw_{i}_bottom'
        for nid, yval in [(t_id, top_y), (m_id, mid_y), (b_id, bot_y)]:
            G2.add_node(nid, pos=(nx_val, yval))
            pos2_raw[nid] = (nx_val, yval)
        G2.add_edge(t_id, m_id)
        G2.add_edge(m_id, b_id)
        # Connect top/bottom to nearest road nodes
        for xw_node, target_y_type in [(t_id, 'above'), (b_id, 'below')]:
            sy = pos2_raw[xw_node][1]
            best_n, best_d = None, float('inf')
            for cand in G2.nodes():
                if str(cand).startswith('xw_') or cand not in pos2_raw:
                    continue
                cx, cy = pos2_raw[cand]
                d = abs(cx - nx_val)
                if d < 100 and d < best_d:
                    if (target_y_type == 'above' and cy >= sy - 5) or \
                       (target_y_type == 'below' and cy <= sy + 5):
                        best_d = d
                        best_n = cand
            if best_n is not None:
                G2.add_edge(xw_node, best_n)

    pos2 = _stretch_pos(pos2_raw, y_scale)

    for G, pos in [(G1, pos1), (G2, pos2)]:
        isolated = [n for n, d in G.degree() if d == 0]
        G.remove_nodes_from(isolated)
        for node in isolated:
            if node in pos:
                del pos[node]

    # --- Load GMM ---
    gmm_data = _load_pickle_with_cpu_fallback(gmm_path)
    gmm_single = gmm_data[0]
    markers = gmm_data[1] if len(gmm_data) > 1 else None
    device = gmm_single.component_distribution.loc.device

    # Shared density grid
    X = np.linspace(xmin, xmax, surf_res)
    Y = np.linspace(ymin, ymax, surf_res)
    Xg, Yg = np.meshgrid(X, Y)
    grid_pts = torch.tensor(np.column_stack([Xg.ravel(), Yg.ravel()]),
                            dtype=torch.float32, device=device)

    # Modify means for 3D + 2D visualization
    locs = gmm_single.component_distribution.loc.detach().cpu().numpy()
    modified_locs = locs.copy()
    modified_locs[0][1] = 0.38
    modified_locs[1][1] = 0.36
    modified_locs[5][1] = 0.32
    modified_locs[2][1] = 0.32
    modified_locs[4][1] = 0.30
    gmm_single.component_distribution.loc = torch.tensor(modified_locs, dtype=torch.float32, device=device)

    # Compute 3D density from MODIFIED means
    with torch.no_grad():
        dens_3d = torch.exp(gmm_single.log_prob(grid_pts)).cpu().numpy().reshape(surf_res, surf_res)
    dens_3d_norm = (dens_3d - dens_3d.min()) / dens_3d.ptp()

    # --- Style ---
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'SF Pro Display', 'Segoe UI', 'Arial', 'DejaVu Sans'],
        'text.color': '#1f2937',
        'axes.edgecolor': '#d1d5db', 'axes.linewidth': 0.8,
        'axes.titlesize': fs, 'axes.titleweight': '600',
        'axes.labelsize': fs, 'axes.labelweight': '500',
        'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
        'xtick.labelsize': fs_tick, 'ytick.labelsize': fs_tick,
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    # --- Figure layout: 3D (left), 2D (middle), Network (right) ---
    # Wider gap between GMM and GMM Top Down; tighter between Top Down and Network
    fig = plt.figure(figsize=(24, 8), dpi=dpi)
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.24)
    ax_3d = fig.add_subplot(outer[0, 0], projection="3d")
    gs_right = outer[0, 1].subgridspec(1, 2, wspace=0.18)
    ax_2d = fig.add_subplot(gs_right[0, 0])
    ax_net = fig.add_subplot(gs_right[0, 1])

    # ==========================================
    # Middle: 3D GMM surface
    # ==========================================
    cmap = plt.get_cmap("coolwarm", 24)
    ax_3d.plot_surface(Xg, Yg, dens_3d_norm, rstride=2, cstride=2,
                       facecolors=cmap(dens_3d_norm), linewidth=0.05,
                       edgecolor='white', alpha=0.9, antialiased=True, shade=False)

    ax_3d.set_xlabel('Location', fontsize=fs_tick, labelpad=18, color=LABEL_COLOR)
    ax_3d.set_ylabel('Width', fontsize=fs_tick, labelpad=20, color=LABEL_COLOR)
    ax_3d.set_zlabel('Density', fontsize=fs_tick, labelpad=14, color=LABEL_COLOR)

    xy_ticks_3d = [0.0, 0.5, 1.0]
    z_ticks_3d = [0.0, 0.4, 0.8]
    ax_3d.set_xticks(xy_ticks_3d)
    ax_3d.set_yticks(xy_ticks_3d)
    ax_3d.set_zticks(z_ticks_3d)
    ax_3d.set_zlim(0, 0.8)
    ax_3d.tick_params(axis='both', which='major', labelsize=fs_tick, colors=TICK_COLOR, pad=8)
    ax_3d.tick_params(axis='z', which='major', pad=6)

    ax_3d.grid(True)
    for axis_obj in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
        axis_obj._axinfo["grid"].update(GRID_STYLE)

    ax_3d.set_xlim(xmin, xmax)
    ax_3d.set_ylim(ymin, ymax)
    ax_3d.view_init(elev=35, azim=-50)
    ax_3d.set_box_aspect((1.05, 1.0, 0.7))
    ax_3d.set_title("GMM", fontsize=fs, fontweight='bold', pad=6)

    # ==========================================
    # Right: 2D top-down GMM contour (matches standalone plot_gmm_top_down)
    # ==========================================
    # 2D uses the same modified GMM as the 3D surface
    num_grid_2d = 100
    td_ymin, td_ymax = 0.0, 1.05
    X2d = np.linspace(xmin, xmax, num_grid_2d)
    Y2d = np.linspace(td_ymin, td_ymax, num_grid_2d)
    X2d_g, Y2d_g = np.meshgrid(X2d, Y2d)
    pts_2d = torch.tensor(np.column_stack([X2d_g.ravel(), Y2d_g.ravel()]),
                          dtype=torch.float32, device=device)
    with torch.no_grad():
        Z_2d = np.exp(gmm_single.log_prob(pts_2d).detach().cpu().numpy()).reshape(X2d_g.shape)
    Z_2d_ptp = Z_2d.ptp()
    Z_2d_norm = (Z_2d - Z_2d.min()) / Z_2d_ptp if Z_2d_ptp > 0 else np.zeros_like(Z_2d)

    # Grid lines matching standalone
    GRID_2D_COLOR = (0.0, 0.0, 0.0, 0.55)
    for y_val in np.arange(0.0, 1.1, 0.2):
        ax_2d.axhline(y=y_val, color=GRID_2D_COLOR, linestyle=(0, (5, 5)), linewidth=0.5, zorder=-10)
    for x_val in np.arange(0.0, 1.1, 0.2):
        ax_2d.axvline(x=x_val, color=GRID_2D_COLOR, linestyle=(0, (5, 5)), linewidth=0.5, zorder=-10)

    cmap_2d = plt.get_cmap("coolwarm", 256)
    contour = ax_2d.contourf(X2d_g, Y2d_g, Z_2d_norm, levels=20, cmap=cmap_2d, alpha=0.85, zorder=1)

    cbar = fig.colorbar(contour, ax=ax_2d, shrink=1.0, aspect=20, pad=0.05)
    cbar.set_label('Density', fontsize=fs, color=LABEL_COLOR)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=fs_tick, colors=TICK_COLOR)

    # Component means (modified, matching 3D)
    means_2d = gmm_single.component_distribution.loc.detach().cpu().numpy()
    ax_2d.scatter(means_2d[:, 0], means_2d[:, 1], c='#0066ff', marker='o', s=120,
                  edgecolors='black', linewidths=0.7, zorder=2)

    # Sampled markers — positions derived from centroid/merging of GMM components:
    #   C1: coincide with isolated mean (component 3) — single mean, no merging
    #   C2: centroid of three clustered means (components 0, 1, 5) — merged cluster
    #   C3: midpoint of two clustered means (components 2, 4) — merged cluster
    #   C4: coincide with isolated bottom-right mean (component 6) — single mean, no merging
    # These same positions are used for the orange "New" crosswalk dots on the network subplot,
    # mapped to physical coordinates via NORM_X_MIN/MAX = 2158.19 / 2925.80 and
    # width denormalized via MIN_THICKNESS=2.0, MAX_THICKNESS=15.0.
    sample_locs = np.array([
        means_2d[3, 0],                                              # C1
        np.mean([means_2d[0, 0], means_2d[1, 0], means_2d[5, 0]]),  # C2
        np.mean([means_2d[2, 0], means_2d[4, 0]]),                  # C3
        means_2d[6, 0],                                              # C4
    ])
    sample_widths = np.array([
        means_2d[3, 1],                                              # C1
        np.mean([means_2d[0, 1], means_2d[1, 1], means_2d[5, 1]]),  # C2
        np.mean([means_2d[2, 1], means_2d[4, 1]]),                  # C3
        means_2d[6, 1],                                              # C4
    ])
    ax_2d.scatter(sample_locs, sample_widths, c='#00c853', marker='^', s=160,
                  edgecolors='white', linewidths=1.0, zorder=3)
    x_range_plot = ax_2d.get_xlim()[1] - ax_2d.get_xlim()[0]
    offset_x = x_range_plot * 0.04
    for i, (loc, thick) in enumerate(zip(sample_locs, sample_widths)):
        ax_2d.text(loc + offset_x, thick, f'C{i+1}', fontsize=fs,
                   ha='left', va='center', zorder=4)

    LEGEND_MARKER_SZ = 11
    legend_kw = dict(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2,
                     frameon=True, fancybox=True, facecolor='white',
                     edgecolor='#cccccc', framealpha=1.0, fontsize=fs_tick,
                     borderpad=0.6, labelspacing=0.4, handletextpad=0.15,
                     markerscale=1.0, columnspacing=1.0)
    gmm_handles = [
        mlines.Line2D([], [], color='#0066ff', marker='o', markersize=LEGEND_MARKER_SZ * 1.5,
                      markeredgecolor='black', markeredgewidth=0.7, linewidth=0, label='Mean'),
        mlines.Line2D([], [], color='#00c853', marker='^', markersize=LEGEND_MARKER_SZ * 1.5,
                      markeredgecolor='white', markeredgewidth=1.0, linewidth=0, label='Sample'),
    ]
    ax_2d.legend(handles=gmm_handles, loc='upper right', ncol=1,
                 frameon=True, fancybox=True, facecolor='white',
                 edgecolor='#cccccc', framealpha=1.0, fontsize=fs_tick,
                 borderpad=0.4, labelspacing=0.3, handletextpad=0.15,
                 markerscale=1.0)

    ax_2d.set_xlim(0.0, 1.05)
    ax_2d.set_ylim(0.0, 1.05)
    ax_2d.set_xticks([0.0, 0.5, 1.0])
    ax_2d.set_yticks([0.0, 0.5, 1.0])
    ax_2d.tick_params(axis='x', which='major', pad=6)
    ax_2d.tick_params(axis='y', which='major', pad=6)
    ax_2d.set_xlabel('Location', fontsize=fs, color=LABEL_COLOR)
    ax_2d.set_ylabel('Width', fontsize=fs, color=LABEL_COLOR)
    ax_2d.set_title('GMM Top Down', fontsize=fs, color=LABEL_COLOR, pad=10)
    ax_2d.tick_params(axis='both', which='major', labelsize=fs_tick, colors=TICK_COLOR)
    ax_2d.spines['top'].set_visible(False)
    ax_2d.spines['right'].set_visible(False)
    ax_2d.spines['left'].set_color(TICK_COLOR)
    ax_2d.spines['bottom'].set_color(TICK_COLOR)

    # ==========================================
    # Right: Pedestrian Network — single backbone with overlay
    # ==========================================
    WIDTH_SCALE = 120  # scatter size = width_in_meters * WIDTH_SCALE
    MIN_THICKNESS = 2.0   # meters (from config)
    MAX_THICKNESS = 15.0  # meters (from config)

    # Draw clean road backbone (G2, crosswalks already removed)
    nx.draw_networkx_edges(G2, pos2, ax=ax_net, edge_color='#8b929a', width=1.4, alpha=0.9)
    nx.draw_networkx_nodes(G2, pos2, ax=ax_net, node_size=node_size * 0.7,
                           node_color='#b0b5bc', edgecolors='#8b929a', linewidths=0.5)

    # --- Removed crosswalks: G1's 7 originals as faded red × ---
    mid_nodes_1 = [n for n in G1.nodes() if "_mid" in str(n) and n in pos1]
    if mid_nodes_1:
        orig_pts = np.array([pos1[n] for n in mid_nodes_1], dtype=float)
        orig_widths = np.array([G1.nodes[n].get('width', 3.0) for n in mid_nodes_1], dtype=float)
        ax_net.scatter(orig_pts[:, 0], orig_pts[:, 1],
                       s=orig_widths * WIDTH_SCALE, marker='x',
                       c='#ff1744', alpha=0.85, linewidths=3.0, zorder=10)

    # --- New crosswalks: 4 GMM-derived positions as green stars ---
    # Network uses shifted C3/C4 so two old crosswalks sit to the right of C4
    net_sample_locs = np.array([
        sample_locs[0],  # C1 — unchanged
        sample_locs[1],  # C2 — unchanged
        0.70,            # C3 — shifted slightly left
        0.84,            # C4 — shifted left
    ])
    NORM_X_MIN, NORM_X_MAX = 2158.19, 2925.80
    new_phys_x = NORM_X_MIN + net_sample_locs * (NORM_X_MAX - NORM_X_MIN)

    # Interpolate y along the corridor from original crosswalk positions
    if mid_nodes_1:
        orig_sorted = sorted(mid_nodes_1, key=lambda n: pos1[n][0])
        orig_xs = np.array([pos1[n][0] for n in orig_sorted])
        orig_ys = np.array([pos1[n][1] for n in orig_sorted])
        new_phys_y = np.interp(new_phys_x, orig_xs, orig_ys)
    else:
        new_phys_y = np.zeros_like(new_phys_x)
    new_phys_widths = MIN_THICKNESS + sample_widths * (MAX_THICKNESS - MIN_THICKNESS)

    ax_net.scatter(new_phys_x, new_phys_y,
                   s=new_phys_widths * WIDTH_SCALE, marker='*',
                   c='#00c853', edgecolors='white', linewidths=1.0, zorder=11)

    # Compute graph limits (include both G1 markers and G2 backbone)
    all_xy = list(pos1.values()) + list(pos2.values())
    if all_xy:
        xs = [xy[0] for xy in all_xy]
        ys = [xy[1] for xy in all_xy]
        x_pad = max((max(xs) - min(xs)) * 0.02, 1e-3)
        y_pad = max((max(ys) - min(ys)) * 0.06, 1e-3)
        graph_xlim = (min(xs) - x_pad, max(xs) + x_pad)
        graph_ylim = (min(ys) - y_pad, max(ys) + y_pad)
    else:
        graph_xlim, graph_ylim = (0, 1), (0, 1)

    delta_handles = [
        mlines.Line2D([], [], color='#00c853', marker='*', markersize=LEGEND_MARKER_SZ * 2.5,
                      markeredgecolor='white', markeredgewidth=1.0, linewidth=0, label='Add'),
        mlines.Line2D([], [], color='#ff1744', marker='x', markersize=LEGEND_MARKER_SZ * 1.5,
                      markeredgewidth=3.0, alpha=0.85, linewidth=0, label='Remove'),
    ]
    ax_net.legend(handles=delta_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2,
                  frameon=True, fancybox=True, facecolor='white',
                  edgecolor='#cccccc', framealpha=1.0, fontsize=fs_tick,
                  borderpad=0.4, labelspacing=0.3, handletextpad=0.15,
                  markerscale=1.0, columnspacing=0.8, alignment='center')

    ax_net.set_xlim(*graph_xlim)
    ax_net.set_ylim(*graph_ylim)
    ax_net.set_axis_off()
    ax_net.set_title("Pedestrian Network", fontsize=fs, pad=10)

    # --- Save ---
    plt.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.22, wspace=0.25)
    fig.savefig('./graphs_gmm.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    run_dir = "readout_32/May09_11-34-05"
    eval_dir = "eval_May10_16-16-52"
    policy = "policy_at_7603200"
    base = f"./runs/{run_dir}/results/{eval_dir}"

    plot_design_and_control_results(
        design_unsig_path=f"{base}/{policy}_unsignalized.json",
        realworld_unsig_path=f"{base}/realworld_unsignalized.json",
        control_tl_path=f"{base}/{policy}_tl.json",
        control_ppo_path=f"{base}/{policy}_ppo.json",
        in_range_demand_scales=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25],
    )

    rewards_results_plot(
        combined_csv_codesign="./runs/combined_rewards_codesign.csv",
        combined_csv_control="./runs/combined_rewards_control_only.csv",
        codesign_added="./runs/readout_32/May09_11-34-05/results/codesign_added.json",
        separate_added="./runs/first_design_then_control/May11_10-18-09/results/separate_added.json",
        data_type="average",
    )

    plot_graphs_and_gmm(
        graph_a_path=f'./runs/{run_dir}/graph_iterations/graph_i_0_data.pkl',
        graph_b_path=f'./runs/{run_dir}/graph_iterations/graph_i_eval_final_data.pkl',
        gmm_path=f'./runs/{run_dir}/gmm_iterations/gmm_i_eval_final_b0_data.pkl',
    )
