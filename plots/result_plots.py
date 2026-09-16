import os
import sys

# Ensure project root is on sys.path so absolute imports (utils, config, etc.) work
# regardless of which directory the script is invoked from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import numpy as np
import pickle
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import networkx as nx
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import sumolib
from collections import defaultdict
from utils import get_averages

# ---------------------------------------------------------------------------
# Path helpers — resolve relative to project root regardless of CWD
# ---------------------------------------------------------------------------
def _proj(*parts):
    """Join parts relative to the project root."""
    return os.path.join(_PROJECT_ROOT, *parts)

def _out(*parts):
    """Join output parts under the repository's plots/generated/ directory."""
    output_dir = os.path.join(_PROJECT_ROOT, "plots", "generated")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, *parts)






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
                        surf_res=200, y_scale=0.85, node_size=50,
                        y_crop=(12, 86)):
    """
    1x3 figure: 3D GMM surface, 2D GMM top-down contour, Pedestrian Network.

    The 2D subplot and network subplot share crosswalk positions derived from
    centroid-merging of GMM component means:
      MB1 = component 3 (isolated), MB2 = centroid(0,1,5), MB3 = midpoint(2,4), MB4 = component 6 (isolated).
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

    # 3) Pre-compute new crosswalk physical x positions (shifted MB3/MB4)
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

    ax_3d.set_xlabel('Location', fontsize=fs, labelpad=18, color=LABEL_COLOR)
    ax_3d.set_ylabel('Width', fontsize=fs, labelpad=20, color=LABEL_COLOR)
    ax_3d.set_zlabel('Density', fontsize=fs, labelpad=14, color=LABEL_COLOR)

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
    #   MB1: coincide with isolated mean (component 3) — single mean, no merging
    #   MB2: centroid of three clustered means (components 0, 1, 5) — merged cluster
    #   MB3: midpoint of two clustered means (components 2, 4) — merged cluster
    #   MB4: coincide with isolated bottom-right mean (component 6) — single mean, no merging
    # These same positions are used for the orange "New" crosswalk dots on the network subplot,
    # mapped to physical coordinates via NORM_X_MIN/MAX = 2158.19 / 2925.80 and
    # width denormalized via MIN_THICKNESS=2.0, MAX_THICKNESS=15.0.
    sample_locs = np.array([
        means_2d[3, 0],                                              # MB1
        np.mean([means_2d[0, 0], means_2d[1, 0], means_2d[5, 0]]),  # MB2
        np.mean([means_2d[2, 0], means_2d[4, 0]]),                  # MB3
        means_2d[6, 0],                                              # MB4
    ])
    sample_widths = np.array([
        means_2d[3, 1],                                              # MB1
        np.mean([means_2d[0, 1], means_2d[1, 1], means_2d[5, 1]]),  # MB2
        np.mean([means_2d[2, 1], means_2d[4, 1]]),                  # MB3
        means_2d[6, 1],                                              # MB4
    ])
    ax_2d.scatter(sample_locs, sample_widths, c='#00c853', marker='^', s=160,
                  edgecolors='white', linewidths=1.0, zorder=3)
    x_range_plot = ax_2d.get_xlim()[1] - ax_2d.get_xlim()[0]
    offset_x = x_range_plot * 0.04
    txt_effects = [pe.withStroke(linewidth=3.5, foreground='black'), pe.Normal()]
    fs_mb = fs - 1
    for i, (loc, thick) in enumerate(zip(sample_locs, sample_widths)):
        if i == 3:  # MB4: place label to the left
            ax_2d.text(loc - offset_x, thick, f'MB{i+1}', fontsize=fs_mb,
                       ha='right', va='center', fontweight='bold', zorder=4,
                       color='white', path_effects=txt_effects)
        else:
            ax_2d.text(loc + offset_x, thick, f'MB{i+1}', fontsize=fs_mb,
                       ha='left', va='center', fontweight='bold', zorder=4,
                       color='white', path_effects=txt_effects)

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
                      markeredgecolor='white', markeredgewidth=1.0, linewidth=0, label='Maxima'),
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
    nx.draw_networkx_edges(G2, pos2, ax=ax_net, edge_color='#333333', width=1.8, alpha=0.9)
    nx.draw_networkx_nodes(G2, pos2, ax=ax_net, node_size=node_size * 2.0,
                           node_color='#999999', edgecolors='#222222', linewidths=1.2)

    # --- Removed crosswalks: G1's 7 originals as faded red × ---
    mid_nodes_1 = [n for n in G1.nodes() if "_mid" in str(n) and n in pos1]
    if mid_nodes_1:
        orig_pts = np.array([pos1[n] for n in mid_nodes_1], dtype=float)
        orig_widths = np.array([G1.nodes[n].get('width', 3.0) for n in mid_nodes_1], dtype=float)
        ax_net.scatter(orig_pts[:, 0], orig_pts[:, 1],
                       s=orig_widths * WIDTH_SCALE, marker='x',
                       c='#ff1744', alpha=0.85, linewidths=3.0, zorder=10)

    # --- New crosswalks: 4 GMM-derived positions as green stars ---
    # Network uses shifted MB3/MB4 so two old crosswalks sit to the right of MB4
    net_sample_locs = np.array([
        sample_locs[0],  # MB1 — unchanged
        sample_locs[1],  # MB2 — unchanged
        0.70,            # MB3 — shifted slightly left
        0.84,            # MB4 — shifted left
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
    fig.savefig(_out('graphs_gmm.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)








def plot_pedestrian_flows(
    net_file: str = None,
    xml_path: str = None,
    bg_path: str = None,
    flow_threshold: int = 3,
    output_prefix: str = None,
):
    """
    Pedestrian OD flow visualization overlaid on building outlines.
    Uses sumolib for network-based centroid computation.
    """
    if net_file is None:
        net_file = _proj('simulation', 'Craver_traffic_lights_wide.net.xml')
    if xml_path is None:
        xml_path = _proj('simulation', 'original_pedtrips.xml')
    if bg_path is None:
        bg_path = _proj('images', 'outlines.png')
    if output_prefix is None:
        output_prefix = _out('pedestrian_flows')
    # --- TAZ mapping: raw name → (label, edge IDs) ---
    TAZ_MAP = {
        'University_Recreation_Center':         ('Z1',  ['E0']),
        '1':                                    ('Z2',  ['1050677005#1', '1054116928#2']),
        '2':                                    ('Z3',  ['1054116932#1']),
        'Student_Union':                        ('Z4',  ['E1']),
        '3':                                    ('Z5',  ['1058666218', '1054121751#0']),
        'Woodward':                             ('Z6',  ['E2']),
        'College_of_Education':                 ('Z7',  ['E3']),
        'College_of_Health_and_Human_Services': ('Z8',  ['E4']),
        'Burson':                               ('Z9',  ['E7']),
        'Cameron':                              ('Z10', ['1060112727#1']),
        'Auxiliary_Services':                   ('Z11', ['1060166235#5.41']),
        '4':                                    ('Z12', ['1098062400', '1050677007#1',
                                                          '1051046791#2', '1098062411#0']),
        'McMillan_Greenhouse':                  ('Z13', ['E10']),
        '5':                                    ('Z14', ['1060112787#5', '1051046792#2']),
    }

    COLORS = {
        'Z1': '#17BECF', 'Z2': '#F0874A', 'Z3': '#E74C6F', 'Z4': '#2E86C1',
        'Z5': '#D770AD', 'Z6': '#F5B041', 'Z7': '#2ECC71', 'Z8': '#E04848',
        'Z9': '#9B59B6', 'Z10': '#E8ECF0', 'Z11': '#A0522D', 'Z12': '#1ABC9C',
        'Z13': '#82C944', 'Z14': '#E84393',
    }

    dpi = 300

    # --- Load network & compute TAZ centroids ---
    net = sumolib.net.readNet(net_file)
    bbox = net.getBoundary()

    centroids = {}
    for trip_name, (label, edges) in TAZ_MAP.items():
        pts = []
        for eid in edges:
            try:
                pts.extend(net.getEdge(eid).getShape())
            except Exception:
                pass
        if pts:
            centroids[label] = (np.mean([p[0] for p in pts]),
                                np.mean([p[1] for p in pts]))

    # Nudge Z5 centroid slightly south (lower y in network coords)
    if 'Z5' in centroids:
        cx, cy = centroids['Z5']
        centroids['Z5'] = (cx, cy + 12)

    # --- Parse OD flows ---
    trip_to_label = {k: v[0] for k, v in TAZ_MAP.items()}
    tree = ET.parse(xml_path)
    od = defaultdict(int)
    for person in tree.getroot().findall('person'):
        w = person.find('walk')
        if w is not None:
            ft = trip_to_label.get(w.get('fromTaz'))
            tt = trip_to_label.get(w.get('toTaz'))
            if ft and tt and ft != tt:
                od[(ft, tt)] += 1
    max_count = max(od.values()) if od else 1

    # --- Load background ---
    bg = mpimg.imread(bg_path)
    img_h, img_w = bg.shape[:2]

    # Map network coords → image pixels
    sx = img_w / (bbox[2] - bbox[0])
    sy = img_h / (bbox[3] - bbox[1])

    def net_to_px(x, y):
        return (x - bbox[0]) * sx, img_h - (y - bbox[1]) * sy

    # --- Build figure ---
    PAD_X, PAD_Y = 0, 0
    pad_l, pad_r, pad_t, pad_b = 0.01, 0.01, 0.01, 0.01
    inner_w = (img_w + 2 * PAD_X) / dpi
    inner_h = (img_h + 2 * PAD_Y) / dpi
    fig_w = inner_w + pad_l + pad_r
    fig_h = inner_h + pad_t + pad_b

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=pad_l / fig_w, right=1 - pad_r / fig_w,
                        bottom=pad_b / fig_h, top=1 - pad_t / fig_h)

    ax.imshow(bg, extent=[0, img_w, 0, img_h], aspect='auto', zorder=0)

    # Dim background for contrast
    ax.imshow(np.ones((2, 2, 4)) * [1, 1, 1, 0.25],
              extent=[0, img_w, 0, img_h], aspect='auto', zorder=1)

    # --- Draw OD flow arcs (low counts first) ---
    for (src, dst), count in sorted(od.items(), key=lambda x: x[1]):
        if src not in centroids or dst not in centroids or count < flow_threshold:
            continue
        sx_, sy_ = net_to_px(*centroids[src])
        dx_, dy_ = net_to_px(*centroids[dst])
        col = COLORS.get(src, '#AAAAAA')
        frac = count / max_count
        lw = 0.8 + 6.5 * frac
        alpha = 0.85
        # Tighter arc for Z1<->Z12
        if {src, dst} == {'Z6', 'Z11'}:
            rad = 0.18
        elif {src, dst} == {'Z6', 'Z9'}:
            rad = 0.16
        elif {src, dst} == {'Z5', 'Z10'}:
            rad = 0.12
        elif {src, dst} in ({'Z1', 'Z12'}, {'Z4', 'Z12'}):
            rad = 0.12
        elif {src, dst} == {'Z6', 'Z14'}:
            rad = 0.08
        else:
            rad = 0.20
        ax.annotate(
            "", xy=(dx_, dy_), xytext=(sx_, sy_),
            arrowprops=dict(
                arrowstyle="-",
                color=col, lw=lw, alpha=alpha,
                connectionstyle=f"arc3,rad={rad}"),
            zorder=3 + int(frac * 10))

    # --- Draw TAZ markers and labels ---
    for label, (nx_, ny_) in centroids.items():
        ix, iy = net_to_px(nx_, ny_)
        col = COLORS.get(label, '#AAAAAA')

        # Outer glow
        ax.scatter(ix, iy, s=198, c=col, alpha=0.35, edgecolors='none', zorder=16)
        # Main dot
        ax.scatter(ix, iy, s=99, c=col, edgecolors='white', lw=1.5, zorder=17)

        # Label with stroke outline
        if label == 'Z5':
            xt, ha_, va_ = (-10, -4), 'right', 'top'
        elif label == 'Z6':
            xt, ha_, va_ = (-10, -6), 'center', 'top'
        elif label == 'Z7':
            xt, ha_, va_ = (-8, 8), 'center', 'bottom'
        elif label == 'Z8':
            xt, ha_, va_ = (8, 8), 'center', 'bottom'
        elif label == 'Z10':
            xt, ha_, va_ = (-4, 8), 'center', 'bottom'
        elif label == 'Z13':
            xt, ha_, va_ = (-10, 8), 'center', 'bottom'
        elif label == 'Z14':
            xt, ha_, va_ = (-6, 8), 'center', 'bottom'
        else:
            xt, ha_, va_ = (0, 8), 'center', 'bottom'
        ax.annotate(
            label, xy=(ix, iy), fontsize=9, fontweight='bold',
            ha=ha_, va=va_, xytext=xt, textcoords='offset points',
            color='white', zorder=20,
            path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])

    ax.set_xlim(-PAD_X, img_w + PAD_X)
    ax.set_ylim(-PAD_Y, img_h + PAD_Y)
    ax.set_axis_off()

    # --- Save ---
    fig.savefig(f"{output_prefix}.png", dpi=dpi, bbox_inches='tight',
                pad_inches=0.02, facecolor='white', transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    run_dir = "readout_32/May09_11-34-05"


    plot_graphs_and_gmm(
        graph_a_path=_proj('runs', run_dir, 'graph_iterations', 'graph_i_0_data.pkl'),
        graph_b_path=_proj('runs', run_dir, 'graph_iterations', 'graph_i_eval_final_data.pkl'),
        gmm_path=_proj('runs', run_dir, 'gmm_iterations', 'gmm_i_eval_final_b0_data.pkl'),
    )




    plot_pedestrian_flows(
        bg_path=_proj('images', 'outlines.png'),
        output_prefix=_out('pedestrian_flows'),
    )
    plot_pedestrian_flows(
        bg_path=_proj('images', 'No_Veh_No_Ped_cropped.png'),
        output_prefix=_out('pedestrian_flows_aerial'),
    )
