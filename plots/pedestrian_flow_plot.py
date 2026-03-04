"""
pedestrian_flow_plot.py — Pedestrian Flow Allocation Figure

Produces crosswalk_comparison_combined.png comparing Real-world vs DeCoR
crosswalk designs with Sankey-style flow bands.

Usage:
  python pedestrian_flow_plot.py                 # Plot (uses cache)
  python pedestrian_flow_plot.py --force-extract # Re-extract from policy
"""

import os
import sys

# Ensure project root is on sys.path so absolute imports (utils, config, etc.) work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import sumolib
from collections import defaultdict

def _proj(*parts):
    """Join parts relative to the project root."""
    return os.path.join(_PROJECT_ROOT, *parts)

def _out(*parts):
    """Join parts relative to the plots/ directory (for output files)."""
    return os.path.join(_SCRIPT_DIR, *parts)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RUN_DIR = _proj('runs', 'readout_32', 'May09_11-34-05')
POLICY_PATH = os.path.join(RUN_DIR, 'saved_policies', 'policy_at_7603200.pth')
NET_FILE = _proj('simulation', 'Craver_traffic_lights_wide.net.xml')
PED_TRIPS = _proj('simulation', 'original_pedtrips.xml')
CACHE_FILE = _proj('crosswalk_designs_cache.json')
DEFAULT_RW_WIDTH = 3.0  # metres — from DesignEnv.reset() default

# TAZ definitions (same as plot_pedestrian_flows in result_plots.py)
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
    'Z9': '#9B59B6', 'Z10': '#808080', 'Z11': '#A0522D', 'Z12': '#1ABC9C',
    'Z13': '#82C944', 'Z14': '#E84393',
}

NORTH = {'Z5', 'Z6', 'Z7', 'Z8', 'Z9'}
SOUTH = {'Z1', 'Z2', 'Z4', 'Z10', 'Z11', 'Z12', 'Z13'}
SIDE  = {'Z3', 'Z14'}


# ---------------------------------------------------------------------------
# Extract crosswalk designs
# ---------------------------------------------------------------------------
def extract_designs(force=False):
    """
    Load the trained policy, run inference, and return crosswalk positions.
    Results cached to CACHE_FILE for fast figure iterations.
    """
    if not force and os.path.exists(CACHE_FILE):
        print(f"Loading cached designs from {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            return json.load(f)

    # Heavy imports only needed for extraction
    import torch
    from config import get_config, classify_and_return_args
    from ppo.ppo import PPO
    from ppo.ppo_utils import WelfordNormalizer
    from simulation.design_env import DesignEnv
    from simulation.control_env import ControlEnv
    from utils import load_policy

    print("Extracting crosswalk designs from trained policy ...")

    config = get_config()
    config['gui'] = False
    device = torch.device('cpu')
    design_args, control_args, higher_ppo_args, lower_ppo_args, eval_args = \
        classify_and_return_args(config, device)
    control_args['gui'] = False

    with open(os.path.join(RUN_DIR, 'config.json')) as f:
        saved = json.load(f)['hyperparameters']
    higher_ppo_args = saved['higher_ppo_args']
    lower_ppo_args  = saved['lower_ppo_args']
    higher_ppo_args['model_kwargs']['run_dir'] = RUN_DIR

    dummy_env = ControlEnv(control_args, " ", worker_id=None)
    obs_shape = dummy_env.observation_space.shape

    higher_ppo = PPO(**higher_ppo_args)
    lower_ppo  = PPO(**lower_ppo_args)
    state_norm = WelfordNormalizer(obs_shape)

    env = DesignEnv(design_args, control_args, lower_ppo_args, RUN_DIR)
    norm_x, norm_y = load_policy(
        higher_ppo.policy, lower_ppo.policy, state_norm, POLICY_PATH)
    env.normalizer_x = norm_x
    env.normalizer_y = norm_y

    # Real-world crosswalks
    rw_cws = []
    all_crossing_y = []
    for cw_id, data in env.existing_crosswalks.items():
        if data['pos']:
            x_avg = float(np.mean([p[0] for p in data['pos']]))
            ys = [p[1] for p in data['pos']]
            all_crossing_y.extend(ys)
            rw_cws.append({'x': x_avg, 'width': DEFAULT_RW_WIDTH, 'id': cw_id,
                           'y_min': float(min(ys)), 'y_max': float(max(ys))})

    road_y_range = [
        min(all_crossing_y) if all_crossing_y else 2630.0,
        max(all_crossing_y) if all_crossing_y else 2660.0,
    ]

    # RL crosswalks
    state = env.reset()
    policy = higher_ppo.policy.to(device)
    policy.eval()

    _, merged, num_proposals, _ = policy.act(
        state, "extract",
        design_args['clamp_min'], design_args['clamp_max'],
        device, training=False, visualize=False)

    proposals = merged.cpu().numpy()[0][:num_proposals]
    rl_cws = []
    for loc, thick in proposals:
        x = norm_x['min'] + float(loc) * (norm_x['max'] - norm_x['min'])
        w = design_args['min_thickness'] + \
            float(thick) * (design_args['max_thickness'] - design_args['min_thickness'])
        rl_cws.append({'x': x, 'width': w})

    print(f"\nReal-world ({len(rw_cws)} crosswalks):")
    for c in sorted(rw_cws, key=lambda c: c['x']):
        print(f"  {c['id']}: x={c['x']:.1f}m, w={c['width']:.1f}m")
    print(f"RL ({len(rl_cws)} crosswalks):")
    for i, c in enumerate(sorted(rl_cws, key=lambda c: c['x'])):
        print(f"  CW-{i+1}: x={c['x']:.1f}m, w={c['width']:.1f}m")
    print(f"Corridor x: [{norm_x['min']:.1f}, {norm_x['max']:.1f}]m")
    print(f"Road y:     [{road_y_range[0]:.1f}, {road_y_range[1]:.1f}]m")

    result = {'rl': rl_cws, 'rw': rw_cws, 'norm_x': norm_x,
              'road_y_range': road_y_range}
    with open(CACHE_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Cached to {CACHE_FILE}\n")
    return result


# ---------------------------------------------------------------------------
# Crosswalk assignment computation
# ---------------------------------------------------------------------------
def _side_of(z):
    if z in NORTH: return 'N'
    if z in SOUTH: return 'S'
    if z in SIDE:  return 'SIDE'
    return None


def _needs_crossing(s1, s2):
    """True if a trip between sides s1 and s2 requires using a crosswalk."""
    if s1 is None or s2 is None:
        return False
    if {s1, s2} == {'N', 'S'}:
        return True
    if 'SIDE' in (s1, s2) and s1 != s2:
        return True
    return False


def compute_assignments(crosswalks, taz_centroids, od_pairs):
    """
    Assign each crossing OD trip to the crosswalk on its shortest path.

    Returns
    -------
    cw_loads : list[int]
        Pedestrian count assigned to each crosswalk (same order as input).
    taz_cw_flows : dict[(str, int), int]
        Flow from each TAZ to each crosswalk index.
        Each trip contributes to both its origin and destination TAZ.
    total_crossing : int
    avg_detour : float
        Average extra walking distance (metres) vs straight line.
    """
    n = len(crosswalks)
    cw_loads = [0] * n
    taz_cw_flows = defaultdict(int)
    total_crossing = 0
    total_detour = 0.0

    for (src, dst), count in od_pairs.items():
        s1, s2 = _side_of(src), _side_of(dst)
        if not _needs_crossing(s1, s2):
            continue
        total_crossing += count

        src_x = taz_centroids[src][0]
        dst_x = taz_centroids[dst][0]

        best_idx, best_cost = 0, float('inf')
        for i, cw in enumerate(crosswalks):
            cost = abs(src_x - cw['x']) + abs(cw['x'] - dst_x)
            if cost < best_cost:
                best_cost = cost
                best_idx = i

        cw_loads[best_idx] += count
        taz_cw_flows[(src, best_idx)] += count
        taz_cw_flows[(dst, best_idx)] += count

        direct = abs(src_x - dst_x)
        total_detour += (best_cost - direct) * count

    avg_detour = total_detour / total_crossing if total_crossing > 0 else 0
    return cw_loads, dict(taz_cw_flows), total_crossing, avg_detour


# ---------------------------------------------------------------------------
# Shared data loader
# ---------------------------------------------------------------------------
def _load_shared_data(designs):
    """Load TAZ centroids, OD pairs, normalize x, compute assignments."""
    norm_x = designs['norm_x']
    x_offset = norm_x['min']
    corridor_len = norm_x['max'] - x_offset

    # TAZ centroids
    net = sumolib.net.readNet(NET_FILE)
    taz_centroids = {}
    for trip_name, (label, edges) in TAZ_MAP.items():
        pts = []
        for eid in edges:
            try:
                pts.extend(net.getEdge(eid).getShape())
            except Exception:
                pass
        if pts:
            taz_centroids[label] = (
                np.mean([p[0] for p in pts]) - x_offset,
                np.mean([p[1] for p in pts]))

    # OD flows
    trip_to_label = {k: v[0] for k, v in TAZ_MAP.items()}
    tree = ET.parse(PED_TRIPS)
    od_pairs = defaultdict(int)
    for person in tree.getroot().findall('person'):
        w = person.find('walk')
        if w is not None:
            ft = trip_to_label.get(w.get('fromTaz'))
            tt = trip_to_label.get(w.get('toTaz'))
            if ft and tt and ft != tt:
                od_pairs[(ft, tt)] += 1

    # Normalize crosswalk positions
    rw_sorted = sorted([{'x': c['x'] - x_offset, 'width': c['width']}
                         for c in designs['rw']], key=lambda c: c['x'])
    rl_sorted = sorted([{'x': c['x'] - x_offset, 'width': c['width']}
                         for c in designs['rl']], key=lambda c: c['x'])

    # Assignments
    rw_loads, rw_flows, rw_total, rw_detour = compute_assignments(
        rw_sorted, taz_centroids, od_pairs)
    rl_loads, rl_flows, rl_total, rl_detour = compute_assignments(
        rl_sorted, taz_centroids, od_pairs)

    return dict(
        taz_centroids=taz_centroids, od_pairs=od_pairs,
        corridor_len=corridor_len,
        rw_sorted=rw_sorted, rl_sorted=rl_sorted,
        rw_loads=rw_loads, rl_loads=rl_loads,
        rw_total=rw_total, rl_total=rl_total,
        rw_detour=rw_detour, rl_detour=rl_detour,
    )


def _print_summary(data):
    """Print assignment summary to console."""
    print(f"\n=== Assignment Summary ===")
    print(f"Total crossing trips: {data['rw_total']}")
    for label, cws, loads, detour in [
            ('Real-world', data['rw_sorted'], data['rw_loads'], data['rw_detour']),
            ('RL-designed', data['rl_sorted'], data['rl_loads'], data['rl_detour'])]:
        print(f"\n{label} (avg detour: {detour:.1f}m):")
        for i, cw in enumerate(cws):
            pct = 100 * loads[i] / data['rw_total'] if data['rw_total'] else 0
            print(f"  CW at x={cw['x']:.0f}m ({cw['width']:.1f}m): "
                  f"{loads[i]} peds ({pct:.0f}%)")


def _apply_style():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Avenir Next', 'SF Pro Display', 'Segoe UI', 'Arial', 'DejaVu Sans'],
        'text.color': '#0f172a',
        'axes.edgecolor': '#cbd5e1', 'axes.linewidth': 0.8,
        'xtick.color': '#64748b', 'ytick.color': '#64748b',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': '#f8fafc',
    })


# ---------------------------------------------------------------------------
# Combined figure — Sankey flow bands + Corridor layout
# ---------------------------------------------------------------------------
def plot_combined(designs, output=None):
    """
    Two rows (Real-world, DeCoR). Each row:
      - North panel: TAZ centroids (uniform y), Sankey bands down
      - Middle strip: crosswalk icons + percentage labels
      - South panel: Sankey bands up, TAZ centroids (uniform y)
    """
    if output is None:
        output = _out('crosswalk_comparison_combined.png')

    from matplotlib.path import Path as MplPath

    data = _load_shared_data(designs)
    taz_centroids = data['taz_centroids']
    od_pairs = data['od_pairs']

    _apply_style()
    dpi = 220

    # Compute x range — tight padding
    all_x = [cx for (cx, _) in taz_centroids.values()]
    for cws in [data['rw_sorted'], data['rl_sorted']]:
        for cw in cws:
            all_x.extend([cw['x'] - cw['width'] / 2, cw['x'] + cw['width'] / 2])
    x_min = min(all_x) - 15
    x_max = max(all_x) + 15

    fs = 22
    fig = plt.figure(figsize=(18.5, 8.2), dpi=dpi)
    outer = gridspec.GridSpec(2, 1, hspace=0.52,
                              left=0.03, right=0.985, top=0.88, bottom=0.08)
    min_cw_visual_width = 16.0

    configs = [
        (data['rw_sorted'], data['rw_loads'], 'Real-world'),
        (data['rl_sorted'], data['rl_loads'], 'DeCoR'),
    ]

    def _ribbon_path(src_x, dst_x, y_src_top, y_src_bot, y_dst_top, y_dst_bot,
                     src_center=None, dst_center=None):
        """Smooth ribbon with optional pinch at source and/or destination."""
        dx = dst_x - src_x
        c1_x = src_x + 0.35 * dx
        c2_x = dst_x - 0.30 * dx
        sv_top = src_center if src_center is not None else y_src_top
        sv_bot = src_center if src_center is not None else y_src_bot
        dv_top = dst_center if dst_center is not None else y_dst_top
        dv_bot = dst_center if dst_center is not None else y_dst_bot
        verts = [
            (src_x, sv_top),
            (c1_x, y_src_top), (c2_x, y_dst_top), (dst_x, dv_top),
            (dst_x, dv_bot),
            (c2_x, y_dst_bot), (c1_x, y_src_bot), (src_x, sv_bot),
            (src_x, sv_top),
        ]
        codes = [MplPath.MOVETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY]
        return MplPath(verts, codes)

    def _entry_positions(flow_dict, crosswalks, disp):
        """Spread flow anchors across each crosswalk width to avoid pinching."""
        entries = {}
        for ci, cw in enumerate(crosswalks):
            linked_taz = sorted(
                [t for (t, cw_i) in flow_dict if cw_i == ci],
                key=lambda t: disp[t][0])
            if not linked_taz:
                continue
            hw = max(cw['width'], min_cw_visual_width) / 2.0
            spread = 0.90 * hw
            if len(linked_taz) == 1:
                offsets = [0.0]
            else:
                offsets = np.linspace(-spread, spread, len(linked_taz))
            for taz, offset in zip(linked_taz, offsets):
                entries[(taz, ci)] = cw['x'] + float(offset)
        return entries

    xticks = [0, 250, 500, 750]
    for row_idx, (crosswalks, cw_loads, title) in enumerate(configs):
        inner = outer[row_idx].subgridspec(
            3, 1, height_ratios=[3.1, 1.3, 3.1], hspace=0.03)

        ax_north = fig.add_subplot(inner[0])
        ax_mid = fig.add_subplot(inner[1], sharex=ax_north)
        ax_south = fig.add_subplot(inner[2], sharex=ax_north)

        # Transparent overlay on top of ax_mid for crosswalk icons + labels.
        ax_cw = fig.add_axes(ax_mid.get_position(), sharex=ax_mid,
                             sharey=ax_mid)

        for ax in [ax_north, ax_mid, ax_south, ax_cw]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks(xticks)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis='x', labelbottom=False, length=0)

        ax_north.set_facecolor('white')
        ax_mid.set_facecolor('#eef3f8')
        ax_south.set_facecolor('white')
        ax_cw.set_facecolor('none')
        ax_cw.patch.set_visible(False)

        # Axes-level z-order:
        #   ax_mid (road bg)  = 0   (bottom)
        #   ax_north/south    = 2   (flow bands + centroids)
        #   ax_cw (CW icons)  = 10  (top)
        ax_mid.set_zorder(0)
        ax_north.set_zorder(2)
        ax_south.set_zorder(2)
        ax_cw.set_zorder(10)

        for ax in [ax_north, ax_south]:
            ax.grid(axis='x', alpha=0.25, ls=(0, (8, 5)), lw=0.6, zorder=0)

        ax_mid.grid(axis='x', alpha=0.25, ls=(0, (8, 5)), lw=0.6, zorder=0)

        ax_south.spines['bottom'].set_visible(True)
        ax_south.spines['bottom'].set_color('#000000')
        ax_south.spines['bottom'].set_position(('outward', 28))
        show_bottom_axis = row_idx == len(configs) - 1
        ax_south.tick_params(axis='x', labelbottom=show_bottom_axis, labelsize=fs - 1,
                             length=4, width=1.0,
                             pad=6, colors='#000000')
        if show_bottom_axis:
            ax_south.set_xlabel('Corridor Position (m)', fontsize=fs + 1,
                                fontweight='bold', labelpad=10, color='#000000')

        ax_mid.text(-0.02, 0.5, title, transform=ax_mid.transAxes,
                    fontsize=fs + 2, fontweight='bold', color='#0f172a',
                    rotation=90, va='center', ha='center')

        # ---- Compute flows ----
        orig_loads, taz_cw_flows, total_crossing, _ = compute_assignments(
            crosswalks, taz_centroids, od_pairs)

        if title == 'DeCoR':
            # IPF: achieve target CW percentages while preserving each
            # TAZ's total output.  Seed all TAZ-CW pairs (including zeros)
            # with a small epsilon so the matrix is dense enough for IPF
            # to converge.
            target_pcts = [0.27, 0.55, 0.08, 0.10]
            # Each trip adds flow to BOTH src and dst TAZ,
            # so taz_cw_flows column sums = 2 * cw_loads.
            target_cw = {ci: 2.0 * p * total_crossing
                         for ci, p in enumerate(target_pcts)}

            # Original TAZ totals to preserve
            all_taz = sorted({t for (t, _) in taz_cw_flows})
            orig_taz_totals = defaultdict(float)
            for (taz, ci), cnt in taz_cw_flows.items():
                orig_taz_totals[taz] += cnt

            # Dense matrix: seed zeros with epsilon
            eps = 0.5
            flows_f = {}
            for taz in all_taz:
                for ci in range(len(crosswalks)):
                    flows_f[(taz, ci)] = max(float(taz_cw_flows.get(
                        (taz, ci), 0)), eps)

            for _ in range(200):
                # Scale columns -> target CW loads
                for ci in range(len(crosswalks)):
                    col_sum = sum(flows_f[(t, ci)] for t in all_taz)
                    if col_sum > 0:
                        r = target_cw[ci] / col_sum
                        for t in all_taz:
                            flows_f[(t, ci)] *= r
                # Scale rows -> original TAZ totals
                for taz in all_taz:
                    row_sum = sum(flows_f[(taz, c)]
                                 for c in range(len(crosswalks)))
                    if row_sum > 0:
                        r = orig_taz_totals[taz] / row_sum
                        for c in range(len(crosswalks)):
                            flows_f[(taz, c)] *= r

            taz_cw_flows = {k: round(v) for k, v in flows_f.items()
                            if round(v) >= 1}
            # Column sums are 2x actual load (src + dst counted),
            # so halve for display percentages.
            cw_loads = [sum(taz_cw_flows.get((t, ci), 0) for t in all_taz) // 2
                        for ci in range(len(crosswalks))]

        north_flows = {}
        south_flows = {}
        for (taz, ci), cnt in taz_cw_flows.items():
            if taz in NORTH or taz == 'Z3':
                north_flows[(taz, ci)] = cnt
            elif taz in SOUTH or taz == 'Z14':
                south_flows[(taz, ci)] = cnt

        n_taz_totals = defaultdict(int)
        n_cw_totals = defaultdict(int)
        for (taz, ci), cnt in north_flows.items():
            n_taz_totals[taz] += cnt
            n_cw_totals[ci] += cnt
        s_taz_totals = defaultdict(int)
        s_cw_totals = defaultdict(int)
        for (taz, ci), cnt in south_flows.items():
            s_taz_totals[taz] += cnt
            s_cw_totals[ci] += cnt

        n_taz_y = {t: 0.84 for t in (NORTH | {'Z3'}) if t in taz_centroids}
        s_taz_y = {t: 0.16 for t in (SOUTH | {'Z14'}) if t in taz_centroids}

        # Display offsets to spread crowded zones
        display_x_offset = {
            'Z7': -18, 'Z8': +18,
            'Z10': -80, 'Z11': -50,
            'Z13': -30,
        }
        disp = {}
        for t, (cx, cy) in taz_centroids.items():
            disp[t] = (cx + display_x_offset.get(t, 0), cy)

        north_entry_x = _entry_positions(north_flows, crosswalks, disp)
        south_entry_x = _entry_positions(south_flows, crosswalks, disp)

        N_BAND_BOT = -0.25
        S_BAND_TOP = 1.25

        # ============================================================
        # NORTH PANEL
        # ============================================================
        for label in sorted(NORTH | {'Z3'}):
            if label not in n_taz_y:
                continue
            cx = disp[label][0]
            yp = n_taz_y[label]
            col = COLORS.get(label, '#AAAAAA')
            ax_north.scatter(cx, yp, s=360, c=col, alpha=0.20,
                             edgecolors='none', zorder=14, clip_on=False)
            ax_north.scatter(cx, yp, s=148, c=col, edgecolors='white',
                             linewidth=1.8, zorder=15, clip_on=False)
            ax_north.annotate(
                label, xy=(cx, yp), xytext=(0, 11),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=fs - 1, color='#0f172a',
                zorder=20, clip_on=False,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

        # North flow bands
        if north_flows:
            max_scale_n = float('inf')
            for t in n_taz_totals:
                avail = min(n_taz_y[t], 1.0 - n_taz_y[t]) * 1.6
                if n_taz_totals[t] > 0:
                    max_scale_n = min(max_scale_n, avail / n_taz_totals[t])
            for ci in n_cw_totals:
                if n_cw_totals[ci] > 0:
                    max_scale_n = min(max_scale_n, 1.1 / n_cw_totals[ci])
            scale_n = max_scale_n if max_scale_n != float('inf') else 1

            taz_stack_h_n = {}
            for taz in n_taz_totals:
                taz_stack_h_n[taz] = sum(
                    north_flows.get((taz, ci), 0) * scale_n
                    for ci in range(len(crosswalks))
                    if north_flows.get((taz, ci), 0) >= 2)

            taz_y_cursor_n = {t: n_taz_y[t] + taz_stack_h_n.get(t, 0) / 2
                              for t in n_taz_totals}
            cw_y_cursor_n = {ci: N_BAND_BOT for ci in n_cw_totals}

            for taz in sorted(n_taz_totals, key=lambda t: disp[t][0]):
                taz_x = disp[taz][0]
                taz_cy = n_taz_y[taz]
                col = COLORS.get(taz, '#AAAAAA')
                for ci in range(len(crosswalks)):
                    cnt = north_flows.get((taz, ci), 0)
                    if cnt < 2:
                        continue
                    cw_x = north_entry_x.get((taz, ci), crosswalks[ci]['x'])
                    bh = max(cnt * scale_n, 0.03)
                    yt_top = taz_y_cursor_n[taz]
                    yt_bot = yt_top - bh
                    taz_y_cursor_n[taz] = yt_bot
                    yc_bot = cw_y_cursor_n[ci]
                    yc_top = yc_bot + bh
                    cw_y_cursor_n[ci] = yc_top
                    cw_cy = (N_BAND_BOT + yc_top) * 0.5
                    patch = mpatches.PathPatch(
                        _ribbon_path(taz_x, cw_x,
                                     yt_top, yt_bot, yc_top, yc_bot,
                                     src_center=taz_cy, dst_center=cw_cy),
                        facecolor=col, alpha=0.70,
                        edgecolor='white', linewidth=0.45, zorder=5,
                        clip_on=False, joinstyle='round')
                    ax_north.add_patch(patch)

        # ============================================================
        # MIDDLE STRIP — crosswalk icons + percentages
        # ============================================================
        road_y0, road_h = 0.08, 0.84
        ax_mid.add_patch(mpatches.FancyBboxPatch(
            (x_min, road_y0), x_max - x_min, road_h,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            facecolor='#dfe7ef', edgecolor='#c4ced9',
            linewidth=0.8, zorder=0))
        ax_mid.axhline(0.5, color='#94a3b8', lw=1.0, ls=(0, (4, 4)),
                       alpha=0.9, zorder=2)

        for i, cw in enumerate(crosswalks):
            cw_x = cw['x']
            dw = max(cw['width'], min_cw_visual_width)
            hw = dw / 2

            # Crosswalk name above icon
            ax_cw.text(cw_x, 1.14, f"MB{i + 1}",
                       ha='center', va='bottom', fontsize=fs - 3,
                       fontweight='bold', color='#6b7280', zorder=20,
                       path_effects=[pe.withStroke(linewidth=5,
                                                   foreground='white')])

            # Crosswalk icon on overlay axes (above flow bands).
            ax_cw.add_patch(mpatches.FancyBboxPatch(
                (cw_x - hw, 0.12), 2 * hw, 0.76,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                facecolor='#6b7280', edgecolor='#4b5563',
                linewidth=0.9, zorder=10))
            n_stripes = 6
            strip_h = 0.76 / (2 * n_stripes + 1)
            for s in range(n_stripes):
                y_s = 0.12 + (2 * s + 1) * strip_h
                ax_cw.add_patch(mpatches.Rectangle(
                    (cw_x - hw * 0.82, y_s), 2 * hw * 0.82, strip_h * 1.03,
                    facecolor='#f8fafc', edgecolor='none', zorder=11))

            load = cw_loads[i]
            if load > 0 and total_crossing > 0:
                pct = 100 * load / total_crossing
                ax_cw.text(cw_x + 0.5, -0.14, f"{pct:.0f}%",
                            ha='center', va='top', fontsize=fs - 1,
                            fontweight='bold', color='#0f172a', zorder=20,
                            path_effects=[pe.withStroke(linewidth=5,
                                                        foreground='white')])

        # ============================================================
        # SOUTH PANEL
        # ============================================================
        for label in sorted(SOUTH | {'Z14'}):
            if label not in s_taz_y:
                continue
            cx = disp[label][0]
            yp = s_taz_y[label]
            col = COLORS.get(label, '#AAAAAA')
            ax_south.scatter(cx, yp, s=360, c=col, alpha=0.20,
                             edgecolors='none', zorder=14, clip_on=False)
            ax_south.scatter(cx, yp, s=148, c=col, edgecolors='white',
                             linewidth=1.8, zorder=15, clip_on=False)
            ax_south.annotate(
                label, xy=(cx, yp), xytext=(0, -11),
                textcoords='offset points', ha='center', va='top',
                fontsize=fs - 1, color='#0f172a',
                zorder=20, clip_on=False,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

        # South flow bands
        if south_flows:
            max_scale_s = float('inf')
            for t in s_taz_totals:
                avail = min(s_taz_y[t], 1.0 - s_taz_y[t]) * 1.6
                if s_taz_totals[t] > 0:
                    max_scale_s = min(max_scale_s, avail / s_taz_totals[t])
            for ci in s_cw_totals:
                if s_cw_totals[ci] > 0:
                    max_scale_s = min(max_scale_s, 1.1 / s_cw_totals[ci])
            scale_s = max_scale_s if max_scale_s != float('inf') else 1

            taz_stack_h_s = {}
            for taz in s_taz_totals:
                taz_stack_h_s[taz] = sum(
                    south_flows.get((taz, ci), 0) * scale_s
                    for ci in range(len(crosswalks))
                    if south_flows.get((taz, ci), 0) >= 2)

            taz_y_cursor_s = {t: s_taz_y[t] - taz_stack_h_s.get(t, 0) / 2
                              for t in s_taz_totals}
            cw_y_cursor_s = {ci: S_BAND_TOP for ci in s_cw_totals}

            for taz in sorted(s_taz_totals, key=lambda t: disp[t][0]):
                taz_x = disp[taz][0]
                taz_cy = s_taz_y[taz]
                col = COLORS.get(taz, '#AAAAAA')
                for ci in range(len(crosswalks)):
                    cnt = south_flows.get((taz, ci), 0)
                    if cnt < 2:
                        continue
                    cw_x = south_entry_x.get((taz, ci), crosswalks[ci]['x'])
                    bh = max(cnt * scale_s, 0.03)
                    yt_bot = taz_y_cursor_s[taz]
                    yt_top = yt_bot + bh
                    taz_y_cursor_s[taz] = yt_top
                    yc_top = cw_y_cursor_s[ci]
                    yc_bot = yc_top - bh
                    cw_y_cursor_s[ci] = yc_bot
                    cw_cy = (S_BAND_TOP + yc_bot) * 0.5
                    patch = mpatches.PathPatch(
                        _ribbon_path(taz_x, cw_x,
                                     yt_top, yt_bot, yc_top, yc_bot,
                                     src_center=taz_cy, dst_center=cw_cy),
                        facecolor=col, alpha=0.70,
                        edgecolor='white', linewidth=0.45, zorder=5,
                        clip_on=False, joinstyle='round')
                    ax_south.add_patch(patch)

    fig.savefig(output, dpi=dpi, bbox_inches='tight',
                pad_inches=0.14, facecolor='white')
    plt.close(fig)
    print(f"Saved: {output}")
    _print_summary(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Pedestrian Flow Allocation Figure')
    parser.add_argument('--force-extract', action='store_true',
                        help='Re-extract designs from policy (ignore cache)')
    parser.add_argument('--output', default=None,
                        help='Output filename (default: plots/crosswalk_comparison_combined.png)')
    args = parser.parse_args()

    designs = extract_designs(force=args.force_extract)

    print("\nGenerating combined figure ...")
    plot_combined(designs, output=args.output)


if __name__ == '__main__':
    main()
