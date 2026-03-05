#!/usr/bin/env python3
"""
Design baselines: evaluate the RL-designed layout against the two
non-learned baselines used in the paper.

Measures pedestrian arrival time under unsignalized conditions and saves
both average and total arrival statistics across the full demand sweep.
"""
import os, sys, json, random, time, traceback
import numpy as np
import torch

os.environ['SUMO_HOME'] = '/usr/share/sumo'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, classify_and_return_args
from simulation.design_env import DesignEnv
from utils import scale_demand_sliced_window

import traci

ITERATION_COUNTER = 0
ALL_DEMAND_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]

def normalize_pos(x, w, norm_min, norm_max, min_thick=2.0, max_thick=15.0):
    range_x = norm_max - norm_min
    nx = max(0.01, min(0.99, (x - norm_min) / range_x))
    nw = max(0.01, min(0.99, (w - min_thick) / (max_thick - min_thick)))
    return torch.tensor([nx, nw])


def run_single_eval(net_file, control_args, run_dir, demand_scale, seed=42):
    """Run one SUMO episode unsignalized and return arrival-time metrics."""
    random.seed(seed)
    np.random.seed(seed)

    veh_out = os.path.join(run_dir, 'scaled_vehtrips.xml')
    ped_out = os.path.join(run_dir, 'scaled_pedtrips.xml')
    window_size = 590

    scale_demand_sliced_window(
        control_args['vehicle_input_trips'], veh_out,
        demand_scale, 'vehicle', window_size, evaluation=True)
    scale_demand_sliced_window(
        control_args['pedestrian_input_trips'], ped_out,
        demand_scale, 'pedestrian', window_size, evaluation=True)

    cfg_path = os.path.join(run_dir, f'eval_{seed}.sumocfg')
    with open(cfg_path, 'w') as f:
        f.write(f'''<configuration>
    <input>
        <net-file value="{os.path.abspath(net_file)}"/>
        <route-files value="{os.path.abspath(veh_out)},{os.path.abspath(ped_out)}"/>
    </input>
    <time><step-length value="1.0"/></time>
    <processing><lateral-resolution value="0.8"/></processing>
</configuration>''')

    label = f"bl_{demand_scale}_{seed}_{int(time.time()*1000)%100000}"
    traci.start(['sumo', '-c', cfg_path,
                 '--no-warnings', 'true', '--no-step-log', 'true',
                 '--time-to-teleport', '-1', '--start', 'true'], label=label)
    conn = traci.getConnection(label)

    # Set all TLs to green (unsignalized)
    for tl_id in conn.trafficlight.getIDList():
        prog = conn.trafficlight.getAllProgramLogics(tl_id)
        if prog:
            n = len(prog[0].phases[0].state)
            conn.trafficlight.setRedYellowGreenState(tl_id, 'G' * n)

    # Discover midblock crosswalk incoming edges
    # These are pedestrian edges (walkingarea/crossing) near non-intersection TLs
    intersection_tl = 'cluster_172228464_482708521_9687148201_9687148202_#5more'
    mb_incoming_edges = set()
    for tl_id in conn.trafficlight.getIDList():
        if tl_id == intersection_tl:
            continue
        controlled_links = conn.trafficlight.getControlledLinks(tl_id)
        for link_group in controlled_links:
            for link in link_group:
                if len(link) >= 2:
                    incoming = link[0]
                    if incoming:
                        edge = '_'.join(incoming.split('_')[:-1])
                        mb_incoming_edges.add(edge)

    # Also add crossing edges (format :node_c0, :node_w0 etc.)
    for edge_id in conn.edge.getIDList():
        if ':iter' in edge_id and ('_c' in edge_id or '_w' in edge_id):
            mb_incoming_edges.add(edge_id)

    # Warmup
    warmup = 90
    for _ in range(warmup):
        conn.simulationStep()

    # Track pedestrians
    ped_existence = {}
    ped_arrival = {}
    max_steps = 450

    for step in range(max_steps):
        conn.simulationStep()
        for ped_id in conn.person.getIDList():
            if ped_id not in ped_existence:
                ped_existence[ped_id] = 1.0
            else:
                ped_existence[ped_id] += 1.0

            if ped_id not in ped_arrival:
                lane_id = conn.person.getLaneID(ped_id)
                edge_id = '_'.join(lane_id.split('_')[:-1])
                if edge_id in mb_incoming_edges:
                    ped_arrival[ped_id] = ped_existence[ped_id]

    conn.close()

    if ped_arrival:
        total = sum(ped_arrival.values())
        avg = total / len(ped_arrival)
    else:
        avg = float('inf')
        total = float('inf')

    return avg, total


def evaluate_layout(env, proposals, control_args, run_dir, demand_scales, n_runs=5, name=""):
    """Evaluate a crosswalk layout across demand scales."""
    global ITERATION_COUNTER
    ITERATION_COUNTER += 1
    env._apply_action(proposals, ITERATION_COUNTER)
    net_file = env.current_net_file_path
    print(f"  [{name}] Net: {net_file} ({len(proposals)} crosswalks)")

    results = {}
    for scale in demand_scales:
        arrivals = []
        totals = []
        for r in range(n_runs):
            try:
                avg, total = run_single_eval(
                    net_file, control_args, run_dir, scale, seed=42 + r * 137 + int(scale * 1000))
                arrivals.append(avg)
                totals.append(total)
            except Exception as e:
                print(f"    [{name}] scale={scale} run={r} ERROR: {e}")
                traceback.print_exc()
        valid = [x for x in arrivals if x < 1e6]
        valid_totals = [x for x in totals if x < 1e9]
        mean = np.mean(valid) if valid else float('inf')
        std = np.std(valid) if valid else 0
        total_mean = np.mean(valid_totals) if valid_totals else float('inf')
        total_std = np.std(valid_totals) if valid_totals else 0
        results[scale] = {
            'mean': round(mean, 2),
            'std': round(std, 2),
            'total_mean': round(total_mean, 2),
            'total_std': round(total_std, 2),
            'n': len(valid),
        }
        print(f"    [{name}] {scale}x: {mean:.2f} ± {std:.2f} s ({len(valid)} runs)")
    return results


def main():
    config = get_config()
    config['gui'] = False
    config['gpu'] = False
    config['evaluate'] = True
    device = torch.device('cpu')

    design_args, control_args, higher_ppo_args, lower_ppo_args, eval_args = \
        classify_and_return_args(config, device)

    run_dir = './runs/baselines_experiment'
    for d in ['', '/components', '/network_iterations']:
        os.makedirs(run_dir + d, exist_ok=True)
    design_args['save_dir'] = run_dir
    design_args['save_graph_images'] = False
    design_args['save_gmm_plots'] = False
    higher_ppo_args['model_kwargs']['run_dir'] = run_dir

    with open('crosswalk_designs_cache.json') as f:
        cache = json.load(f)
    nmin, nmax = cache['norm_x']['min'], cache['norm_x']['max']

    print("Initializing DesignEnv...")
    env = DesignEnv(design_args, control_args, lower_ppo_args, run_dir)
    env.reset()
    print("DesignEnv ready.\n")

    demand_scales = ALL_DEMAND_SCALES
    n_runs = 5
    all_results = {}

    # 1. RL design
    rl_proposals = [normalize_pos(c['x'], c['width'], nmin, nmax) for c in cache['rl']]
    print("=== RL Design (4 crosswalks) ===")
    all_results['rl_design'] = evaluate_layout(env, rl_proposals, control_args, run_dir, demand_scales, n_runs, "rl")

    # 2. Uniform spacing (4 crosswalks)
    uniform_x = [nmin + o * (nmax - nmin) for o in [0.2, 0.4, 0.6, 0.8]]
    uniform_proposals = [normalize_pos(x, 6.0, nmin, nmax) for x in uniform_x]
    print("\n=== Uniform Spacing (4 crosswalks) ===")
    all_results['uniform'] = evaluate_layout(env, uniform_proposals, control_args, run_dir, demand_scales, n_runs, "uniform")

    # 3. Random search: best of 20
    print("\n=== Random Search (best of 20) ===")
    random.seed(42)
    best_random = {'mean': float('inf')}
    best_random_proposals = None
    for i in range(20):
        rx = sorted([nmin + 0.05 * (nmax - nmin) + random.random() * 0.9 * (nmax - nmin) for _ in range(4)])
        rw = [2.0 + random.random() * 13.0 for _ in range(4)]
        rp = [normalize_pos(x, w, nmin, nmax) for x, w in zip(rx, rw)]
        try:
            res = evaluate_layout(env, rp, control_args, run_dir, [1.0], 3, f"rand_{i}")
            if res[1.0]['mean'] < best_random['mean']:
                best_random = res[1.0]
                best_random_proposals = rp
                print(f"  >> New best random: {res[1.0]['mean']:.2f} s")
        except Exception as e:
            print(f"  rand_{i} failed: {e}")

    if best_random_proposals:
        print("\n=== Best Random (full eval) ===")
        all_results['random_best20'] = evaluate_layout(
            env, best_random_proposals, control_args, run_dir, demand_scales, n_runs, "best_rand")

    # Summary table
    print("\n" + "=" * 75)
    print("RESULTS: Avg Pedestrian Arrival Time (seconds)")
    print("=" * 75)
    header = f"{'Layout':<22}"
    for s in demand_scales:
        header += f"  {s}x".rjust(18)
    print(header)
    print("-" * 75)
    for name, res in all_results.items():
        row = f"{name:<22}"
        for s in demand_scales:
            if s in res:
                row += f"  {res[s]['mean']:>7.2f} ± {res[s]['std']:>5.2f}"
            else:
                row += f"  {'N/A':>15}"
        print(row)

    with open(os.path.join(run_dir, 'baseline_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {run_dir}/baseline_results.json")


if __name__ == '__main__':
    main()
