"""Reproducible, separate review diagnostics; never overwrite historical results.

Uses the corrected ControlEnv and demand scaler. Each manifest fixes the
warmup mode, seeds, demand files and a 450 s measurement horizon. Learned control
requires a compatible observation version; legacy weights remain usable only
for reconstructing the learned design. An opt-in journey protocol fixes the
departure cohort and drains it after measurement. No policy training occurs.
Placement-matched baselines (Uniform and best-of-20 random search) keep the
reference layout's crossing count and west-to-east widths and vary placement
only; random search selects on training-window trials under actuated control.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import traci
from torch_geometric.data import Batch

from ppo.ppo import PPO
from ppo.models import MLP_ActorCritic
from ppo.ppo_utils import WelfordNormalizer
from simulation.control_env import ControlEnv
from simulation.design_env import DesignEnv
from utils import require_exposed_heads, signal_slots_from_network

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/readout_32/May09_11-34-05"
CHECKPOINT = RUN / "saved_policies/policy_at_7603200.pth"
INTERSECTION = "cluster_172228464_482708521_9687148201_9687148202_#5more"
JOURNEY_PROTOCOL = {"version": 1, "warmup_s": 100, "measurement_horizon_s": 450,
                    "drain_cap_s": 1800}
SELECTION_SCALES, SELECTION_SEEDS = (1., 2.), (5100, 5101, 5102)  # training-window selection split
LOCATION_GAP, LOCATION_LOW, LOCATION_HIGH = .08, .01, .99  # learned merge separation and location clamps
GEOMETRY_SEED, RANDOM_CANDIDATES = 42, 20


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def arguments(configuration=None):
    recorded = (copy.deepcopy(configuration) if configuration is not None else
                json.loads((RUN / "config.json").read_text())["hyperparameters"])
    d, ctrl = recorded["design_args"], recorded["control_args"]
    higher, lower = recorded["higher_ppo_args"], recorded["lower_ppo_args"]
    d.update(save_graph_images=False, save_gmm_plots=False)
    higher["device"] = lower["device"] = "cpu"
    return d, ctrl, higher, lower


def physical_proposals(proposals, normalizer_x, design):
    """Normalized (location, width) rows as corridor x in metres and crossing width in metres."""
    span = normalizer_x["max"] - normalizer_x["min"]
    thickness = design["max_thickness"] - design["min_thickness"]
    return [{"x": float(normalizer_x["min"] + x * span), "width": float(design["min_thickness"] + w * thickness),
             "normalized": [float(x), float(w)]} for x, w in proposals]


def random_proposals(count, rng, widths=None):
    """Uniform feasible locations with the learned policy's minimum separation.

    Widths are random unless a west-to-east width vector is given. Fixed widths
    consume only location draws; the existing random-width training path is unchanged.
    """
    locations = LOCATION_LOW + np.arange(count) * LOCATION_GAP
    locations += np.sort(rng.random(count)) * (LOCATION_HIGH - LOCATION_LOW - (count - 1) * LOCATION_GAP)
    if widths is None:
        widths = rng.uniform(LOCATION_LOW, LOCATION_HIGH, count)
    proposals = torch.full((1, 10, 2), -1.0)
    proposals[0, :count] = torch.tensor(np.column_stack((locations, widths)), dtype=torch.float32)
    return proposals, torch.tensor(count)


def uniform_proposals(count, widths):
    """Evenly spaced interior locations k/(count+1): the historical .2/.4/.6/.8 form for any count."""
    locations = np.arange(1, count + 1) / (count + 1)
    return np.column_stack((locations, widths)).astype(np.float32)


def check_matched(proposals, widths):
    """Placement-only variation: ascending feasible locations and exactly the reference widths."""
    proposals = np.asarray(proposals, dtype=np.float32)
    if len(proposals) != len(widths) or not np.array_equal(proposals[:, 1], np.asarray(widths, dtype=np.float32)):
        raise ValueError("Baseline widths must equal the reference west-to-east width vector.")
    locations = proposals[:, 0]
    if locations.min() < LOCATION_LOW or locations.max() > LOCATION_HIGH or np.any(np.diff(locations) < LOCATION_GAP - 1e-6):
        raise ValueError("Baseline locations must be ascending, clamped and separated like learned proposals.")


def reference_proposals(manifest, layout="learned"):
    """Declared reference geometry west to east; its crossing count and width vector are matched."""
    proposals = np.array(sorted(p["normalized"] for p in manifest["evaluation_proposals"]), dtype=np.float32)
    if len(proposals) != manifest["layouts"][layout]["num_proposals"]:
        raise ValueError("Manifest proposals do not match the reference layout crossing count.")
    return proposals


def baseline_manifest(directory):
    """Derive Uniform and predeclared random placements matched to the reference; reuse if already derived."""
    parent_path = directory / "manifest.json"
    parent_sha256 = digest(parent_path)
    path = directory / "baselines" / "manifest.json"
    if path.exists():
        if json.loads(path.read_text())["parent_manifest_sha256"] != parent_sha256:
            raise ValueError("Baseline manifest derives from a different study manifest. Use a fresh study directory.")
        return path
    parent = json.loads(parent_path.read_text())
    reference = reference_proposals(parent)
    count, widths = len(reference), reference[:, 1]
    torch.set_num_threads(1)
    d, ctrl, _, lower = arguments(parent["configuration"])
    d["save_dir"] = str(directory / "baselines" / "geometry")
    env = DesignEnv(d, ctrl, lower, d["save_dir"])
    env.normalizer_x = parent["normalizer_x"]
    env.reset()
    rng = np.random.default_rng(GEOMETRY_SEED)  # Geometry only; every trial seeds its own simulation.
    candidates = {"uniform": uniform_proposals(count, widths)}
    for index in range(RANDOM_CANDIDATES):
        padded, _ = random_proposals(count, rng, widths)
        candidates[f"random_{index:02d}"] = padded[0, :count].numpy()
    layouts = {}
    for name, proposals in candidates.items():
        check_matched(proposals, widths)
        env._apply_action(proposals, name)
        network = str(Path(env.current_net_file_path).resolve())
        layouts[name] = {"network": network, "sha256": digest(network), "iteration": name,
                         "num_proposals": count, "real_world": False,
                         "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
                         "crossing_ids": list(env.crossing_ids), "signal_slots": dict(env.signal_slots),
                         "proposals": physical_proposals(proposals, env.normalizer_x, d)}
    search = {"comparison": "placement-matched: reference crossing count and west-to-east widths are kept; only locations vary. Count/width search is a separate study.",
              "reference_layout": "learned", "crossing_count": count,
              "reference_proposals": physical_proposals(reference, env.normalizer_x, d),
              "uniform": "interior locations k/(n+1)",
              "random": {"candidates": RANDOM_CANDIDATES, "geometry_seed": GEOMETRY_SEED,
                         "sampler": f"random_proposals: ascending locations in [{LOCATION_LOW}, {LOCATION_HIGH}] with at least {LOCATION_GAP} normalized separation; independent of simulation and demand seeds"},
              "selection": {"arm": "actuated", "split": "training", "scales": list(SELECTION_SCALES), "seeds": list(SELECTION_SEEDS),
                            "metric": "mean over selection trials of the sum over road-user classes of (wait_sum_1s + backlog_age_at_end_s) / max(scheduled, 1); lower is better. A selection metric, not travel time or full-cohort delay.",
                            "tie_break": "lowest candidate index", "failed_or_incomplete_candidates": "recorded and ineligible"}}
    save(path, dict(parent, layouts=layouts, parent_manifest=str(parent_path), parent_manifest_sha256=parent_sha256,
                    baseline_search=search))
    return path


def search_jobs(directory, manifest):
    """Training-window actuated trials for every predeclared random candidate; no held-out demand."""
    return [dict(manifest=str(manifest), layout=f"random_{index:02d}", arm="actuated", scale=scale, seed=seed,
                 split="training", directory=str(directory / "baselines" / "search" / f"random_{index:02d}_{scale}_{seed}"))
            for index in range(RANDOM_CANDIDATES) for scale in SELECTION_SCALES for seed in SELECTION_SEEDS]


def selection_score(result):
    """Equal weight per road-user class; include outstanding insertion backlog. Lower is better."""
    return sum((x["wait_sum_1s"] + x["backlog_age_at_end_s"]) / max(x["scheduled"], 1) for x in result["traffic"].values())


def select_baseline(directory, manifest, jobs, failures):
    """Rank candidates with every selection trial complete; failures stay recorded and ineligible."""
    selection_path = directory / "baselines" / "selection.json"
    if selection_path.exists():
        raise FileExistsError("Baseline selection is frozen. Use a fresh study directory.")
    layouts = json.loads(manifest.read_text())["layouts"]
    candidates = []
    for index in range(RANDOM_CANDIDATES):
        name = f"random_{index:02d}"
        trials = []
        for job in jobs:
            if job["layout"] != name:
                continue
            entry = {"scale": job["scale"], "seed": job["seed"]}
            path = Path(job["directory"]) / "result.json"
            if job["directory"] in failures:
                entry["error"] = failures[job["directory"]]
            elif not path.exists():
                entry["error"] = "missing result"
            else:
                entry["score"] = selection_score(json.loads(path.read_text()))
            trials.append(entry)
        scores = [t["score"] for t in trials if "score" in t]
        eligible = len(scores) == len(trials) == len(SELECTION_SCALES) * len(SELECTION_SEEDS)
        candidates.append({"layout": name, "index": index, "network_sha256": layouts[name]["sha256"],
                           "proposals": layouts[name]["proposals"], "trials": trials, "eligible": eligible,
                           "score": statistics.mean(scores) if eligible else None})
    eligible = [c for c in candidates if c["eligible"]]
    winner = min(eligible, key=lambda c: (c["score"], c["index"])) if eligible else None
    selection = {"manifest_sha256": digest(manifest), "uniform": "uniform",
                 "random_best20": winner["layout"] if winner else None,
                 "random_best20_network_sha256": winner["network_sha256"] if winner else None,
                 "random_best20_score": winner["score"] if winner else None,
                 "eligible_candidates": len(eligible), "failed_or_incomplete_candidates": len(candidates) - len(eligible),
                 "metric": "training-window selection score; not held-out performance", "tie_break": "lowest candidate index",
                 "candidates": candidates}
    save(selection_path, selection)
    if winner is None:
        raise RuntimeError("No random candidate completed every selection trial; nothing to select.")
    return selection


def baseline_rows(directory, scales, seeds):
    """Held-out actuated rows for the Uniform layout and the frozen random-search winner."""
    selection_path = directory / "baselines" / "selection.json"
    if not selection_path.exists():
        print("No baseline selection; run search to add uniform and random_best20 rows.", flush=True)
        return []
    selection = json.loads(selection_path.read_text())
    manifest = directory / "baselines" / "manifest.json"
    if digest(manifest) != selection["manifest_sha256"]:
        raise ValueError("Baseline selection does not match the baseline manifest. Rerun search in a fresh study directory.")
    if selection["random_best20"] is None:
        raise RuntimeError("Baseline search has no eligible winner. Use a fresh study directory.")
    return [dict(manifest=str(manifest), layout=selection[name], arm="actuated", scale=scale, seed=seed,
                 split="evaluation", directory=str(directory / "trials" / f"{name}_actuated_{scale}_{seed}"))
            for name in ("uniform", "random_best20") for scale in scales for seed in seeds]


def restrict_departure_cohort(path, tag, cutoff):
    """Retain scheduled departures before cutoff, before SUMO reads the file."""
    tree = ET.parse(path)
    due = {}
    for parent in tree.iter():
        for entry in list(parent):
            if entry.tag == tag:
                departure = float(entry.get("depart"))
                if departure < cutoff:
                    due[entry.get("id")] = departure
                else:
                    parent.remove(entry)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return due


def cohort_summary(due, entries, kind, end_time):
    """Include insertion delay and every scheduled trip, even without tripinfo."""
    records = {entry.get("id"): entry for entry in entries}
    inserted, completed = set(), set()
    duration_sum = insertion_delay_sum = time_loss_sum = 0.0
    for identifier, departure in due.items():
        entry = records.get(identifier)
        arrival = -1.0
        if entry is not None and float(entry.get("depart")) >= 0:
            inserted.add(identifier)
            insertion_delay_sum += float(entry.get("depart")) - departure
            arrival = float((entry if kind == "vehicle" else entry[-1]).get("arrival"))
            if kind == "vehicle":
                time_loss_sum += float(entry.get("timeLoss"))
        else:
            insertion_delay_sum += end_time - departure
        if arrival >= 0:
            completed.add(identifier)
            duration_sum += arrival - departure
        else:
            duration_sum += end_time - departure
    count = len(due)
    complete = len(completed) == count
    result = {"scheduled": count, "inserted": len(inserted), "completed": len(completed),
              "unfinished_inserted": len(inserted - completed), "not_inserted": count - len(inserted),
              "censored": count - len(completed), "all_completed": complete,
              "completion_fraction": len(completed) / count if count else None,
              "journey_mean_lower_bound_s": duration_sum / count if count else None,
              "journey_mean_s": duration_sum / count if count and complete else None,
              "insertion_delay_mean_lower_bound_s": insertion_delay_sum / count if count else None}
    if kind == "vehicle":
        delay = (time_loss_sum + insertion_delay_sum) / count if count else None
        result["time_loss_plus_insertion_delay_mean_lower_bound_s"] = delay
        result["time_loss_plus_insertion_delay_mean_s"] = delay if complete else None
    return result


def prepare(destination):
    destination = Path(destination).resolve()
    if (destination / "manifest.json").exists():
        raise FileExistsError("Use a fresh destination or the existing manifest.")
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260914)
    d, ctrl, higher, lower = arguments()
    d["save_dir"] = str(destination / "geometry")
    higher["model_kwargs"]["run_dir"] = d["save_dir"]
    env = DesignEnv(d, ctrl, lower, d["save_dir"])
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    control_version = checkpoint["lower"].get("observation_version", 1)
    active_arms = ["fixed", "actuated", "tuned_fixed"]
    skip_reason = None
    if control_version == MLP_ActorCritic.observation_version:
        active_arms.insert(1, "learned")
    else:
        skip_reason = (f"Learned control is disabled: checkpoint observation version {control_version} "
                       f"is incompatible with version {MLP_ActorCritic.observation_version}. Fresh training is required.")
    env.normalizer_x = checkpoint["higher"]["norm_x"]
    env.normalizer_y = checkpoint["higher"]["norm_y"]
    state = env.reset()
    original_network = Path(env.network_dir) / "network_iteration_0.net.xml"
    original_slots = signal_slots_from_network(original_network, INTERSECTION)
    original = {"network": str(original_network),
                "iteration": "0", "num_proposals": len(original_slots), "real_world": True,
                "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
                "crossing_ids": [tid.removesuffix("_mid") for tid in original_slots],
                "signal_slots": original_slots}
    policy = PPO(**higher).policy
    policy.load_state_dict(checkpoint["higher"]["state_dict"])
    policy.eval()
    with torch.no_grad():
        gmm = policy.get_gmm_distribution(Batch.from_data_list([state]), "cpu")[0]
        _, merged, count, _ = policy.act(state, None, d["clamp_min"], d["clamp_max"], "cpu", training=False)
        proposals = merged[0, :int(count.item())].numpy()
        counts = Counter()
        for _ in range(1000):
            _, _, n, _ = policy.act(state, None, d["clamp_min"], d["clamp_max"], "cpu", training=True)
            counts[int(n.item())] += 1
    span = env.normalizer_x["max"] - env.normalizer_x["min"]
    physical = physical_proposals(proposals, env.normalizer_x, d)
    env._apply_action(proposals, "review")
    learned = {"network": str(Path(env.current_net_file_path).resolve()),
               "iteration": "review", "num_proposals": len(proposals), "real_world": False,
               "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
               "crossing_ids": list(env.crossing_ids), "signal_slots": dict(env.signal_slots)}
    for layout in (original, learned):
        layout["network"] = str(Path(layout["network"]).resolve())
        layout["sha256"] = digest(layout["network"])
    manifest = {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
                "observation_version": MLP_ActorCritic.observation_version,
                "checkpoint_control_version": control_version, "active_arms": active_arms,
                "learned_control_skip_reason": skip_reason,
                "configuration": dict(design_args=d, control_args=ctrl,
                                      higher_ppo_args=higher, lower_ppo_args=lower),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_hashes": {p: digest(ROOT / p) for p in [
                    "ppo/models.py", "ppo/ppo.py", "ppo/ppo_utils.py", "simulation/control_env.py",
                    "simulation/design_env.py", "simulation/worker.py", "simulation/env_utils.py",
                    "simulation/sim_setup.py", "utils.py", "review_validation.py", "uv.lock",
                    ctrl["vehicle_input_trips"], ctrl["pedestrian_input_trips"]]},
                "layouts": {"original": original, "learned": learned},
                "normalizer_x": env.normalizer_x,
                "sigma_normalized": float(np.exp(-2.5)), "sigma_location_m": float(np.exp(-2.5)*span),
                "sigma_width_m": float(np.exp(-2.5)*13), "merge_location_m": 0.08*span,
                "mixture_means": gmm.component_distribution.mean.tolist(),
                "mixture_probabilities": gmm.mixture_distribution.probs.tolist(),
                "evaluation_proposals": physical,
                "training_sample_merged_counts_1000_draws": dict(sorted(counts.items())),
                "sampling_diagnostic": "Frozen checkpoint conditioned on the original graph, seed 20260914; not historical training counts.",
                "protocol": {"warmup": "same fixed-time program for every arm; seeded 40-140 s, rounded down to 10 s",
                             "measurement_horizon_s": 450,
                             "demand": "Compression and repetition with horizon filtering; training [0,2400), evaluation [2400,3600). Report realized counts for nonuniform demand.",
                             "learned_control": skip_reason or "Same compatible frozen controller on both layouts; zero-shot comparison, no per-layout retraining.",
                             "measurement": "Environment and independent telemetry both accumulate 1 s waiting increments during the measurement period. Tripinfo retains complete and unfinished journeys including warmup.",
                             "actuated": "Intersection gap actuation with 10-45 s greens; mid-block pedestrian request, 10 s minimum vehicle green, original 4/16/2 s transition/pedestrian phases.",
                             "tuned_fixed": "Select symmetric intersection/mid-block vehicle greens using training-window trials only; retain all other phase durations."}}
    save(destination / "manifest.json", manifest)
    if skip_reason:
        print(skip_reason)
    print(json.dumps({k: manifest[k] for k in ["sigma_location_m", "merge_location_m", "evaluation_proposals", "training_sample_merged_counts_1000_draws"]}, indent=2))


class Telemetry(traci.StepListener):
    def __init__(self, env, arm):
        self.env, self.arm = env, arm
        self.active = False
        self.controller_active = False
        self.inserted = {"vehicle": set(), "pedestrian": set()}
        self.completed = {"vehicle": set(), "pedestrian": set()}
        self.present = {"vehicle": set(), "pedestrian": set()}
        self.prev_wait = {"vehicle": {}, "pedestrian": {}}
        self.wait = {"vehicle": 0.0, "pedestrian": 0.0}
        self.departures = {}
        self.first_approach = {}
        self.teleports, self.collisions = [], []
        self.mb_phase = {}
        self.phase_started = {}

    def step(self, unused=0):
        now = traci.simulation.getTime()
        for kind, domain, departed, arrived in [
            ("vehicle", traci.vehicle, traci.simulation.getDepartedIDList(), traci.simulation.getArrivedIDList()),
            ("pedestrian", traci.person, traci.simulation.getDepartedPersonIDList(), traci.simulation.getArrivedPersonIDList())]:
            self.inserted[kind].update(departed)
            self.completed[kind].update(arrived)
            ids = domain.getIDList()
            if self.active:
                self.present[kind].update(ids)
            for identifier in ids:
                value = domain.getWaitingTime(identifier)
                previous = self.prev_wait[kind].get(identifier, 0.0)
                if self.active:
                    self.wait[kind] += value-previous if value >= previous else value
                self.prev_wait[kind][identifier] = value
                if kind == "pedestrian":
                    self.departures.setdefault(identifier, now)
                    edge = domain.getRoadID(identifier)
                    if edge in self.env.mb_ped_incoming_edges_all:
                        self.first_approach.setdefault(identifier, now-self.departures[identifier])
        self.teleports.extend({"time": now, "id": x} for x in traci.simulation.getStartingTeleportIDList())
        self.collisions.extend({"time": now, "id": x} for x in traci.simulation.getCollidingVehiclesIDList())
        if self.controller_active and self.arm == "actuated":
            for tid in self.env.tl_ids[1:]:
                phase = traci.trafficlight.getPhase(tid)
                if phase != self.mb_phase.get(tid):
                    self.mb_phase[tid] = phase
                    self.phase_started[tid] = now
                if phase == 0 and now-self.phase_started[tid] >= 10:
                    approach = self.env.tl_lane_dict[tid]["pedestrian"]["incoming"]["north"]["main"]
                    crossing = self.env.junction_pos_cache[tid]
                    request = any(traci.person.getRoadID(p) in approach and
                                  np.linalg.norm(np.array(traci.person.getPosition(p))-crossing) < 15
                                  for p in traci.person.getIDList())
                    if request:
                        traci.trafficlight.setPhase(tid, 1)
        return True


def install_controller(env, arm, parameters):
    if arm == "learned" or arm == "fixed":
        return
    for tid in env.tl_ids:
        logic = traci.trafficlight.getAllProgramLogics(tid)[0]
        logic.programID = "review_" + arm
        if arm == "tuned_fixed":
            for phase in logic.phases:
                if tid == INTERSECTION and float(phase.duration) == 90:
                    phase.duration = parameters[0]
                elif tid != INTERSECTION and phase.state == "GGr":
                    phase.duration = parameters[1]
        elif arm == "actuated":
            if tid == INTERSECTION:
                logic.type = 3  # SUMO trafficlight type ACTUATED
                for phase in logic.phases:
                    if float(phase.duration) == 90:
                        phase.duration, phase.minDur, phase.maxDur = 10, 10, 45
            else:
                # Pedestrian requests release the otherwise resting vehicle phase.
                logic.phases[0].duration = 100000
        traci.trafficlight.setProgramLogic(tid, logic)
        traci.trafficlight.setProgram(tid, logic.programID)
        traci.trafficlight.setPhase(tid, 0)


def trial(job):
    os.chdir(ROOT)
    torch.set_num_threads(1)
    manifest_bytes = Path(job["manifest"]).read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    m = json.loads(manifest_bytes)
    for name, expected in m["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source changed since prepare: {name}. Use a fresh study directory.")
    if digest(m["checkpoint"]) != m["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since prepare. Use a fresh study directory.")
    if m.get("observation_version") != MLP_ActorCritic.observation_version:
        raise ValueError("Use prepare in a fresh directory for the current observation protocol.")
    configuration = m.get("configuration")
    if configuration is None:
        raise ValueError("Manifest lacks an embedded configuration. Use prepare in a fresh study directory.")
    if job["arm"] not in m["active_arms"]:
        raise ValueError(m["learned_control_skip_reason"] or "Controller is not enabled in this manifest.")
    layout = m["layouts"][job["layout"]]
    if "sha256" in layout and digest(layout["network"]) != layout["sha256"]:
        raise ValueError("Network changed since the final layout was recorded. Use a fresh study directory.")
    family = layout.get("family_slots")
    if family is not None:
        # Paired variants of one base layout keep every surviving signal in its base slot.
        slots = layout.get("signal_slots") or {}
        moved = sorted(tid for tid in slots.keys() & family.keys() if slots[tid] != family[tid])
        if moved:
            raise ValueError(f"Signals {moved} leave their family base slots; shared identities must keep their slots.")
        if any(tid not in family and slot in family.values() for tid, slot in slots.items()):
            raise ValueError("New signals cannot reuse reserved family base slots.")
    warmup_control = m.get("warmup_control", "fixed")
    if warmup_control not in ("fixed", "random"):
        raise ValueError("warmup_control must be 'fixed' or 'random'.")
    if warmup_control == "random" and m["active_arms"] != ["learned"]:
        raise ValueError("Random warmup requires a learned-only manifest.")
    journey_protocol = m.get("journey_protocol")
    if journey_protocol is not None and journey_protocol != JOURNEY_PROTOCOL:
        raise ValueError(f"journey_protocol must declare {JOURNEY_PROTOCOL}.")
    folder = Path(job["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing["job"] != job:
            raise ValueError("Existing trial uses a different specification.")
        if existing.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Existing trial uses a different manifest or configuration. Use a fresh trial directory.")
        return str(result_path)
    network_dir = folder / "network_iterations"
    network_dir.mkdir(exist_ok=True)
    net = network_dir / f"network_iteration_{layout['iteration']}.net.xml"
    if not net.exists():
        net.symlink_to(layout["network"])
    seed = job["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    d, ctrl, higher, lower = arguments(configuration)
    ctrl.update(gui=False, max_timesteps=450, total_action_timesteps_per_episode=45,
                vehicle_output_trips=str(folder / "vehicles.xml"),
                pedestrian_output_trips=str(folder / "pedestrians.xml"),
                manual_demand_veh=job["scale"], manual_demand_ped=job["scale"])
    if journey_protocol is not None:
        ctrl["warmup_steps"] = [journey_protocol["warmup_s"]] * 2
    if job["arm"] == "learned":
        checkpoint = torch.load(m["checkpoint"], map_location="cpu")
        state_stats = checkpoint["lower"]
        if state_stats.get("observation_version", 1) != MLP_ActorCritic.observation_version:
            raise ValueError("Legacy control weights and Welford statistics require fresh training for this observation protocol.")
        require_exposed_heads(state_stats.get("provenance"), layout.get("signal_slots"))
        policy = PPO(**lower).policy
        policy.load_state_dict(state_stats["state_dict"]); policy.eval()
        normalizer = WelfordNormalizer(state_stats["state_normalizer_mean"].shape)
        normalizer.manual_load(torch.from_numpy(state_stats["state_normalizer_mean"]),
                               torch.from_numpy(state_stats["state_normalizer_M2"]), state_stats["state_normalizer_count"])
        normalizer.eval()
    env = ControlEnv(ctrl, str(folder), worker_id=0, network_iteration=layout["iteration"], current_net_file_path=str(net))
    tracker = Telemetry(env, job["arm"])
    cohort = {}
    original_start = traci.start
    def instrumented_start(command, *args, **kwargs):
        if journey_protocol is not None:
            cutoff = journey_protocol["warmup_s"] + journey_protocol["measurement_horizon_s"]
            for kind, file, tag in [("vehicle", env.vehicle_output_trips, "trip"),
                                    ("pedestrian", env.pedestrian_output_trips, "person")]:
                cohort[kind] = restrict_departure_cohort(file, tag, cutoff)
        command = list(command) + ["--seed", str(seed), "--no-step-log", "true", "--duration-log.disable", "true",
                                   "--tripinfo-output", str(folder / "tripinfo.xml"), "--tripinfo-output.write-unfinished", "true"]
        value = original_start(command, *args, **kwargs)
        traci.addStepListener(tracker)
        return value
    traci.start = instrumented_start
    started = time.monotonic()
    try:
        with (folder / "stdout.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            state, _ = env.reset(layout["extreme_edges"], layout["num_proposals"], tl=warmup_control == "fixed",
                                 real_world=layout["real_world"], eval_mode=job["split"] == "evaluation",
                                 signal_slots=layout.get("signal_slots"))
            warmup = traci.simulation.getTime()
            if journey_protocol is not None:
                assert warmup == journey_protocol["warmup_s"]
            # Every arm starts from the same traffic state for a layout/seed/scale.
            initial_state_hash = hashlib.sha256(np.asarray(state).tobytes()).hexdigest()
            install_controller(env, job["arm"], job.get("parameters"))
            tracker.active = True
            tracker.controller_active = True
            measured_wait = {"vehicle": 0.0, "pedestrian": 0.0}
            action = np.zeros(1 + layout["num_proposals"], dtype=np.int32)
            for _ in range(45):
                if job["arm"] == "learned":
                    with torch.no_grad():
                        action, _ = policy.act(normalizer.normalize(torch.as_tensor(state)), layout["num_proposals"],
                                               training=False, active_slots=env.active_slots)
                    action = action.cpu()
                state, _, done, _, info = env.eval_step(action, tl=job["arm"] != "learned")
                for kind in measured_wait:
                    measured_wait[kind] += info[kind + "_wait"]
            assert env.step_count == 450 and done
            assert measured_wait == tracker.wait, (measured_wait, tracker.wait)
            end_time = traci.simulation.getTime()
            records = {}
            for kind, file, tag in [("vehicle", env.vehicle_output_trips, "trip"), ("pedestrian", env.pedestrian_output_trips, "person")]:
                elements = ET.parse(file).getroot().findall(".//" + tag)
                scheduled = {e.get("id") for e in elements if float(e.get("depart")) < end_time}
                due = {e.get("id"): float(e.get("depart")) for e in elements if float(e.get("depart")) < end_time}
                not_inserted = scheduled - tracker.inserted[kind]
                records[kind] = {"scheduled": len(scheduled), "inserted": len(tracker.inserted[kind]),
                                 "completed": len(tracker.completed[kind]), "not_inserted": len(not_inserted),
                                 "unfinished_inserted": len(tracker.inserted[kind] - tracker.completed[kind]),
                                 "observed_during_measurement": len(tracker.present[kind]),
                                 "wait_sum_1s": tracker.wait[kind],
                                 "wait_mean_1s": tracker.wait[kind]/len(tracker.present[kind]) if tracker.present[kind] else None,
                                 "environment_wait_sum_1s": measured_wait[kind],
                                 "backlog_age_at_end_s": sum(end_time-due[i] for i in not_inserted),
                                 "demand_sha256": digest(file)}
            result = {"job": job, "manifest_sha256": manifest_sha256, "warmup_s": warmup,
                      "warmup_control": warmup_control, "measurement_s": env.step_count,
                      "initial_state_sha256": initial_state_hash, "traffic": records,
                      "teleports": list(tracker.teleports), "collisions": list(tracker.collisions),
                      "environment_approach_mean": statistics.mean(env.pedestrian_arrival_times.values()) if env.pedestrian_arrival_times else None,
                      "environment_approach_n": len(env.pedestrian_arrival_times),
                      "clock_arrival_mean": statistics.mean(tracker.first_approach.values()) if tracker.first_approach else None,
                      "clock_arrival_n": len(tracker.first_approach), "elapsed_s": time.monotonic()-started}
            if journey_protocol is not None:
                # Measurement has ended; controller service continues without a reset.
                tracker.active = False
                deadline = end_time + journey_protocol["drain_cap_s"]
                env.max_timesteps += journey_protocol["drain_cap_s"]
                while (traci.simulation.getTime() < deadline and
                       any(set(cohort[kind]) - tracker.completed[kind] for kind in cohort)):
                    if job["arm"] == "learned":
                        with torch.no_grad():
                            action, _ = policy.act(normalizer.normalize(torch.as_tensor(state)), layout["num_proposals"],
                                                   training=False, active_slots=env.active_slots)
                        action = action.cpu()
                    state, _, _, _, _ = env.eval_step(action, tl=job["arm"] != "learned")
                simulation_end = traci.simulation.getTime()
                assert simulation_end <= deadline
                assert measured_wait == tracker.wait
                result["elapsed_s"] = time.monotonic()-started
    finally:
        traci.start = original_start
        if env.sumo_running:
            traci.switch(env.traci_label)
            traci.close()  # Wait for SUMO to finish writing complete and unfinished trip records.
            env.sumo_running = False
    tripinfo = ET.parse(folder / "tripinfo.xml").getroot()
    journeys = {"scope": "Full simulation including warmup; duration summaries use completed journeys.",
                "tripinfo_sha256": digest(folder / "tripinfo.xml")}
    for kind, tag in [("vehicle", "tripinfo"), ("pedestrian", "personinfo")]:
        entries = tripinfo.findall(tag)
        inserted = [entry for entry in entries if float(entry.get("depart")) >= 0]
        completed = [entry for entry in inserted
                     if float((entry if kind == "vehicle" else entry[-1]).get("arrival")) >= 0]
        assert {entry.get("id") for entry in inserted} == tracker.inserted[kind]
        assert {entry.get("id") for entry in completed} == tracker.completed[kind]
        journeys[kind] = {"completed": len(completed), "unfinished_inserted": len(inserted) - len(completed),
                          "not_departed_records": len(entries) - len(inserted)}
        if kind == "vehicle":
            for attribute in ("duration", "timeLoss", "departDelay"):
                journeys[kind]["completed_" + attribute + "_sum_s"] = sum(float(entry.get(attribute)) for entry in completed)
        else:
            journeys[kind]["completed_duration_sum_s"] = sum(float(entry[-1].get("arrival")) - float(entry.get("depart")) for entry in completed)
            journeys[kind]["completed_walk_duration_sum_s"] = sum(float(stage.get("duration")) for entry in completed for stage in entry.findall("walk"))
    result["journeys"] = journeys
    if journey_protocol is not None:
        journeys["scope"] = "Full simulation including warmup and drain; legacy duration sums use completed journeys only."
        summaries = {}
        for kind, tag in [("vehicle", "tripinfo"), ("pedestrian", "personinfo")]:
            assert tracker.inserted[kind] <= set(cohort[kind])
            assert tracker.completed[kind] <= set(cohort[kind])
            summaries[kind] = cohort_summary(cohort[kind], tripinfo.findall(tag), kind, simulation_end)
        journeys["cohort"] = {
            "protocol": journey_protocol,
            "scope": "All trips scheduled in [0, measurement_end_s), including warmup. No later departures are supplied. Journey time starts at scheduled departure and includes insertion delay.",
            "censoring": "At the common drain cap, incomplete journeys contribute their elapsed age to a lower bound; journey_mean_s is null unless every scheduled trip completes.",
            "measurement_end_s": end_time, "simulation_end_s": simulation_end,
            "drain_s": simulation_end - end_time,
            "stop_reason": "cohort_complete" if all(s["all_completed"] for s in summaries.values()) else "drain_cap",
            "traffic": summaries, "teleports": tracker.teleports, "collisions": tracker.collisions}
    save(result_path, result)
    return str(result_path)


def run_jobs(jobs, fail_fast=True):
    """Run trials in parallel. Failures raise, or with fail_fast=False are returned by directory."""
    failures = {}
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as executor:
        pending = {executor.submit(trial, job): job for job in jobs}
        for future in as_completed(pending):
            try:
                path = future.result()  # Fail visibly; never turn a failed run into zero delay.
            except Exception as error:
                if fail_fast:
                    raise
                directory = pending[future]["directory"]
                failures[directory] = f"{type(error).__name__}: {error}"
                print(f"FAILED {directory}: {failures[directory]}", flush=True)
                continue
            print(path, flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "smoke", "tune", "search", "matrix"],
                        help="prepare: freeze a study manifest; smoke: one held-out seed; tune: fixed-time greens on training trials; "
                             "search: count/width-matched Uniform and best-of-20 random placements selected on training trials; "
                             "matrix: held-out rows, including baseline layouts once searched")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    directory = args.directory.resolve()
    if args.operation == "prepare":
        prepare(directory)
        return
    manifest = directory / "manifest.json"
    metadata = json.loads(manifest.read_text())
    if metadata.get("observation_version") != MLP_ActorCritic.observation_version:
        raise ValueError("Use prepare in a fresh directory for the current observation protocol.")
    if metadata["learned_control_skip_reason"]:
        print(metadata["learned_control_skip_reason"], flush=True)
    jobs = []
    if args.operation == "search":
        if (directory / "baselines" / "selection.json").exists():
            raise FileExistsError("Baseline selection is frozen. Use a fresh study directory.")
        baseline = baseline_manifest(directory)
        jobs = search_jobs(directory, baseline)
        failures = run_jobs(jobs, fail_fast=False)
        selection = select_baseline(directory, baseline, jobs, failures)
        print(json.dumps({k: selection[k] for k in ("random_best20", "random_best20_score", "eligible_candidates",
                                                     "failed_or_incomplete_candidates")}), flush=True)
        return
    if args.operation == "tune":
        # Training-only selection at 1x and 2x, three seeds. No test feedback.
        variants = [(i, b) for i in (15, 30, 60, 90) for b in (10, 20, 40)]
        for layout in ("original", "learned"):
            for i, b in variants:
                for scale in SELECTION_SCALES:
                    for seed in SELECTION_SEEDS:
                        jobs.append(dict(manifest=str(manifest), layout=layout, arm="tuned_fixed", scale=scale, seed=seed,
                                         split="training", parameters=[i,b], directory=str(directory/"tuning"/f"{layout}_{i}_{b}_{scale}_{seed}")))
    else:
        arms = [arm for arm in metadata["active_arms"] if arm != "tuned_fixed"]
        scales = [.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75]
        seeds = range(6100, 6110)
        if args.operation == "smoke":
            scales, seeds = [1.0], [6100]
        else:
            arms.append("tuned_fixed")
            selected = json.loads((directory/"selected_timing.json").read_text())
        for layout in ("original", "learned"):
            for arm in arms:
                for scale in scales:
                    for seed in seeds:
                        job = dict(manifest=str(manifest), layout=layout, arm=arm, scale=scale, seed=seed,
                                   split="evaluation", directory=str(directory/"trials"/f"{layout}_{arm}_{scale}_{seed}"))
                        if arm == "tuned_fixed":
                            job["parameters"] = selected[layout]
                        jobs.append(job)
        jobs += baseline_rows(directory, scales, seeds)
    run_jobs(jobs)
    if args.operation == "tune":
        selected = {}
        scores = {}
        for layout in ("original", "learned"):
            candidates = []
            for params in variants:
                values = []
                for j in jobs:
                    if j["layout"] == layout and j["parameters"] == list(params):
                        r = json.loads((Path(j["directory"])/"result.json").read_text())
                        values.append(selection_score(r))
                candidates.append((statistics.mean(values), params))
            candidates.sort()
            selected[layout] = list(candidates[0][1])
            scores[layout] = candidates
        save(directory/"selected_timing.json", selected)
        save(directory/"tuning_scores.json", scores)


if __name__ == "__main__":
    main()
