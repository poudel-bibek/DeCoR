"""Reproducible, separate review diagnostics; never overwrite historical results.

Uses the corrected ControlEnv and demand scaler. Each manifest fixes the
warmup mode, seeds, demand files and a 450 s measurement horizon. Learned control
requires a compatible observation version, and decoding a design checkpoint
requires compatible readout semantics. An opt-in journey protocol fixes the
departure cohort and drains it after measurement. No policy training occurs.
Placement-matched baselines (Uniform and best-of-20 random search) keep the
reference layout's crossing count and west-to-east widths and vary placement
only; random search selects on training-window trials under actuated control.
The feedback comparison uses an explicit matched placement grid, three design
selection scores, equal-class full-journey evaluation and a frozen selection.
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
import re
import statistics
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import traci
from torch_geometric.data import Batch
from scipy.stats import t as student_t

from config import classify_and_return_args, get_config

from ppo.ppo import PPO
from ppo.models import MLP_ActorCritic
from ppo.ppo_utils import WelfordNormalizer
from simulation.control_env import ControlEnv
from simulation.design_env import DesignEnv
from simulation.signal_control import PROTOCOL as SHARED_SIGNAL_PROTOCOL, CoordinatedSchedule, LocalActuated, coordinated_grid
from utils import load_design_policy, require_exposed_heads, signal_slots_from_network

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


def completed_training_reference(path):
    """Read an immutable training endpoint, not a newly decoded or selected design."""
    path = Path(path).resolve()
    artifact_bytes = path.read_bytes()
    record = json.loads(artifact_bytes)
    if record.get("complete") is not True:
        raise ValueError("The training artifact must contain a completed arm.")
    study_path = path.parents[2] / "study.json"
    study_bytes = study_path.read_bytes()
    study = json.loads(study_bytes)
    if study["settings"]["development"]:
        raise ValueError("Development artifacts are training-only gates, not held-out baseline references.")
    if record["seed"] not in study["seeds"] or record["source_hashes"] != study["source_hashes"]:
        raise ValueError("Training artifact provenance does not match its declared study.")
    if not study["evaluation_scales"] or not study["evaluation_seeds"]:
        raise ValueError("The training study must declare held-out scales and seeds.")
    if digest(record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("The completed training checkpoint does not match its recorded hash.")
    layout = record["layout"]
    if digest(layout["network"]) != layout["sha256"]:
        raise ValueError("The completed training layout does not match its recorded hash.")
    count, slots = layout["num_proposals"], layout["signal_slots"]
    proposals = np.asarray(layout["proposals"], dtype=np.float32)
    if proposals.shape != (count, 2) or not np.isfinite(proposals).all():
        raise ValueError("Recorded layout proposals must match its crossing count.")
    limit = record["configuration"]["design_args"]["max_proposals"]
    if (len(slots) != count or len(layout["crossing_ids"]) != count
            or set(slots) != {f"{cid}_mid" for cid in layout["crossing_ids"]}
            or set(slots) != set(signal_slots_from_network(layout["network"], INTERSECTION))
            or any(not isinstance(slot, int) or not 0 <= slot < limit for slot in slots.values())
            or len(set(slots.values())) != count):
        raise ValueError("Recorded crossing identities and slots do not match the compiled final layout.")
    provenance = {"path": str(path), "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                  "study": str(study_path), "study_sha256": hashlib.sha256(study_bytes).hexdigest(),
                  "arm": record["arm"], "seed": record["seed"], "settings": study["settings"],
                  "source_hashes": record["source_hashes"]}
    return record, study, provenance


def prepare(destination, training_artifact=None):
    destination = Path(destination).resolve()
    if (destination / "manifest.json").exists():
        raise FileExistsError("Use a fresh destination or the existing manifest.")
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260914)
    reference = study = provenance = None
    checkpoint_path = CHECKPOINT
    if training_artifact is not None:
        reference, study, provenance = completed_training_reference(training_artifact)
        checkpoint_path = Path(reference["checkpoint"])
    d, ctrl, higher, lower = arguments(reference["configuration"] if reference is not None else None)
    if reference is not None:
        # Scientific inputs come from the training snapshot; evaluator source bytes are independent.
        snapshot = Path(provenance["study"]).parent / "source_snapshot"
        for args, key in ((d, "original_net_file"), (ctrl, "vehicle_input_trips"), (ctrl, "pedestrian_input_trips")):
            name = str(Path(args[key]))
            source = (snapshot / name).resolve()
            if digest(source) != reference["source_hashes"].get(name):
                raise ValueError(f"Training input does not match its recorded hash: {name}")
            args[key] = str(source)
        ctrl.update(vehicle_output_trips=str(destination / "vehicles.xml"),
                    pedestrian_output_trips=str(destination / "pedestrians.xml"))
    d["save_dir"] = str(destination / "geometry")
    higher["model_kwargs"]["run_dir"] = d["save_dir"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    control_version = checkpoint["lower"].get("observation_version", 1)
    active_arms = ["fixed", "actuated", "tuned_fixed"]
    skip_reason = None
    if control_version == MLP_ActorCritic.observation_version:
        active_arms.insert(1, "learned")
    else:
        skip_reason = (f"Learned control is disabled: checkpoint observation version {control_version} "
                       f"is incompatible with version {MLP_ActorCritic.observation_version}. Fresh training is required.")
    if reference is not None:
        active_arms = ["actuated"]
        skip_reason = "Placement baselines use common actuated control, not learned-policy performance."
    policy = PPO(**higher).policy
    norm_x, norm_y = load_design_policy(policy, checkpoint["higher"])
    env = DesignEnv(d, ctrl, lower, d["save_dir"])
    env.normalizer_x, env.normalizer_y = norm_x, norm_y
    state = env.reset()
    original_network = Path(env.network_dir) / "network_iteration_0.net.xml"
    original_slots = signal_slots_from_network(original_network, INTERSECTION)
    original = {"network": str(original_network),
                "iteration": "0", "num_proposals": len(original_slots), "real_world": True,
                "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
                "crossing_ids": [tid.removesuffix("_mid") for tid in original_slots],
                "signal_slots": original_slots}
    diagnostics = {}
    if reference is None:
        policy.eval()
        with torch.no_grad():
            gmm = policy.get_gmm_distribution(Batch.from_data_list([state]), "cpu")[0]
            _, merged, count, _ = policy.act(state, None, d["clamp_min"], d["clamp_max"], "cpu", training=False)
            proposals = merged[0, :int(count.item())].numpy()
            counts = Counter()
            for _ in range(1000):
                _, _, n, _ = policy.act(state, None, d["clamp_min"], d["clamp_max"], "cpu", training=True)
                counts[int(n.item())] += 1
        env._apply_action(proposals, "review")
        learned = {"network": str(Path(env.current_net_file_path).resolve()),
                   "iteration": "review", "num_proposals": len(proposals), "real_world": False,
                   "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
                   "crossing_ids": list(env.crossing_ids), "signal_slots": dict(env.signal_slots)}
        span = env.normalizer_x["max"] - env.normalizer_x["min"]
        diagnostics = {"sigma_normalized": float(np.exp(-2.5)), "sigma_location_m": float(np.exp(-2.5)*span),
                       "sigma_width_m": float(np.exp(-2.5)*13), "merge_location_m": 0.08*span,
                       "mixture_means": gmm.component_distribution.mean.tolist(),
                       "mixture_probabilities": gmm.mixture_distribution.probs.tolist(),
                       "training_sample_merged_counts_1000_draws": dict(sorted(counts.items())),
                       "sampling_diagnostic": "Frozen checkpoint conditioned on the original graph, seed 20260914; not historical training counts."}
    else:
        learned = copy.deepcopy(reference["layout"])
        proposals = np.asarray(learned["proposals"], dtype=np.float32)
    physical = physical_proposals(proposals, env.normalizer_x, d)
    for layout in ((original, learned) if reference is None else (original,)):
        layout["network"] = str(Path(layout["network"]).resolve())
        layout["sha256"] = digest(layout["network"])
    manifest = {"checkpoint": str(checkpoint_path),
                "checkpoint_sha256": reference["checkpoint_sha256"] if reference is not None else digest(checkpoint_path),
                "observation_version": MLP_ActorCritic.observation_version,
                "checkpoint_control_version": control_version, "active_arms": active_arms,
                "learned_control_skip_reason": skip_reason,
                "configuration": dict(design_args=d, control_args=ctrl,
                                      higher_ppo_args=higher, lower_ppo_args=lower),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_hashes": {p: digest(ROOT / p) for p in [
                    "ppo/models.py", "ppo/ppo.py", "ppo/ppo_utils.py", "simulation/control_env.py",
                    "simulation/design_env.py", "simulation/worker.py", "simulation/env_utils.py",
                    "simulation/sim_setup.py", "utils.py", "review_validation.py", "uv.lock",
                    ctrl["vehicle_input_trips"], ctrl["pedestrian_input_trips"]]},
                "layouts": {"original": original, "learned": learned},
                "normalizer_x": env.normalizer_x, "normalizer_y": env.normalizer_y,
                "evaluation_proposals": physical, **diagnostics,
                "protocol": {"warmup": "same fixed-time program for every arm; seeded 40-140 s, rounded down to 10 s",
                             "measurement_horizon_s": 450,
                             "demand": "Compression and repetition with horizon filtering; training [0,2400), evaluation [2400,3600). Report realized counts for nonuniform demand.",
                             "learned_control": skip_reason or "Same compatible frozen controller on both layouts; zero-shot comparison, no per-layout retraining.",
                             "measurement": "Environment and independent telemetry both accumulate 1 s waiting increments during the measurement period. Tripinfo retains complete and unfinished journeys including warmup.",
                             "actuated": "Intersection gap actuation with 10-45 s greens; mid-block pedestrian request, 10 s minimum vehicle green, original 4/16/2 s transition/pedestrian phases.",
                             "tuned_fixed": "Select symmetric intersection/mid-block vehicle greens using training-window trials only; retain all other phase durations."}}
    if reference is not None:
        manifest.update(training_artifact=provenance,
                        evaluation_scales=study["evaluation_scales"], evaluation_seeds=study["evaluation_seeds"],
                        journey_protocol=study["journey_protocol"])
        manifest["source_hashes"][provenance["path"]] = provenance["sha256"]
        manifest["source_hashes"][d["original_net_file"]] = digest(d["original_net_file"])
        manifest["protocol"].update(
            warmup=f"Common fixed-time warmup for every placement, {study['journey_protocol']['warmup_s']} s.",
            learned_control=skip_reason,
            reference="The completed training endpoint is reused without policy sampling or held-out selection.",
            measurement=f"Declared finite cohort {study['journey_protocol']}; compare these common-actuated rows together, not against random-warmup learned-policy results.")
    save(destination / "manifest.json", manifest)
    if skip_reason:
        print(skip_reason)
    print(json.dumps({k: manifest[k] for k in ["evaluation_proposals", "training_sample_merged_counts_1000_draws"] if k in manifest}, indent=2))


def prepare_feedback(destination, protocol_path):
    """Freeze an explicit placement grid and non-learning comparison; no historical weights."""
    destination = Path(destination).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    config = get_config()
    config.update(gui=False, gpu=False, evaluate=False, save_graph_images=False, save_gmm_plots=False)
    d, ctrl, higher, lower, _ = classify_and_return_args(config, "cpu")
    placements = [np.asarray(p, dtype=np.float32) for p in protocol["placements"]]
    if len(placements) < 2:
        raise ValueError("Declare at least two candidate placements.")
    for proposals in placements:
        if (proposals.ndim != 2 or proposals.shape[1] != 2 or
                not 1 <= len(proposals) <= d["max_proposals"] or
                not np.isfinite(proposals).all() or np.any((proposals[:, 1] < 0) | (proposals[:, 1] > 1))):
            raise ValueError("Placements must contain finite normalized location/width pairs.")
        check_matched(proposals, placements[0][:, 1])
    for key in ("selection_scales", "evaluation_scales"):
        values = protocol[key]
        if not values or len(set(values)) != len(values) or any(not np.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"{key} must contain distinct positive scales.")
    for key in ("selection_seeds", "evaluation_seeds"):
        values = protocol[key]
        if not values or len(set(values)) != len(values) or any(not isinstance(x, int) or x < 0 for x in values):
            raise ValueError(f"{key} must contain distinct nonnegative integer seeds.")
    if set(protocol["selection_seeds"]) & set(protocol["evaluation_seeds"]):
        raise ValueError("Use different random seeds for selection and evaluation.")
    timings = protocol["timings"]
    if not timings or any(len(p) != 2 or any(not np.isfinite(x) or x <= 0 for x in p) for p in timings):
        raise ValueError("Declare positive intersection/mid-block green-duration pairs.")
    if not np.isfinite(protocol["practical_margin_s"]) or protocol["practical_margin_s"] <= 0:
        raise ValueError("Declare a positive practical performance margin in seconds.")
    destination.mkdir(parents=True, exist_ok=False)
    sources = ("config.py", "review_validation.py", "utils.py", "uv.lock", "ppo/models.py", "ppo/ppo.py",
               "ppo/ppo_utils.py", "simulation/control_env.py", "simulation/design_env.py",
               "simulation/worker.py", "simulation/env_utils.py", "simulation/sim_setup.py")
    source_hashes = {}
    for name in sources:
        target = destination / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        source_hashes[name] = digest(target)
    for settings, key in ((d, "original_net_file"), (ctrl, "vehicle_input_trips"), (ctrl, "pedestrian_input_trips")):
        source = (ROOT / settings[key]).resolve()
        target = destination / "source_snapshot" / "simulation" / source.name
        shutil.copy2(source, target)
        settings[key] = str(target)
        source_hashes[str(target)] = digest(target)
    ctrl.update(vehicle_output_trips=str(destination / "vehicles.xml"),
                pedestrian_output_trips=str(destination / "pedestrians.xml"))
    d["save_dir"] = higher["model_kwargs"]["run_dir"] = str(destination / "geometry")
    torch.set_num_threads(1)
    torch.manual_seed(GEOMETRY_SEED)
    env = DesignEnv(d, ctrl, lower, d["save_dir"])
    env.reset()
    layouts = {}
    for index, proposals in enumerate(placements):
        name = f"placement_{index:03d}"
        env._apply_action(proposals, name)
        network = str(Path(env.current_net_file_path).resolve())
        layouts[name] = {"network": network, "sha256": digest(network), "iteration": name,
                         "num_proposals": len(proposals), "real_world": False,
                         "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
                         "crossing_ids": list(env.crossing_ids), "signal_slots": dict(env.signal_slots),
                         "proposals": physical_proposals(proposals, env.normalizer_x, d)}
    manifest = {"feedback_protocol": protocol, "checkpoint": None, "checkpoint_sha256": None,
                "observation_version": MLP_ActorCritic.observation_version,
                "active_arms": ["tuned_fixed", "actuated"],
                "learned_control_skip_reason": "This comparison uses non-learning controllers.",
                "warmup_control": "fixed", "journey_protocol": JOURNEY_PROTOCOL,
                "configuration": dict(design_args=d, control_args=ctrl, higher_ppo_args=higher, lower_ppo_args=lower),
                "source_hashes": source_hashes, "layouts": layouts,
                "normalizer_x": env.normalizer_x, "normalizer_y": env.normalizer_y,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "criterion": "Mean pedestrian scheduled-departure-to-arrival journey time plus mean vehicle "
                             "time loss and insertion delay; equal weight per class, not per traveler.",
                "eligibility": "Every selection cohort must fully complete without teleports or collisions "
                               "other than person-person overlaps, which remain diagnostic. "
                               "Undefined score components are not zero. All failed outcomes remain recorded.",
                "selection": "Tune fixed-time parameters for each layout using the common journey criterion, "
                             "then compare access-only, access-plus-approach-wait and vehicle-inclusive layout "
                             "selection scores on the same candidate trials. "
                             "Average per-trial scores uniformly over the declared scale-by-seed grid. "
                             "This ablates layout-ranking scores, not all operational information: fixed-time "
                             "tuning uses journey outcomes in every rule. Access is a simulated first-approach "
                             "age, not a geometric or controller-independent distance. "
                             "Actuation uses the existing common request/gap procedure. Ties keep declared candidate order.",
                "scope": "Within-recording exploratory/mechanistic comparison. Training window [0,2400); "
                         "evaluation window [2400,3600) has been inspected in prior studies. New seeds are "
                         "new simulation draws, not new observed data."}
    save(destination / "manifest.json", manifest)
    return destination / "manifest.json"


def feedback_scores(result):
    """Common full-journey criterion and information ablations; never score incomplete service as success."""
    cohort = result["journeys"]["cohort"]
    traffic = cohort["traffic"]
    serious_collision = any(event.get("type") != "person-person" for event in cohort["collisions"])
    if cohort["teleports"] or serious_collision or not all(x["all_completed"] for x in traffic.values()):
        return None
    pedestrian = traffic["pedestrian"]["journey_mean_s"]
    vehicle = traffic["vehicle"]["time_loss_plus_insertion_delay_mean_s"]
    if pedestrian is None or vehicle is None:
        return None
    access = result["feedback"]["access_mean_s"]
    wait = result["feedback"]["approach_wait_mean_s"]
    scores = {"journey": pedestrian + vehicle, "access": access,
              "access_wait": access + wait if access is not None and wait is not None else None,
              "access_wait_vehicle": access + wait + vehicle if access is not None and wait is not None else None}
    if any(value is not None and not np.isfinite(value) for value in scores.values()):
        return None
    return scores


def feedback_jobs(directory, stage):
    directory = Path(directory).resolve()
    manifest = directory / "manifest.json"
    metadata = json.loads(manifest.read_text())
    protocol = metadata["feedback_protocol"]
    variants = []
    if stage == "selection":
        for layout in metadata["layouts"]:
            variants.append((layout, "actuated", None, "actuated"))
            variants.extend((layout, "tuned_fixed", parameters, f"timing_{i:03d}")
                            for i, parameters in enumerate(protocol["timings"]))
    elif stage == "evaluation":
        selection = json.loads((directory / "feedback_selection.json").read_text())
        if selection["manifest_sha256"] != digest(manifest):
            raise ValueError("Frozen feedback selection does not match the manifest.")
        for path, expected in selection["result_sha256s"].items():
            if digest(path) != expected:
                raise ValueError("A selection result changed after choices were frozen.")
        for arm, choices in selection["choices"].items():
            if any(choice is None for choice in choices.values()):
                raise RuntimeError("A declared selection rule has no eligible layout; evaluation is not released.")
            for layout in dict.fromkeys(choices.values()):
                parameters = selection["timings"][layout] if arm == "tuned_fixed" else None
                variants.append((layout, arm, parameters, arm))
    else:
        raise ValueError("Feedback stage must be selection or evaluation.")
    jobs = []
    for layout, arm, parameters, label in variants:
        for scale in protocol[stage + "_scales"]:
            for seed in protocol[stage + "_seeds"]:
                job = dict(manifest=str(manifest), layout=layout, arm=arm, scale=scale, seed=seed,
                           split="training" if stage == "selection" else "evaluation",
                           directory=str(directory / stage / f"{layout}_{label}_{scale}_{seed}"))
                if parameters is not None:
                    job["parameters"] = parameters
                jobs.append(job)
    return jobs


def feedback_results(jobs, failures):
    """Read complete declared trials, preserving failure rows and checking paired demand and provenance."""
    manifest_sha256 = digest(jobs[0]["manifest"])
    records, demands = [], {}
    for job in jobs:
        path = Path(job["directory"]) / "result.json"
        row = {"job": job, "scores": None}
        if job["directory"] in failures:
            row["error"] = failures[job["directory"]]
        elif not path.exists():
            row["error"] = "missing result"
        else:
            result = json.loads(path.read_text())
            if result["job"] != job or result["manifest_sha256"] != manifest_sha256:
                raise ValueError(f"Trial provenance does not match the declared comparison: {path}")
            demand = {kind: value["demand_sha256"] for kind, value in result["traffic"].items()}
            key = job["scale"], job["seed"]
            if demands.setdefault(key, demand) != demand:
                raise ValueError("Demand differs between paired layout/controller trials.")
            cohort = result["journeys"]["cohort"]
            row.update(scores=feedback_scores(result), result=str(path), sha256=digest(path),
                       censored={kind: value["censored"] for kind, value in cohort["traffic"].items()},
                       simulation_s=cohort["simulation_end_s"], warmup_s=result["warmup_s"],
                       measurement_s=result["measurement_s"], drain_s=cohort["drain_s"],
                       elapsed_s=result["elapsed_s"],
                       scheduled={kind: value["scheduled"] for kind, value in cohort["traffic"].items()},
                       teleports=len(cohort["teleports"]), collisions=len(cohort["collisions"]))
        records.append(row)
    return records


def select_feedback(directory, failures):
    directory = Path(directory).resolve()
    path = directory / "feedback_selection.json"
    if path.exists():
        raise FileExistsError("Feedback choices are frozen. Use a fresh study directory.")
    manifest = directory / "manifest.json"
    metadata = json.loads(manifest.read_text())
    records = feedback_results(feedback_jobs(directory, "selection"), failures)
    rules = ("access", "access_wait", "access_wait_vehicle")
    choices = {arm: {} for arm in metadata["active_arms"]}
    candidates, timings = [], {}
    for layout in metadata["layouts"]:
        for arm in metadata["active_arms"]:
            variants = []
            for parameters in (metadata["feedback_protocol"]["timings"] if arm == "tuned_fixed" else [None]):
                rows = [r for r in records if r["job"]["layout"] == layout and r["job"]["arm"] == arm
                        and r["job"].get("parameters") == parameters]
                valid = all(r["scores"] is not None for r in rows)
                variants.append({"parameters": parameters, "journey": statistics.mean(r["scores"]["journey"] for r in rows)
                                 if valid else None, "rows": rows})
            eligible = [v for v in variants if v["journey"] is not None]
            best = min(eligible, key=lambda v: v["journey"]) if eligible else None
            if arm == "tuned_fixed":
                timings[layout] = best["parameters"] if best else None
            scores = {}
            for rule in rules:
                scores[rule] = (statistics.mean(r["scores"][rule] for r in best["rows"])
                                if best and all(r["scores"][rule] is not None for r in best["rows"]) else None)
            candidates.append({"layout": layout, "arm": arm, "parameters": best["parameters"] if best else None,
                               "scores": scores, "timing_trials": variants})
    for arm in choices:
        for rule in rules:
            eligible = [c for c in candidates if c["arm"] == arm and c["scores"][rule] is not None]
            choices[arm][rule] = min(eligible, key=lambda c: c["scores"][rule])["layout"] if eligible else None
    selection = {"manifest_sha256": digest(manifest), "choices": choices, "timings": timings, "candidates": candidates,
                 "result_sha256s": {r["result"]: r["sha256"] for r in records if "result" in r},
                 "budget": {"declared_trials": len(records), "returned_trials": sum("result" in r for r in records),
                            "returned_trial_simulation_s": sum(r["simulation_s"] for r in records if "result" in r),
                            "scope": "Count each shared candidate/timing trial once, not once per selection score. "
                                     "Failed execution may consume additional unaccounted simulation; retain its logs."},
                 "complete": all(layout is not None for arm in choices.values() for layout in arm.values())}
    save(path, selection)
    if not selection["complete"]:
        raise RuntimeError("No eligible layout for at least one declared rule; all outcomes are retained.")
    return selection


def summarize_feedback(directory, failures):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "manifest.json").read_text())
    selection_path = directory / "feedback_selection.json"
    selection = json.loads(selection_path.read_text())
    protocol = metadata["feedback_protocol"]
    records = feedback_results(feedback_jobs(directory, "evaluation"), failures)
    lookup = {(r["job"]["arm"], r["job"]["layout"], r["job"]["scale"], r["job"]["seed"]): r for r in records}
    comparisons = []
    for arm, choices in selection["choices"].items():
        for rule in ("access_wait", "access_wait_vehicle"):
            blocks = []
            for seed in protocol["evaluation_seeds"]:
                differences = []
                for scale in protocol["evaluation_scales"]:
                    baseline = lookup[arm, choices["access"], scale, seed]["scores"]
                    feedback = lookup[arm, choices[rule], scale, seed]["scores"]
                    differences.append(baseline["journey"] - feedback["journey"]
                                       if baseline is not None and feedback is not None else None)
                blocks.append({"seed": seed, "differences_by_scale_s": differences,
                               "mean_difference_s": statistics.mean(differences) if all(x is not None for x in differences) else None})
            values = [b["mean_difference_s"] for b in blocks]
            complete = all(v is not None for v in values)
            mean = statistics.mean(values) if complete else None
            interval = None
            if complete and choices["access"] == choices[rule]:
                interval = [0.0, 0.0]
            elif complete and len(values) >= 2:
                half = float(student_t.ppf(.975, len(values) - 1)) * statistics.stdev(values) / len(values) ** .5
                interval = [mean - half, mean + half]
            verdict = "incomplete_service" if not complete else "undetermined"
            if interval is not None:
                margin = protocol["practical_margin_s"]
                if interval[0] > margin:
                    verdict = "feedback_benefit"
                elif interval[1] < -margin:
                    verdict = "feedback_harm"
                elif interval[0] >= -margin and interval[1] <= margin:
                    verdict = "practically_negligible"
            comparisons.append({"arm": arm, "selection_rule": rule, "blocks": blocks,
                                "access_layout": choices["access"], "feedback_layout": choices[rule],
                                "identical_selection": choices["access"] == choices[rule],
                                "mean_difference_s": mean, "conditional_t95_interval_s": interval, "verdict": verdict})
    summary = {"manifest_sha256": digest(directory / "manifest.json"), "selection_sha256": digest(selection_path),
               "criterion": metadata["criterion"], "scope": metadata["scope"], "trials": records,
               "comparisons": comparisons,
               "budget": {"selection": selection["budget"],
                          "evaluation": {"declared_trials": len(records), "returned_trials": sum("result" in r for r in records),
                                         "returned_trial_simulation_s": sum(r["simulation_s"] for r in records if "result" in r)},
                          "scope": "Identical selected layout/controller cases are evaluated once and reused across "
                                   "score comparisons. Returned trials include ineligible outcomes; failed execution "
                                   "may consume additional unaccounted simulation."},
               "uncertainty": "Positive differences favor feedback. Average over the fixed scale grid within each "
                              "evaluation-seed block, then use a Student t interval across independent seed blocks. "
                              "Conditional on this recording and frozen selections, with approximate small-sample "
                              "coverage; not selection-algorithm uncertainty. No incomplete or failed pair is dropped. "
                              "Verdicts are per comparison, not a simultaneous claim across all comparisons. "
                              "Identical selections have zero difference by construction when service is eligible; "
                              "this says nothing about selection stability or information value on other candidate sets."}
    save(directory / "feedback_results.json", summary)
    return summary


def _person_collision_events(log_path):
    """Read person-person overlaps that SUMO warns about but does not register in TraCI."""
    pattern = re.compile(
        r"^Warning: Collision of person '(.*?)' and person '(.*?)', "
        r"lane='(.*?)', time=(\d+(?:\.\d+)?)\.\s*$")
    events = []
    with Path(log_path).open() as log:
        for line in log:
            match = pattern.match(line)
            if match:
                collider, victim, lane, time_s = match.groups()
                events.append(dict(time=float(time_s), collider=collider, victim=victim,
                                   type="person-person", lane=lane))
    return events


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
        self.feedback = False
        self.approach_wait = {}
        self.mechanism_file = None
        self.mechanism_lanes = {}
        self.previous_lane_vehicles = {}

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
                increment = value-previous if value >= previous else value
                if self.active:
                    self.wait[kind] += increment
                self.prev_wait[kind][identifier] = value
                if kind == "pedestrian":
                    self.departures.setdefault(identifier, now)
                    edge = domain.getRoadID(identifier)
                    if edge in self.env.mb_ped_incoming_edges_all:
                        self.first_approach.setdefault(identifier, now-self.departures[identifier])
                        if self.feedback:
                            self.approach_wait[identifier] = self.approach_wait.get(identifier, 0.0) + increment
        self.teleports.extend({"time": now, "id": x} for x in traci.simulation.getStartingTeleportIDList())
        self.collisions.extend(dict(time=now, collider=event.collider, victim=event.victim,
                                    type=event.type, lane=event.lane)
                               for event in traci.simulation.getCollisions())
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
        if self.mechanism_file is not None:
            if not self.mechanism_lanes:
                lanes = sorted({lane for tid in self.env.tl_ids for lane in traci.trafficlight.getControlledLanes(tid)})
                self.mechanism_lanes = {lane: traci.lane.getLength(lane) for lane in lanes}
                self.mechanism_file.write(json.dumps({
                    "type": "geometry",
                    "signals": {tid: traci.junction.getPosition(tid) for tid in self.env.tl_ids},
                    "lanes": {lane: {"length_m": length, "shape": traci.lane.getShape(lane)}
                              for lane, length in self.mechanism_lanes.items()}}) + "\n")
            queues = {}
            for lane, length in self.mechanism_lanes.items():
                vehicles = set(traci.lane.getLastStepVehicleIDs(lane))
                stopped = [v for v in vehicles if traci.vehicle.getSpeed(v) < .1]
                rear = min((traci.vehicle.getLanePosition(v) - traci.vehicle.getLength(v) for v in stopped),
                           default=length)
                previous = self.previous_lane_vehicles.get(lane)
                queues[lane] = {"stopped": len(stopped), "queue_extent_m": max(0., length - rear),
                                "entered": len(vehicles) if previous is None else len(vehicles - previous)}
                self.previous_lane_vehicles[lane] = vehicles
            self.mechanism_file.write(json.dumps({
                "type": "step", "time_s": now,
                "stage": "measurement" if self.active else "drain" if self.controller_active else "warmup",
                "signals": {tid: {"phase": traci.trafficlight.getPhase(tid),
                                  "state": traci.trafficlight.getRedYellowGreenState(tid)} for tid in self.env.tl_ids},
                "lanes": queues}) + "\n")
        return True


def install_controller(env, arm, parameters):
    if arm in ("learned", "learned_initial", "fixed"):
        return
    if arm in ("coordinated_schedule", "local_actuated"):
        if env.signal_control_protocol != SHARED_SIGNAL_PROTOCOL:
            raise ValueError("Catalogue controllers require the shared signal executor")
        return CoordinatedSchedule(env, parameters) if arm == "coordinated_schedule" else LocalActuated(env)
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
    learned_arm = job["arm"] in ("learned", "learned_initial")
    shared_classical = job["arm"] in ("coordinated_schedule", "local_actuated")
    for name, expected in m["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source changed since prepare: {name}. Use a fresh study directory.")
    if m.get("checkpoint") is not None and digest(m["checkpoint"]) != m["checkpoint_sha256"]:
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
    if job["arm"] == "learned_initial":
        artifact = m.get("training_artifact")
        if not artifact or digest(artifact) != m.get("training_artifact_sha256"):
            raise ValueError("Initial-policy diagnostics require a frozen catalogue training record")
        training = json.loads(Path(artifact).read_text())
        initial = next((row for row in training.get("checkpoints", []) if row["round"] == 0), None)
        if (job["split"] != "diagnostic" or m.get("checkpoint_round") != 0 or
                not initial or not initial["diagnostic_only"] or
                initial["sha256"] != m["checkpoint_sha256"] or
                training["layout"]["sha256"] != layout["sha256"]):
            raise ValueError("An initial policy is diagnostic-only, never a trained comparator")
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
    if learned_arm:
        checkpoint = torch.load(m["checkpoint"], map_location="cpu")
        state_stats = checkpoint["lower"]
        if state_stats.get("observation_version", 1) != MLP_ActorCritic.observation_version:
            raise ValueError("Legacy control weights and Welford statistics require fresh training for this observation protocol.")
        saved_protocol = (state_stats.get("provenance") or {}).get("signal_control_protocol")
        if saved_protocol != ctrl.get("signal_control_protocol"):
            raise ValueError("Checkpoint signal-control protocol differs from the requested executor")
        if job["arm"] == "learned_initial":
            from review_training import policy_digest
            provenance = state_stats.get("provenance", {})
            if (any(provenance.get("head_decisions", [1])) or any(provenance.get("head_updates", [1])) or
                    state_stats["state_normalizer_count"] != 0):
                raise ValueError("Initial diagnostic checkpoint must have zero training exposure")
        else:
            require_exposed_heads(state_stats.get("provenance"), layout.get("signal_slots"))
        policy = PPO(**lower).policy
        policy.load_state_dict(state_stats["state_dict"]); policy.eval()
        normalizer = WelfordNormalizer(state_stats["state_normalizer_mean"].shape)
        normalizer.manual_load(torch.from_numpy(state_stats["state_normalizer_mean"]),
                               torch.from_numpy(state_stats["state_normalizer_M2"]), state_stats["state_normalizer_count"])
        normalizer.eval()
        if job["arm"] == "learned_initial" and policy_digest(policy) != training["initial_controller_sha256"]:
            raise ValueError("Initial diagnostic policy does not match its declared initialization")
    if (shared_classical or "catalogue_protocol" in m) and ctrl.get("signal_control_protocol") != SHARED_SIGNAL_PROTOCOL:
        raise ValueError("Catalogue comparison lacks its shared signal-control configuration")
    env = ControlEnv(ctrl, str(folder), worker_id=0, network_iteration=layout["iteration"], current_net_file_path=str(net))
    tracker = Telemetry(env, job["arm"])
    tracker.feedback = "feedback_protocol" in m or "catalogue_protocol" in m
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
    action_trace = logits_hook = None
    probabilities = None
    try:
        if tracker.feedback:
            tracker.mechanism_file = (folder / "mechanism.jsonl").open("w")
        with (folder / "stdout.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            state, _ = env.reset(layout["extreme_edges"], layout["num_proposals"], tl=warmup_control == "fixed",
                                 real_world=layout["real_world"], eval_mode=job["split"] == "evaluation",
                                 signal_slots=layout.get("signal_slots"))
            warmup = traci.simulation.getTime()
            if journey_protocol is not None:
                assert warmup == journey_protocol["warmup_s"]
            # Every arm starts from the same traffic state for a layout/seed/scale.
            initial_state_hash = hashlib.sha256(np.asarray(state).tobytes()).hexdigest()
            controller = install_controller(env, job["arm"], job.get("parameters"))
            if "catalogue_protocol" in m:
                action_trace = (folder / "actions.jsonl").open("w")
                if learned_arm:
                    def capture_probabilities(module, inputs, logits):
                        nonlocal probabilities
                        logits = logits.detach()[0]
                        midblock = logits[4 + torch.as_tensor(env.active_slots, dtype=torch.long)].sigmoid().tolist()
                        probabilities = {env.tl_ids[0]: logits[:4].softmax(dim=0).tolist()}
                        probabilities.update({tid: [1-p, p] for tid, p in zip(env.tl_ids[1:], midblock)})
                    logits_hook = policy.actor_logits.register_forward_hook(capture_probabilities)

            def choose_action(observation):
                if learned_arm:
                    with torch.no_grad():
                        requested, _ = policy.act(normalizer.normalize(torch.as_tensor(observation)),
                                                  layout["num_proposals"], training=False,
                                                  active_slots=env.active_slots)
                    requested = requested.cpu()
                elif controller is not None:
                    requested = controller.act()
                else:
                    requested = np.zeros(1 + layout["num_proposals"], dtype=np.int32)
                if action_trace is not None:
                    action_trace.write(json.dumps(dict(time_s=traci.simulation.getTime(),
                                                       measured=tracker.active,
                                                       requests=dict(zip(env.tl_ids, np.asarray(requested).tolist())),
                                                       probabilities=probabilities), allow_nan=False) + "\n")
                return requested

            tracker.active = True
            tracker.controller_active = True
            measured_wait = {"vehicle": 0.0, "pedestrian": 0.0}
            for _ in range(45):
                action = choose_action(state)
                state, _, done, _, info = env.eval_step(action, tl=not (learned_arm or shared_classical))
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
            result["demand_windows"] = env.demand_windows
            if controller is not None:
                result["controller"] = controller.evidence
            if journey_protocol is not None:
                # Measurement has ended; controller service continues without a reset.
                tracker.active = False
                deadline = end_time + journey_protocol["drain_cap_s"]
                env.max_timesteps += journey_protocol["drain_cap_s"]
                while (traci.simulation.getTime() < deadline and
                       any(set(cohort[kind]) - tracker.completed[kind] for kind in cohort)):
                    action = choose_action(state)
                    state, _, _, _, _ = env.eval_step(action, tl=not (learned_arm or shared_classical))
                simulation_end = traci.simulation.getTime()
                assert simulation_end <= deadline
                assert measured_wait == tracker.wait
                result["elapsed_s"] = time.monotonic()-started
            if env.signal_executor is not None:
                result["signal_service"] = env.signal_executor.stats
    finally:
        traci.start = original_start
        if logits_hook is not None:
            logits_hook.remove()
        if action_trace is not None:
            action_trace.close()
        if env.sumo_running:
            traci.switch(env.traci_label)
            traci.close()  # Wait for SUMO to finish writing complete and unfinished trip records.
            env.sumo_running = False
        if tracker.mechanism_file is not None:
            tracker.mechanism_file.close()
    error_log = folder / f"sumo_errorlog{env.traci_label}.txt"
    person_collisions = _person_collision_events(error_log)
    tracker.collisions.extend(person_collisions)
    tracker.collisions.sort(key=lambda event: event["time"])
    # SUMO warning timestamps mark the start of the step, not its completed boundary.
    result["collisions"].extend(event for event in person_collisions if event["time"] < end_time)
    result["collisions"].sort(key=lambda event: event["time"])
    result["sumo_errorlog_sha256"] = digest(error_log)
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
    if tracker.feedback:
        approaches = env.pedestrian_arrival_times
        result["feedback"] = {
            "access_mean_s": statistics.mean(approaches.values()) if approaches else None,
            "approach_wait_mean_s": sum(tracker.approach_wait.values()) / len(approaches) if approaches else None,
            "approach_n": len(approaches), "approach_ids": sorted(approaches),
            "scope": "Full cohort, including warmup and drain. Access is first recorded mid-block approach age "
                     "since insertion, not a geometric or controller-independent distance; approach wait is "
                     "accumulated stopped time on the declared mid-block pedestrian approach edges per recorded "
                     "approach pedestrian, not all journey waiting. The approach cohort may differ by layout.",
            "mechanism_sha256": digest(folder / "mechanism.jsonl")}
    if action_trace is not None:
        result["action_trace_sha256"] = digest(folder / "actions.jsonl")
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


def catalogue_study(directory):
    """Bind mutable progress to the immutable preparation and its exact inputs."""
    directory = Path(directory)
    study = json.loads((directory / "study.json").read_text())
    prepared_path = directory / "preparation.json"
    if study.get("kind") != "catalogue":
        raise ValueError("Use catalogue-prepare or catalogue-smoke first")
    if digest(prepared_path) != study["preparation_sha256"]:
        raise ValueError("Frozen catalogue preparation changed")
    prepared = json.loads(prepared_path.read_text())
    if any(study.get(key) != value for key, value in prepared.items() if key != "status"):
        raise ValueError("Catalogue inputs differ from the immutable preparation")
    for name, checksum in study["source_hashes"].items():
        if digest(ROOT / name) != checksum or digest(directory / "source_snapshot" / name) != checksum:
            raise ValueError(f"Catalogue source changed: {name}")
    for name in ("protocol", "catalogue"):
        if digest(directory / f"{name}.json") != study[f"{name}_sha256"]:
            raise ValueError(f"Frozen catalogue {name} changed")
    for layout in study["layouts"].values():
        if digest(layout["network"]) != layout["sha256"]:
            raise ValueError("Frozen catalogue network changed")
    return study


def catalogue_jobs(directory, stage):
    """Declared catalogue calibration or development diagnostics, never held-out selection."""
    directory = Path(directory).resolve()
    study = catalogue_study(directory)
    protocol = study["protocol"]
    learning_rate_screen = protocol.get("study_type") == "learning_rate_screen"
    if learning_rate_screen and stage == "calibration":
        raise ValueError("Learning-rate screens do not calibrate classical controllers; use catalogue-diagnose")
    smoke = study["settings"]["smoke"]
    scales = [1.] if smoke else protocol["scenarios"]["scales"]
    seeds = protocol["randomness"]["calibration_seeds" if stage == "calibration" else "diagnostic_seeds"]
    if smoke:
        seeds = seeds[:1]
    manifests = directory / "catalogue_manifests"
    manifests.mkdir(exist_ok=True)
    base = dict(source_hashes=study["source_hashes"], layouts=study["layouts"],
                configuration=study["configuration"], observation_version=MLP_ActorCritic.observation_version,
                active_arms=["coordinated_schedule", "local_actuated"], learned_control_skip_reason=None,
                journey_protocol=JOURNEY_PROTOCOL, warmup_control="fixed",
                preparation_sha256=study["preparation_sha256"],
                catalogue_protocol=protocol, checkpoint=None)

    def manifest(name, value):
        path = manifests / (name + ".json")
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen catalogue manifest changed: {path}")
        if not path.exists():
            save(path, value)
        return str(path)

    jobs = []
    classical_manifest = manifest("classical", base) if not learning_rate_screen else None

    def add(layout, arm, manifest_path, identity, **extra):
        for scale in scales:
            for seed in seeds:
                jobs.append(dict(directory=str(directory / stage / layout / identity / f"{scale}_{seed}"),
                                 manifest=manifest_path, split="selection" if stage == "calibration" else "diagnostic",
                                 layout=layout, arm=arm, seed=seed, scale=scale, **extra))

    if stage == "calibration":
        grid = coordinated_grid(protocol)
        if smoke:
            grid = [grid[0], grid[-1]]
        for layout in study["layouts"]:
            for index, parameters in enumerate(grid):
                add(layout, "coordinated_schedule", classical_manifest, f"setting_{index}",
                    parameters=parameters, parameter_index=index)
    else:
        if study["status"] != "complete":
            raise ValueError("Catalogue diagnostics require every declared training job to complete")
        if not learning_rate_screen:
            selection = json.loads((directory / "catalogue_calibration.json").read_text())
            if selection["preparation_sha256"] != study["preparation_sha256"]:
                raise ValueError("Catalogue calibration belongs to a different frozen study")
        for layout in study["layouts"]:
            if not learning_rate_screen:
                chosen = selection["selected"][layout]
                if chosen is None:
                    raise ValueError(f"No eligible coordinated controller for {layout}; preserve and review calibration failures")
                add(layout, "coordinated_schedule", classical_manifest, "coordinated", parameters=chosen["parameters"])
                add(layout, "local_actuated", classical_manifest, "local")
            for learner_seed in study["seeds"]:
                artifact = directory / str(learner_seed) / layout / "training.json"
                training = json.loads(artifact.read_text())
                if not training["complete"] or training["layout"]["sha256"] != study["layouts"][layout]["sha256"]:
                    raise ValueError("Diagnostics require a complete, layout-matched training record")
                expected_checkpoints = [0, 3] if smoke else [0, 48, 96]
                if learning_rate_screen and sorted(row["round"] for row in training["checkpoints"]) != expected_checkpoints:
                    raise ValueError(f"Learning-rate screen diagnostics require exactly checkpoints {expected_checkpoints}")
                for checkpoint in training["checkpoints"]:
                    round_number = checkpoint["round"]
                    arm = "learned_initial" if round_number == 0 else "learned"
                    value = dict(base, active_arms=[arm], configuration=training["configuration"],
                                 checkpoint=checkpoint["path"], checkpoint_sha256=checkpoint["sha256"],
                                 checkpoint_round=round_number, training_artifact=str(artifact),
                                 training_artifact_sha256=digest(artifact))
                    path = manifest(f"{layout}_{learner_seed}_{round_number}", value)
                    add(layout, arm, path, f"learned_{learner_seed}_{round_number}",
                        learner_seed=learner_seed, checkpoint_round=round_number)
    return jobs


def summarize_catalogue(directory, stage, jobs, failures):
    """Keep the complete matrix, including ineligible cells and paired learner changes."""
    directory = Path(directory).resolve()
    records = []
    groups = {}
    for job in jobs:
        groups.setdefault(job["manifest"], []).append(job)
    for group in groups.values():
        records.extend(feedback_results(group, failures))
    demands, initial_states = {}, {}
    for row in records:
        if "result" not in row:
            continue
        result = json.loads(Path(row["result"]).read_text())
        job = row["job"]
        demand = {kind: entry["demand_sha256"] for kind, entry in result["traffic"].items()}
        block = job["scale"], job["seed"]
        if demands.setdefault(block, demand) != demand:
            raise ValueError("Catalogue demand differs across a paired scenario")
        key = job["layout"], *block
        if initial_states.setdefault(key, result["initial_state_sha256"]) != result["initial_state_sha256"]:
            raise ValueError("Catalogue arms have different warmup observations")
    study = catalogue_study(directory)
    report = dict(stage=stage, preparation_sha256=study["preparation_sha256"],
                  scope="Development only; no layout selection, superiority, equivalence or convergence claim.",
                  records=records, failures=failures, trials=len(jobs),
                  measured_steps=sum(row.get("measurement_s", 0) for row in records),
                  total_simulated_steps=sum(row.get("simulation_s", 0) for row in records))
    if stage == "calibration":
        selected = {}
        for layout in study["layouts"]:
            candidates = {}
            for row in records:
                if row["job"]["layout"] == layout:
                    candidates.setdefault(row["job"]["parameter_index"], []).append(row)
            eligible = []
            for index, rows in candidates.items():
                if all(row["scores"] is not None for row in rows):
                    eligible.append((statistics.mean(row["scores"]["journey"] for row in rows), index, rows[0]["job"]["parameters"]))
            best = min(eligible, key=lambda item: item[:2]) if eligible else None
            selected[layout] = None if best is None else dict(score=best[0], parameter_index=best[1], parameters=best[2])
        report["selected"] = selected
    else:
        pairs = {}
        for row in records:
            job = row["job"]
            if "learner_seed" in job:
                key = (job["layout"], job["learner_seed"], job["scale"], job["seed"])
                pairs.setdefault(key, {})[job["checkpoint_round"]] = None if row["scores"] is None else row["scores"]["journey"]
        report["paired_final_minus_initial"] = [
            dict(layout=key[0], learner_seed=key[1], scale=key[2], seed=key[3], by_round=values,
                 difference_s=(values[max(values)]-values[0]
                               if values.get(0) is not None and values.get(max(values)) is not None else None))
            for key, values in pairs.items()]
    target = directory / f"catalogue_{stage}.json"
    if target.exists() and json.loads(target.read_text()) != report:
        raise ValueError(f"Catalogue {stage} is already frozen; use a new study for changed outcomes")
    save(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "smoke", "tune", "search", "matrix",
                                              "feedback-prepare", "feedback-select", "feedback-evaluate",
                                              "catalogue-calibrate", "catalogue-diagnose"],
                        help="prepare/smoke/tune/search/matrix: existing review comparisons; "
                             "feedback-prepare: freeze a declared placement protocol without checkpoints; "
                             "feedback-select: tune controls and freeze the three layout choices on training data; "
                             "feedback-evaluate: evaluate those frozen choices and report paired seed-block differences")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--training-artifact", type=Path,
                        help="prepare only: completed study/<seed>/<arm>/training.json; reuse its final layout for common-actuated placement baselines")
    parser.add_argument("--protocol", type=Path,
                        help="feedback-prepare only: JSON placements, timings, selection/evaluation scales and seeds, practical_margin_s")
    args = parser.parse_args()
    if args.training_artifact is not None and args.operation != "prepare":
        parser.error("--training-artifact is only valid with prepare")
    if (args.protocol is not None) != (args.operation == "feedback-prepare"):
        parser.error("--protocol is required only with feedback-prepare")
    os.chdir(ROOT)
    directory = args.directory.resolve()
    if args.operation in ("catalogue-calibrate", "catalogue-diagnose"):
        stage = "calibration" if args.operation == "catalogue-calibrate" else "diagnostic"
        jobs = catalogue_jobs(directory, stage)
        failures = run_jobs(jobs, fail_fast=False)
        report = summarize_catalogue(directory, stage, jobs, failures)
        print(json.dumps({key: value for key, value in report.items() if key not in ("records",)}, indent=2))
        return
    if args.operation == "feedback-prepare":
        print(prepare_feedback(directory, args.protocol))
        return
    if args.operation in ("feedback-select", "feedback-evaluate"):
        if args.operation == "feedback-select" and (directory / "feedback_selection.json").exists():
            raise FileExistsError("Feedback choices are frozen. Use a fresh study directory.")
        stage = "selection" if args.operation == "feedback-select" else "evaluation"
        jobs = feedback_jobs(directory, stage)
        failures = run_jobs(jobs, fail_fast=False)
        report = select_feedback(directory, failures) if stage == "selection" else summarize_feedback(directory, failures)
        print(json.dumps(report["choices"] if stage == "selection" else report["comparisons"], indent=2))
        return
    if args.operation == "prepare":
        prepare(directory, args.training_artifact)
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
    if metadata.get("training_artifact"):
        if args.operation == "tune":
            raise ValueError("This matched-placement manifest declares actuated control, not fixed-time tuning.")
        if not (directory / "baselines" / "selection.json").exists():
            raise ValueError("Freeze baseline selection before evaluating the declared held-out demand.")
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
        scales = metadata.get("evaluation_scales", [.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75])
        seeds = metadata.get("evaluation_seeds", list(range(6100, 6110)))
        if args.operation == "smoke":
            scales, seeds = [1.0 if 1.0 in scales else scales[0]], [seeds[0]]
        elif "tuned_fixed" in metadata["active_arms"]:
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
