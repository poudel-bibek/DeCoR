"""Fresh joint/sequential comparison; historical pilots remain unchanged.

Run `smoke DIRECTORY`, then `develop DIRECTORY` to inspect 48 joint rounds
with the longer learning-rate schedule before `train DIRECTORY` and `evaluate DIRECTORY`.
Each learner receives 480 rounds. Sequential training receives 480 design
rounds plus 480 control rounds, with its extra simulation cost recorded.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import time

os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch

from config import classify_and_return_args, get_config
from ppo.ppo import PPO
from ppo.ppo_utils import Memory, WelfordNormalizer
from torch_geometric.data import Batch
from review_validation import ROOT, JOURNEY_PROTOCOL, digest, run_jobs, save
from simulation.design_env import DesignEnv
from utils import save_policy


ARMS = ("joint", "sequential", "fixed_layout", "random_layout")
SOURCES = ("config.py", "main.py", "utils.py", "ppo/ppo.py", "ppo/models.py", "ppo/ppo_utils.py",
           "simulation/design_env.py", "simulation/control_env.py", "simulation/worker.py",
           "simulation/env_utils.py", "simulation/sim_setup.py",
           "review_training.py", "review_validation.py", "uv.lock",
           "simulation/Craver_traffic_lights_wide.net.xml",
           "simulation/original_vehtrips.xml", "simulation/original_pedtrips.xml")


def policy_digest(policy):
    checksum = hashlib.sha256()
    for name, value in policy.state_dict().items():
        checksum.update(name.encode())
        checksum.update(value.detach().cpu().numpy().tobytes())
    return checksum.hexdigest()


def verify_sources(expected):
    for name, checksum in expected.items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f"Study source changed: {name}")


def design_diagnostics(ppo, memory):
    """Measure policy change over the whole rollout without advancing training RNG."""
    was_training = ppo.policy.training
    ppo.policy.eval()
    with torch.random.fork_rng(), torch.no_grad():
        current, _, _ = ppo.policy.evaluate(Batch.from_data_list(memory.states),
                                             torch.stack(memory.actions), device="cpu")
        logratio = current - torch.tensor(memory.logprobs)
        ratio = logratio.exp()
        diagnostics = {"approx_kl": float((ratio - 1 - logratio).mean()),
                       "clip_fraction": float(((ratio - 1).abs() > ppo.eps_clip).float().mean()),
                       "max_abs_logratio": float(logratio.abs().max())}
    ppo.policy.train(was_training)
    return diagnostics


def random_proposals(count, rng):
    """Uniform feasible locations with the learned policy's minimum separation."""
    gap, low, high = .08, .01, .99
    locations = low + np.arange(count) * gap
    locations += np.sort(rng.random(count)) * (high - low - (count - 1) * gap)
    widths = rng.uniform(low, high, count)
    proposals = torch.full((1, 10, 2), -1.0)
    proposals[0, :count] = torch.tensor(np.column_stack((locations, widths)), dtype=torch.float32)
    return proposals, torch.tensor(count)


def layout_record(env, proposals, count):
    return {"network": str(Path(env.current_net_file_path).resolve()),
            "sha256": digest(env.current_net_file_path),
            "iteration": env.current_network_iteration,
            "num_proposals": int(count), "real_world": False,
            "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
            "proposals": proposals[0, :int(count)].tolist()}


def use_layout(env, layout):
    proposals = torch.full((1, 10, 2), -1.0)
    count = torch.tensor(layout["num_proposals"])
    proposals[0, :count] = torch.tensor(layout["proposals"])
    env._apply_action(proposals[0, :count].numpy(), layout["iteration"])
    # Reuse identical signal identities and network bytes in the fixed-layout arm.
    if Path(layout["network"]).resolve() != Path(env.current_net_file_path).resolve():
        shutil.copyfile(layout["network"], env.current_net_file_path)
    assert digest(env.current_net_file_path) == layout["sha256"]
    env.extreme_edge_dict = copy.deepcopy(layout["extreme_edges"])
    return proposals, count


def run_arm(directory, arm, seed, settings, sources):
    started = time.monotonic()
    verify_sources(sources)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rounds = settings["rounds"] * (2 if arm == "sequential" else 1)
    design_rounds = settings["rounds"] if arm in ("joint", "sequential") else 0
    folder = directory / str(seed) / arm
    folder.mkdir(parents=True, exist_ok=False)
    config = get_config()
    config.update(gui=False, gpu=False, evaluate=False, seed=seed,
                  save_graph_images=False, save_gmm_plots=False,
                  lower_num_processes=settings["workers"],
                  total_timesteps=rounds * settings["workers"] * 360)
    if settings["smoke"]:
        config.update(higher_update_freq=2, lower_update_freq=settings["workers"] * 36)
    d, ctrl, higher_args, lower_args, _ = classify_and_return_args(config, "cpu")
    d["save_dir"] = higher_args["model_kwargs"]["run_dir"] = str(folder)
    ctrl.update(global_seed=seed, total_action_timesteps_per_episode=36,
                vehicle_output_trips=str(folder / "vehicles.xml"),
                pedestrian_output_trips=str(folder / "pedestrians.xml"))
    env = DesignEnv(d, ctrl, lower_args, str(folder))
    env.lower_state_normalizer = WelfordNormalizer((10, 123))
    higher = PPO(**higher_args)
    initial_controller = policy_digest(env.lower_ppo.policy)
    initial_design = policy_digest(higher.policy)
    original_state = state = env.reset()
    memory = Memory()
    design_updates = 0
    configuration = dict(design_args=d, control_args=ctrl,
                         higher_ppo_args=higher_args, lower_ppo_args=lower_args)
    record = {"arm": arm, "seed": seed, "configuration": configuration,
              "initial_controller_sha256": initial_controller,
              "initial_design_sha256": initial_design, "source_hashes": sources,
              "rounds": [], "complete": False}
    joint = None
    if arm in ("fixed_layout", "random_layout"):
        joint = json.loads((directory / str(seed) / "joint" / "training.json").read_text())
        assert initial_controller == joint["initial_controller_sha256"]
        proposals, count = use_layout(env, joint["layout"])
    rng = np.random.default_rng(seed + 90210)

    def select_final_layout(iteration):
        higher.policy_old.eval()
        with torch.no_grad():
            _, chosen, n, _ = higher.policy_old.act(
                original_state, iteration, d["clamp_min"], d["clamp_max"], "cpu",
                training=False, visualize=False)
        env._apply_action(chosen[0, :int(n)].numpy(), iteration)
        return chosen, n

    for iteration in range(1, rounds + 1):
        verify_sources(sources)
        optimize_design = arm == "joint" or (arm == "sequential" and iteration <= design_rounds)
        fixed_control = arm == "sequential" and optimize_design
        if arm == "sequential" and iteration == design_rounds + 1:
            # No controller parameters or statistics may change in the design-only stage.
            assert policy_digest(env.lower_ppo.policy) == initial_controller
            assert env.lower_state_normalizer.count.value == env.lower_reward_normalizer.count.value == 0
            assert env.lower_update_count == env.action_timesteps == 0
            proposals, count = select_final_layout(rounds + 1)
        if optimize_design:
            higher.policy_old.eval()
            # Keep design-sampling randomness separate from controller PPO shuffling.
            with torch.random.fork_rng(), torch.no_grad():
                torch.manual_seed(seed + iteration * 7919)
                raw, proposals, count, logprob = higher.policy_old.act(
                    state, iteration, d["clamp_min"], d["clamp_max"], "cpu",
                    training=True, visualize=False)
                value = higher.policy_old.critic(state, device="cpu").item()
        elif arm == "random_layout":
            proposals, count = random_proposals(joint["rounds"][iteration - 1]["crossings"], rng)
        old_updates = env.lower_update_count
        old_steps = env.global_step
        # Match controller-training episode seeds by experience, including sequential stage two.
        rollout_round = iteration - design_rounds if arm == "sequential" and not fixed_control else iteration
        next_state, reward, raw_reward, done, info = env.step(
            proposals, count, rollout_round, fixed_control=fixed_control,
            update_layout=optimize_design or arm == "random_layout")
        assert env.global_step - old_steps == settings["workers"] * 360
        losses = None
        if optimize_design:
            memory.append(state, raw, count, value, logprob, reward, done)
            if iteration % d["higher_update_freq"] == 0:
                design_updates += 1
                if d["higher_anneal_lr"]:
                    higher.update_learning_rate(design_updates, settings["design_horizon_rounds"] // d["higher_update_freq"])
                before = design_diagnostics(higher, memory)
                with torch.no_grad():
                    bootstrap = higher.policy_old.critic(next_state, device="cpu").item()
                losses = {key: float(value) for key, value in higher.update(memory, bootstrap_value=bootstrap).items()}
                losses["rollout_before_update"] = before
                losses["rollout_after_update"] = design_diagnostics(higher, memory)
                memory = Memory()
        state = next_state
        record["rounds"].append({"iteration": iteration, "rollout_round": rollout_round, "fixed_control": fixed_control,
                                "crossings": int(count), "proposals": proposals[0, :int(count)].tolist(),
                                "simulation_steps": env.global_step, "design_reward": float(raw_reward),
                                "design_updates": design_updates, "design_loss": losses,
                                "control_updates": env.lower_update_count,
                                "control_loss": {key: float(value) for key, value in info.items()} if env.lower_update_count > old_updates else None,
                                "elapsed_s": time.monotonic() - started})
        save(folder / "training.json", record)
        print(f"{seed} {arm} round {iteration}/{rounds}: {env.global_step} simulation steps", flush=True)
    assert not memory.states and not env.lower_memories.states
    if arm == "joint":
        proposals, count = select_final_layout(rounds + 1)
    elif arm == "random_layout":
        proposals, count = use_layout(env, joint["layout"])
    assert env.global_step == settings["workers"] * rounds * 360
    checkpoint = folder / "final.pth"
    save_policy(higher.policy, env.lower_ppo.policy, env.lower_state_normalizer,
                env.normalizer_x, env.normalizer_y, str(checkpoint))
    control_rounds = sum(not row["fixed_control"] for row in record["rounds"])
    assert control_rounds == settings["rounds"]
    assert env.lower_state_normalizer.count.value == control_rounds * settings["workers"] * 36
    record.update(complete=True, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                  budget={"design_rounds": design_rounds, "control_rounds": control_rounds,
                          "simulation_steps": env.global_step,
                          "control_decisions": control_rounds * settings["workers"] * 36},
                  layout=layout_record(env, proposals, count),
                  controller_sha256=policy_digest(env.lower_ppo.policy),
                  design_sha256=policy_digest(higher.policy),
                  state_normalizer_count=env.lower_state_normalizer.count.value,
                  elapsed_s=time.monotonic() - started)
    save(folder / "training.json", record)
    env.close()
    return record


def run_seed(job):
    directory, seed, settings, sources = job
    os.chdir(ROOT)
    if settings["development"]:
        run_arm(directory, "joint", seed, settings, sources)
        return seed
    records = [run_arm(directory, arm, seed, settings, sources) for arm in ARMS]
    assert len({r["initial_controller_sha256"] for r in records}) == 1
    assert len({r["initial_design_sha256"] for r in records}) == 1
    assert len({r["budget"]["control_decisions"] for r in records}) == 1
    assert records[0]["budget"]["design_rounds"] == records[1]["budget"]["design_rounds"]
    assert records[0]["rounds"][-1]["design_updates"] == records[1]["rounds"][-1]["design_updates"]
    assert len({r["rounds"][-1]["control_updates"] for r in records}) == 1
    assert records[1]["budget"]["simulation_steps"] == 2 * records[0]["budget"]["simulation_steps"]
    schedules = [[row["rollout_round"] for row in r["rounds"] if not row["fixed_control"]] for r in records]
    assert all(schedule == schedules[0] for schedule in schedules)
    assert records[2]["layout"]["sha256"] == records[3]["layout"]["sha256"] == records[0]["layout"]["sha256"]
    return seed


def evaluate(directory):
    study = json.loads((directory / "study.json").read_text())
    verify_sources(study["source_hashes"])
    if study["settings"]["development"]:
        raise ValueError("Development runs are inspected using training diagnostics, not held-out evaluation.")
    jobs = []
    for seed in study["seeds"]:
        for arm in ARMS:
            folder = directory / str(seed) / arm
            training = json.loads((folder / "training.json").read_text())
            assert training["complete"]
            layout = training["layout"]
            assert digest(layout["network"]) == layout["sha256"]
            manifest = {"checkpoint": training["checkpoint"],
                        "checkpoint_sha256": training["checkpoint_sha256"],
                        "observation_version": 2, "active_arms": ["learned"],
                        "learned_control_skip_reason": None, "source_hashes": study["source_hashes"],
                        "configuration": training["configuration"], "warmup_control": "random",
                        "journey_protocol": study["journey_protocol"],
                        "layouts": {arm: layout}}
            manifest_path = folder / "evaluation_manifest.json"
            save(manifest_path, manifest)
            scales = study["evaluation_scales"]
            test_seeds = study["evaluation_seeds"]
            for scale in scales:
                for test_seed in test_seeds:
                    jobs.append(dict(manifest=str(manifest_path), layout=arm, arm="learned",
                                     scale=scale, seed=test_seed, split="evaluation",
                                     directory=str(folder / "evaluation" / f"{scale}_{test_seed}")))
    run_jobs(jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["smoke", "develop", "train", "evaluate"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    os.chdir(ROOT)
    directory = args.directory.resolve()
    if args.operation == "evaluate":
        evaluate(directory)
        return
    smoke = args.operation == "smoke"
    development = args.operation == "develop"
    settings = dict(smoke=smoke, development=development,
                    rounds=4 if smoke else 48 if development else 480,
                    design_horizon_rounds=4 if smoke else 480,
                    workers=2 if smoke else 10)
    seeds = [14900] if development else [14100] if smoke else [14100, 14200, 14300]
    directory.mkdir(parents=True, exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    save(directory / "study.json", {"settings": settings, "seeds": seeds, "source_hashes": sources,
                                    "journey_protocol": JOURNEY_PROTOCOL,
                                    "evaluation_scales": [] if development else [1.] if smoke else [.5, 1., 1.75, 2.75],
                                    "evaluation_seeds": [] if development else [19100] if smoke else list(range(19100, 19105)),
                                    "budget": "Measured simulation steps; warmup and evaluation excluded. Sequential costs twice as much as joint. Fixed-layout control excludes the shared cost of obtaining the joint layout.",
                                    "comparison": "Training-only stability check; no held-out comparisons." if development else "Matched design and controller experience; sequential receives a full budget for each stage. Random layouts match joint crossing counts, not its location/width distribution.",
                                    "evaluation": "Final policies selected by the declared training endpoint, never held-out performance. Common 100 s random-action warmup, 450 s measurement, then at most 1800 s to finish the fixed demand cohort; report every run and all remaining trips."})
    for name in SOURCES:
        target = directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    with ProcessPoolExecutor(max_workers=len(seeds), mp_context=mp.get_context("spawn")) as pool:
        for seed in pool.map(run_seed, [(directory, seed, settings, sources) for seed in seeds]):
            print(f"Completed {'development' if development else 'all four training arms'} for seed {seed}", flush=True)


if __name__ == "__main__":
    main()
