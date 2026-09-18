"""Fresh joint/sequential comparison; historical pilots remain unchanged.

Run `smoke DIRECTORY`, then `develop DIRECTORY` to inspect 48 joint rounds
with the longer learning-rate schedule before `train DIRECTORY` and `evaluate DIRECTORY`.
Each learner receives 480 rounds. Sequential training receives 480 design
rounds plus 480 control rounds, with its extra simulation cost recorded.
Catalogue operations use --catalogue layouts.json and --protocol pilot_protocol.json:
catalogue-prepare freezes inputs only, catalogue-smoke runs two layouts for three
rounds, and catalogue-train runs the separately authorized fixed-layout protocol.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch

from config import classify_and_return_args, get_config
from ppo.ppo import PPO
from ppo.ppo_utils import Memory, WelfordNormalizer
from torch_geometric.data import Batch
from review_validation import ROOT, JOURNEY_PROTOCOL, digest, random_proposals, run_jobs, save
from simulation.design_env import DesignEnv
from utils import SLOT_PROTOCOL, load_policy, require_exposed_heads, save_policy


ARMS = ("joint", "sequential", "fixed_layout", "random_layout")
SOURCES = ("config.py", "main.py", "utils.py", "ppo/ppo.py", "ppo/models.py", "ppo/ppo_utils.py",
           "simulation/design_env.py", "simulation/control_env.py", "simulation/worker.py",
           "simulation/env_utils.py", "simulation/sim_setup.py",
           "review_training.py", "review_validation.py", "uv.lock",
           "simulation/Craver_traffic_lights_wide.net.xml",
           "simulation/original_vehtrips.xml", "simulation/original_pedtrips.xml")
CATALOGUE_SOURCES = SOURCES + ("simulation/signal_control.py", "pyproject.toml")


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


def layout_record(env, proposals, count):
    return {"network": str(Path(env.current_net_file_path).resolve()),
            "sha256": digest(env.current_net_file_path),
            "iteration": env.current_network_iteration,
            "num_proposals": int(count), "real_world": False,
            "extreme_edges": copy.deepcopy(env.extreme_edge_dict),
            "proposals": proposals[0, :int(count)].tolist(),
            "crossing_ids": list(env.crossing_ids), "signal_slots": dict(env.signal_slots)}


def use_layout(env, layout):
    proposals = torch.full((1, 10, 2), -1.0)
    count = torch.tensor(layout["num_proposals"])
    proposals[0, :count] = torch.tensor(layout["proposals"])
    # Reuse identical signal identities, slots and network bytes in the fixed-layout arm.
    env._apply_action(proposals[0, :count].numpy(), layout["iteration"],
                      crossing_ids=layout["crossing_ids"], signal_slots=layout["signal_slots"])
    assert env.crossing_ids == layout["crossing_ids"] and env.signal_slots == layout["signal_slots"]
    if Path(layout["network"]).resolve() != Path(env.current_net_file_path).resolve():
        shutil.copyfile(layout["network"], env.current_net_file_path)
    assert digest(env.current_net_file_path) == layout["sha256"]
    env.extreme_edge_dict = copy.deepcopy(layout["extreme_edges"])
    return proposals, count


class CatalogueGateFailure(RuntimeError):
    pass


def catalogue_progress(env, record, started):
    record.update(budget={"design_rounds": 0,
                          "control_rounds": env.global_step // (env.control_args["lower_num_processes"] *
                                                                 env.control_args["max_timesteps"]),
                          "saved_successful_rounds": sum(row.get("status") != "failed" for row in record["rounds"]),
                          "simulation_steps": env.global_step,
                          "executed_simulation_steps": env.executed_simulation_steps,
                          "executed_total_steps": env.executed_total_steps,
                          "uncollected_simulation_steps": env.executed_simulation_steps - env.global_step,
                          "execution_accounting_complete": env.execution_accounting_complete,
                          "control_decisions": env.global_step // env.control_args["lower_action_duration"],
                          "control_update_attempts": env.lower_update_count,
                          "control_updates_completed": sum(g["status"] == "passed" for g in record["gates"]),
                          "optimizer_steps": sum(g["optimizer_steps"] for g in record["gates"])},
                  control_head_decisions=env.control_head_decisions.tolist(),
                  control_head_updates=env.control_head_updates.tolist(),
                  rollout_execution=env.rollout_execution,
                  state_normalizer_count=env.lower_state_normalizer.count.value,
                  reward_normalizer_count=env.lower_reward_normalizer.count.value,
                  elapsed_s=time.monotonic() - started)


def catalogue_checkpoint(env, higher, record, folder, iteration, final_round):
    path = folder / ("final.pth" if iteration == final_round else f"round_{iteration:03d}.pth")
    save_policy(higher.policy, env.lower_ppo.policy, env.lower_state_normalizer,
                env.normalizer_x, env.normalizer_y, str(path),
                env.control_head_decisions, env.control_head_updates, signal_control_protocol="shared_v1")
    record["checkpoints"].append(dict(round=iteration, path=str(path), sha256=digest(path),
                                      controller_sha256=policy_digest(env.lower_ppo.policy),
                                      diagnostic_only=iteration != final_round))


def install_catalogue_gates(env, record, folder, protocol):
    """Gate this learner's existing diagnostic calls and optimizer steps, without global patches."""
    ppo = env.lower_ppo
    original_update, original_diagnostics = ppo.update, ppo._control_diagnostics
    limits = protocol["diagnostics"]["numerical_gates"]
    training = protocol["training"]

    def require(condition, message):
        if not condition:
            raise CatalogueGateFailure(message)

    def metrics_record(values):
        # JSON cannot carry NaN/Infinity; retain their literal spellings on failed gates.
        return {key: float(value) if math.isfinite(float(value)) else str(float(value))
                for key, value in values.items()}

    def snapshot(memory, path, error=None):
        payload = dict(seed=record["seed"], layout_id=record["layout_id"],
                       round=record["current_round"], gates=record["gates"], error=error,
                       memory=vars(memory), policy=ppo.policy.state_dict(),
                       policy_old=ppo.policy_old.state_dict(), optimizer=ppo.optimizer.state_dict(),
                       state_normalizer=dict(mean=env.lower_state_normalizer.mean,
                                             M2=env.lower_state_normalizer.M2,
                                             count=env.lower_state_normalizer.count.value),
                       reward_normalizer=dict(mean=env.lower_reward_normalizer.mean,
                                              M2=env.lower_reward_normalizer.M2,
                                              count=env.lower_reward_normalizer.count.value),
                       control_head_decisions=env.control_head_decisions.tolist(),
                       control_head_updates=env.control_head_updates.tolist(),
                       simulation_steps=env.global_step,
                       rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                                python=random.getstate()))
        temporary = path.with_suffix(".pth.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def diagnostics(*args, **kwargs):
        values = original_diagnostics(*args, **kwargs)
        gate = record["gates"][-1]
        before = "before" not in gate
        gate["before" if before else "after"] = metrics_record(values)
        require(all(math.isfinite(float(value)) for value in values.values()),
                "Non-finite full-rollout controller diagnostics.")
        if before:
            require(float(values["max_abs_logratio"]) <= limits["maximum_preupdate_abs_logratio"],
                    "Pre-update likelihood error exceeds the protocol gate.")
        else:
            for metric, threshold in (("approx_kl", "maximum_postupdate_sampled_kl"),
                                      ("exact_kl", "maximum_postupdate_exact_kl"),
                                      ("clip_fraction", "maximum_clip_fraction")):
                # Sampled KL deliberately retains the declared one-sided upper bound.
                require(float(values[metric]) <= limits[threshold], f"Post-update {metric} exceeds the protocol gate.")
        return values

    def update(memory, *args, **kwargs):
        gate = dict(round=record["current_round"], update=env.lower_update_count,
                    status="started", rollout_decisions=len(memory.states),
                    simulation_steps=env.global_step, optimizer_steps=0,
                    gradient_norms_after_clipping=[])
        record["gates"].append(gate)
        active = (torch.stack(memory.actions)[:, 1:] >= 0).any(dim=0).cpu().numpy()
        gate["active_heads_in_rollout"] = np.flatnonzero(active).tolist()

        def before_step(optimizer, positional, named):
            parameters = list(ppo.policy.parameters())
            gradients = [p.grad for p in parameters if p.grad is not None]
            require(all(torch.isfinite(p).all().item() for p in parameters), "Non-finite pre-step parameters.")
            require(bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients),
                    "Non-finite or absent optimizer gradients.")
            norm = float(torch.sqrt(sum(g.double().square().sum() for g in gradients)))
            gate["gradient_norms_after_clipping"].append(norm if math.isfinite(norm) else str(norm))
            require(math.isfinite(norm) and norm <= limits["maximum_gradient_norm_after_clipping"],
                    "Post-clipping gradient norm exceeds the protocol gate.")

        def after_step(optimizer, positional, named):
            gate["optimizer_steps"] += 1
            require(all(torch.isfinite(p).all().item() for p in ppo.policy.parameters()),
                    "Non-finite post-step parameters.")

        try:
            snapshot(memory, folder / "latest_preupdate_lower.pth")
            require(len(memory.states) == training["transitions_per_update"], "Unexpected controller rollout size.")
            for field in ("values", "logprobs", "rewards"):
                require(torch.isfinite(torch.as_tensor(getattr(memory, field))).all().item(),
                        f"Non-finite rollout {field}.")
            require(all(torch.isfinite(state).all().item() for state in memory.states), "Non-finite rollout states.")
            require(all(torch.isfinite(p).all().item() for p in ppo.policy.parameters()),
                    "Non-finite pre-update parameters.")
            pre = ppo.optimizer.register_step_pre_hook(before_step)
            post = ppo.optimizer.register_step_post_hook(after_step)
            try:
                result = original_update(memory, *args, **kwargs)
            finally:
                pre.remove()
                post.remove()
            gate["losses"] = metrics_record(result)
            require(all(math.isfinite(float(value)) for value in result.values()), "Non-finite optimizer losses.")
            require("before" in gate and "after" in gate, "Missing full-rollout diagnostic calls.")
            require(gate["optimizer_steps"] == training["optimizer_steps_per_update"], "Unexpected optimizer-step count.")
            gate.update(status="passed", all_finite=True)
            return result
        except BaseException as error:
            gate.update(status="failed", error=f"{type(error).__name__}: {error}")
            if gate["optimizer_steps"]:
                # DesignEnv increments this only after update returns; retain failed-update exposure too.
                env.control_head_updates += active
            snapshot(memory, folder / "failed_update_lower.pth", gate["error"])
            raise

    ppo._control_diagnostics = diagnostics
    ppo.update = update


def run_arm(directory, arm, seed, settings, sources, layout=None, protocol=None):
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
    config = copy.deepcopy(settings["config"]) if layout is not None else get_config()
    config.update(gui=False, gpu=False, evaluate=False, seed=seed,
                  save_graph_images=False, save_gmm_plots=False,
                  lower_num_processes=settings["workers"],
                  total_timesteps=rounds * settings["workers"] * 360)
    if settings["smoke"] and layout is None:
        config.update(higher_update_freq=2, lower_update_freq=settings["workers"] * 36)
    d, ctrl, higher_args, lower_args, _ = classify_and_return_args(config, "cpu")
    d["save_dir"] = higher_args["model_kwargs"]["run_dir"] = str(folder)
    ctrl.update(global_seed=seed, total_action_timesteps_per_episode=36,
                vehicle_output_trips=str(folder / "vehicles.xml"),
                pedestrian_output_trips=str(folder / "pedestrians.xml"))
    if layout is not None:
        ctrl["signal_control_protocol"] = "shared_v1"
    env = DesignEnv(d, ctrl, lower_args, str(folder))
    env.lower_state_normalizer = WelfordNormalizer((10, 123))
    higher = PPO(**higher_args)
    initial_controller = policy_digest(env.lower_ppo.policy)
    initial_design = policy_digest(higher.policy)
    # One-shot design: every proposal, value and final extraction uses the original-crossing context.
    context = env.reset() if layout is None else None
    memory = Memory()
    design_updates = 0
    configuration = dict(design_args=d, control_args=ctrl,
                         higher_ppo_args=higher_args, lower_ppo_args=lower_args)
    record = {"arm": arm, "seed": seed, "configuration": configuration,
              "controller_interface": {"observation_version": env.lower_ppo.policy.observation_version,
                                       "slot_protocol": SLOT_PROTOCOL, "permutation_augmentation": False},
              "initial_controller_sha256": initial_controller,
              "initial_design_sha256": initial_design, "source_hashes": sources,
              "rounds": [], "complete": False}
    joint = None
    if layout is not None:
        proposals, count = use_layout(env, layout)
        record.update(kind="catalogue", layout_id=arm, layout=copy.deepcopy(layout),
                      checkpoints=[], gates=[], status="running",
                      smoke=settings["smoke"], protocol=protocol)
        record["controller_interface"]["signal_control_protocol"] = "shared_v1"
        record["configuration"]["config"] = config
        install_catalogue_gates(env, record, folder, protocol)
        catalogue_checkpoint(env, higher, record, folder, 0, rounds)
        catalogue_progress(env, record, started)
        save(folder / "training.json", record)
        expected_initial = settings.get("expected_initial_controller_sha256", initial_controller)
        if initial_controller != expected_initial:
            raise CatalogueGateFailure("Initial controller parameters differ across layouts for this seed.")
    elif arm in ("fixed_layout", "random_layout"):
        joint = json.loads((directory / str(seed) / "joint" / "training.json").read_text())
        assert initial_controller == joint["initial_controller_sha256"]
        proposals, count = use_layout(env, joint["layout"])
    rng = np.random.default_rng(seed + 90210)

    def select_final_layout(iteration):
        higher.policy_old.eval()
        with torch.no_grad():
            _, chosen, n, _ = higher.policy_old.act(
                context, iteration, d["clamp_min"], d["clamp_max"], "cpu",
                training=False, visualize=False)
        env._apply_action(chosen[0, :int(n)].numpy(), iteration)
        return chosen, n

    for iteration in range(1, rounds + 1):
        rollout_round = iteration - design_rounds if arm == "sequential" and iteration > design_rounds else iteration
        try:
            verify_sources(sources)
            if layout is not None:
                record["current_round"] = iteration
                if digest(env.current_net_file_path) != layout["sha256"]:
                    raise ValueError("Sealed catalogue network changed during training.")
            optimize_design = layout is None and (arm == "joint" or (arm == "sequential" and iteration <= design_rounds))
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
                        context, iteration, d["clamp_min"], d["clamp_max"], "cpu",
                        training=True, visualize=False)
                    value = higher.policy_old.critic(context, device="cpu").item()
            elif arm == "random_layout":
                proposals, count = random_proposals(joint["rounds"][iteration - 1]["crossings"], rng)
            old_updates = env.lower_update_count
            old_steps = env.global_step
            _, reward, raw_reward, done, info = env.step(
                proposals, count, rollout_round, fixed_control=fixed_control,
                update_layout=optimize_design or (layout is None and arm == "random_layout"))
            assert env.global_step - old_steps == settings["workers"] * 360
            losses = None
            if optimize_design:
                memory.append(context, raw, count, value, logprob, reward, done)
                if iteration % d["higher_update_freq"] == 0:
                    design_updates += 1
                    if d["higher_anneal_lr"]:
                        higher.update_learning_rate(design_updates, settings["design_horizon_rounds"] // d["higher_update_freq"])
                    before = design_diagnostics(higher, memory)
                    losses = {key: float(value) for key, value in higher.update(memory).items()}
                    losses["rollout_before_update"] = before
                    losses["rollout_after_update"] = design_diagnostics(higher, memory)
                    memory = Memory()
            # Finish checkpoints before publishing this round to append-only consumers.
            if layout is not None and (iteration == rounds or iteration in protocol["training"]["diagnostic_rounds"]):
                catalogue_checkpoint(env, higher, record, folder, iteration, rounds)
            row = {"iteration": iteration, "rollout_round": rollout_round, "fixed_control": fixed_control,
                   "crossings": int(count), "proposals": proposals[0, :int(count)].tolist(),
                   "signal_slots": dict(env.signal_slots),
                   "simulation_steps": env.global_step, "design_reward": float(raw_reward),
                   "design_updates": design_updates, "design_loss": losses,
                   "control_updates": env.lower_update_count,
                   "control_loss": {key: float(value) for key, value in info.items()} if env.lower_update_count > old_updates else None,
                   "control_head_decisions": env.control_head_decisions.tolist(),
                   "control_head_updates": env.control_head_updates.tolist(),
                   "elapsed_s": time.monotonic() - started}
            if layout is not None:
                row.update(worker_seeds=[seed + rollout_round * 1000 + rank for rank in range(settings["workers"])],
                           executed_simulation_steps=env.executed_simulation_steps,
                           uncollected_simulation_steps=env.executed_simulation_steps - env.global_step,
                           execution_accounting_complete=env.execution_accounting_complete)
            record["rounds"].append(row)
            if layout is not None:
                catalogue_progress(env, record, started)
            save(folder / "training.json", record)
        except BaseException as error:
            if layout is not None:
                record.update(status="failed", error=f"{type(error).__name__}: {error}",
                              traceback=traceback.format_exc())
                # The current round has not been published unless save itself failed.
                if record["rounds"] and record["rounds"][-1]["iteration"] == iteration:
                    record["rounds"].pop()
                record["rounds"].append(dict(iteration=iteration, rollout_round=rollout_round,
                                            status="failed", simulation_steps=env.global_step,
                                            executed_simulation_steps=env.executed_simulation_steps,
                                            uncollected_simulation_steps=env.executed_simulation_steps - env.global_step,
                                            execution_accounting_complete=env.execution_accounting_complete,
                                            worker_seeds=[seed + rollout_round * 1000 + rank
                                                          for rank in range(settings["workers"])],
                                            control_updates=env.lower_update_count,
                                            control_head_decisions=env.control_head_decisions.tolist(),
                                            control_head_updates=env.control_head_updates.tolist()))
                catalogue_progress(env, record, started)
                save(folder / "training.json", record)
                env.close()
            raise
        print(f"{seed} {arm} round {iteration}/{rounds}: {env.global_step} collected simulation steps", flush=True)
    assert not memory.states and not env.lower_memories.states
    if arm == "joint":
        proposals, count = select_final_layout(rounds + 1)
    elif arm == "random_layout":
        proposals, count = use_layout(env, joint["layout"])
    assert env.global_step == settings["workers"] * rounds * 360
    checkpoint = folder / "final.pth"
    if layout is None:
        save_policy(higher.policy, env.lower_ppo.policy, env.lower_state_normalizer,
                    env.normalizer_x, env.normalizer_y, str(checkpoint),
                    env.control_head_decisions, env.control_head_updates)
    control_rounds = sum(not row["fixed_control"] for row in record["rounds"])
    assert control_rounds == settings["rounds"]
    assert env.lower_state_normalizer.count.value == control_rounds * settings["workers"] * 36
    record.update(complete=True, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                  budget={"design_rounds": design_rounds, "control_rounds": control_rounds,
                          "simulation_steps": env.global_step,
                          "control_decisions": control_rounds * settings["workers"] * 36},
                  layout=layout_record(env, proposals, count),
                  control_head_decisions=env.control_head_decisions.tolist(),
                  control_head_updates=env.control_head_updates.tolist(),
                  controller_sha256=policy_digest(env.lower_ppo.policy),
                  design_sha256=policy_digest(higher.policy),
                  state_normalizer_count=env.lower_state_normalizer.count.value,
                  elapsed_s=time.monotonic() - started)
    if layout is not None:
        assert policy_digest(higher.policy) == initial_design and design_updates == 0
        assert env.lower_update_count == rounds // 3
        assert all(gate["status"] == "passed" for gate in record["gates"])
        assert env.execution_accounting_complete and env.executed_simulation_steps == env.global_step
        if settings["smoke"]:
            restored_higher, restored_lower = copy.deepcopy(higher.policy), copy.deepcopy(env.lower_ppo.policy)
            with torch.no_grad():
                for parameter in list(restored_higher.parameters()) + list(restored_lower.parameters()):
                    parameter.zero_()
            restored_normalizer = WelfordNormalizer((10, 123))
            norm_x, norm_y, provenance = load_policy(restored_higher, restored_lower, restored_normalizer, checkpoint,
                                                    signal_control_protocol="shared_v1")
            assert policy_digest(restored_lower) == record["controller_sha256"]
            assert policy_digest(restored_higher) == initial_design
            assert norm_x == env.normalizer_x and norm_y == env.normalizer_y
            assert torch.equal(restored_normalizer.mean, env.lower_state_normalizer.mean)
            assert torch.equal(restored_normalizer.M2, env.lower_state_normalizer.M2)
            assert restored_normalizer.count.value == env.lower_state_normalizer.count.value
            require_exposed_heads(provenance, env.signal_slots)
            record["checkpoint_roundtrip_verified"] = True
        record["layout"] = dict(layout, network=str(Path(env.current_net_file_path).resolve()))
        record["status"] = "complete"
        catalogue_progress(env, record, started)
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
    assert records[2]["layout"]["signal_slots"] == records[3]["layout"]["signal_slots"] == records[0]["layout"]["signal_slots"]
    return seed


def evaluate(directory):
    study = json.loads((directory / "study.json").read_text())
    verify_sources(study["source_hashes"])
    if study.get("kind") == "catalogue":
        raise ValueError("Catalogue diagnostics use the separate per-layout evaluation protocol.")
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
                        "observation_version": training["controller_interface"]["observation_version"],
                        "active_arms": ["learned"],
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


def prepare_catalogue(directory, catalogue_path, protocol_path, smoke=False):
    """Freeze only: no environment, policy, simulator, normalizer or optimizer construction."""
    catalogue = json.loads(catalogue_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    training = protocol["training"]
    overrides = training["configuration_overrides"]
    study_type = protocol.get("study_type")
    if study_type not in (None, "learning_rate_screen"):
        raise ValueError(f"Unknown catalogue study_type: {study_type}")
    screen = study_type == "learning_rate_screen"
    expected_overrides = dict(lower_lr=1e-4, lower_batch_size=256, lower_K_epochs=2,
                              lower_anneal_lr=False, demand_scale_min=.5, demand_scale_max=2.)
    if screen:
        # Declared development comparison: one learner seed, the two validated layouts, one screened rate.
        if (len(training["learner_seeds"]) != 1 or overrides.get("lower_lr") not in (1e-4, 3e-4, 1e-3)
                or protocol["catalogue"]["candidate_ids"] != ["two_spread", "six_central"]):
            raise ValueError("Learning-rate screen requires one learner seed, two_spread/six_central and a rate in {1e-4, 3e-4, 1e-3}.")
        expected_overrides["lower_lr"] = overrides["lower_lr"]
    if (training["rounds_per_layout_seed"] != 96 or training["workers"] != 10
            or len(training["learner_seeds"]) != (1 if screen else 2) or training["episode_measured_s"] != 360
            or training["decision_s"] != 10 or training["design_updates"] != 0
            or training["layout_updates"] or training["transitions_per_update"] != 1080
            or training["optimizer_steps_per_update"] != 10 or overrides != expected_overrides
            or training["diagnostic_rounds"] != [0, 48, 96] or training["endpoint_round"] != 96):
        raise ValueError("Catalogue training requires the recorded 96-round, ten-worker protocol.")
    layouts = catalogue["layouts"]
    if list(layouts) != protocol["catalogue"]["candidate_ids"]:
        raise ValueError("Catalogue candidate order differs from the protocol.")
    if catalogue["family_slots"] != protocol["catalogue"]["family_slots"]:
        raise ValueError("Catalogue and protocol signal families differ.")
    for name, layout in layouts.items():
        if name in ARMS or Path(name).name != name:
            raise ValueError(f"Invalid catalogue layout ID: {name}")
        if digest(layout["network"]) != layout["sha256"]:
            raise ValueError(f"Sealed catalogue network changed: {name}")
        expected_slots = {f"{crossing}_mid": catalogue["family_slots"][f"{crossing}_mid"]
                          for crossing in layout["crossing_ids"]}
        if (layout["signal_slots"] != expected_slots or layout["family_slots"] != catalogue["family_slots"]
                or layout["num_proposals"] != len(layout["proposals"])
                or layout["num_proposals"] != len(layout["crossing_ids"])):
            raise ValueError(f"Invalid sealed catalogue identity/slot record: {name}")
    if smoke:
        ordered = sorted(layouts, key=lambda name: layouts[name]["num_proposals"])
        layouts = {name: layouts[name] for name in (ordered[0], ordered[-1])}
    rounds = 3 if smoke else training["rounds_per_layout_seed"]
    seeds = training["learner_seeds"][:1] if smoke else training["learner_seeds"]
    config = get_config()
    config.update(overrides)
    config.update(gui=False, gpu=False, evaluate=False, save_graph_images=False, save_gmm_plots=False,
                  lower_num_processes=10, lower_update_freq=1080, lower_max_timesteps=360,
                  lower_action_duration=10, lower_step_length=1., lower_warmup_steps=[40, 140],
                  signal_control_protocol="shared_v1", total_timesteps=rounds * 10 * 360)
    sources = {name: digest(ROOT / name) for name in CATALOGUE_SOURCES}
    directory.mkdir(parents=True, exist_ok=False)
    for name in CATALOGUE_SOURCES:
        target = directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if digest(target) != sources[name]:
            raise ValueError(f"Source changed while freezing: {name}")
    for key in ("original_net_file", "vehicle_input_trips", "pedestrian_input_trips"):
        config[key] = str(directory / "source_snapshot" / Path(config[key]))
    d, ctrl, higher_args, lower_args, _ = classify_and_return_args(config, "cpu")
    ctrl["signal_control_protocol"] = config["signal_control_protocol"]
    configuration = dict(design_args=d, control_args=ctrl,
                         higher_ppo_args=higher_args, lower_ppo_args=lower_args)
    frozen_layouts = copy.deepcopy(layouts)
    for name, layout in frozen_layouts.items():
        target = directory / "networks" / f"{name}.net.xml"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(layout["network"], target)
        if digest(target) != layout["sha256"]:
            raise ValueError(f"Network changed while freezing: {name}")
        layout["prepared_network"] = layout["network"]
        layout["network"] = str(target)
    shutil.copyfile(catalogue_path, directory / "catalogue.json")
    shutil.copyfile(protocol_path, directory / "protocol.json")
    study = dict(kind="catalogue", status="prepared", settings=dict(rounds=rounds, workers=10, smoke=smoke, config=config),
                 configuration=configuration, protocol=protocol, layouts=frozen_layouts,
                 seeds=seeds, source_hashes=sources,
                 catalogue_sha256=digest(directory / "catalogue.json"),
                 protocol_sha256=digest(directory / "protocol.json"),
                 comparison="Bounded engineering smoke; never a pilot result." if smoke else
                            f"Development learning-rate screen at lower_lr={overrides['lower_lr']}; one learner seed, two validated layouts, not a pilot or paper result." if screen else
                            "Fresh per-layout controller adaptation; no joint design optimization or layout selection.")
    save(directory / "preparation.json", study)
    study["preparation_sha256"] = digest(directory / "preparation.json")
    save(directory / "study.json", study)
    verify_sources(sources)
    return study


def verify_catalogue_preparation(directory, study):
    path = directory / "preparation.json"
    if digest(path) != study["preparation_sha256"]:
        raise ValueError("Frozen catalogue preparation changed.")
    preparation = json.loads(path.read_text())
    if any(study.get(key) != value for key, value in preparation.items() if key != "status"):
        raise ValueError("Study differs from its frozen preparation.")
    verify_sources(study["source_hashes"])
    if (digest(directory / "protocol.json") != study["protocol_sha256"]
            or digest(directory / "catalogue.json") != study["catalogue_sha256"]
            or json.loads((directory / "protocol.json").read_text()) != study["protocol"]):
        raise ValueError("Frozen catalogue/protocol provenance changed.")
    for name, checksum in study["source_hashes"].items():
        if digest(directory / "source_snapshot" / name) != checksum:
            raise ValueError(f"Frozen study source changed: {name}")
    for layout in study["layouts"].values():
        if digest(layout["network"]) != layout["sha256"]:
            raise ValueError("Frozen catalogue network changed.")


def run_catalogue_job(job):
    directory, seed, layout_id, study = job
    os.chdir(ROOT)
    folder = directory / str(seed) / layout_id
    try:
        verify_catalogue_preparation(directory, study)
        settings = dict(study["settings"])
        if seed in study.get("initial_controller_sha256", {}):
            settings["expected_initial_controller_sha256"] = study["initial_controller_sha256"][seed]
        record = run_arm(directory, layout_id, seed, settings, study["source_hashes"],
                         study["layouts"][layout_id], study["protocol"])
        save(folder / "result.json", dict(status="complete", seed=seed, layout_id=layout_id,
                                         training=str(folder / "training.json")))
        return record["initial_controller_sha256"]
    except BaseException as error:
        result = dict(status="failed", seed=seed, layout_id=layout_id,
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        # Also retain failures before the environment or initial training record exists.
        save(folder / "result.json", result)
        path = folder / "training.json"
        if path.exists():
            record = json.loads(path.read_text())
            record.update(complete=False, status="failed", error=result["error"], traceback=result["traceback"])
            save(path, record)
        raise


def run_catalogue(directory, study):
    if study["status"] != "prepared":
        raise ValueError("Catalogue execution requires an unused prepared study; failed jobs are never retried.")
    verify_catalogue_preparation(directory, study)
    study["status"] = "running"
    save(directory / "study.json", study)
    initial = {}
    study["initial_controller_sha256"] = initial
    try:
        # Sequential jobs stop at the first failure; each job still owns ten parallel rollout workers.
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as pool:
            for seed in study["seeds"]:
                for layout_id in study["layouts"]:
                    checksum = pool.submit(run_catalogue_job, (directory, seed, layout_id, study)).result()
                    if seed in initial and initial[seed] != checksum:
                        raise ValueError(f"Initial controller parameters differ across layouts for seed {seed}.")
                    initial[seed] = checksum
        study.update(status="complete", initial_controller_sha256=initial)
    except BaseException as error:
        study.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        save(directory / "study.json", study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["smoke", "develop", "train", "evaluate",
                                               "catalogue-prepare", "catalogue-smoke", "catalogue-train"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    os.chdir(ROOT)
    directory = args.directory.resolve()
    if args.operation.startswith("catalogue-"):
        if args.catalogue is None or args.protocol is None:
            parser.error("Catalogue operations require --catalogue and --protocol.")
        if args.operation == "catalogue-train" and directory.exists():
            study = json.loads((directory / "study.json").read_text())
            if (study.get("kind") != "catalogue" or study["settings"]["smoke"]
                    or digest(args.catalogue) != study["catalogue_sha256"]
                    or digest(args.protocol) != study["protocol_sha256"]):
                raise ValueError("Prepared study differs from the requested catalogue/protocol.")
        else:
            study = prepare_catalogue(directory, args.catalogue.resolve(), args.protocol.resolve(),
                                      smoke=args.operation == "catalogue-smoke")
        if args.operation != "catalogue-prepare":
            run_catalogue(directory, study)
        return
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
