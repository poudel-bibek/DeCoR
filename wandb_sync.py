#!/usr/bin/env python3
"""Compact, immutable per-learner dashboards without the scientific environment.

Run with: uv run --no-project --with wandb==0.30.0 python wandb_sync.py STUDY --once
Add --mode online only for an authorized private upload; --follow polls until Ctrl-C.
Local JSON/checkpoints remain authoritative. No checkpoint or source files are uploaded.

Schema v2 uses one run per (study, layout, learner seed), never per diagnostic trial.
Only METRIC_AXES metrics and sync/event identity/hash bookkeeping are transmitted.
Controller rewards are the means for the round that triggered an actual update,
not the entire multi-round PPO buffer. Evaluation is the existing greedy diagnostic
policy; training remains sampled. No action selection or simulation is performed.

A checkpoint is emitted only when every declared scale/seed cell has a result or
a published failure. Primary journey cost is null unless every cell is eligible;
completion fractions pool native completed/scheduled counts, not per-cell fractions.
Missing failed-cell counters make pooled fractions and incident totals null, not zero.
The effective smoke grid follows catalogue_jobs. Classical/calibration trials stay local.

The fsynced ledger under .wandb_sync/v2/ledgers is logically append-only: an atomic
snapshot adds new event IDs without rewriting their order, payload or source hash.
Late evaluation can therefore follow later training without changing logged prefixes.
Offline acceptance means a finished, fsynced W&B segment under .wandb_sync/v2/offline.
Segments have disjoint explicit steps; interrupted staging directories are uncommitted.
To publish later, rerun this CLI online against the original JSON, not `wandb sync`
over the segment/staging tree. Online and offline cursors are deliberately independent.
Online acceptance includes finish() and server history readback. A restart reconciles
accepted steps before logging, including a crash between remote acceptance and cursor save.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time


SDK_VERSION = "0.30.0"
SCHEMA_VERSION = 2
PPO_FIELDS = {
    "policy_loss": "lower_policy_loss",
    "value_loss": "lower_value_loss",
    "entropy": "lower_entropy_loss",
    "learning_rate": "lower_current_lr",
    "approx_kl": "lower_approx_kl",
    "exact_kl": "lower_exact_kl",
    "clip_fraction": "lower_clip_fraction",
}
OPTIONAL_PPO_FIELDS = ("actor_grad_norm", "critic_grad_norm", "actor_update_norm", "entropy_fraction")
METRIC_AXES = {
    "training/round": (
        "training/measured_steps", "training/controller_reward_unnormalized",
        "training/controller_reward_normalized", "training/status", "training/complete", "training/failed",
    ),
    "ppo/update": tuple("ppo/" + field for field in PPO_FIELDS) + (
        "ppo/gradient_norm_after_clipping_max",
    ) + tuple("ppo/" + field for field in OPTIONAL_PPO_FIELDS),
    "eval/checkpoint_round": (
        "eval/eligible_blocks", "eval/total_blocks", "eval/failed_blocks",
        "eval/teleports", "eval/collisions", "eval/pedestrian_completion_fraction",
        "eval/vehicle_completion_fraction", "eval/primary_journey_mean_s",
    ),
}
IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, value):
    with path.with_suffix(".tmp").open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.with_suffix(".tmp").replace(path)
    sync_directory(path.parent)


def identifier(value):
    value = str(value)
    if not IDENTIFIER.fullmatch(value) or value in (".", ".."):
        raise ValueError("Study, seed, layout and controller identities must be simple names")
    return value


def metric(value):
    """Keep explicitly unavailable/non-finite observations null, never zero."""
    if value is None or value in ("nan", "inf", "-inf"):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    raise ValueError("Expected a finite numeric metric or an explicit non-finite marker")


def hashes(value):
    """Transmit hash values only; never transmit or follow source/checkpoint paths."""
    if isinstance(value, dict):
        return sorted({item for child in value.values() for item in hashes(child)})
    if isinstance(value, list):
        return sorted({item for child in value for item in hashes(child)})
    return [value] if isinstance(value, str) and HASH.fullmatch(value) else []


def selected(record, fields):
    return {key: record[key] for key in fields if key in record}


def event(event_id, source, metrics):
    return {"sync/event_id": f"v{SCHEMA_VERSION}:{event_id}",
            "sync/source_sha256": digest(source), **metrics}


def by_round(entries):
    result = {entry["round"]: entry for entry in entries}
    if len(result) != len(entries):
        raise ValueError("Duplicate checkpoint or optimizer-gate round")
    return result


def training_events(record):
    if record is None:
        return []
    checkpoints = by_round(record.get("checkpoints", []))
    gates = by_round(record.get("gates", []))
    initial = selected(record, ("initial_controller_sha256", "initial_design_sha256"))
    events = [event("initial", [initial, checkpoints.get(0)], {
        "training/round": 0, "training/measured_steps": 0,
        "training/status": "running", "training/complete": False, "training/failed": False,
    })]
    previous_update = 0
    for row in record["rounds"]:
        iteration = row["iteration"]
        gate = gates.get(iteration, {})
        values = {"training/round": iteration, "training/measured_steps": row["simulation_steps"]}
        if row.get("status") == "failed":
            values["training/failed"] = True
        update = row.get("control_updates", previous_update)
        loss = row.get("control_loss")
        if update > previous_update:
            if loss is not None:
                values.update({
                    "training/controller_reward_unnormalized": metric(loss["lower_avg_reward_unnorm"]),
                    "training/controller_reward_normalized": metric(loss["lower_avg_reward_norm"]),
                })
            if loss is not None or gate:
                values["ppo/update"] = update
                for name, field in PPO_FIELDS.items():
                    if loss is not None and field in loss:
                        values["ppo/" + name] = metric(loss[field])
                    else:
                        source_field = field.removeprefix("lower_")
                        diagnostics = gate.get("after", {})
                        losses = gate.get("losses", {})
                        if source_field in losses:
                            values["ppo/" + name] = metric(losses[source_field])
                        elif source_field in diagnostics:
                            values["ppo/" + name] = metric(diagnostics[source_field])
                for name in OPTIONAL_PPO_FIELDS:
                    if name in gate.get("losses", {}):
                        values["ppo/" + name] = metric(gate["losses"][name])
                norms = [metric(value) for value in gate.get("gradient_norms_after_clipping", [])]
                if norms:
                    values["ppo/gradient_norm_after_clipping_max"] = (
                        max(norms) if all(value is not None for value in norms) else None)
        previous_update = update
        events.append(event(f"training/{iteration}", [row, gate, checkpoints.get(iteration)], values))
    return events


def evaluation_events(directory, study, record, identity, report):
    if record is None:
        return []
    checkpoints = by_round(record.get("checkpoints", []))
    protocol = study["protocol"]
    scales = [1.] if study["settings"].get("smoke") else protocol["scenarios"]["scales"]
    seeds = protocol["randomness"]["diagnostic_seeds"]
    if study["settings"].get("smoke"):
        seeds = seeds[:1]
    expected = {(scale, seed) for scale in scales for seed in seeds}
    if not expected or len(expected) != len(scales) * len(seeds):
        raise ValueError("Empty or duplicate declared diagnostic scale/seed matrix")
    groups = {}

    def key(job):
        if (job["split"] != "diagnostic" or identifier(job["layout"]) != identity["layout"]
                or identifier(job["learner_seed"]) != identity["seed"]):
            raise ValueError("Diagnostic result belongs to a different learner or split")
        checkpoint = job["checkpoint_round"]
        if checkpoint not in checkpoints or job["arm"] != ("learned_initial" if checkpoint == 0 else "learned"):
            raise ValueError("Diagnostic result is not a declared learner checkpoint")
        block = job["scale"], job["seed"]
        if block not in expected:
            raise ValueError("Unexpected diagnostic scale/seed cell")
        return checkpoint, block

    base = directory / "diagnostic" / identity["layout"]
    for folder in sorted(base.glob(f"learned_{identity['seed']}_*")):
        for path in sorted(folder.rglob("result.json")):
            path.resolve().relative_to(directory)
            payload = path.read_bytes()
            result = json.loads(payload)
            checkpoint, block = key(result["job"])
            cells = groups.setdefault(checkpoint, {})
            if block in cells:
                raise ValueError("Duplicate diagnostic result cell")
            cells[block] = dict(result=result, sha256=hashlib.sha256(payload).hexdigest(), failure=None)

    declared = set()
    for row in report.get("records", []):
        job = row["job"]
        if ("learner_seed" not in job or identifier(job["layout"]) != identity["layout"]
                or identifier(job["learner_seed"]) != identity["seed"]):
            continue
        checkpoint, block = key(job)
        if (checkpoint, block) in declared:
            raise ValueError("Duplicate diagnostic summary cell")
        declared.add((checkpoint, block))
        cells = groups.setdefault(checkpoint, {})
        if block in cells and cells[block]["result"] is not None and job != cells[block]["result"]["job"]:
            raise ValueError("Diagnostic summary job provenance differs from its result")
        if "error" in row:
            cell = cells.setdefault(block, dict(result=None, sha256=None, failure=None))
            cell["failure"] = row
        elif block not in cells or row["sha256"] != cells[block]["sha256"]:
            raise ValueError("Diagnostic summary result is missing or its exact hash differs")
        else:
            cells[block]["reported_scores"] = row["scores"]

    events = []
    for checkpoint, cells in sorted(groups.items()):
        if set(cells) != expected:
            continue
        ordered = [cells[block] for block in sorted(expected)]
        eligible = []
        cohorts = []
        for cell in ordered:
            cohort = cell["result"]["journeys"].get("cohort", {}) if cell["result"] is not None else {}
            cohorts.append(cohort)
            traffic = cohort.get("traffic", {})
            pedestrian = traffic.get("pedestrian", {})
            vehicle = traffic.get("vehicle", {})
            parts = (pedestrian.get("journey_mean_s"), vehicle.get("time_loss_plus_insertion_delay_mean_s"))
            valid = (cell["failure"] is None and pedestrian.get("all_completed", False)
                     and vehicle.get("all_completed", False) and "teleports" in cohort and "collisions" in cohort
                     and not cohort["teleports"]
                     and all(event.get("type") == "person-person" for event in cohort["collisions"])
                     and all(value is not None and metric(value) is not None for value in parts))
            # Match feedback_scores eligibility without importing the scientific stack.
            if valid:
                feedback = cell["result"].get("feedback", {})
                access, wait = feedback.get("access_mean_s"), feedback.get("approach_wait_mean_s")
                scores = [sum(parts), access]
                if access is not None and wait is not None:
                    if metric(access) is None or metric(wait) is None:
                        valid = False
                    else:
                        scores.extend((access + wait, access + wait + parts[1]))
                valid = valid and all(value is None or metric(value) is not None for value in scores)
            cost = sum(parts) if valid else None
            if "reported_scores" in cell:
                reported = cell["reported_scores"]
                if (None if reported is None else reported["journey"]) != cost:
                    raise ValueError("Diagnostic summary eligibility/primary cost differs from its result")
            eligible.append(cost)
        values = {
            "eval/checkpoint_round": checkpoint,
            "eval/eligible_blocks": sum(value is not None for value in eligible),
            "eval/total_blocks": len(expected),
            "eval/failed_blocks": sum(cell["failure"] is not None for cell in ordered),
            "eval/primary_journey_mean_s": (sum(eligible) / len(expected)
                                          if all(value is not None for value in eligible) else None),
        }
        for name in ("teleports", "collisions"):
            values["eval/" + name] = (sum(len(cohort[name]) for cohort in cohorts)
                                     if all(name in cohort for cohort in cohorts) else None)
        for kind in ("pedestrian", "vehicle"):
            counts = [cohort.get("traffic", {}).get(kind, {}) for cohort in cohorts]
            complete = all("scheduled" in count and "completed" in count for count in counts)
            scheduled = sum(count["scheduled"] for count in counts) if complete else 0
            values[f"eval/{kind}_completion_fraction"] = (
                sum(count["completed"] for count in counts) / scheduled if scheduled else None)
        source = [checkpoints[checkpoint], [[cell["sha256"], cell["failure"]] for cell in ordered]]
        events.append(event(f"eval/{checkpoint}", source, values))
    return events


def records(directory, study):
    summary = directory / "catalogue_diagnostic.json"
    summary.resolve().relative_to(directory)
    report = read(summary) if summary.exists() else {}
    if report and (report["stage"] != "diagnostic"
                   or report["preparation_sha256"] != study["preparation_sha256"]):
        raise ValueError("Diagnostic summary belongs to a different frozen study")
    for seed in study["seeds"]:
        for layout in study["layouts"]:
            identity = dict(seed=identifier(seed), layout=identifier(layout), controller="learned")
            path = directory / identity["seed"] / identity["layout"] / "training.json"
            path.resolve().relative_to(directory)
            record = read(path) if path.exists() else None
            if record is not None and (record["kind"] != "catalogue"
                                       or identifier(record["seed"]) != identity["seed"]
                                       or identifier(record["layout_id"]) != identity["layout"]):
                raise ValueError("Training record belongs to a different learner")
            events = training_events(record)
            status_path = path.with_name("result.json")
            status_path.resolve().relative_to(directory)
            status = read(status_path) if status_path.exists() else None
            if status is None and record is not None and record.get("status") == "stopped":
                status = selected(record, ("status", "seed", "layout_id", "error", "traceback"))
            if status is not None:
                if (identifier(status["seed"]) != identity["seed"]
                        or identifier(status["layout_id"]) != identity["layout"]
                        or status["status"] not in ("complete", "failed", "stopped")):
                    raise ValueError("Terminal status belongs to a different learner or is not terminal")
                last = record["rounds"][-1] if record is not None and record["rounds"] else {}
                values = {"training/round": last.get("iteration", 0),
                          "training/measured_steps": last.get("simulation_steps"),
                          "training/status": status["status"], "training/complete": status["status"] == "complete",
                          "training/failed": status["status"] == "failed"}
                events.append(event("terminal", [status, values], values))
            events.extend(evaluation_events(directory, study, record, identity, report))
            yield path, record, identity, events


def learner_run_id(directory, identity):
    return digest([SCHEMA_VERSION, directory.name, identity["layout"], identity["seed"]])[:24]


def append_events(path, run_id, contract, candidates):
    """Atomically persist an append-only logical stream before any remote submission."""
    current = {item["sync/event_id"]: item for item in candidates}
    if len(current) != len(candidates):
        raise ValueError("Duplicate candidate event ID")
    ledger = read(path) if path.exists() else dict(
        schema_version=SCHEMA_VERSION, run_id=run_id, contract=contract, events=[])
    if (ledger["schema_version"] != SCHEMA_VERSION or ledger["run_id"] != run_id
            or ledger["contract"] != contract):
        raise ValueError("Frozen learner ledger contract changed")
    events = ledger["events"]
    if len({item["sync/event_id"] for item in events}) != len(events):
        raise ValueError("Duplicate saved ledger event ID")
    for item in events:
        if current.pop(item["sync/event_id"], None) != item:
            raise ValueError(f"Previously observed source event changed or disappeared: {item['sync/event_id']}")
    if current:
        events.extend(current.values())
        path.parent.mkdir(parents=True, exist_ok=True)
        sync_directory(path.parent.parent)
        save(path, ledger)
    return events


def private_project(api, entity, project):
    """0.30's public create_project omits access; use its shipped GraphQL schema.

    Api._service_api is a version-specific authenticated SDK bridge, not a public
    stability promise. No credential is inspected. Never alter an existing project's
    access. Verify the server's explicit PRIVATE answer before initializing any run.
    """
    from wandb.apis._generated import CREATE_PROJECT_GQL, UpsertModelInput

    query = """query SidecarProject($entity: String!, $name: String!) {
      project(entityName: $entity, name: $name) { name access }
    }"""
    variables = {"entity": entity, "name": project}
    result = api._service_api.execute_graphql(query, variables)["project"]
    if result is None:
        values = UpsertModelInput(name=project, entity_name=entity, access="PRIVATE")
        api._service_api.execute_graphql(CREATE_PROJECT_GQL, {"input": values.model_dump(by_alias=True)})
        result = api._service_api.execute_graphql(query, variables)["project"]
    if not result or result.get("access") != "PRIVATE":
        raise RuntimeError(f"Refusing upload: {entity}/{project} is not verified PRIVATE")


def verify_history(api, destination, events, start, end):
    if start == end:
        return
    api.flush()
    history = api.run(destination).scan_history(keys=["_step", "sync/event_sha256"],
                                                min_step=start, max_step=end, use_cache=False)
    received = {int(row["_step"]): row["sync/event_sha256"] for row in history}
    expected = {step: digest(events[step]) for step in range(start, end)}
    if received != expected:
        raise RuntimeError("Remote history is missing or differs from local records; cursor retained, retry sync")


def sync_record(wandb, api, args, directory, study, state, path, record, identity, events):
    if not path.resolve().is_relative_to(directory):
        raise ValueError("Refusing a record symlink outside the study")
    relative = path.relative_to(directory).as_posix()
    run_id = learner_run_id(directory, identity)
    record_configuration = record.get("configuration") if record is not None else study.get("configuration")
    frozen_study = selected(study, (
        "kind", "settings", "configuration", "protocol", "layouts", "seeds", "source_hashes", "preparation_sha256"))
    contract = digest([SCHEMA_VERSION, frozen_study, record_configuration, identity])
    ledger = directory / ".wandb_sync" / f"v{SCHEMA_VERSION}" / "ledgers" / f"{run_id}.json"
    events = append_events(ledger, run_id, contract, events)
    if not events:
        return
    root = state / run_id
    root.mkdir(exist_ok=True)
    cursor_file = root / "cursor.json"
    cursor = read(cursor_file) if cursor_file.exists() else {
        "schema_version": SCHEMA_VERSION, "next": 0, "prefix": digest([]), "contract": contract}
    start = cursor["next"]
    if (cursor["schema_version"] != SCHEMA_VERSION or cursor["contract"] != contract
            or start > len(events) or cursor["prefix"] != digest(events[:start])):
        raise ValueError(f"Previously logged records changed for {relative}; preserve the original study")
    if args.mode == "offline":
        # Recover the directory commit if interruption preceded cursor.json replacement.
        while (root / f"{start:08d}" / "receipt.json").exists():
            receipt = read(root / f"{start:08d}" / "receipt.json")
            end = receipt["next"]
            if (receipt["schema_version"] != SCHEMA_VERSION or receipt["run_id"] != run_id
                    or receipt["contract"] != contract or end <= start or end > len(events)
                    or receipt["prefix"] != digest(events[:end])):
                raise ValueError("Offline segment differs from the authoritative records")
            save(cursor_file, receipt)
            start = end
    if start == len(events):
        return
    configuration = {
        "study": directory.name, **identity, "kind": "catalogue", "metric_schema_version": SCHEMA_VERSION,
        "study_type": study["protocol"].get("study_type", "catalogue_pilot"),
        "learning_rate": (record_configuration or {}).get("lower_ppo_args", {}).get("lr"),
        "training_action_selection": "sampled", "evaluation_action_selection": "greedy",
        "evaluation_split": "diagnostic", "evaluation_scope": "development only; not held-out selection",
        "reward_semantics": "controller reward mean for update-triggering round; not full-cohort journey cost",
        "policy_loss_semantics": "persisted positive PPO surrogate; minimized objective negates it",
        "study_contract_sha256": digest(frozen_study), "configuration_sha256": digest(record_configuration),
        "protocol_sha256": digest(study["protocol"]), "source_hashes": hashes(study.get("source_hashes", {})),
        "network_hashes": hashes(study["layouts"][identity["layout"]]),
    }
    if api is not None:
        private_project(api, args.entity, args.project)
    staging = Path(tempfile.mkdtemp(prefix="pending-", dir=root))
    run = wandb.init(entity=args.entity, project=args.project, group=directory.name,
                     id=run_id, name=f"{identity['layout']}/learner_{identity['seed']}", job_type="learner",
                     config=configuration, dir=str(staging), mode=args.mode,
                     resume="allow" if api is not None else None)
    if run is None or run.offline != (args.mode == "offline"):
        raise RuntimeError("W&B did not initialize in the requested mode; cursor retained")
    try:
        for axis, metrics in METRIC_AXES.items():
            run.define_metric(axis)
            for name in metrics:
                run.define_metric(name, step_metric=axis, step_sync=False)
        destination = f"{args.entity}/{args.project}/{run_id}"
        next_step = start
        if api is not None:
            # The init receipt carries the resumed offset; 0.30's live step query can still return zero.
            next_step = run.starting_step
            if not start <= next_step <= len(events):
                raise RuntimeError("Remote step disagrees with the durable cursor; cursor retained")
            verify_history(api, destination, events, start, next_step)
        for step in range(next_step, len(events)):
            run.log({**events[step], "sync/event_sha256": digest(events[step])}, step=step, commit=True)
        run.finish()
        if api is not None:
            verify_history(api, destination, events, start, len(events))
    except BaseException:
        # Finish may itself fail. Never turn that into cursor advancement.
        try:
            run.finish(exit_code=1)
        except Exception:
            pass
        raise
    receipt = {"schema_version": SCHEMA_VERSION, "next": len(events), "prefix": digest(events),
               "run_id": run_id, "contract": contract}
    if args.mode == "offline":
        files = list(staging.rglob("*.wandb"))
        if not files:
            raise RuntimeError("W&B produced no offline log; cursor retained")
        for file in files:
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
            for parent in file.parents:
                sync_directory(parent)
                if parent == staging:
                    break
        save(staging / "receipt.json", receipt)
        staging.rename(root / f"{start:08d}")
        sync_directory(root)
    save(cursor_file, receipt)
    print(f"Accepted {args.mode} {relative}: events {start}..{len(events) - 1}, run {run_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--follow", action="store_true")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--entity", default="bibek-poudel")
    parser.add_argument("--project", default="decor-design-control-pilot")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive and finite")
    directory = args.study.resolve()
    identifier(directory.name)
    identifier(args.entity)
    identifier(args.project)
    study = read(directory / "study.json")
    if study.get("kind") != "catalogue":
        parser.error("Only kind=catalogue studies are supported; historical workflows stay separate")
    state_root = directory / ".wandb_sync"
    state_root.mkdir(exist_ok=True)
    sync_directory(directory)
    with (state_root / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another sidecar owns this study; stop it before resuming") from None
        try:
            import wandb
        except ImportError:
            raise RuntimeError("Use a separate environment: uv run --no-project --with wandb==0.30.0 python wandb_sync.py ...") from None
        if wandb.__version__ != SDK_VERSION:
            raise RuntimeError(f"This sidecar requires wandb=={SDK_VERSION}, found {wandb.__version__}; use --no-project --with wandb==0.30.0")
        wandb.setup(settings=wandb.Settings(
            mode=args.mode, console="off", disable_code=True, disable_git=True, save_code=False,
            x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
            login_timeout=1, init_timeout=60, finish_timeout=60, finish_timeout_raises=True,
            program="wandb_sync.py", silent=False, quiet=False, symlink=False))
        state = state_root / f"v{SCHEMA_VERSION}" / args.mode / digest([args.entity, args.project])[:16]
        state.mkdir(parents=True, exist_ok=True)
        sync_directory(state_root)
        api = None
        while True:
            try:
                study = read(directory / "study.json")
                if study.get("kind") != "catalogue":
                    raise ValueError("The frozen catalogue study kind changed")
                if args.mode == "online" and api is None:
                    api = wandb.Api(timeout=30)
                    private_project(api, args.entity, args.project)
                failures = 0
                for path, record, identity, events in records(directory, study):
                    try:
                        sync_record(wandb, api, args, directory, study, state, path, record, identity, events)
                    except Exception as error:
                        failures += 1
                        print(f"SYNC ERROR {path}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                if args.once:
                    return 1 if failures else 0
            except Exception as error:
                print(f"SYNC ERROR: {type(error).__name__}: {error}; local records untouched", file=sys.stderr, flush=True)
                if args.once:
                    return 1
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Sidecar stopped; rerun the same command to reconcile and resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"SYNC ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
