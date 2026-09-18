import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import wandb_sync as sync


class MemoryRemote:
    """A resumable append-only history boundary; no SDK, network or scientific imports."""

    def __init__(self):
        self.histories = {}
        self.fail_finish = False

    def init(self, **options):
        history = self.histories.setdefault(options["id"], [])
        remote = self

        class Run:
            offline = False
            starting_step = len(history)

            def define_metric(self, *args, **kwargs):
                pass

            def log(self, values, step, commit):
                if step != len(history):
                    raise AssertionError("History overwrite or duplicate submission")
                history.append({"_step": step, **copy.deepcopy(values)})

            def finish(self, exit_code=0):
                if remote.fail_finish:
                    remote.fail_finish = False
                    raise RuntimeError("Remote accepted history before the connection failed")

        return Run()

    def flush(self):
        pass

    def run(self, destination):
        history = self.histories[destination.rsplit("/", 1)[1]]

        def scan_history(keys, min_step, max_step, use_cache):
            return [{key: row[key] for key in keys} for row in history
                    if min_step <= row["_step"] < max_step]

        return SimpleNamespace(scan_history=scan_history)


class CompactLoggingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "synthetic_study"
        self.directory.mkdir()
        self.study = {
            "kind": "catalogue", "settings": {"smoke": False}, "configuration": {"frozen": True},
            "protocol": {"scenarios": {"scales": [1, 2]}, "randomness": {"diagnostic_seeds": [11, 12]}},
            "layouts": {"layout": {"sha256": "a" * 64}}, "seeds": [7],
            "source_hashes": {"source.py": "b" * 64}, "preparation_sha256": "c" * 64,
        }
        self.identity = dict(seed="7", layout="layout", controller="learned")
        self.record = {
            "kind": "catalogue", "seed": 7, "layout_id": "layout", "configuration": {"frozen": True},
            "rounds": [], "gates": [], "checkpoints": [self.checkpoint(0)],
            "initial_controller_sha256": "d" * 64, "initial_design_sha256": "e" * 64,
            "status": "running", "complete": False,
        }
        self.training_path = self.directory / "7" / "layout" / "training.json"
        self.write(self.directory / "study.json", self.study)
        self.publish_training()
        self.state = self.directory / ".wandb_sync" / "v2" / "online" / "destination"
        self.state.mkdir(parents=True)
        self.remote = MemoryRemote()
        self.args = SimpleNamespace(mode="online", entity="private_owner", project="existing_project")

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        return path

    def checkpoint(self, number):
        return {"round": number, "path": f"unread_checkpoint_{number}.pth",
                "sha256": hashlib.sha256(f"checkpoint-{number}".encode()).hexdigest()}

    def publish_training(self):
        self.write(self.training_path, self.record)

    def add_round(self, number, checkpoint=False, update=0, loss=None):
        row = dict(iteration=number, simulation_steps=number * 100, control_updates=update,
                   control_loss=loss, design_reward=12345, worker_seeds=[number + 1000])
        self.record["rounds"].append(row)
        if checkpoint:
            self.record["checkpoints"].append(self.checkpoint(number))
        self.publish_training()
        return row

    def job(self, checkpoint, scale, seed):
        folder = self.directory / "diagnostic" / "layout" / f"learned_7_{checkpoint}" / f"{scale}_{seed}"
        return dict(directory=str(folder), manifest="unread_manifest.json", split="diagnostic", layout="layout",
                    arm="learned_initial" if checkpoint == 0 else "learned", learner_seed=7,
                    checkpoint_round=checkpoint, scale=scale, seed=seed)

    def diagnostic(self, checkpoint, scale, seed, pedestrian=(10, 10), vehicle=(20, 20), cost=30):
        job = self.job(checkpoint, scale, seed)
        traffic = {}
        for kind, (scheduled, completed) in (("pedestrian", pedestrian), ("vehicle", vehicle)):
            traffic[kind] = dict(scheduled=scheduled, completed=completed,
                                 all_completed=scheduled == completed, completion_fraction=completed / scheduled)
        traffic["pedestrian"]["journey_mean_s"] = cost - 10
        traffic["vehicle"]["time_loss_plus_insertion_delay_mean_s"] = 10
        result = dict(job=job, journeys={"cohort": dict(traffic=traffic, teleports=[], collisions=[])},
                      feedback=dict(access_mean_s=2, approach_wait_mean_s=3),
                      unlogged_debug="private local detail")
        path = self.write(Path(job["directory"]) / "result.json", result)
        return path, result

    def complete_checkpoint(self, number):
        return [self.diagnostic(number, scale, seed, cost=scale * 10 + seed)
                for scale in (1, 2) for seed in (11, 12)]

    def summary(self, results=(), failures=()):
        rows = []
        for path, result in results:
            cohort = result["journeys"]["cohort"]
            eligible = (not cohort["teleports"]
                        and all(event.get("type") == "person-person" for event in cohort["collisions"])
                        and all(item["all_completed"] for item in cohort["traffic"].values()))
            traffic = cohort["traffic"]
            cost = traffic["pedestrian"]["journey_mean_s"] + traffic["vehicle"]["time_loss_plus_insertion_delay_mean_s"]
            rows.append(dict(job=result["job"], scores={"journey": cost} if eligible else None,
                             result=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rows.extend(failures)
        return dict(stage="diagnostic", preparation_sha256=self.study["preparation_sha256"], records=rows)

    def evaluations(self, report=None):
        return sync.evaluation_events(self.directory, self.study, self.record, self.identity, report or {})

    def synchronize(self):
        with patch.object(sync, "private_project"):
            for path, record, identity, events in sync.records(self.directory, self.study):
                sync.sync_record(self.remote, self.remote, self.args, self.directory, self.study,
                                 self.state, path, record, identity, events)
        return self.remote.histories[sync.learner_run_id(self.directory, self.identity)]

    def test_rewards_and_ppo_only_on_actual_updates(self):
        self.add_round(1)
        loss = dict(lower_avg_reward_unnorm=-9, lower_avg_reward_norm=-0.4,
                    lower_policy_loss=0.2, lower_value_loss=3, lower_entropy_loss=0.8,
                    lower_current_lr=0.001, lower_approx_kl=0.01, lower_exact_kl=0.02,
                    lower_clip_fraction=0.1, ignored_numeric_detail=999)
        self.add_round(2, update=1, loss=loss)
        self.add_round(3, update=1, loss=loss)  # A carried-forward info dict is not another update.
        self.record["gates"] = [dict(round=2, update=1, gradient_norms_after_clipping=[0.3, 0.7, 0.2])]
        events = sync.training_events(self.record)
        self.assertEqual([event["training/round"] for event in events if "ppo/update" in event], [2])
        self.assertEqual(events[2]["training/controller_reward_unnormalized"], -9)
        self.assertEqual(events[2]["training/controller_reward_normalized"], -0.4)
        self.assertEqual(events[2]["ppo/gradient_norm_after_clipping_max"], 0.7)
        self.assertEqual(events[2]["ppo/entropy"], 0.8)
        self.assertNotIn("ppo/actor_grad_norm", events[2])
        self.assertFalse(any("reward" in key or key.startswith("ppo/") for key in events[3]))
        self.assertNotIn("ignored_numeric_detail", json.dumps(events))
        self.assertNotIn("design_reward", json.dumps(events))


    def test_failed_ppo_keeps_nonfinite_metrics_null_without_inventing_rewards(self):
        self.add_round(1, update=1)["status"] = "failed"
        self.record["gates"] = [dict(round=1, update=1, status="failed", after={"exact_kl": "inf"},
                                     losses={"policy_loss": "nan"}, gradient_norms_after_clipping=["nan"])]
        event = sync.training_events(self.record)[1]
        self.assertTrue(event["training/failed"])
        self.assertIsNone(event["ppo/exact_kl"])
        self.assertIsNone(event["ppo/policy_loss"])
        self.assertIsNone(event["ppo/gradient_norm_after_clipping_max"])
        self.assertNotIn("training/controller_reward_unnormalized", event)
        self.assertNotIn("ppo/learning_rate", event)

    def test_checkpoint_waits_for_exact_complete_matrix(self):
        results = [self.diagnostic(0, scale, seed, cost=cost)
                   for scale, seed, cost in ((1, 11, 10), (1, 12, 20), (2, 11, 30))]
        self.assertEqual(self.evaluations(), [])
        results.append(self.diagnostic(0, 2, 12, cost=40))
        event, = self.evaluations()
        self.assertEqual((event["eval/eligible_blocks"], event["eval/total_blocks"]), (4, 4))
        self.assertEqual(event["eval/primary_journey_mean_s"], 25)
        self.assertEqual(event["eval/pedestrian_completion_fraction"], 1)
        self.assertEqual(event["eval/vehicle_completion_fraction"], 1)

    def test_ineligible_cells_keep_null_cost_and_pooled_native_counts(self):
        results = [self.diagnostic(0, scale, seed, pedestrian=pedestrian, vehicle=vehicle)
                   for scale, seed, pedestrian, vehicle in (
                       (1, 11, (1, 0), (2, 2)), (1, 12, (9, 9), (2, 1)),
                       (2, 11, (10, 5), (6, 6)), (2, 12, (20, 20), (10, 10)))]
        results[0][1]["journeys"]["cohort"]["teleports"] = [{"id": "a"}, {"id": "b"}]
        results[1][1]["journeys"]["cohort"]["collisions"] = [{"id": "c"}]
        for path, result in results:
            self.write(path, result)
        event, = self.evaluations(self.summary(results))
        self.assertEqual(event["eval/eligible_blocks"], 1)
        self.assertIsNone(event["eval/primary_journey_mean_s"])
        self.assertAlmostEqual(event["eval/pedestrian_completion_fraction"], 34 / 40)
        self.assertAlmostEqual(event["eval/vehicle_completion_fraction"], 19 / 20)
        self.assertEqual((event["eval/teleports"], event["eval/collisions"]), (2, 1))

    def test_person_overlap_keeps_score_but_serious_or_unknown_collision_excludes_it(self):
        results = self.complete_checkpoint(0)
        path, result = results[0]
        collisions = result["journeys"]["cohort"]["collisions"]
        collisions.append({"type": "person-person"})
        self.write(path, result)
        event, = self.evaluations(self.summary(results))
        self.assertEqual(event["eval/primary_journey_mean_s"], 26.5)
        self.assertEqual((event["eval/eligible_blocks"], event["eval/collisions"]), (4, 1))

        for serious in ({"type": "vehicle-person"}, {"id": "unclassified"}):
            with self.subTest(collision=serious):
                collisions[:] = [{"type": "person-person"}, serious]
                self.write(path, result)
                event, = self.evaluations(self.summary(results))
                self.assertIsNone(event["eval/primary_journey_mean_s"])
                self.assertEqual((event["eval/eligible_blocks"], event["eval/collisions"]), (3, 2))

    def test_published_execution_failure_completes_matrix_without_zero_imputation(self):
        results = [self.diagnostic(0, scale, seed) for scale, seed in ((1, 11), (1, 12), (2, 11))]
        failure = dict(job=self.job(0, 2, 12), scores=None, error="Synthetic execution failure")
        event, = self.evaluations(self.summary(results, [failure]))
        self.assertEqual((event["eval/eligible_blocks"], event["eval/total_blocks"], event["eval/failed_blocks"]),
                         (3, 4, 1))
        for name in ("primary_journey_mean_s", "pedestrian_completion_fraction", "vehicle_completion_fraction",
                     "teleports", "collisions"):
            self.assertIsNone(event["eval/" + name])

    def test_duplicate_native_cell_is_not_overwritten(self):
        results = self.complete_checkpoint(0)
        duplicate = results[0][0].parent.parent / "duplicate" / "result.json"
        self.write(duplicate, results[0][1])
        with self.assertRaisesRegex(ValueError, "Duplicate diagnostic result"):
            self.evaluations()

    def test_duplicate_summary_cell_is_rejected_even_with_identical_results(self):
        report = self.summary(self.complete_checkpoint(0))
        report["records"].append(copy.deepcopy(report["records"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate diagnostic summary"):
            self.evaluations(report)

    def test_matching_cell_count_does_not_replace_exact_expected_keys(self):
        for scale, seed in ((1, 11), (1, 12), (2, 11), (3, 12)):
            self.diagnostic(0, scale, seed)
        with self.assertRaisesRegex(ValueError, "Unexpected diagnostic scale/seed"):
            self.evaluations()

    def test_duplicate_declared_matrix_is_rejected(self):
        self.study["protocol"]["scenarios"]["scales"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "duplicate declared diagnostic"):
            self.evaluations()

    def test_smoke_uses_existing_reduced_grid(self):
        self.study["settings"]["smoke"] = True
        self.diagnostic(0, 1, 11)
        event, = self.evaluations()
        self.assertEqual((event["eval/eligible_blocks"], event["eval/total_blocks"]), (1, 1))

    def test_late_evaluation_and_later_training_share_immutable_run_history(self):
        self.add_round(1)
        history = self.synchronize()
        first = copy.deepcopy(history)
        self.add_round(2, checkpoint=True)
        self.synchronize()
        before_evaluation = copy.deepcopy(history)
        self.complete_checkpoint(0)  # Arrives after training already reached round two.
        self.synchronize()
        after_evaluation = copy.deepcopy(history)
        self.add_round(3)
        self.synchronize()
        self.complete_checkpoint(2)
        self.record.update(status="complete", complete=True)
        self.publish_training()
        self.write(self.training_path.with_name("result.json"),
                   dict(status="complete", seed=7, layout_id="layout", training=str(self.training_path)))
        self.write(self.directory / "calibration" / "ignored" / "result.json", {"not": "a learner"})
        self.write(self.directory / "diagnostic" / "layout" / "local" / "cell" / "result.json", {"not": "a learner"})
        self.synchronize()
        self.synchronize()
        self.assertEqual(len(self.remote.histories), 1)
        self.assertEqual(history[:len(first)], first)
        self.assertEqual(history[:len(before_evaluation)], before_evaluation)
        self.assertEqual(history[:len(after_evaluation)], after_evaluation)
        self.assertEqual([row["sync/event_id"] for row in history], [
            "v2:initial", "v2:training/1", "v2:training/2", "v2:eval/0",
            "v2:training/3", "v2:terminal", "v2:eval/2",
        ])
        self.assertTrue(history[-2]["training/complete"])
        self.assertNotIn("training/round", history[-1])
        self.assertEqual(history[-1]["eval/checkpoint_round"], 2)
        self.assertEqual([row["_step"] for row in history], list(range(len(history))))

    def test_remote_acceptance_before_cursor_save_recovers_without_duplicate_submission(self):
        self.add_round(1)
        self.synchronize()
        self.add_round(2)
        self.remote.fail_finish = True
        with self.assertRaisesRegex(RuntimeError, "Remote accepted history"):
            self.synchronize()
        run_id = sync.learner_run_id(self.directory, self.identity)
        accepted = copy.deepcopy(self.remote.histories[run_id])
        cursor = self.state / run_id / "cursor.json"
        self.assertEqual(sync.read(cursor)["next"], 2)
        self.assertEqual(self.synchronize(), accepted)
        self.assertEqual(sync.read(cursor)["next"], 3)

    def test_unallowlisted_source_mutations_are_rejected_without_rewriting_history(self):
        self.add_round(1)
        results = self.complete_checkpoint(0)
        original_history = copy.deepcopy(self.synchronize())
        for path, source in ((self.training_path, lambda value: value["rounds"][0]),
                             (results[0][0], lambda value: value)):
            with self.subTest(path=path):
                original = path.read_bytes()
                changed = json.loads(original)
                source(changed)["unlogged_debug"] = "changed outside the metric allowlist"
                self.write(path, changed)
                with self.assertRaisesRegex(ValueError, "source event changed or disappeared"):
                    self.synchronize()
                self.assertEqual(next(iter(self.remote.histories.values())), original_history)
                path.write_bytes(original)
        self.assertEqual(self.synchronize(), original_history)

    def test_disappearing_training_source_is_not_silently_skipped(self):
        self.add_round(1)
        self.synchronize()
        self.training_path.unlink()
        with self.assertRaisesRegex(ValueError, "source event changed or disappeared"):
            self.synchronize()

    def test_late_success_summary_does_not_rewrite_native_result_events(self):
        results = self.complete_checkpoint(0)
        before = copy.deepcopy(self.synchronize())
        self.write(self.directory / "catalogue_diagnostic.json", self.summary(results))
        self.assertEqual(self.synchronize(), before)

    def test_failure_before_training_record_stays_in_the_learner_run(self):
        self.training_path.unlink()
        self.write(self.training_path.with_name("result.json"),
                   dict(status="failed", seed=7, layout_id="layout", error="private traceback detail"))
        history = self.synchronize()
        self.assertEqual(len(self.remote.histories), 1)
        self.assertEqual([row["sync/event_id"] for row in history], ["v2:terminal"])
        self.assertTrue(history[0]["training/failed"])
        self.assertFalse(history[0]["training/complete"])
        self.assertIsNone(history[0]["training/measured_steps"])
        self.assertNotIn("private traceback detail", json.dumps(history))


if __name__ == "__main__":
    unittest.main()
