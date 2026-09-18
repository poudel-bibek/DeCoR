import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_validation as review
import wandb_sync


class LearningRateScreenEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.folder = self.root / "screen"
        self.folder.mkdir()
        self.live = self.root / "live_sources"
        self.live.mkdir()
        self.enterContext(patch.object(review, "ROOT", self.live))
        source = self.live / "fixture_source.py"
        source.write_text("# immutable synthetic source\n")
        snapshot = self.folder / "source_snapshot" / source.name
        snapshot.parent.mkdir()
        snapshot.write_bytes(source.read_bytes())
        self.protocol = {
            "study_type": "learning_rate_screen",
            "scenarios": {"scales": [.5, 1., 2.]},
            "randomness": {"calibration_seeds": [710100], "diagnostic_seeds": list(range(810100, 810108))},
            "training": {"rounds": 96, "diagnostic_rounds": [0, 48, 96]},
            "classical": {"grid": {
                "cycle_s": [60], "intersection_action_order": [0, 1, 2, 3],
                "intersection_action_ticks": {"60": [[2, 2, 1, 1]]},
                "midblock_vehicle_fraction": [.5], "progression_direction": ["eastbound", "westbound"]}},
        }
        layouts = {}
        for name in ("two_spread", "six_central"):
            network = self.folder / f"{name}.xml"
            network.write_text("<net/>")
            layouts[name] = dict(network=str(network), sha256=review.digest(network), iteration=name)
        self.study = dict(
            kind="catalogue", status="prepared", settings={"smoke": False}, seeds=[610100],
            protocol=self.protocol, layouts=layouts, source_hashes={source.name: review.digest(source)},
            configuration=dict(design_args={}, control_args={"signal_control_protocol": "shared_v1"},
                               higher_ppo_args={}, lower_ppo_args={"lr": 3e-4}),
        )
        self.freeze()
        for layout in layouts:
            checkpoints = []
            for number in (0, 48, 96):
                path = self.folder / "610100" / layout / f"round_{number}.pth"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"synthetic checkpoint {layout}/{number}".encode())
                checkpoints.append(dict(round=number, path=str(path), sha256=review.digest(path),
                                        diagnostic_only=number != 96))
            review.save(self.training_path(layout), dict(
                kind="catalogue", complete=True, seed=610100, layout_id=layout,
                configuration=self.study["configuration"], layout=layouts[layout], checkpoints=checkpoints,
            ))

    def freeze(self):
        review.save(self.folder / "protocol.json", self.protocol)
        review.save(self.folder / "catalogue.json", {"candidate_ids": list(self.study["layouts"])})
        self.study.update(protocol_sha256=review.digest(self.folder / "protocol.json"),
                          catalogue_sha256=review.digest(self.folder / "catalogue.json"))
        prepared = {key: value for key, value in self.study.items() if key != "preparation_sha256"}
        prepared["status"] = "prepared"
        review.save(self.folder / "preparation.json", prepared)
        self.study.update(status="complete", preparation_sha256=review.digest(self.folder / "preparation.json"))
        review.save(self.folder / "study.json", self.study)

    def training_path(self, layout="two_spread"):
        return self.folder / "610100" / layout / "training.json"

    def jobs(self):
        return review.catalogue_jobs(self.folder, "diagnostic")

    def reject_trial(self, job, message):
        # These rejection paths must stop before RNG reseeding, weight loading or a simulator.
        with patch.object(review.os, "chdir"), patch.object(review.torch, "set_num_threads"), \
                patch.object(review, "ControlEnv", side_effect=AssertionError("Unexpected simulator construction")):
            with self.assertRaisesRegex(ValueError, message):
                review.trial(job)

    def test_screen_has_exactly_144_learned_jobs_without_classical_calibration(self):
        jobs = self.jobs()
        expected = {(layout, 610100, number, scale, seed, "learned_initial" if number == 0 else "learned")
                    for layout in self.study["layouts"] for number in (0, 48, 96)
                    for scale in (.5, 1., 2.) for seed in range(810100, 810108)}
        actual = [(job["layout"], job["learner_seed"], job["checkpoint_round"], job["scale"], job["seed"], job["arm"])
                  for job in jobs]
        self.assertEqual(len(actual), 144)
        self.assertEqual(set(actual), expected)
        self.assertTrue(all(job["split"] == "diagnostic" for job in jobs))
        self.assertFalse((self.folder / "catalogue_calibration.json").exists())
        self.assertFalse((self.folder / "catalogue_manifests" / "classical.json").exists())

    def test_screen_smoke_uses_its_actual_two_checkpoint_development_grid(self):
        self.study["settings"]["smoke"] = True
        self.freeze()
        for layout in self.study["layouts"]:
            path = self.training_path(layout)
            training = json.loads(path.read_text())
            training["checkpoints"] = [training["checkpoints"][0],
                                       dict(training["checkpoints"][-1], round=3)]
            review.save(path, training)
        jobs = self.jobs()
        self.assertEqual(
            {(job["layout"], job["checkpoint_round"], job["scale"], job["seed"]) for job in jobs},
            {(layout, checkpoint, 1., 810100) for layout in self.study["layouts"] for checkpoint in (0, 3)})
        self.assertEqual(len(jobs), 4)

    def test_screen_calibration_is_rejected_before_creating_manifests(self):
        with self.assertRaisesRegex(ValueError, "do not calibrate classical controllers"):
            review.catalogue_jobs(self.folder, "calibration")
        self.assertFalse((self.folder / "catalogue_manifests").exists())

    def test_screen_still_requires_the_whole_training_study_to_complete(self):
        self.study["status"] = "running"
        review.save(self.folder / "study.json", self.study)
        with self.assertRaisesRegex(ValueError, "every declared training job"):
            self.jobs()

    def test_screen_still_rejects_incomplete_learner_records(self):
        path = self.training_path()
        training = json.loads(path.read_text())
        training["complete"] = False
        review.save(path, training)
        with self.assertRaisesRegex(ValueError, "complete, layout-matched training record"):
            self.jobs()

    def test_screen_still_rejects_layout_mismatched_learner_records(self):
        path = self.training_path()
        training = json.loads(path.read_text())
        training["layout"]["sha256"] = "0" * 64
        review.save(path, training)
        with self.assertRaisesRegex(ValueError, "complete, layout-matched training record"):
            self.jobs()

    def test_three_checkpoint_entries_do_not_replace_exact_0_48_96_membership(self):
        path = self.training_path()
        training = json.loads(path.read_text())
        training["checkpoints"][1] = copy.deepcopy(training["checkpoints"][0])
        review.save(path, training)
        with self.assertRaises(ValueError):
            self.jobs()

    def test_screen_does_not_bypass_frozen_source_checks(self):
        (self.folder / "source_snapshot" / "fixture_source.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "Catalogue source changed"):
            self.jobs()

    def test_screen_jobs_retain_native_checkpoint_hash_rejection(self):
        job = next(job for job in self.jobs() if job["checkpoint_round"] == 96)
        manifest = json.loads(Path(job["manifest"]).read_text())
        Path(manifest["checkpoint"]).write_bytes(b"changed checkpoint")
        self.reject_trial(job, "Checkpoint changed since prepare")

    def test_initial_screen_jobs_retain_native_training_artifact_binding(self):
        job = next(job for job in self.jobs() if job["checkpoint_round"] == 0)
        manifest = json.loads(Path(job["manifest"]).read_text())
        artifact = Path(manifest["training_artifact"])
        training = json.loads(artifact.read_text())
        training["unlogged_change"] = True
        review.save(artifact, training)
        self.reject_trial(job, "Initial-policy diagnostics require a frozen catalogue training record")

    def test_ordinary_catalogue_still_requires_calibration_and_includes_classical_jobs(self):
        self.protocol.pop("study_type")
        self.freeze()
        with self.assertRaises(FileNotFoundError):
            self.jobs()
        calibration = review.catalogue_jobs(self.folder, "calibration")
        selected = {layout: dict(parameters=next(job["parameters"] for job in calibration if job["layout"] == layout))
                    for layout in self.study["layouts"]}
        review.save(self.folder / "catalogue_calibration.json",
                    dict(preparation_sha256=self.study["preparation_sha256"], selected=selected))
        jobs = self.jobs()
        self.assertEqual(len(jobs), 240)
        self.assertEqual({job["arm"] for job in jobs},
                         {"learned_initial", "learned", "coordinated_schedule", "local_actuated"})
        self.assertEqual(sum("learner_seed" not in job for job in jobs), 96)

    def result(self, job, incomplete=False):
        cost = 100 - job["checkpoint_round"] / 48
        traffic = {
            "pedestrian": dict(scheduled=5, completed=4 if incomplete else 5, censored=int(incomplete),
                               all_completed=not incomplete, journey_mean_s=None if incomplete else cost - 10),
            "vehicle": dict(scheduled=10, completed=10, censored=0, all_completed=True,
                            time_loss_plus_insertion_delay_mean_s=10),
        }
        cohort = dict(traffic=traffic, teleports=[], collisions=[], simulation_end_s=550, drain_s=0)
        result = dict(job=job, manifest_sha256=review.digest(job["manifest"]),
                      traffic={kind: {"demand_sha256": f"{kind}/{job['scale']}/{job['seed']}"} for kind in traffic},
                      initial_state_sha256=f"{job['layout']}/{job['scale']}/{job['seed']}",
                      journeys={"cohort": cohort}, feedback=dict(access_mean_s=2, approach_wait_mean_s=3),
                      warmup_s=100, measurement_s=450, elapsed_s=1)
        review.save(Path(job["directory"]) / "result.json", result)

    def test_learned_only_native_report_preserves_failures_pairs_and_compact_logger_contract(self):
        jobs = self.jobs()
        failed = next(job for job in jobs if job["layout"] == "two_spread" and job["checkpoint_round"] == 96)
        incomplete = next(job for job in jobs if job["layout"] == "two_spread" and job["checkpoint_round"] == 96
                          and job["scale"] == failed["scale"] and job["seed"] != failed["seed"])
        for job in jobs:
            if job != failed:
                self.result(job, incomplete=job == incomplete)
        failures = {failed["directory"]: "synthetic execution failure"}
        report = review.summarize_catalogue(self.folder, "diagnostic", jobs, failures)
        self.assertEqual((report["trials"], len(report["records"])), (144, 144))
        self.assertNotIn("selected", report)
        self.assertEqual(sum(row["scores"] is None for row in report["records"]), 2)
        self.assertEqual(len(report["paired_final_minus_initial"]), 48)
        self.assertEqual(sum(row["difference_s"] is None for row in report["paired_final_minus_initial"]), 2)
        self.assertEqual({row["difference_s"] for row in report["paired_final_minus_initial"]
                          if row["difference_s"] is not None}, {-2})
        native = json.loads((self.folder / "catalogue_diagnostic.json").read_text())
        training = json.loads(self.training_path().read_text())
        events = wandb_sync.evaluation_events(self.folder, self.study, training,
                                             dict(layout="two_spread", seed="610100", controller="learned"), native)
        final = next(event for event in events if event["eval/checkpoint_round"] == 96)
        self.assertEqual((final["eval/eligible_blocks"], final["eval/total_blocks"], final["eval/failed_blocks"]),
                         (22, 24, 1))
        self.assertIsNone(final["eval/primary_journey_mean_s"])
        self.assertIsNone(final["eval/pedestrian_completion_fraction"])


if __name__ == "__main__":
    unittest.main()
