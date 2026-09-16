import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np
from config import classify_and_return_args, get_config
from utils import save_policy

import review_validation as review


class TrainingArtifactPreparationTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(review.torch.random.fork_rng(devices=[]))
        review.torch.manual_seed(173)
        self.study_dir = self.folder / "training"
        self.arm_dir = self.study_dir / "14100" / "joint"
        self.arm_dir.mkdir(parents=True)
        snapshot = self.study_dir / "source_snapshot" / "simulation"
        snapshot.mkdir(parents=True)
        self.network(snapshot / "Craver_traffic_lights_wide.net.xml", {"west_mid": 20, "east_mid": 80})
        for name in ("original_vehtrips.xml", "original_pedtrips.xml"):
            (snapshot / name).write_text("<routes/>")
        d, ctrl, higher, lower, _ = classify_and_return_args(get_config(), "cpu")
        d["save_dir"] = str(self.arm_dir)
        higher["model_kwargs"].update(run_dir=str(self.arm_dir), num_mixtures=2,
                                      hidden_channels=4, out_channels=4, initial_heads=1,
                                      second_heads=1, readout_k=3, model_size="small")
        lower["model_kwargs"]["model_size"] = "small"
        self.configuration = dict(design_args=d, control_args=ctrl, higher_ppo_args=higher, lower_ppo_args=lower)
        self.checkpoint = self.arm_dir / "final.pth"
        save_policy(review.PPO(**higher).policy, review.PPO(**lower).policy,
                    review.WelfordNormalizer((10, 123)), {"min": 0.0, "max": 100.0},
                    {"min": -10.0, "max": 10.0}, self.checkpoint, [36] * 10, [1] * 10)
        self.final_network = self.arm_dir / "network_iteration_481.net.xml"
        self.network(self.final_network, {"final_east_mid": 80, "final_west_mid": 20})
        source_hashes = {f"simulation/{name}": review.digest(snapshot / name) for name in
                         ("Craver_traffic_lights_wide.net.xml", "original_vehtrips.xml", "original_pedtrips.xml")}
        source_hashes["review_validation.py"] = "0" * 64  # Training source differs from this evaluator.
        self.record = {"complete": True, "arm": "joint", "seed": 14100,
                       "configuration": self.configuration, "source_hashes": source_hashes,
                       "checkpoint": str(self.checkpoint), "checkpoint_sha256": review.digest(self.checkpoint),
                       "layout": {"network": str(self.final_network), "sha256": review.digest(self.final_network),
                                  "iteration": 481, "num_proposals": 2, "real_world": False, "extreme_edges": {},
                                  "proposals": [[0.8, 0.6], [0.2, 0.4]],
                                  "crossing_ids": ["final_east", "final_west"],
                                  "signal_slots": {"final_east_mid": 7, "final_west_mid": 2}}}
        self.study = {"settings": {"development": False, "rounds": 480}, "seeds": [14100, 14200, 14300],
                      "source_hashes": source_hashes, "evaluation_scales": [.5, 1., 1.75, 2.75],
                      "evaluation_seeds": [19100, 19101], "journey_protocol": review.JOURNEY_PROTOCOL}
        self.artifact = self.arm_dir / "training.json"
        review.save(self.artifact, self.record)
        review.save(self.study_dir / "study.json", self.study)
        networks = self.folder / "prepared_networks"
        networks.mkdir()
        self.network(networks / "network_iteration_0.net.xml", {"west_mid": 20, "east_mid": 80})
        self.env = Mock(network_dir=str(networks), extreme_edge_dict={})

    @staticmethod
    def network(path, signals):
        root = ET.Element("net")
        for tid, x in {review.INTERSECTION: 50, **signals}.items():
            ET.SubElement(root, "junction", id=tid, x=str(x), y="0")
            ET.SubElement(root, "tlLogic", id=tid, programID="0")
        ET.ElementTree(root).write(path)

    def prepare(self, destination):
        with patch.object(review, "DesignEnv", return_value=self.env), patch("builtins.print"):
            review.prepare(destination, self.artifact)
        return json.loads((destination / "manifest.json").read_text())

    def test_completed_endpoint_keeps_geometry_and_freezes_distinct_evaluator_inputs(self):
        destination = self.folder / "baseline"
        manifest = self.prepare(destination)
        self.assertEqual(manifest["layouts"]["learned"], self.record["layout"])
        self.assertEqual(manifest["active_arms"], ["actuated"])
        self.assertEqual(manifest["training_artifact"]["source_hashes"], self.record["source_hashes"])
        self.assertEqual(manifest["source_hashes"]["review_validation.py"], review.digest(review.ROOT / "review_validation.py"))
        np.testing.assert_allclose(review.reference_proposals(manifest), [[.2, .4], [.8, .6]], rtol=1e-6)
        job = dict(manifest=str(destination / "manifest.json"), layout="learned", arm="actuated",
                   scale=1., seed=19100, split="evaluation", directory=str(destination / "cached_trial"))
        result = Path(job["directory"]) / "result.json"
        review.save(result, {"job": job, "manifest_sha256": review.digest(job["manifest"])})
        self.assertEqual(review.trial(job), str(result))
        # Every scientific input and the original recorded configuration remain cache prerequisites.
        paths = [self.artifact, self.checkpoint, self.final_network,
                 Path(manifest["configuration"]["design_args"]["original_net_file"]),
                 Path(manifest["configuration"]["control_args"]["vehicle_input_trips"]),
                 Path(manifest["configuration"]["control_args"]["pedestrian_input_trips"])]
        for path in paths:
            original = path.read_bytes()
            with self.subTest(input=path.name):
                try:
                    path.write_bytes(original + b"\n")
                    with self.assertRaises(ValueError):
                        review.trial(job)
                finally:
                    path.write_bytes(original)

    def test_incomplete_development_or_mismatched_artifacts_cannot_prepare(self):
        for fault in ("incomplete", "development", "checkpoint", "network", "slots", "input"):
            record, study = copy.deepcopy(self.record), copy.deepcopy(self.study)
            if fault == "incomplete":
                record["complete"] = False
            elif fault == "development":
                study["settings"]["development"] = True
            elif fault == "checkpoint":
                record["checkpoint_sha256"] = "1" * 64
            elif fault == "network":
                record["layout"]["sha256"] = "1" * 64
            elif fault == "slots":
                record["layout"]["signal_slots"]["final_west_mid"] = 7
            else:
                record["source_hashes"]["simulation/original_vehtrips.xml"] = "1" * 64
                study["source_hashes"] = record["source_hashes"]
            review.save(self.artifact, record)
            review.save(self.study_dir / "study.json", study)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.prepare(self.folder / fault)

    def test_current_design_with_unused_legacy_control_is_valid_but_old_readout_is_not(self):
        checkpoint = review.torch.load(self.checkpoint)
        checkpoint["lower"]["observation_version"] = 1
        review.torch.save(checkpoint, self.checkpoint)
        self.record["checkpoint_sha256"] = review.digest(self.checkpoint)
        review.save(self.artifact, self.record)
        manifest = self.prepare(self.folder / "unused_control")
        self.assertEqual(manifest["active_arms"], ["actuated"])
        self.assertEqual(manifest["checkpoint_control_version"], 1)
        checkpoint["higher"]["readout_version"] = 1
        review.torch.save(checkpoint, self.checkpoint)
        self.record["checkpoint_sha256"] = review.digest(self.checkpoint)
        review.save(self.artifact, self.record)
        with self.assertRaisesRegex(ValueError, "readout"):
            self.prepare(self.folder / "old_readout")

    def test_cli_uses_declared_heldout_only_after_frozen_selection(self):
        destination = self.folder / "cli_baseline"
        with patch("sys.argv", ["review_validation.py", "prepare", str(destination),
                               "--training-artifact", str(self.artifact)]), \
             patch.object(review, "DesignEnv", return_value=self.env), \
             patch.object(review.os, "chdir"), patch("builtins.print"):
            review.main()
        with patch("sys.argv", ["review_validation.py", "matrix", str(destination)]), \
             patch.object(review.os, "chdir"), patch("builtins.print"), \
             patch.object(review, "run_jobs") as run:
            with self.assertRaises(ValueError):
                review.main()
            manifest = json.loads((destination / "manifest.json").read_text())
            baseline = destination / "baselines" / "manifest.json"
            review.save(baseline, dict(manifest, layouts={"uniform": self.record["layout"],
                                                        "random_03": self.record["layout"]}))
            review.save(destination / "baselines" / "selection.json",
                        {"manifest_sha256": review.digest(baseline), "uniform": "uniform", "random_best20": "random_03"})
            review.main()
        jobs = run.call_args.args[0]
        self.assertEqual({(job["scale"], job["seed"]) for job in jobs},
                         {(scale, seed) for scale in self.study["evaluation_scales"] for seed in self.study["evaluation_seeds"]})
        self.assertEqual({(job["arm"], job["split"]) for job in jobs}, {("actuated", "evaluation")})
        self.assertEqual({job["layout"] for job in jobs}, {"original", "learned", "uniform", "random_03"})


class ReviewConfigurationTest(unittest.TestCase):
    def test_prepared_manifest_freezes_configuration_and_demand(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            inputs = [folder / name for name in ['vehicles.xml', 'pedestrians.xml']]
            for path in inputs:
                path.write_text('<routes/>')
            configuration = {
                'design_args': {'clamp_min': 0.01, 'clamp_max': 0.99, 'min_thickness': 2.0, 'max_thickness': 15.0},
                'control_args': {'warmup_steps': [40, 140],
                                 'vehicle_input_trips': str(inputs[0]),
                                 'pedestrian_input_trips': str(inputs[1])},
                'higher_ppo_args': {'model_kwargs': {}}, 'lower_ppo_args': {},
            }
            historical = folder / 'historical'
            historical.mkdir()
            config = historical / 'config.json'
            config.write_text(json.dumps({'hyperparameters': configuration}))
            checkpoint = folder / 'checkpoint.pth'
            checkpoint.write_bytes(b'prepare checkpoint fixture')
            state = {'lower': {}, 'higher': {'readout_version': 2, 'state_dict': {},
                     'norm_x': {'min': 0, 'max': 100}, 'norm_y': {'min': 0, 'max': 100}}}
            network_dir = folder / 'networks'
            network_dir.mkdir()
            for iteration in ['0', 'review']:
                (network_dir / f'network_iteration_{iteration}.net.xml').write_text('<net/>')
            original_network = ET.Element('net')
            for tid, x in [(review.INTERSECTION, 50), ('east_mid', 80), ('west_mid', 20)]:
                ET.SubElement(original_network, 'junction', id=tid, x=str(x), y='0')
                ET.SubElement(original_network, 'tlLogic', id=tid, programID='0')
            ET.SubElement(original_network, 'tlLogic', id='west_mid', programID='alternative')
            ET.ElementTree(original_network).write(network_dir / 'network_iteration_0.net.xml')
            env = Mock(network_dir=str(network_dir), extreme_edge_dict={},
                       current_net_file_path=str(network_dir / 'network_iteration_review.net.xml'),
                       crossing_ids=['iterreview_0', 'iterreview_1'],
                       signal_slots={'iterreview_0_mid': 0, 'iterreview_1_mid': 1})
            proposals = review.torch.tensor([[[0.25, 0.5], [0.75, 0.5]]])
            policy = Mock(readout_version=2)
            policy.act.return_value = (None, proposals, review.torch.tensor(2), None)
            policy.get_gmm_distribution.return_value = [Mock(
                component_distribution=Mock(mean=proposals[0]),
                mixture_distribution=Mock(probs=review.torch.tensor([0.5, 0.5])))]
            with patch.object(review, 'RUN', historical), patch.object(review, 'CHECKPOINT', checkpoint), \
                 patch.object(review, 'DesignEnv', return_value=env), \
                 patch.object(review, 'PPO', return_value=Mock(policy=policy)), \
                 patch.object(review.torch, 'load', return_value=state), \
                 patch.object(review.Batch, 'from_data_list'), patch('builtins.print'):
                destination = folder / 'study'
                review.prepare(destination)
                manifest = destination / 'manifest.json'
                layouts = json.loads(manifest.read_text())['layouts']
                self.assertEqual(layouts['learned']['signal_slots'], {'iterreview_0_mid': 0, 'iterreview_1_mid': 1})
                self.assertEqual(layouts['original'].get('signal_slots'), {'west_mid': 0, 'east_mid': 1})
                self.assertEqual(layouts['original']['num_proposals'], 2)
                job = dict(manifest=str(manifest), arm='fixed', layout='original', seed=6100,
                           scale=1.0, split='evaluation', directory=str(destination / 'trial'))
                configuration['control_args']['warmup_steps'] = [100, 100]
                config.write_text(json.dumps({'hyperparameters': configuration}))
                with patch.object(review, 'ControlEnv', side_effect=RuntimeError('stop before SUMO')) as control:
                    with self.assertRaisesRegex(RuntimeError, 'stop before SUMO'):
                        review.trial(job)
                    self.assertEqual(control.call_args.args[0]['warmup_steps'], [40, 140])

                result = Path(job['directory']) / 'result.json'
                result.write_text(json.dumps({'job': job, 'manifest_sha256': review.digest(manifest)}))
                self.assertEqual(review.trial(job), str(result))
                for path in inputs:
                    with self.subTest(input=path.name):
                        path.write_text('<routes changed="true"/>')
                        with patch.object(review, 'ControlEnv') as control:
                            with self.assertRaisesRegex(ValueError, 'Source changed'):
                                review.trial(job)
                            control.assert_not_called()
                        path.write_text('<routes/>')

    def test_embedded_configuration_and_cache_provenance(self):
        configuration = {
            'design_args': {'save_graph_images': True, 'save_gmm_plots': True},
            'control_args': {'per_timestep_state_dim': 123},
            'higher_ppo_args': {'device': 'cuda'},
            'lower_ppo_args': {'device': 'cuda', 'model_kwargs': {'per_timestep_state_dim': 123}},
        }
        original = copy.deepcopy(configuration)
        design, control, higher, lower = review.arguments(configuration)
        self.assertEqual(configuration, original)
        self.assertEqual(control, configuration['control_args'])
        self.assertEqual(higher['device'], 'cpu')
        self.assertEqual(lower['device'], 'cpu')
        self.assertFalse(design['save_graph_images'])
        lower['model_kwargs']['per_timestep_state_dim'] = 999
        self.assertEqual(configuration, original)

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / 'checkpoint.pth'
            checkpoint.write_bytes(b'cached-result checkpoint fixture')
            network = folder / 'network.net.xml'
            network.write_text('<net/>')
            manifest = folder / 'manifest.json'
            metadata = {
                'source_hashes': {'review_validation.py': review.digest(review.ROOT / 'review_validation.py')},
                'checkpoint': str(checkpoint), 'checkpoint_sha256': review.digest(checkpoint),
                'observation_version': review.MLP_ActorCritic.observation_version,
                'active_arms': ['fixed'], 'learned_control_skip_reason': None,
                'layouts': {'final': {'network': str(network), 'iteration': 'final', 'sha256': review.digest(network)}},
                'configuration': configuration,
            }
            manifest.write_text(json.dumps(metadata))
            job = dict(manifest=str(manifest), arm='fixed', layout='final', seed=6100,
                       scale=1.0, split='evaluation', directory=str(folder))
            result = folder / 'result.json'
            result.write_text(json.dumps({'job': job, 'manifest_sha256': review.digest(manifest)}))
            self.assertEqual(review.trial(job), str(result))
            network.write_text('<net changed="true"/>')
            with self.assertRaisesRegex(ValueError, 'Network changed'):
                review.trial(job)
            network.write_text('<net/>')

            for changed, message in [('configuration', 'different manifest'), ('version', 'observation protocol'),
                                     ('source', 'Source changed'), ('checkpoint', 'Checkpoint changed'),
                                     ('missing_configuration', 'embedded configuration')]:
                candidate = copy.deepcopy(metadata)
                if changed == 'configuration':
                    candidate['configuration']['control_args']['per_timestep_state_dim'] = 999
                elif changed == 'version':
                    candidate['observation_version'] = 1
                elif changed == 'source':
                    candidate['source_hashes']['review_validation.py'] = 'changed'
                elif changed == 'missing_configuration':
                    candidate.pop('configuration')
                else:
                    candidate['checkpoint_sha256'] = 'changed'
                manifest.write_text(json.dumps(candidate))
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, message):
                    review.trial(job)

            manifest.write_text(json.dumps(metadata))
            result.unlink()
            with patch.object(review, 'arguments', side_effect=RuntimeError('stop before SUMO')) as arguments:
                with self.assertRaisesRegex(RuntimeError, 'stop before SUMO'):
                    review.trial(job)
                arguments.assert_called_once_with(configuration)

    def test_warmup_mode_selects_reset_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / 'checkpoint.pth'
            checkpoint.write_bytes(b'checkpoint fixture')
            network = folder / 'network.net.xml'
            network.write_text('<net/>')
            manifest = folder / 'manifest.json'
            slots = {'a_mid': 0, 'b_mid': 1, 'c_mid': 2, 'd_mid': 3}
            metadata = {
                'source_hashes': {}, 'checkpoint': str(checkpoint), 'checkpoint_sha256': review.digest(checkpoint),
                'observation_version': review.MLP_ActorCritic.observation_version, 'learned_control_skip_reason': None,
                'configuration': {},
                'layouts': {'final': {'network': str(network), 'iteration': 'final', 'num_proposals': 4,
                                      'real_world': False, 'extreme_edges': {}, 'signal_slots': slots}},
            }
            job = dict(manifest=str(manifest), layout='final', seed=6100,
                       scale=1.0, split='evaluation', directory=str(folder))
            state = {'lower': {'observation_version': review.MLP_ActorCritic.observation_version, 'state_dict': {},
                               'provenance': {'slot_protocol': 'explicit_fixed_map', 'permutation_augmentation': False,
                                              'head_decisions': [36] * 4 + [0] * 6, 'head_updates': [1] * 4 + [0] * 6},
                               'state_normalizer_mean': np.zeros(1), 'state_normalizer_M2': np.zeros(1),
                               'state_normalizer_count': 2}}
            env = Mock(sumo_running=False)
            env.reset.side_effect = RuntimeError('stop at reset')
            with patch.object(review, 'arguments', return_value=({}, {}, {}, {})), \
                 patch.object(review, 'PPO'), patch.object(review.torch, 'load', return_value=state), \
                 patch.object(review, 'ControlEnv', return_value=env):
                for arm, mode, fixed in [('fixed', None, True), ('learned', None, True),
                                         ('learned', 'fixed', True), ('learned', 'random', False)]:
                    metadata['active_arms'] = [arm]
                    job['arm'] = arm
                    if mode is None:
                        metadata.pop('warmup_control', None)
                    else:
                        metadata['warmup_control'] = mode
                    manifest.write_text(json.dumps(metadata))
                    env.reset.reset_mock()
                    with self.subTest(arm=arm, mode=mode), self.assertRaisesRegex(RuntimeError, 'stop at reset'):
                        review.trial(job)
                    env.reset.assert_called_once_with({}, 4, tl=fixed, real_world=False, eval_mode=True,
                                                      signal_slots=slots)
                metadata['active_arms'].append('fixed')
                manifest.write_text(json.dumps(metadata))
                env.reset.reset_mock()
                with self.assertRaisesRegex(ValueError, 'learned-only'):
                    review.trial(job)
                env.reset.assert_not_called()

    def test_learned_evaluation_requires_declared_map_and_exposed_heads(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / 'checkpoint.pth'
            checkpoint.write_bytes(b'checkpoint fixture')
            network = folder / 'network.net.xml'
            network.write_text('<net/>')
            manifest = folder / 'manifest.json'
            base = {'west_mid': 0, 'east_mid': 7}
            layout = {'network': str(network), 'iteration': 'final', 'num_proposals': 2,
                      'real_world': False, 'extreme_edges': {}, 'signal_slots': dict(base)}
            metadata = {
                'source_hashes': {}, 'checkpoint': str(checkpoint), 'checkpoint_sha256': review.digest(checkpoint),
                'observation_version': review.MLP_ActorCritic.observation_version, 'learned_control_skip_reason': None,
                'active_arms': ['learned', 'fixed'], 'configuration': {}, 'layouts': {'final': layout},
            }
            provenance = {'slot_protocol': 'explicit_fixed_map', 'permutation_augmentation': False,
                          'head_decisions': [36, 0, 0, 0, 0, 0, 0, 36, 0, 0],
                          'head_updates': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]}
            state = {'lower': {'observation_version': review.MLP_ActorCritic.observation_version, 'state_dict': {},
                               'provenance': provenance,
                               'state_normalizer_mean': np.zeros(1), 'state_normalizer_M2': np.zeros(1),
                               'state_normalizer_count': 2}}
            job = dict(manifest=str(manifest), layout='final', arm='learned', seed=6100,
                       scale=1.0, split='evaluation', directory=str(folder))
            env = Mock(sumo_running=False)
            env.reset.side_effect = RuntimeError('stop at reset')
            cases = [
                ('exposed heads on the declared map', 'learned', {}, {}, RuntimeError, 'stop at reset'),
                ('missing map', 'learned', {'signal_slots': None}, {}, ValueError, 'explicit signal-slot map'),
                ('never-updated head', 'learned', {'signal_slots': {'west_mid': 0, 'east_mid': 5}}, {},
                 ValueError, r'heads \[5\]'),
                ('updated head without collected decisions', 'learned', {},
                 {'provenance': dict(provenance, head_decisions=[0] * 10)}, ValueError, r'heads \[0, 7\]'),
                ('incomplete exposure provenance', 'learned', {},
                 {'provenance': {'slot_protocol': 'explicit_fixed_map', 'head_updates': provenance['head_updates']}},
                 ValueError, 'provenance'),
                ('all crossings removed from a declared family', 'learned',
                 {'signal_slots': {}, 'num_proposals': 0, 'family_slots': base}, {}, RuntimeError, 'stop at reset'),
                ('legacy checkpoint without provenance', 'learned', {}, {'provenance': None}, ValueError, 'provenance'),
                ('variant moves a shared signal', 'fixed', {'family_slots': {'west_mid': 3}}, {},
                 ValueError, 'family base slots'),
                ('variant keeps shared signals', 'fixed', {'family_slots': {'west_mid': 0, 'gone_mid': 4}}, {},
                 RuntimeError, 'stop at reset'),
                ('new signal steals a removed signal slot', 'fixed',
                 {'signal_slots': {'west_mid': 0, 'new_mid': 7}, 'family_slots': base}, {}, ValueError, 'reserved'),
                ('new signal uses a free family slot', 'fixed',
                 {'signal_slots': {'west_mid': 0, 'new_mid': 5}, 'family_slots': base}, {}, RuntimeError, 'stop at reset'),
            ]
            for name, arm, layout_change, lower_change, error, message in cases:
                candidate = copy.deepcopy(metadata)
                candidate['layouts']['final'].update(layout_change)
                manifest.write_text(json.dumps(candidate))
                loaded = copy.deepcopy(state)
                loaded['lower'].update(lower_change)
                job['arm'] = arm
                env.reset.reset_mock()
                with self.subTest(case=name), \
                     patch.object(review, 'arguments', return_value=({}, {}, {}, {})), \
                     patch.object(review, 'PPO'), patch.object(review.torch, 'load', return_value=loaded), \
                     patch.object(review, 'ControlEnv', return_value=env):
                    with self.assertRaisesRegex(error, message):
                        review.trial(job)
                    if error is ValueError:
                        env.reset.assert_not_called()
                    else:
                        env.reset.assert_called_once()


class MatchedBaselineTest(unittest.TestCase):
    widths = np.array([0.975, 0.575, 0.05, 0.4875], dtype=np.float32)

    def test_matched_placements_keep_count_widths_and_feasibility(self):
        uniform = review.uniform_proposals(4, self.widths)
        np.testing.assert_allclose(uniform[:, 0], [.2, .4, .6, .8], rtol=1e-6)
        np.testing.assert_array_equal(uniform[:, 1], self.widths)
        rng = np.random.default_rng(review.GEOMETRY_SEED)
        for _ in range(review.RANDOM_CANDIDATES):
            padded, count = review.random_proposals(4, rng, self.widths)
            self.assertEqual(int(count), 4)
            np.testing.assert_array_equal(padded[0, :4, 1], self.widths)
            self.assertGreaterEqual(float(padded[0, :4, 0].min()), .01)
            self.assertLessEqual(float(padded[0, :4, 0].max()), .99)
            self.assertTrue(np.all(np.diff(padded[0, :4, 0].numpy()) >= .08 - 1e-6))
            np.testing.assert_array_equal(padded[0, 4:], -1)
        # The first location draw does not depend on whether widths are supplied.
        matched, _ = review.random_proposals(3, np.random.default_rng(7), self.widths[:3])
        free, _ = review.random_proposals(3, np.random.default_rng(7))
        np.testing.assert_array_equal(matched[0, :3, 0], free[0, :3, 0])
        with self.assertRaises(ValueError):
            review.check_matched(review.uniform_proposals(4, self.widths[::-1]), self.widths)
        for locations in ([.2, .25, .6, .8], [.0, .4, .6, .8], [.2, .4, .6, .995]):
            with self.subTest(locations=locations), self.assertRaises(ValueError):
                review.check_matched(np.column_stack((locations, self.widths)), self.widths)

    def test_baseline_manifest_derives_matched_layouts_from_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            networks = folder / 'geometry'
            networks.mkdir()
            reference = [[0.85, 0.05], [0.125, 0.975], [0.4, 0.4875], [0.725, 0.575]]  # merged order, not west to east
            parent = {'configuration': {'design_args': {'min_thickness': 2.0, 'max_thickness': 15.0},
                                        'control_args': {}, 'higher_ppo_args': {}, 'lower_ppo_args': {}},
                      'normalizer_x': {'min': 2000.0, 'max': 3000.0},
                      'layouts': {'learned': {'num_proposals': 4}},
                      'evaluation_proposals': [{'normalized': p} for p in reference]}
            manifest = folder / 'manifest.json'
            manifest.write_text(json.dumps(parent))
            applied = []
            env = Mock(extreme_edge_dict={'leftmost': {'old': 'a', 'new': None}})

            def apply(proposals, iteration):
                applied.append((iteration, np.array(proposals, dtype=np.float32)))
                path = networks / f'network_iteration_{iteration}.net.xml'
                path.write_text(f'<net id="{iteration}"/>')
                env.current_net_file_path = str(path)
                env.crossing_ids = [f'iter{iteration}_{i}' for i in range(len(proposals))]
                env.signal_slots = {f'{cid}_mid': i for i, cid in enumerate(env.crossing_ids)}
            env._apply_action.side_effect = apply
            with patch.object(review, 'DesignEnv', return_value=env):
                path = review.baseline_manifest(folder)
                self.assertEqual(review.baseline_manifest(folder), path)  # Reused, not rebuilt.
            derived = json.loads(path.read_text())
            self.assertEqual(derived['parent_manifest_sha256'], review.digest(manifest))
            names = ['uniform'] + [f'random_{i:02d}' for i in range(20)]
            self.assertEqual([name for name, _ in applied], names)
            self.assertEqual(list(derived['layouts']), names)
            widths = np.array([0.975, 0.4875, 0.575, 0.05], dtype=np.float32)  # reference widths west to east
            for name, proposals in applied:
                with self.subTest(layout=name):
                    np.testing.assert_array_equal(proposals[:, 1], widths)
                    layout = derived['layouts'][name]
                    self.assertEqual((layout['num_proposals'], layout['iteration'], layout['real_world']), (4, name, False))
                    self.assertEqual(layout['sha256'], review.digest(layout['network']))
                    self.assertEqual([p['normalized'] for p in layout['proposals']], proposals.tolist())
                    self.assertEqual(len(layout['signal_slots']), 4)
            np.testing.assert_allclose(applied[0][1][:, 0], [.2, .4, .6, .8], rtol=1e-6)
            search = derived['baseline_search']
            self.assertEqual(search['crossing_count'], 4)
            np.testing.assert_allclose([p['x'] for p in search['reference_proposals']], [2125, 2400, 2725, 2850], rtol=1e-6)
            parent['layouts']['learned']['num_proposals'] = 5
            manifest.write_text(json.dumps(parent))
            with self.assertRaises(ValueError):
                review.baseline_manifest(folder)

    def test_search_selects_lowest_complete_training_score_and_retains_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = folder / 'baselines' / 'manifest.json'
            review.save(manifest, {'layouts': {f'random_{i:02d}': {'sha256': f'sha{i}', 'proposals': [[i, 0.5]]}
                                               for i in range(20)}})
            jobs = review.search_jobs(folder, manifest)
            self.assertEqual(len(jobs), 120)
            self.assertEqual({(j['arm'], j['split']) for j in jobs}, {('actuated', 'training')})
            self.assertEqual({(j['scale'], j['seed']) for j in jobs},
                             {(s, d) for s in (1., 2.) for d in (5100, 5101, 5102)})
            # Score equals the index, except: 00 scores 20, 01 and 02 tie at 1; 05 has the lowest partial scores but one
            # failed trial; 07 lacks one result. Zero scheduled pedestrians exercise the max(scheduled, 1) rule.
            failures = {}
            for job in jobs:
                index = int(job['layout'][-2:])
                if index == 5 and (job['scale'], job['seed']) == (2., 5102):
                    failures[job['directory']] = 'RuntimeError: SUMO aborted'
                    continue
                if index == 7 and (job['scale'], job['seed']) == (1., 5100):
                    continue
                score = {0: 20, 2: 1, 5: 0}.get(index, index)
                traffic = {'vehicle': {'wait_sum_1s': score * 10, 'backlog_age_at_end_s': 0, 'scheduled': 10},
                           'pedestrian': {'wait_sum_1s': 0, 'backlog_age_at_end_s': 0, 'scheduled': 0}}
                review.save(Path(job['directory']) / 'result.json', {'traffic': traffic})
            selection = review.select_baseline(folder, manifest, jobs, failures)
            self.assertEqual((selection['random_best20'], selection['random_best20_network_sha256'],
                              selection['random_best20_score']), ('random_01', 'sha1', 1))
            self.assertEqual((selection['eligible_candidates'], selection['failed_or_incomplete_candidates']), (18, 2))
            candidates = {c['layout']: c for c in selection['candidates']}
            self.assertEqual(len(candidates), 20)
            self.assertEqual((candidates['random_05']['eligible'], candidates['random_05']['score']), (False, None))
            self.assertEqual([t['error'] for t in candidates['random_05']['trials'] if 'error' in t], ['RuntimeError: SUMO aborted'])
            self.assertEqual([t['error'] for t in candidates['random_07']['trials'] if 'error' in t], ['missing result'])
            self.assertEqual(len(candidates['random_07']['trials']), 6)
            self.assertEqual(candidates['random_02']['score'], 1)
            self.assertEqual(json.loads((folder / 'baselines' / 'selection.json').read_text()), selection)
            frozen = (folder / 'baselines' / 'selection.json').read_bytes()
            with self.assertRaises(FileExistsError):
                review.select_baseline(folder, manifest, jobs, {j['directory']: 'lost' for j in jobs})
            self.assertEqual((folder / 'baselines' / 'selection.json').read_bytes(), frozen)

    def test_failed_search_retains_every_outcome_without_evaluation_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = folder / 'baselines' / 'manifest.json'
            review.save(manifest, {'layouts': {f'random_{i:02d}': {'sha256': f'sha{i}', 'proposals': [[i, 0.5]]}
                                               for i in range(20)}})
            jobs = review.search_jobs(folder, manifest)
            failures = {job['directory']: 'RuntimeError: SUMO aborted' for job in jobs}
            with self.assertRaises(RuntimeError):
                review.select_baseline(folder, manifest, jobs, failures)
            result = json.loads((folder / 'baselines' / 'selection.json').read_text())
            self.assertIsNone(result['random_best20'])
            self.assertEqual(result['eligible_candidates'], 0)
            self.assertEqual({candidate['layout'] for candidate in result['candidates']},
                             {f'random_{i:02d}' for i in range(20)})
            for candidate in result['candidates']:
                self.assertFalse(candidate['eligible'])
                self.assertEqual({(trial['scale'], trial['seed']) for trial in candidate['trials']},
                                 {(scale, seed) for scale in (1., 2.) for seed in (5100, 5101, 5102)})
                self.assertTrue(all(trial['error'] == 'RuntimeError: SUMO aborted' for trial in candidate['trials']))
            with self.assertRaises(RuntimeError):
                review.baseline_rows(folder, [1.], [6100])

    def test_baseline_rows_use_frozen_winner_on_held_out_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertEqual(review.baseline_rows(folder, [1.0], [6100]), [])
            manifest = folder / 'baselines' / 'manifest.json'
            review.save(manifest, {'layouts': {}})
            review.save(folder / 'baselines' / 'selection.json',
                        {'manifest_sha256': review.digest(manifest), 'uniform': 'uniform', 'random_best20': 'random_11'})
            rows = review.baseline_rows(folder, [1.0, 2.0], [6100])
            self.assertEqual([(r['layout'], r['scale'], r['seed']) for r in rows],
                             [('uniform', 1.0, 6100), ('uniform', 2.0, 6100), ('random_11', 1.0, 6100), ('random_11', 2.0, 6100)])
            self.assertEqual({(r['arm'], r['split'], r['manifest']) for r in rows}, {('actuated', 'evaluation', str(manifest))})
            self.assertEqual(rows[2]['directory'], str(folder / 'trials' / 'random_best20_actuated_1.0_6100'))
            review.save(manifest, {'layouts': {'changed': True}})
            with self.assertRaises(ValueError):
                review.baseline_rows(folder, [1.0], [6100])


class JourneyCohortTest(unittest.TestCase):
    def test_departure_cutoff_excludes_boundary_and_preserves_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'demand.xml'
            path.write_text('''<routes><vType id="type"/>
                <person id="warmup" depart="0"><walk from="a" to="b"/></person>
                <person id="last" depart="549.9"><walk from="a" to="c"/></person>
                <person id="boundary" depart="550"><walk from="a" to="b"/></person>
                <person id="later" depart="600"><walk from="a" to="b"/></person>
                </routes>''')
            due = review.restrict_departure_cohort(path, 'person', 550)
            self.assertEqual(due, {'warmup': 0, 'last': 549.9})
            root = ET.parse(path).getroot()
            self.assertEqual([p.get('id') for p in root.findall('person')], ['warmup', 'last'])
            self.assertEqual(root.find('person/walk').attrib, {'from': 'a', 'to': 'b'})
            self.assertIsNotNone(root.find('vType'))

    def test_incomplete_cohort_keeps_missing_and_not_inserted_trips(self):
        entries = ET.fromstring('''<tripinfos>
            <tripinfo id="complete" depart="12" arrival="30" timeLoss="5"/>
            <tripinfo id="unfinished" depart="25" arrival="-1" timeLoss="10"/>
            <tripinfo id="pending" depart="-1" arrival="-1" timeLoss="0"/>
            </tripinfos>''')
        due = {'complete': 10, 'unfinished': 20, 'pending': 30, 'no_record': 40}
        summary = review.cohort_summary(due, list(entries), 'vehicle', 100)
        self.assertEqual(summary['scheduled'], 4)
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['unfinished_inserted'], 1)
        self.assertEqual(summary['not_inserted'], 2)
        self.assertEqual(summary['censored'], 3)
        self.assertEqual(summary['completion_fraction'], .25)
        # Arrival minus requested departure includes insertion delay. The other
        # three trips contribute elapsed age rather than disappearing from the mean.
        self.assertEqual(summary['journey_mean_lower_bound_s'], (20 + 80 + 70 + 60) / 4)
        self.assertIsNone(summary['journey_mean_s'])
        self.assertEqual(summary['insertion_delay_mean_lower_bound_s'], (2 + 5 + 70 + 60) / 4)
        self.assertEqual(summary['time_loss_plus_insertion_delay_mean_lower_bound_s'], (15 + 137) / 4)
        self.assertIsNone(summary['time_loss_plus_insertion_delay_mean_s'])

    def test_full_journey_mean_requires_all_stages_and_trips_complete(self):
        entries = ET.fromstring('''<tripinfos>
            <personinfo id="a" depart="2"><walk arrival="10"/><walk arrival="22"/></personinfo>
            <personinfo id="b" depart="12"><walk arrival="20"/><walk arrival="-1"/></personinfo>
            </tripinfos>''')
        due = {'a': 0, 'b': 10}
        summary = review.cohort_summary(due, list(entries), 'pedestrian', 50)
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['journey_mean_lower_bound_s'], 31)
        self.assertIsNone(summary['journey_mean_s'])
        entries[1][-1].set('arrival', '40')
        summary = review.cohort_summary(due, list(entries), 'pedestrian', 50)
        self.assertTrue(summary['all_completed'])
        self.assertEqual(summary['journey_mean_s'], 26)
        self.assertEqual(summary['journey_mean_lower_bound_s'], 26)
        self.assertEqual(summary['not_inserted'], 0)

    def test_drain_keeps_actuation_but_excludes_wait_from_measurement(self):
        env = Mock(tl_ids=['intersection', 'crossing'], mb_ped_incoming_edges_all=[])
        env.tl_lane_dict = {'crossing': {'pedestrian': {'incoming': {'north': {'main': ['edge']}}}}}
        env.junction_pos_cache = {'crossing': (0, 0)}
        tracker = review.Telemetry(env, 'actuated')
        tracker.active = False
        tracker.controller_active = True
        tracker.mb_phase['crossing'] = 0
        tracker.phase_started['crossing'] = 0
        with patch.object(review, 'traci') as traci:
            traci.simulation.getTime.return_value = 600
            for name in ('getDepartedIDList', 'getArrivedIDList', 'getDepartedPersonIDList',
                         'getArrivedPersonIDList', 'getStartingTeleportIDList', 'getCollidingVehiclesIDList'):
                getattr(traci.simulation, name).return_value = []
            traci.vehicle.getIDList.return_value = []
            traci.person.getIDList.return_value = ['p']
            traci.person.getWaitingTime.return_value = 10
            traci.person.getRoadID.return_value = 'edge'
            traci.person.getPosition.return_value = (0, 0)
            traci.trafficlight.getPhase.return_value = 0
            tracker.step()
            self.assertEqual(tracker.wait, {'vehicle': 0, 'pedestrian': 0})
            self.assertEqual(tracker.present['pedestrian'], set())
            traci.trafficlight.setPhase.assert_called_once_with('crossing', 1)


class FeedbackComparisonTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.manifest = self.folder / "manifest.json"
        self.protocol = {"timings": [[15, 10], [30, 20]], "selection_scales": [1.0],
                         "selection_seeds": [1, 2], "evaluation_scales": [.5, 2.0],
                         "evaluation_seeds": [11, 12], "practical_margin_s": 1.0}
        review.save(self.manifest, {"feedback_protocol": self.protocol,
                                   "active_arms": ["tuned_fixed", "actuated"],
                                   "layouts": {f"placement_{i:03d}": {} for i in range(5)},
                                   "criterion": "equal-class journey cost", "scope": "test fixture"})
        for job in review.feedback_jobs(self.folder, "selection"):
            index = int(job["layout"].split("_")[-1])
            access, wait, vehicle = [(1, 20, 30), (5, 1, 20), (7, 2, 1), (0, 0, 0), (0, 0, 0)][index]
            extra = 30 if job.get("parameters") == [15, 10] else 0
            self.record(job, access, wait + extra, vehicle, 100 + access + wait + extra,
                        incomplete=index == 3 and job["seed"] == 2, teleport=index == 4)

    def record(self, job, access, wait, vehicle, journey, incomplete=False, teleport=False):
        result = {"job": job, "manifest_sha256": review.digest(self.manifest),
                  "warmup_s": 100, "measurement_s": 450, "elapsed_s": .1,
                  "traffic": {kind: {"demand_sha256": f"{kind}/{job['seed']}/{job['scale']}"}
                              for kind in ("vehicle", "pedestrian")},
                  "feedback": {"access_mean_s": access, "approach_wait_mean_s": wait},
                  "journeys": {"cohort": {
                      "simulation_end_s": 650, "drain_s": 100,
                      "traffic": {"pedestrian": {"all_completed": not incomplete, "censored": int(incomplete), "scheduled": 1,
                                                "journey_mean_s": None if incomplete else journey},
                                  "vehicle": {"all_completed": True, "censored": 0, "scheduled": 1,
                                              "time_loss_plus_insertion_delay_mean_s": vehicle}},
                      "teleports": [{"id": "teleported"}] if teleport else [], "collisions": []}}}
        review.save(Path(job["directory"]) / "result.json", result)

    def test_selection_uses_complete_cohorts_and_all_three_information_scores(self):
        selected = review.select_feedback(self.folder, {})
        expected = {"access": "placement_000", "access_wait": "placement_001",
                    "access_wait_vehicle": "placement_002"}
        self.assertEqual(selected["choices"], {"tuned_fixed": expected, "actuated": expected})
        self.assertEqual(selected["timings"]["placement_000"], [30, 20])
        self.assertIsNone(selected["timings"]["placement_003"])
        self.assertIsNone(selected["timings"]["placement_004"])
        for candidate in selected["candidates"]:
            if candidate["layout"] in ("placement_003", "placement_004"):
                self.assertEqual(candidate["scores"], {rule: None for rule in expected})
        frozen = (self.folder / "feedback_selection.json").read_bytes()
        with self.assertRaises(FileExistsError):
            review.select_feedback(self.folder, {})
        self.assertEqual((self.folder / "feedback_selection.json").read_bytes(), frozen)

    def test_no_eligible_layout_preserves_failures_and_blocks_evaluation(self):
        jobs = review.feedback_jobs(self.folder, "selection")
        failures = {job["directory"]: "SUMO failed" for job in jobs}
        with self.assertRaises(RuntimeError):
            review.select_feedback(self.folder, failures)
        selected = json.loads((self.folder / "feedback_selection.json").read_text())
        self.assertFalse(selected["complete"])
        self.assertTrue(all(value is None for choices in selected["choices"].values() for value in choices.values()))
        self.assertTrue(all(row["error"] == "SUMO failed" for candidate in selected["candidates"]
                            for variant in candidate["timing_trials"] for row in variant["rows"]))
        with self.assertRaises(RuntimeError):
            review.feedback_jobs(self.folder, "evaluation")

    def test_changed_selection_evidence_cannot_release_evaluation(self):
        review.select_feedback(self.folder, {})
        path = Path(review.feedback_jobs(self.folder, "selection")[0]["directory"]) / "result.json"
        result = json.loads(path.read_text())
        result["feedback"]["access_mean_s"] = 0
        review.save(path, result)
        with self.assertRaises(ValueError):
            review.feedback_jobs(self.folder, "evaluation")

    def test_unmatched_demand_cannot_select_a_winner(self):
        path = Path(review.feedback_jobs(self.folder, "selection")[0]["directory"]) / "result.json"
        result = json.loads(path.read_text())
        result["traffic"]["vehicle"]["demand_sha256"] = "different demand"
        review.save(path, result)
        with self.assertRaises(ValueError):
            review.select_feedback(self.folder, {})
        self.assertFalse((self.folder / "feedback_selection.json").exists())

    def test_evaluation_uses_seed_blocks_and_never_drops_incomplete_pairs(self):
        review.select_feedback(self.folder, {})
        jobs = review.feedback_jobs(self.folder, "evaluation")
        for job in jobs:
            gap = {(11, .5): 2, (11, 2.): 4, (12, .5): 10, (12, 2.): 12}[job["seed"], job["scale"]]
            journey = 200 if job["layout"] == "placement_000" else 200 - gap
            self.record(job, 1, 1, 0, journey)
        summary = review.summarize_feedback(self.folder, {})
        for comparison in summary["comparisons"]:
            self.assertEqual([b["mean_difference_s"] for b in comparison["blocks"]], [3, 11])
            self.assertEqual(comparison["mean_difference_s"], 7)
            np.testing.assert_allclose(comparison["conditional_t95_interval_s"],
                                       [7 - 12.706204736 * 4, 7 + 12.706204736 * 4], rtol=1e-8)
            self.assertEqual(comparison["verdict"], "undetermined")
        failed = next(job for job in jobs if job["arm"] == "actuated" and job["layout"] == "placement_001")
        self.record(failed, 0, 0, 0, 0, incomplete=True)
        summary = review.summarize_feedback(self.folder, {})
        comparison = next(c for c in summary["comparisons"] if c["arm"] == "actuated" and c["selection_rule"] == "access_wait")
        self.assertIsNone(comparison["mean_difference_s"])
        self.assertIsNone(comparison["conditional_t95_interval_s"])
        self.assertEqual(comparison["verdict"], "incomplete_service")
        self.assertEqual(len(comparison["blocks"]), 2)

    def test_verdict_requires_the_whole_interval_to_clear_the_practical_margin(self):
        review.select_feedback(self.folder, {})
        jobs = review.feedback_jobs(self.folder, "evaluation")
        for mean, spread, expected in [(6., .1, "feedback_benefit"), (-6., .1, "feedback_harm"),
                                      (0., .01, "practically_negligible"), (1., .1, "undetermined")]:
            with self.subTest(expected=expected):
                for job in jobs:
                    gap = mean + (-spread if job["seed"] == 11 else spread)
                    self.record(job, 1, 1, 0, 200 if job["layout"] == "placement_000" else 200 - gap)
                report = review.summarize_feedback(self.folder, {})
                for comparison in report["comparisons"]:
                    self.assertAlmostEqual(comparison["mean_difference_s"], mean)
                    self.assertEqual(comparison["verdict"], expected)

    def test_coincident_choices_report_identity_not_distinct_layout_equivalence(self):
        for job in review.feedback_jobs(self.folder, "selection"):
            if job["layout"] == "placement_000":
                self.record(job, 0, 0, 0, 100)
        review.select_feedback(self.folder, {})
        for job in review.feedback_jobs(self.folder, "evaluation"):
            self.record(job, 1, 1, 0, 200)
        report = review.summarize_feedback(self.folder, {})
        for comparison in report["comparisons"]:
            self.assertTrue(comparison["identical_selection"])
            self.assertEqual(comparison["access_layout"], comparison["feedback_layout"])
            self.assertEqual(comparison["mean_difference_s"], 0)
            self.assertEqual(comparison["conditional_t95_interval_s"], [0, 0])
            self.assertEqual(comparison["verdict"], "practically_negligible")

    def test_single_seed_distinguishes_exact_identity_from_unestimated_variance(self):
        root = self.folder
        metadata = json.loads(self.manifest.read_text())
        metadata["feedback_protocol"]["evaluation_seeds"] = [11]
        metadata["layouts"] = {"placement_000": {}, "placement_001": {}}
        for identical in (False, True):
            with self.subTest(identical=identical):
                self.folder = root / str(identical)
                self.manifest = self.folder / "manifest.json"
                review.save(self.manifest, metadata)
                for job in review.feedback_jobs(self.folder, "selection"):
                    first = job["layout"] == "placement_000"
                    self.record(job, 1 if first else 5, 20 if first else 30 if identical else 0, 0, 200)
                review.select_feedback(self.folder, {})
                jobs = review.feedback_jobs(self.folder, "evaluation")
                for job in jobs:
                    self.record(job, 1, 1, 0, 200)
                report = review.summarize_feedback(self.folder, {})
                for comparison in report["comparisons"]:
                    self.assertEqual(comparison["identical_selection"], identical)
                    self.assertEqual(comparison["conditional_t95_interval_s"], [0, 0] if identical else None)
                    self.assertEqual(comparison["verdict"], "practically_negligible" if identical else "undetermined")
                # Identity does not excuse an ineligible service outcome.
                self.record(jobs[0], 1, 1, 0, 200, incomplete=True)
                report = review.summarize_feedback(self.folder, {})
                for comparison in report["comparisons"]:
                    if comparison["arm"] == jobs[0]["arm"]:
                        self.assertIsNone(comparison["conditional_t95_interval_s"])
                        self.assertEqual(comparison["verdict"], "incomplete_service")

    def test_no_approach_observations_do_not_become_a_perfect_access_score(self):
        job = review.feedback_jobs(self.folder, "selection")[0]
        self.record(job, None, None, 10, 100)
        scores = review.feedback_scores(json.loads((Path(job["directory"]) / "result.json").read_text()))
        self.assertEqual(scores["journey"], 110)
        self.assertIsNone(scores["access"])
        self.assertIsNone(scores["access_wait"])

    def test_full_cohort_approach_wait_does_not_pollute_measurement_wait(self):
        env = Mock(tl_ids=["intersection"], mb_ped_incoming_edges_all=["approach"])
        tracker = review.Telemetry(env, "fixed")
        tracker.feedback = True
        with patch.object(review, "traci") as traci:
            for name in ("getDepartedIDList", "getArrivedIDList", "getDepartedPersonIDList",
                         "getArrivedPersonIDList", "getStartingTeleportIDList", "getCollidingVehiclesIDList"):
                getattr(traci.simulation, name).return_value = []
            traci.vehicle.getIDList.return_value = []
            traci.person.getIDList.return_value = ["p"]
            for time, wait, edge, active in [(1, 1, "approach", False), (2, 2, "approach", False),
                                             (3, 0, "approach", True), (4, 1, "approach", True),
                                             (5, 2, "elsewhere", False)]:
                traci.simulation.getTime.return_value = time
                traci.person.getWaitingTime.return_value = wait
                traci.person.getRoadID.return_value = edge
                tracker.active = active
                tracker.step()
        self.assertEqual(tracker.approach_wait["p"], 3)
        self.assertEqual(tracker.wait["pedestrian"], 1)


if __name__ == '__main__':
    unittest.main()
