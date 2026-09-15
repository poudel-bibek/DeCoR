import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np

import review_validation as review


class ReviewConfigurationTest(unittest.TestCase):
    def test_prepared_manifest_freezes_configuration_and_demand(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            inputs = [folder / name for name in ['vehicles.xml', 'pedestrians.xml']]
            for path in inputs:
                path.write_text('<routes/>')
            configuration = {
                'design_args': {'clamp_min': 0.01, 'clamp_max': 0.99},
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
            state = {'lower': {}, 'higher': {'state_dict': {},
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
            policy = Mock()
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


if __name__ == '__main__':
    unittest.main()
