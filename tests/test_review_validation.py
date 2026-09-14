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
                'observation_version': 2, 'active_arms': ['fixed'], 'learned_control_skip_reason': None,
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
                                     ('source', 'Source changed'), ('checkpoint', 'Checkpoint changed')]:
                candidate = copy.deepcopy(metadata)
                if changed == 'configuration':
                    candidate['configuration']['control_args']['per_timestep_state_dim'] = 999
                elif changed == 'version':
                    candidate['observation_version'] = 1
                elif changed == 'source':
                    candidate['source_hashes']['review_validation.py'] = 'changed'
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
            metadata = {
                'source_hashes': {}, 'checkpoint': str(checkpoint), 'checkpoint_sha256': review.digest(checkpoint),
                'observation_version': 2, 'learned_control_skip_reason': None,
                'layouts': {'final': {'network': str(network), 'iteration': 'final', 'num_proposals': 4,
                                      'real_world': False, 'extreme_edges': {}}},
            }
            job = dict(manifest=str(manifest), layout='final', seed=6100,
                       scale=1.0, split='evaluation', directory=str(folder))
            state = {'lower': {'observation_version': 2, 'state_dict': {},
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
                    env.reset.assert_called_once_with({}, 4, tl=fixed, real_world=False, eval_mode=True)
                metadata['active_arms'].append('fixed')
                manifest.write_text(json.dumps(metadata))
                env.reset.reset_mock()
                with self.assertRaisesRegex(ValueError, 'learned-only'):
                    review.trial(job)
                env.reset.assert_not_called()


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
