"""Preparation and failure contracts; synthetic workers make no SUMO performance claims."""
import copy
import json
import queue
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ppo.ppo_utils import Memory, WelfordNormalizer
from review_training import (CatalogueGateFailure, catalogue_progress, install_catalogue_gates,
                             prepare_catalogue, run_catalogue, verify_catalogue_preparation)
from review_validation import arguments, digest, save
from config import get_config
from simulation.design_env import DesignEnv
from simulation.worker import parallel_train_worker


LIMITS = dict(maximum_preupdate_abs_logratio=1e-4, maximum_postupdate_sampled_kl=.02,
              maximum_postupdate_exact_kl=.02, maximum_clip_fraction=.25,
              maximum_gradient_norm_after_clipping=.7501)


class TinyUpdate:
    """Real optimizer hooks with independently controlled probability diagnostics."""

    def __init__(self, before=None, after=None, gradient=.5):
        self.policy = torch.nn.Linear(1, 1, bias=False)
        self.policy_old = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=.01)
        self.metrics = [dict(approx_kl=0., exact_kl=0., clip_fraction=0., max_abs_logratio=0.)
                        for _ in range(2)]
        self.metrics[0].update(before or {})
        self.metrics[1].update(after or {})
        self.gradient = gradient

    def _control_diagnostics(self):
        return self.metrics.pop(0)

    def update(self, memory):
        self._control_diagnostics()
        self.policy.weight.grad = torch.full_like(self.policy.weight, self.gradient)
        self.optimizer.step()
        self._control_diagnostics()
        return dict(total_loss=0.)


class CatalogueGateTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(torch.random.fork_rng(devices=[]))

    def learner(self, **kwargs):
        ppo = TinyUpdate(**kwargs)
        env = SimpleNamespace(lower_ppo=ppo, lower_state_normalizer=WelfordNormalizer((1,)),
                              lower_reward_normalizer=WelfordNormalizer(1),
                              control_head_decisions=np.array([0, 1080, 0, 0, 0, 0, 0, 1080, 0, 0]),
                              control_head_updates=np.zeros(10, dtype=np.int64), lower_update_count=1,
                              executed_simulation_steps=10800, executed_total_steps=12000,
                              execution_accounting_complete=True, rollout_execution=[],
                              global_step=10800, control_args={"lower_action_duration": 10,
                                                              "lower_num_processes": 10, "max_timesteps": 360})
        record = dict(seed=510100, layout_id="synthetic", current_round=3, gates=[], rounds=[{}, {}])
        memory = Memory()
        for _ in range(2):
            action = torch.full((11,), -1)
            action[[0, 2, 8]] = 0
            memory.append(torch.zeros(1), action, 2, 0., 0., 0., True)
        protocol = dict(diagnostics={"numerical_gates": LIMITS},
                        training={"transitions_per_update": 2, "optimizer_steps_per_update": 1})
        install_catalogue_gates(env, record, self.folder, protocol)
        return env, record, memory

    def test_pre_likelihood_gate_prevents_optimizer_and_keeps_failure_snapshot(self):
        env, record, memory = self.learner(before={"max_abs_logratio": 1.01e-4})
        original = env.lower_ppo.policy.weight.detach().clone()
        with self.assertRaisesRegex(CatalogueGateFailure, "Pre-update likelihood"):
            env.lower_ppo.update(memory)
        torch.testing.assert_close(env.lower_ppo.policy.weight, original, rtol=0, atol=0)
        self.assertEqual(record["gates"][0]["optimizer_steps"], 0)
        self.assertTrue((self.folder / "latest_preupdate_lower.pth").exists())
        failed = torch.load(self.folder / "failed_update_lower.pth", weights_only=False)
        self.assertEqual(len(failed["memory"]["states"]), 2)
        self.assertEqual(failed["simulation_steps"], 10800)

    def test_failed_post_gate_keeps_update_head_exposure_and_full_rollout_budget(self):
        for metric in ("exact_kl", "approx_kl", "clip_fraction"):
            with self.subTest(metric=metric):
                env, record, memory = self.learner(after={metric: .26})
                with self.assertRaisesRegex(CatalogueGateFailure, metric):
                    env.lower_ppo.update(memory)
                catalogue_progress(env, record, 0.)
                self.assertEqual(record["budget"]["control_decisions"], 1080)
                self.assertEqual(record["budget"]["simulation_steps"], 10800)
                self.assertEqual(record["budget"]["control_rounds"], 3)
                self.assertEqual(record["budget"]["optimizer_steps"], 1)
                self.assertEqual(record["budget"]["control_updates_completed"], 0)
                self.assertEqual(record["control_head_updates"], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
                self.assertEqual(record["gates"][0]["status"], "failed")

    def test_sampled_kl_remains_one_sided_and_gate_state_is_instance_local(self):
        env, record, memory = self.learner(after={"approx_kl": -.03})
        other, other_record, _ = self.learner()
        env.lower_ppo.update(memory)
        self.assertEqual(record["gates"][0]["status"], "passed")
        self.assertEqual(other_record["gates"], [])

    def test_nonfinite_or_excessive_postclip_gradients_cannot_reach_optimizer(self):
        for gradient in (float("nan"), float("inf"), .751):
            with self.subTest(gradient=gradient):
                env, record, memory = self.learner(gradient=gradient)
                original = env.lower_ppo.policy.weight.detach().clone()
                with self.assertRaises(CatalogueGateFailure):
                    env.lower_ppo.update(memory)
                torch.testing.assert_close(env.lower_ppo.policy.weight, original, rtol=0, atol=0)
                self.assertEqual(record["gates"][0]["optimizer_steps"], 0)


class CatalogueWorkerTests(unittest.TestCase):
    def test_episode_provenance_survives_log_reuse_without_mutating_caller_seed(self):
        with TemporaryDirectory() as directory, patch('simulation.worker.ControlEnv') as constructor, \
                patch('simulation.worker.traci') as traci:
            folder = Path(directory)
            network = folder / 'network.net.xml'
            network.write_text('<net/>')
            log = folder / 'sumo_logfile_0.txt'
            error_log = folder / 'sumo_errorlog_0.txt'
            ctrl = dict(signal_control_protocol='shared_v1', global_seed=100,
                        total_action_timesteps_per_episode=1, sumo_seed=-1)
            env = constructor.return_value
            env.traci_label, env.sumo_running, env.step_count = '_0', True, 10
            env.demand_windows = {'vehicle': {'realized_trips': 3}}
            env.signal_executor = SimpleNamespace(stats={'site_01_mid': {'transitions': 1}})
            def step(action, **kwargs):
                for _ in range(10):
                    traci.simulationStep()
                return np.zeros(1), 0., True, False, {}

            env.eval_step.side_effect = step
            env._get_design_reward.return_value = 0.
            traci.simulation.getTime.side_effect = [40., 50., 60., 70.]

            def reset(*args, **kwargs):
                env.sumo_running = True
                log.write_text('episode log')
                error_log.write_text('retained warning')
                for _ in range(4):
                    traci.simulationStep()
                return np.zeros(1), {}

            env.reset.side_effect = reset
            queue = Mock()
            for seed in (1100, 2100):
                parallel_train_worker(0, directory, None, ctrl, queue, seed, torch.tensor(2), 10,
                                      None, {}, 'cpu', 'synthetic', str(network), fixed_control=True,
                                      signal_slots={'site_01_mid': 1, 'site_07_mid': 7})
            rows = [json.loads(line) for line in (folder / 'worker_0_episodes.jsonl').read_text().splitlines()]
            self.assertEqual([row['worker_seed'] for row in rows], [1100, 2100])
            self.assertEqual([row['round'] for row in rows], [1, 2])
            self.assertEqual([row['warmup_s'] for row in rows], [40., 60.])
            self.assertEqual(rows[0]['sumo_logs']['error_log']['text'], 'retained warning')
            self.assertEqual([row['measured_steps'] for row in rows], [10, 10])
            self.assertEqual([row['confirmed_total_steps'] for row in rows], [14, 14])
            self.assertEqual(ctrl['sumo_seed'], -1)
            self.assertEqual(queue.put.call_args.args[0], (0, None, 0., 1))


    def test_learned_worker_keeps_partial_action_on_failure_without_success_packet(self):
        with TemporaryDirectory() as directory, patch('simulation.worker.ControlEnv') as constructor, \
                patch('simulation.worker.traci') as traci:
            network = Path(directory) / 'network.net.xml'
            network.write_text('<net/>')
            env = constructor.return_value
            env.traci_label, env.sumo_running, env.step_count = '_0', True, 0
            env.active_slots, env.demand_windows, env.signal_executor = np.array([1, 7]), {}, None
            ticks = [0]
            traci.simulationStep.side_effect = lambda: ticks.__setitem__(0, ticks[0] + 1)
            traci.simulation.getTime.side_effect = lambda: ticks[0]

            def reset(*args, **kwargs):
                for _ in range(4):
                    traci.simulationStep()
                return np.zeros(1), {}

            def step(action):
                steps = 10 if env.step_count == 0 else 3
                for _ in range(steps):
                    traci.simulationStep()
                    env.step_count += 1
                if steps == 3:
                    raise RuntimeError('injected partial action failure')
                return np.zeros(1), 0., False, False, {}

            env.reset.side_effect, env.train_step.side_effect = reset, step
            policy = SimpleNamespace(act=lambda *args, **kwargs: (torch.zeros(3), torch.tensor(0.)),
                                     critic=lambda state: torch.tensor(0.))
            ctrl = dict(signal_control_protocol='shared_v1', global_seed=100,
                        total_action_timesteps_per_episode=2)
            packets, progress = queue.Queue(), [0] * 5
            with self.assertRaisesRegex(RuntimeError, 'partial action'):
                parallel_train_worker(0, directory, policy, ctrl, packets, 1100, torch.tensor(2), 10,
                                      SimpleNamespace(normalize=lambda state: state), {}, 'cpu',
                                      'synthetic', str(network), signal_slots={'a_mid': 1, 'b_mid': 7},
                                      progress=progress)
            episode = json.loads((Path(directory) / 'worker_0_episodes.jsonl').read_text())
            self.assertEqual(episode['status'], 'failed')
            self.assertEqual((episode['measured_steps'], episode['executed_decisions']), (13, 1))
            self.assertEqual(episode['confirmed_total_steps'], 17)
            self.assertFalse(episode['simulator_call_unresolved'])
            self.assertIn('partial action', packets.get_nowait()['error'])
            self.assertTrue(packets.empty())
            self.assertEqual(progress[:3], [17, 13, 1])


class CatalogueCollectionTests(unittest.TestCase):
    def test_failed_or_timed_out_worker_cost_is_not_learner_experience(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                env = DesignEnv.__new__(DesignEnv)
                env.control_args = dict(signal_control_protocol='shared_v1', lower_num_processes=2,
                                        global_seed=100, lower_action_duration=10, lower_update_freq=2,
                                        max_timesteps=10, total_action_timesteps_per_episode=1)
                env.lower_ppo_args = {'device': 'cpu'}
                env.run_dir, env.max_proposals = '/unused', 10
                env.current_net_file_path, env.current_network_iteration = '/unused/network.xml', 'fixed'
                env.extreme_edge_dict, env.signal_slots = {}, {'a_mid': 1, 'b_mid': 7}
                env.lower_ppo = Mock()
                env.lower_ppo.policy_old.to.return_value = env.lower_ppo.policy_old
                env.lower_memories = Memory()
                env.lower_state_normalizer = WelfordNormalizer((1,))
                env.lower_reward_normalizer = env.higher_reward_normalizer = WelfordNormalizer(1)
                env.control_head_decisions = np.zeros(10, dtype=np.int64)
                env.control_head_updates = np.zeros(10, dtype=np.int64)
                env.action_timesteps = env.lower_update_count = env.global_step = 0
                env.executed_simulation_steps = env.executed_total_steps = 0
                env.execution_accounting_complete, env.rollout_execution = True, []
                env.info = {}
                episode = Memory()
                action = np.full(11, -1)
                action[[0, 2, 8]] = 0
                episode.append(np.zeros(1), action, 2, 0., 0., 1., True)
                packets = Mock()
                packets.get.side_effect = [(0, episode, -2., 1),
                                           queue.Empty() if timeout else dict(rank=1, status='failed', error='injected')]
                processes = []

                def process(target, args):
                    rank, counts = args[0], args[-1]
                    p = Mock()
                    p.exitcode = None if rank == 1 and timeout else rank
                    p.is_alive.side_effect = lambda: p.exitcode is None
                    p.terminate.side_effect = lambda: setattr(p, 'exitcode', -15)
                    p.start.side_effect = lambda: counts.__setitem__(
                        slice(None), [50, 10, 1, 1, 0] if rank == 0 else
                        [47, 7, 0, 0, 1] if timeout else [43, 3, 0, -1, 0])
                    processes.append(p)
                    return p

                with patch('simulation.design_env.mp.Queue', return_value=packets), \
                        patch('simulation.design_env.mp.Process', side_effect=process):
                    with self.assertRaises(queue.Empty if timeout else RuntimeError):
                        env.step(torch.zeros((1, 10, 2)), torch.tensor(2), 1, update_layout=False)
                record = dict(rounds=[dict(status='failed')], gates=[])
                catalogue_progress(env, record, 0.)
                budget = record['budget']
                self.assertEqual(budget['simulation_steps'], 10)
                self.assertEqual(budget['control_decisions'], 1)
                self.assertEqual(budget['executed_simulation_steps'], 17 if timeout else 13)
                self.assertEqual(budget['uncollected_simulation_steps'], 7 if timeout else 3)
                self.assertEqual(budget['executed_total_steps'], 97 if timeout else 93)
                self.assertEqual(budget['execution_accounting_complete'], not timeout)
                self.assertEqual(budget['saved_successful_rounds'], 0)
                self.assertEqual(len(env.lower_memories.states), 1)
                self.assertEqual(env.rollout_execution[0]['status'], 'failed')
                self.assertTrue(all(not p.is_alive() for p in processes))
                env.lower_ppo.update.assert_not_called()


class CataloguePreparationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch('review_training.ROOT', self.root))
        config = get_config()
        sources = tuple(config[key] for key in ('original_net_file', 'vehicle_input_trips', 'pedestrian_input_trips')) + ('model.py',)
        for name in sources:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<synthetic/>')
        self.enterContext(patch('review_training.CATALOGUE_SOURCES', sources))
        self.catalogue = self.root / 'catalogue.json'
        self.protocol = self.root / 'protocol.json'
        family = {f'site_{i}_mid': i for i in range(6)}
        layouts = {}
        for name, count in [('two', 2), ('six', 6)]:
            network = self.root / f'{name}.net.xml'
            network.write_text(f'<net id="{name}"/>')
            ids = [f'site_{i}' for i in range(count)]
            layouts[name] = dict(network=str(network), sha256=digest(network), num_proposals=count,
                                 proposals=[[.1 * (i + 1), .5] for i in range(count)], crossing_ids=ids,
                                 signal_slots={f'{site}_mid': family[f'{site}_mid'] for site in ids},
                                 family_slots=family)
        save(self.catalogue, dict(layouts=layouts, family_slots=family))
        save(self.protocol, dict(catalogue=dict(candidate_ids=list(layouts), family_slots=family),
                                 training=dict(rounds_per_layout_seed=96, workers=10, learner_seeds=[510100, 510200],
                                               episode_measured_s=360, decision_s=10, design_updates=0, layout_updates=False,
                                               transitions_per_update=1080, optimizer_steps_per_update=10,
                                               diagnostic_rounds=[0, 48, 96], endpoint_round=96,
                                               configuration_overrides=dict(lower_lr=1e-4, lower_batch_size=256,
                                                                            lower_K_epochs=2, lower_anneal_lr=False,
                                                                            demand_scale_min=.5, demand_scale_max=2.))))
        self.destination = self.root / 'study'

    def test_preparation_is_evaluator_compatible_and_status_changes_do_not_change_identity(self):
        with patch('review_training.DesignEnv', side_effect=AssertionError('environment constructed')), \
                patch('review_training.PPO', side_effect=AssertionError('policy constructed')):
            study = prepare_catalogue(self.destination, self.catalogue, self.protocol, smoke=True)
        d, ctrl, _, _ = arguments(study['configuration'])
        self.assertEqual(ctrl['signal_control_protocol'], 'shared_v1')
        self.assertEqual(Path(d['original_net_file']).read_text(), '<synthetic/>')
        self.assertTrue(Path(d['original_net_file']).is_relative_to(self.destination / 'source_snapshot'))
        frozen = (self.destination / 'preparation.json').read_bytes()
        self.assertNotIn('preparation_sha256', json.loads(frozen))
        identity = study['preparation_sha256']
        for status in ('running', 'complete', 'failed'):
            study.update(status=status, initial_controller_sha256={'510100': 'initial'})
            verify_catalogue_preparation(self.destination, study)
        self.assertEqual(digest(self.destination / 'preparation.json'), identity)
        study['settings']['config']['demand_scale_max'] = 9
        with self.assertRaises(ValueError):
            verify_catalogue_preparation(self.destination, study)

    def test_corrupt_frozen_inputs_are_rejected_before_starting_a_learner(self):
        study = prepare_catalogue(self.destination, self.catalogue, self.protocol, smoke=True)
        for relative in ('source_snapshot/model.py', 'networks/two.net.xml', 'preparation.json'):
            with self.subTest(relative=relative):
                path = self.destination / relative
                original = path.read_bytes()
                path.write_bytes(original + b'\nchanged')
                try:
                    with patch('review_training.ProcessPoolExecutor', side_effect=AssertionError('learner started')):
                        with self.assertRaises(ValueError):
                            run_catalogue(self.destination, study)
                finally:
                    path.write_bytes(original)

    def screen_inputs(self, lower_lr, names=('two_spread', 'six_central'), **training):
        catalogue = json.loads(self.catalogue.read_text())
        catalogue['layouts'] = dict(zip(names, catalogue['layouts'].values()))
        protocol = json.loads(self.protocol.read_text())
        protocol['study_type'] = 'learning_rate_screen'
        protocol['catalogue']['candidate_ids'] = list(names)
        protocol['training'].update({'learner_seeds': [610100], **training})
        protocol['training']['configuration_overrides']['lower_lr'] = lower_lr
        paths = self.root / f'screen_{lower_lr}_catalogue.json', self.root / f'screen_{lower_lr}_protocol.json'
        save(paths[0], catalogue)
        save(paths[1], protocol)
        return paths

    def test_learning_rate_screen_admits_only_the_declared_single_seed_two_layout_rates(self):
        for lower_lr in (1e-4, 3e-4, 1e-3):
            with self.subTest(lower_lr=lower_lr):
                catalogue, protocol = self.screen_inputs(lower_lr)
                study = prepare_catalogue(self.root / f'screen_{lower_lr}', catalogue, protocol)
                self.assertEqual(study['seeds'], [610100])
                self.assertEqual(list(study['layouts']), ['two_spread', 'six_central'])
                self.assertEqual((study['settings']['rounds'], study['settings']['smoke']), (96, False))
                self.assertEqual(arguments(study['configuration'])[3]['lr'], lower_lr)
                self.assertEqual(study['protocol']['study_type'], 'learning_rate_screen')
        rejected = {
            'unscreened rate': lambda: self.screen_inputs(5e-4),
            'two learner seeds': lambda: self.screen_inputs(3e-4, learner_seeds=[610100, 610200]),
            'other candidates': lambda: self.screen_inputs(3e-4, names=('two_inner', 'six_central')),
            'shortened horizon': lambda: self.screen_inputs(3e-4, rounds_per_layout_seed=48),
        }
        for name, inputs in rejected.items():
            with self.subTest(rejected=name), self.assertRaises(ValueError):
                prepare_catalogue(self.root / f'rejected_{name}', *inputs())

    def test_legacy_protocol_keeps_two_seed_base_rate_contract_and_unknown_study_types_fail(self):
        protocol = json.loads(self.protocol.read_text())
        variants = {
            'unknown study_type': dict(protocol, study_type='pilot'),
            'screen rate without declaration': json.loads(json.dumps(protocol)),
            'single seed without declaration': json.loads(json.dumps(protocol)),
        }
        variants['screen rate without declaration']['training']['configuration_overrides']['lower_lr'] = 3e-4
        variants['single seed without declaration']['training']['learner_seeds'] = [510100]
        for name, variant in variants.items():
            with self.subTest(variant=name):
                path = self.root / f'{name}.json'
                save(path, variant)
                with self.assertRaises(ValueError):
                    prepare_catalogue(self.root / f'legacy_{name}', self.catalogue, path)
                self.assertFalse((self.root / f'legacy_{name}').exists())


if __name__ == "__main__":
    unittest.main()
