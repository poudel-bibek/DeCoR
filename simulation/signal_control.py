"""Catalogue control: common clearance executor and non-learning request policies."""
import math
import statistics

import numpy as np
import traci

from .sim_setup import get_intersection_phase_groups

PROTOCOL = "shared_v1"
YELLOW_S = 4
ALL_RED_S = 1
MIN_GREEN_S = 5


class SignalExecutor:
    """Close all entries before changing service; wait for crossing/junction clearance.

    Requests arrive every 10 s, but clearance persists across decisions. No timeout
    releases conflicting traffic into an occupied crossing. This is a simulator
    service contract, not a real-world signal safety certification.
    """
    def __init__(self, env):
        self.env = env
        vehicle, pedestrian = get_intersection_phase_groups()
        self.targets = [[vehicle[action] + p['A'] + 'r' + p['B'] + p['C'] + 'r' + p['D']
                         for action, p in pedestrian.items()]]
        self.targets.extend([["rrG", "GGr"] for _ in env.tl_ids[1:]])
        self.states = []
        self.stats = {}
        now = traci.simulation.getTime()
        for index, tid in enumerate(env.tl_ids):
            width = 16 if index == 0 else 2
            current = traci.trafficlight.getRedYellowGreenState(tid)
            if len(current) != len(self.targets[index][0]):
                raise ValueError(f"{PROTOCOL} does not support signal geometry {tid}")
            links = traci.trafficlight.getControlledLinks(tid)
            inside = sorted({edge for movements in links[:width]
                             for _, outgoing, via in movements if via
                             for edge in env._internal_edges(via, outgoing)})
            crossings = sorted({edge for direction in env.tl_lane_dict[tid]['pedestrian']['outgoing'].values()
                                for edge in direction['main']})
            self.states.append(dict(action=None, green_since=now-MIN_GREEN_S,
                                    pending=None, yellow_until=now, red_until=now,
                                    physical=current, width=width, inside=inside, crossings=crossings))
            self.stats[tid] = dict(requested_s=[0]*len(self.targets[index]),
                                   green_s=[0]*len(self.targets[index]), yellow_s=0,
                                   all_red_s=0, occupied_clearance_s=0, transitions=0)

    def _occupied(self, state):
        return (any(traci.edge.getLastStepPersonIDs(edge) for edge in state['crossings']) or
                any(traci.edge.getLastStepVehicleNumber(edge) for edge in state['inside']))

    def apply(self, actions):
        if len(actions) != len(self.env.tl_ids):
            raise ValueError("One request is required for every active signal")
        now = traci.simulation.getTime()
        phases = []
        for index, (tid, requested) in enumerate(zip(self.env.tl_ids, actions)):
            requested = int(requested)
            state, stats = self.states[index], self.stats[tid]
            if not 0 <= requested < len(self.targets[index]):
                raise ValueError(f"Invalid signal request {requested} for {tid}")
            stats['requested_s'][requested] += self.env.step_length
            if (state['pending'] is None and requested != state['action'] and
                    now-state['green_since'] >= MIN_GREEN_S):
                # Commit service through clearance; newer requests cannot cancel it.
                state['pending'] = requested
                vehicle = state['physical'][:state['width']]
                yellow = any(c in 'gGyY' for c in vehicle)
                state['yellow_state'] = ''.join('y' if c in 'gGyY' else 'r' for c in vehicle) + 'r'*(len(state['physical'])-state['width'])
                state['yellow_until'] = now + (YELLOW_S if yellow else 0)
                state['red_until'] = state['yellow_until'] + ALL_RED_S
                stats['transitions'] += 1
            if state['pending'] is not None:
                if now < state['yellow_until']:
                    physical, phase = state['yellow_state'], 4 if index == 0 else 2
                    stats['yellow_s'] += self.env.step_length
                else:
                    occupied = now >= state['red_until'] and self._occupied(state)
                    if now < state['red_until'] or occupied:
                        physical, phase = 'r'*len(state['physical']), 5 if index == 0 else 3
                        stats['all_red_s'] += self.env.step_length
                        stats['occupied_clearance_s'] += self.env.step_length if occupied else 0
                    else:
                        state.update(action=state['pending'], pending=None, green_since=now)
                        physical, phase = self.targets[index][state['action']], state['action']
                        stats['green_s'][phase] += self.env.step_length
            else:
                physical, phase = self.targets[index][state['action']], state['action']
                stats['green_s'][phase] += self.env.step_length
            traci.trafficlight.setRedYellowGreenState(tid, physical)
            state['physical'] = physical
            phases.append(phase)
        return phases


def coordinated_grid(protocol):
    grid = protocol['classical']['grid']
    return [dict(cycle_s=cycle, intersection_action_order=grid['intersection_action_order'],
                 intersection_action_ticks=ticks, midblock_vehicle_fraction=fraction,
                 progression_direction=direction)
            for cycle in grid['cycle_s'] for ticks in grid['intersection_action_ticks'][str(cycle)]
            for fraction in grid['midblock_vehicle_fraction'] for direction in grid['progression_direction']]


class CoordinatedSchedule:
    """One global request clock, with explicit geometry-derived progression offsets."""
    def __init__(self, env, parameters):
        self.env, self.parameters = env, parameters
        self.cycle = parameters['cycle_s']
        self.ticks = parameters['intersection_action_ticks']
        self.order = parameters['intersection_action_order']
        if sum(self.ticks)*10 != self.cycle or sorted(self.order) != [0, 1, 2, 3] or self.order[0] != 0:
            raise ValueError('Invalid common-cycle intersection allocation')
        self.vehicle_ticks = math.floor(parameters['midblock_vehicle_fraction']*self.cycle/10 + .5)
        if not 0 < self.vehicle_ticks < self.cycle/10:
            raise ValueError('Every cycle must request both midblock service modes')
        speeds = [traci.lane.getMaxSpeed(movement[0]) for tid in env.tl_ids[1:]
                  for links in traci.trafficlight.getControlledLinks(tid)[:2] for movement in links]
        speed = statistics.median(speeds)
        origin = env.junction_pos_cache[env.tl_ids[0]][0]
        direction = 1 if parameters['progression_direction'] == 'eastbound' else -1
        self.offsets = {tid: (math.floor(direction*(env.junction_pos_cache[tid][0]-origin)/speed/10+.5)*10) % self.cycle
                        for tid in env.tl_ids[1:]}
        self.started = traci.simulation.getTime()
        self.evidence = dict(parameters=parameters, offsets_s=self.offsets,
                             progression_speed_m_s=speed, origin_x_m=origin,
                             clock_origin_s=self.started,
                             scope='Requested phases; common clearance can delay actual greens.')

    def act(self):
        elapsed = traci.simulation.getTime()-self.started
        tick = int(elapsed % self.cycle // 10)
        boundary = 0
        for action, duration in zip(self.order, self.ticks):
            boundary += duration
            if tick < boundary:
                break
        return np.asarray([action] + [int((elapsed-self.offsets[tid]) % self.cycle < self.vehicle_ticks*10)
                                      for tid in self.env.tl_ids[1:]], dtype=np.int32)


class LocalActuated:
    """Demand-actuated cyclic service, using the same local occupancy features as RL.

    Requests hold 10--40 s; empty phases are skipped, not predicted. This reference
    is deliberately local, unlike the common-clock coordinated comparator.
    """
    def __init__(self, env):
        self.env = env
        self.actions = np.asarray([0] + [1]*len(env.tl_ids[1:]), dtype=np.int32)
        self.started = np.full(len(env.tl_ids), traci.simulation.getTime())
        self.evidence = dict(minimum_request_s=10, maximum_request_s=40,
                             intersection_order=[0, 1, 2, 3], information='current local incoming occupancy')

    def act(self):
        now = traci.simulation.getTime()
        for index, tid in enumerate(self.env.tl_ids):
            occupancy = self.env.corrected_occupancy_map[tid]
            incoming = occupancy['vehicle']['incoming']
            pedestrian = occupancy['pedestrian']['incoming']
            if index == 0:
                ped = {d: len(v['main'])+len(v.get('vicinity', [])) for d, v in pedestrian.items()}
                demands = [sum(len(ids) for d, ids in incoming.items() if d.startswith(('east-', 'west-'))) + ped['north']+ped['south'],
                           sum(len(incoming[d+'-straight'])+len(incoming[d+'-right']) for d in ('north', 'south')) + ped['east']+ped['west'],
                           sum(len(incoming[d+'-left']) for d in ('north', 'south')),
                           sum(ped.values())]
            else:
                demands = [len(pedestrian['north']['main']), sum(len(ids) for ids in incoming.values())]
            current = int(self.actions[index])
            age = now-self.started[index]
            if age >= 10 and (not demands[current] or age >= 40):
                for offset in range(1, len(demands)):
                    candidate = (current+offset) % len(demands)
                    if demands[candidate]:
                        self.actions[index], self.started[index] = candidate, now
                        break
        return self.actions.copy()
