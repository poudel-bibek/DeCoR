import os
import copy
import math
import xml.dom.minidom
import time
import json
import torch
import logging
import random
import numpy as np
import seaborn as sns
from scipy import stats
import xml.etree.ElementTree as ET

def save_config(higher_ppo_args, lower_ppo_args, control_args, design_args, run_dir):
    """
    Save hyperparameters to json.
    """
    save_path = os.path.join(run_dir, 'config.json')
    config_to_save = {
        "hyperparameters": {
            "higher_ppo_args": higher_ppo_args,
            "lower_ppo_args": lower_ppo_args,
            "control_args": control_args,
            "design_args": design_args,
        },
    }
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)

SLOT_PROTOCOL = 'explicit_fixed_map'  # Declared per-layout signal-slot maps; no permutation augmentation.


class LearnedEvaluationUnavailable(ValueError):
    """The controller lacks required slot provenance or exposure for a layout."""


def signal_slots_from_network(network_path, intersection_id):
    """Read crossing traffic-light IDs and deterministic spatial slots without running SUMO."""
    root = ET.parse(network_path).getroot()
    positions = {node.get('id'): float(node.get('x')) for node in root.findall('junction')}
    signals = {logic.get('id') for logic in root.findall('tlLogic')} - {intersection_id}
    ordered = sorted(signals, key=lambda tid: (positions[tid], tid))
    return {tid: slot for slot, tid in enumerate(ordered)}


def require_exposed_heads(provenance, signal_slots):
    """Every active crossing head must have collected decisions and entered an update batch."""
    if signal_slots is None:
        raise LearnedEvaluationUnavailable('Learned evaluation requires an explicit signal-slot map for the layout.')
    if not provenance or provenance.get('slot_protocol') != SLOT_PROTOCOL:
        raise LearnedEvaluationUnavailable('Checkpoint lacks explicit slot-protocol provenance; train under the current protocol.')
    decisions, updates = provenance.get('head_decisions'), provenance.get('head_updates')
    if decisions is None or updates is None:
        raise LearnedEvaluationUnavailable('Checkpoint lacks complete per-head exposure provenance.')
    slots = signal_slots.values()
    if any(not isinstance(slot, (int, np.integer)) or slot < 0 for slot in slots):
        raise LearnedEvaluationUnavailable('Signal slots must be nonnegative integer head indices.')
    missing = sorted({slot for slot in slots
                      if slot >= len(decisions) or slot >= len(updates)
                      or decisions[slot] <= 0 or updates[slot] <= 0})
    if missing:
        raise LearnedEvaluationUnavailable(f'Action heads {missing} lack collected decisions or controller update batches.')


def save_policy(higher_policy, lower_policy, lower_state_normalizer, norm_x, norm_y, save_path,
                head_decisions, head_updates):
    """
    Save both policies with the controller's Welford statistics and literal per-head exposure.
    """
    torch.save(
    {'higher': {
        'state_dict': higher_policy.state_dict(),  
        'norm_x': norm_x,
        'norm_y': norm_y
    },
    'lower': {
        'observation_version': lower_policy.observation_version,
        'provenance': {'slot_protocol': SLOT_PROTOCOL, 'permutation_augmentation': False,
                       'head_decisions': [int(n) for n in head_decisions],
                       'head_updates': [int(n) for n in head_updates]},
        'state_dict': lower_policy.state_dict(),  
        'state_normalizer_mean': lower_state_normalizer.mean.numpy(),  
        'state_normalizer_M2': lower_state_normalizer.M2.numpy(),  
        'state_normalizer_count': lower_state_normalizer.count.value  
    }}, save_path)

def load_policy(higher_policy, lower_policy, lower_state_normalizer, load_path):
    """
    Load policy state dict and welford normalizer stats; return design normalizers and controller provenance.
    """
    checkpoint = torch.load(load_path)
    if checkpoint['lower'].get('observation_version', 1) != lower_policy.observation_version:
        raise ValueError(
            'Checkpoint observation packing and Welford statistics are incompatible with '
            'this controller. Use fresh training or the historical checkout.'
        )
    # In place operations
    higher_policy.load_state_dict(checkpoint['higher']['state_dict'])
    lower_policy.load_state_dict(checkpoint['lower']['state_dict'])
    lower_state_normalizer.manual_load(
        mean=torch.from_numpy(checkpoint['lower']['state_normalizer_mean']),  
        M2=torch.from_numpy(checkpoint['lower']['state_normalizer_M2']),  
        count=checkpoint['lower']['state_normalizer_count']
    )
    return checkpoint['higher']['norm_x'], checkpoint['higher']['norm_y'], checkpoint['lower'].get('provenance')
    
def convert_demand_to_scale_factor(demand, demand_type, input_file):
    """
    Convert the demand to a scaling factor number.
    For vehicles: (veh/hr) that want to enter the network
    For pedestrians: (ped/hr) that want to enter the network
    """

    if demand <= 0:
        raise ValueError("Demand must be a positive number")
    
    if demand_type not in ['vehicle', 'pedestrian']:
        raise ValueError("Demand type must be either 'vehicle' or 'pedestrian'")
    
    # Calculate the original demand from the input file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    if demand_type == 'vehicle':
        original_demand = len(root.findall("trip"))
    else:  # pedestrian
        original_demand = len(root.findall(".//person"))
    
    if original_demand == 0:
        raise ValueError(f"No {demand_type} demand found in the input file")
    
    # Calculate the time span of the original demand
    if demand_type == 'vehicle':
        elements = root.findall("trip")
    else:
        elements = root.findall(".//person")
    
    # Find the start and end time of the demand
    start_time = min(float(elem.get('depart')) for elem in elements)
    end_time = max(float(elem.get('depart')) for elem in elements)
    time_span = (end_time - start_time) / 3600  # Convert to hours
    
    # Calculate the original demand per hour
    original_demand_per_hour = original_demand / time_span if time_span > 0 else 0
    print(f"\nOriginal {demand_type} demand per hour: {original_demand_per_hour:.2f}")

    if original_demand_per_hour == 0:
        raise ValueError(f"Cannot calculate original {demand_type} demand per hour")
    
    # Calculate the scale factor
    scale_factor = demand / original_demand_per_hour
    
    return scale_factor

def scale_demand(input_file, output_file, scale_factor, demand_type):
    """
    This function was causing some errors, so there is a new version as well.
    """
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    if demand_type == "vehicle":
        # Vehicle demand
        trips = root.findall("trip")
        for trip in trips:
            current_depart = float(trip.get('depart'))
            new_depart = current_depart / scale_factor
            trip.set('depart', f"{new_depart:.2f}")

        original_trip_count = len(trips)
        for i in range(1, int(scale_factor)):
            for trip in trips[:original_trip_count]:
                new_trip = ET.Element('trip')
                for attr, value in trip.attrib.items():
                    if attr == 'id':
                        new_trip.set(attr, f"{value}_{i}")
                    elif attr == 'depart':
                        new_depart = float(value) + (3600 * i / scale_factor)
                        new_trip.set(attr, f"{new_depart:.2f}")
                    else:
                        new_trip.set(attr, value)
                root.append(new_trip)

    elif demand_type == "pedestrian":
        # Pedestrian demand
        persons = root.findall(".//person")
        for person in persons:
            current_depart = float(person.get('depart'))
            new_depart = current_depart / scale_factor
            person.set('depart', f"{new_depart:.2f}")

        original_person_count = len(persons)
        for i in range(1, int(scale_factor)):
            for person in persons[:original_person_count]:
                new_person = ET.Element('person')
                for attr, value in person.attrib.items():
                    if attr == 'id':
                        new_person.set(attr, f"{value}_{i}")
                    elif attr == 'depart':
                        new_depart = float(value) + (3600 * i / scale_factor)
                        new_person.set(attr, f"{new_depart:.2f}")
                    else:
                        new_person.set(attr, value)
                
                # Copy all child elements (like <walk>)
                for child in person:
                    new_child = ET.SubElement(new_person, child.tag, child.attrib)
                    # Ensure 'from' attribute is present for walk elements
                    if child.tag == 'walk' and 'from' not in child.attrib:
                        # If 'from' is missing, use the first edge in the route
                        edges = child.get('edges', '').split()
                        if edges:
                            new_child.set('from', edges[0])
                        else:
                            logging.warning(f"Walk element for person {new_person.get('id')} is missing both 'from' and 'edges' attributes.")
                
                # Find the correct parent to append the new person
                parent = root.find(".//routes")
                if parent is None:
                    parent = root
                parent.append(new_person)

    else:
        print("Invalid demand type. Please specify 'vehicle' or 'pedestrian'.")
        return

    # Convert to string
    xml_str = ET.tostring(root, encoding='unicode')
   
    # Pretty print the XML string
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="    ")
   
    # Remove extra newlines between elements
    pretty_xml_str = '\n'.join([line for line in pretty_xml_str.split('\n') if line.strip()])
    
    # If there are folders in the path that dont exist, create them
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the formatted XML to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)
    
    # print(f"{demand_type.capitalize()} demand scaled by factor {scale_factor}.") # Output written to {output_file}")
    
    # Wait for the file writing operations to finish (it could be large)
    time.sleep(1)

def clear_elements(parent, tag):
    """
    Helper to clear out old elements.
    """
    for elem in parent.findall(tag):
        parent.remove(elem)

def scale_demand_sliced_window(input_file, output_file, scale_factor, demand_type, window_size, evaluation=False):
    """
    Compress and repeat demand from a random window, including warm-up.
    Training uses [0, 2400); evaluation uses the held-out [2400, 3600).
    Only departures within the output window are retained. For nonuniform
    demand, a partial repetition need not contain exactly scale_factor times
    the original number of trips.
    """
    START_SPAN, END_SPAN = (2400, 3600) if evaluation else (0, 2400)
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    if not math.isfinite(window_size) or not 0 < window_size <= END_SPAN - START_SPAN:
        raise ValueError("window_size must be positive and fit within the selected data partition")
    if demand_type not in ("vehicle", "pedestrian"):
        raise ValueError("Invalid demand_type: must be 'vehicle' or 'pedestrian'")
    t_start = random.uniform(START_SPAN, END_SPAN - window_size)
    t_end = t_start + window_size
    tree = ET.parse(input_file)
    root = tree.getroot()
    tag = "trip" if demand_type == "vehicle" else "person"
    routes_parent = root.find(".//routes") if demand_type == "pedestrian" else root
    if routes_parent is None:
        routes_parent = root
    originals = routes_parent.findall(tag)
    used_ids = {trip.get("id") for trip in originals}
    clear_elements(routes_parent, tag)
    windowed = []
    for trip in originals:
        depart = float(trip.get("depart"))
        if t_start <= depart < t_end:
            windowed.append((trip, depart - t_start))

    scaled = []
    for i in range(math.ceil(scale_factor)):
        for trip, shifted in windowed:
            new_depart = (shifted + window_size * i) / scale_factor
            if new_depart >= window_size:
                continue
            new_trip = copy.deepcopy(trip)
            new_trip.set("depart", str(new_depart))
            if i:
                new_id = f"{trip.get('id')}_{i}"
                while new_id in used_ids:
                    new_id += "_"
                used_ids.add(new_id)
                new_trip.set("id", new_id)
            for child in new_trip:
                if child.tag == "walk" and "from" not in child.attrib:
                    edges = child.get("edges", "").split()
                    if edges:
                        child.set("from", edges[0])
                    else:
                        logging.warning(
                            f"Walk element for {new_trip.get('id')} missing both 'from' and 'edges'."
                        )
            scaled.append(new_trip)
    routes_parent.extend(sorted(scaled, key=lambda trip: float(trip.get("depart"))))

    xml_str = ET.tostring(root, encoding="unicode")
    dom = xml.dom.minidom.parseString(xml_str)
    pretty = "\n".join(line for line in dom.toprettyxml(indent="    ").split("\n") if line.strip())

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty)

def get_averages(result_json_path, total=False):
    """
    Helper function that reads a JSON file with results and returns the scales,
    means and standard deviations for :
    - Lower agent: Vehicle wait time, pedestrian wait time
    - Higher agent: Pedestrian arrival time
    """
    with open(result_json_path, 'r') as f:
        results = json.load(f)

    scales, veh_wait_mean, ped_wait_mean, ped_arrival_mean = [], [], [], []
    veh_wait_std, ped_wait_std, ped_arrival_std = [], [], []
    
    for scale_str, runs in results.items():
        scale = float(scale_str)
        scales.append(scale)
        veh_wait_vals = []
        ped_wait_vals = []
        ped_arrival_vals = []
        
        for run in runs.values():
            if total:
                veh_wait_vals.append(run["total_veh_waiting_time"])
                ped_wait_vals.append(run["total_ped_waiting_time"])
                ped_arrival_vals.append(run["total_ped_arrival_time"])
            else:
                veh_wait_vals.append(run["veh_avg_waiting_time"])
                ped_wait_vals.append(run["ped_avg_waiting_time"])
                ped_arrival_vals.append(run["average_arrival_time_per_ped"])
                
        veh_wait_mean.append(np.mean(veh_wait_vals))
        ped_wait_mean.append(np.mean(ped_wait_vals))
        ped_arrival_mean.append(np.mean(ped_arrival_vals))
        veh_wait_std.append(np.std(veh_wait_vals))
        ped_wait_std.append(np.std(ped_wait_vals))
        ped_arrival_std.append(np.std(ped_arrival_vals))

    # Convert to numpy arrays and sort by scale
    scales = np.array(scales)
    sort_idx = np.argsort(scales)
    
    return (scales[sort_idx], 
            np.array(veh_wait_mean)[sort_idx], 
            np.array(ped_wait_mean)[sort_idx],
            np.array(ped_arrival_mean)[sort_idx],
            np.array(veh_wait_std)[sort_idx],
            np.array(ped_wait_std)[sort_idx],
            np.array(ped_arrival_std)[sort_idx])

