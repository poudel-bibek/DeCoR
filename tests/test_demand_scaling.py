import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from utils import scale_demand_sliced_window


class DemandScalingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / 'input.xml'
        self.output = Path(self.directory.name) / 'output.xml'

    def scale(self, root, factor, demand_type='vehicle', window=100, start=0, evaluation=False):
        ET.ElementTree(root).write(self.input)
        with patch('utils.random.uniform', return_value=start) as sample:
            scale_demand_sliced_window(
                self.input, self.output, factor, demand_type, window, evaluation=evaluation
            )
        return ET.parse(self.output).getroot(), sample

    def test_uniform_counts_sorted_and_within_window(self):
        for demand_type, tag in [('vehicle', 'trip'), ('pedestrian', 'person')]:
            root = ET.Element('routes')
            for i in reversed(range(100)):
                ET.SubElement(root, tag, id=f'user{i}', depart=str(i + 0.5))
            for factor in [0.5, 1, 1.75, 2, 2.75]:
                with self.subTest(demand_type=demand_type, factor=factor):
                    output, _ = self.scale(root, factor, demand_type)
                    trips = output.findall(tag)
                    departures = [float(trip.get('depart')) for trip in trips]
                    self.assertEqual(len(trips), int(100 * factor))
                    self.assertEqual(departures, sorted(departures))
                    self.assertTrue(all(0 <= depart < 100 for depart in departures))
                    self.assertEqual(len({trip.get('id') for trip in trips}), len(trips))

    def test_preserves_vehicle_types_attributes_and_children(self):
        root = ET.fromstring('''<routes>
            <vType id="car" accel="2.6"><param key="type-data" value="keep"/></vType>
            <trip id="v" depart="25" type="car" fromTaz="west" toTaz="east">
                <param key="trip-data" value="keep"/>
            </trip>
        </routes>''')
        output, _ = self.scale(root, 2)
        self.assertEqual(output.find('vType').attrib, root.find('vType').attrib)
        self.assertEqual(output.find('vType/param').attrib, root.find('vType/param').attrib)
        for trip in output.findall('trip'):
            self.assertEqual(trip.get('type'), 'car')
            self.assertEqual(trip.get('fromTaz'), 'west')
            self.assertEqual(trip.get('toTaz'), 'east')
            self.assertEqual(trip.find('param').attrib, {'key': 'trip-data', 'value': 'keep'})

    def test_preserves_person_stages_and_supplies_walk_origin(self):
        root = ET.fromstring('''<routes><person id="p" depart="25" type="walker">
            <walk edges="a b"><param key="stage-data" value="keep"/></walk>
            <stop lane="b_0" duration="5"/>
        </person></routes>''')
        output, _ = self.scale(root, 2, 'pedestrian')
        for person in output.findall('person'):
            self.assertEqual(person.get('type'), 'walker')
            self.assertEqual(person.find('walk').attrib, {'edges': 'a b', 'from': 'a'})
            self.assertEqual(person.find('walk/param').get('value'), 'keep')
            self.assertEqual(person.find('stop').attrib, {'lane': 'b_0', 'duration': '5'})

    def test_duplicate_ids_do_not_collide_with_original_ids(self):
        root = ET.fromstring('''<routes>
            <trip id="v" depart="25"/><trip id="v_1" depart="75"/>
        </routes>''')
        output, _ = self.scale(root, 2)
        ids = [trip.get('id') for trip in output.findall('trip')]
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids)), 4)
        self.assertTrue({'v', 'v_1'}.issubset(ids))

    def test_half_open_window_does_not_round_departure_to_horizon(self):
        root = ET.fromstring('''<routes>
            <trip id="before" depart="-1"/><trip id="start" depart="0"/>
            <trip id="last" depart="99.9999999"/><trip id="end" depart="100"/>
        </routes>''')
        output, _ = self.scale(root, 1)
        trips = output.findall('trip')
        self.assertEqual([trip.get('id') for trip in trips], ['start', 'last'])
        self.assertLess(float(trips[-1].get('depart')), 100)

    def test_evaluation_samples_full_held_out_partition(self):
        root = ET.fromstring('''<routes>
            <trip id="early" depart="2400"/><trip id="start" depart="3010"/>
            <trip id="last" depart="3599.9"/><trip id="end" depart="3600"/>
        </routes>''')
        output, sample = self.scale(root, 1, window=590, start=3010, evaluation=True)
        sample.assert_called_once_with(2400, 3010)
        self.assertEqual([trip.get('id') for trip in output.findall('trip')], ['start', 'last'])
        for evaluation, window, start in [(False, 2400, 0), (True, 1200, 2400)]:
            _, sample = self.scale(root, 1, window=window, start=start, evaluation=evaluation)
            sample.assert_called_once_with(start, start)

    def test_rejects_invalid_scale_and_window(self):
        root = ET.Element('routes')
        for factor in [0, -1, float('nan'), float('inf')]:
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                self.scale(root, factor)
        for window in [0, -1, float('nan'), float('inf'), 2401]:
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.scale(root, 1, window=window)
        with self.assertRaises(ValueError):
            self.scale(root, 1, window=1201, evaluation=True)
        with self.assertRaises(ValueError):
            self.scale(root, 1, demand_type='invalid')


if __name__ == '__main__':
    unittest.main()
