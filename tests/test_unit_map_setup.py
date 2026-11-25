import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from submit_updated_unit_map import write_files
from run_best_fit_galacticus_model import _command_from_template


class UnitSetupTest(unittest.TestCase):
    def test_tree_groups_and_weights(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            for batch in range(10):
                files = [Path(f'/trees/Tree_UNIT_001_SF10000_d{i}.hdf5')
                         for i in range(10*batch+1,10*batch+11)]
                path = Path(tmp) / f'{batch}.xml'
                write_files(path, files)
                nodes = ET.parse(path).getroot().findall('change')
                self.assertEqual(nodes[0].get('value').split(), [str(f) for f in files])
                self.assertEqual(nodes[1].get('value'), '1000.0')
                self.assertIn("[@value='scaleWeight']", nodes[1].get('path'))
                seen.extend(files)
        self.assertEqual(len(set(seen)),100)

    def test_conversion_compatibility_and_physics(self):
        root = ET.parse(ROOT / 'configs/unit/paper_EPStoUNIT.xml').getroot()
        self.assertEqual(root.find("change[@path='hotHaloRamPressureForce']").get('value'), 'orbitalPosition')
        self.assertEqual(root.find("change[@path='coolingFunction']/coolingFunction").get('value'), 'atomicCIECloudy')
        self.assertEqual(root.find("change[@path='mergerTreeConstructor']/mergerTreeConstructor").get('value'), 'read')
        self.assertEqual(root.find("change[@path='nodeOperator']/nodeOperator/nbodyHaloMassError/massParticle").get('value'), '=[simulation/massParticle]')


if __name__ == '__main__':
    unittest.main()
