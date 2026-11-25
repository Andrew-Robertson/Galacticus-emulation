import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import h5py

SPEC = importlib.util.spec_from_file_location(
    'best_fit_runner', Path(__file__).resolve().parents[1] / 'scripts/run_best_fit_galacticus_model.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PreparedCampaignTest(unittest.TestCase):
    def test_prepared_run_preserves_training_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            evaluation = root / 'evaluations/model-eval-0000'
            evaluation.mkdir(parents=True)
            original = evaluation / 'input.xml'
            original.write_text('<parameters><outputFileName value="/training/output.hdf5"/>'
                                '<outputScratchFileName value="/training/output.hdf5"/>'
                                '<randomNumberGenerator value="GSL"><seed value="1234"/>'
                                '</randomNumberGenerator></parameters>')
            original_bytes = original.read_bytes()
            (root / 'assembly.json').write_text(json.dumps({'configuration': {'environment': {
                'GALACTICUS_DATA_PATH': '/pinned/datasets'}}}))
            mcmc = root / 'mcmc'
            mcmc.mkdir()
            changes = mcmc / 'maximum_a_posteriori_model_changes.xml'
            changes.write_text('<changes/>')
            extra = root / 'unit_changes.xml'
            extra.write_text('<changes/>')
            exe = root / 'source/Galacticus.exe'
            exe.parent.mkdir()
            exe.write_text('test executable')
            out = root / 'map'
            calls = []

            def simulate(command, *, cwd, environment):
                calls.append(command)
                self.assertEqual(Path(command[2]).name, 'additional_changes_00.xml')
                self.assertEqual(command[3], changes.name)
                self.assertEqual(Path(command[2]).read_bytes(), extra.read_bytes())
                self.assertEqual(environment['GALACTICUS_DATA_PATH'], '/pinned/datasets')
                self.assertEqual(environment['GALACTICUS_EXEC_PATH'], str(exe.parent))
                params = ET.parse(command[1]).getroot()
                for tag in ('outputFileName', 'outputScratchFileName'):
                    self.assertEqual(params.find(tag).get('value'), str(out / 'romanEPS_massFunction.hdf5'))
                self.assertEqual(params.find('randomNumberGenerator/seed').get('value'), '1234')
                (cwd / 'params.xml').write_text('<parameters/>')
                with h5py.File(cwd / 'romanEPS_massFunction.hdf5', 'w') as f:
                    f.attrs['statusCompletion'] = 0

            argv = ['runner', '--campaign-root', str(root), '--mcmc-dir', str(mcmc),
                    '--output-dir', str(out), '--galacticus-executable', str(exe),
                    '--no-postprocess-emission-line-dust', '--changes-file', str(extra)]
            with patch('sys.argv', argv), patch.object(runner, '_run', simulate):
                runner.main()
                with self.assertRaises(FileExistsError):
                    runner.main()
            self.assertEqual(len(calls), 1)
            self.assertEqual(original.read_bytes(), original_bytes)
            proof = json.loads((out / 'run_provenance.json').read_text())
            self.assertEqual(proof['map_source_sha256'], runner._sha(changes))
            self.assertEqual(proof['input_sha256'][str(original)], runner._sha(original))
            self.assertEqual(proof['executable_sha256'], runner._sha(exe))
            self.assertTrue((out / 'complete.json').is_file())

    def test_final_overlay_uses_identical_absolute_output_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            overlay = runner._write_output_change(root / 'output.xml', 'romanUNIT.hdf5')
            # Emulate the previous layers, including the relative UNIT output.
            values = {'outputFileName': 'romanUNIT.hdf5',
                      'outputScratchFileName': str(root / 'romanUNIT.hdf5')}
            for change in ET.parse(overlay).getroot():
                self.assertEqual(change.get('type'), 'replaceOrAppend')
                values[change.get('path')] = change.get('value')
            self.assertEqual(values['outputFileName'], values['outputScratchFileName'])
            self.assertEqual(values['outputFileName'], str(root / 'romanUNIT.hdf5'))

    def test_host_extension_changes_only_lower_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.xml'
            text = ('<parameters><mergerTreeBuildMasses value="sampledDistributionUniform">'
                    '<massTreeMinimum value="1e11"/><massTreeMaximum value="3e14"/>'
                    '<treesPerDecade value="5000"/></mergerTreeBuildMasses>'
                    '<mergerTreeMassResolution value="fixed"><massResolution value="2e10"/>'
                    '</mergerTreeMassResolution><randomNumberGenerator><seed value="1234"/>'
                    '</randomNumberGenerator></parameters>')
            path.write_text(text)
            with self.assertRaises(ValueError):
                runner._extend_root_hosts(path, 1e10)
            self.assertEqual(path.read_text(), text)
            proof = runner._extend_root_hosts(path, 2e10)
            actual = ET.parse(path).getroot()
            expected = ET.fromstring(text)
            expected.find('mergerTreeBuildMasses/massTreeMinimum').set('value', '20000000000')
            self.assertEqual(ET.tostring(actual), ET.tostring(expected))
            self.assertEqual(proof['old_minimum'], 1e11)

    def test_legacy_command_still_replaces_model_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            evaluation = root / 'evaluations/model-eval-0000'
            evaluation.mkdir(parents=True)
            (evaluation / 'run_eval.sh').write_text(
                '/old/Galacticus.exe /base.xml /old/model_changes.xml /old/output.xml '
                '--output-processed-parameters /old/params.xml\n')
            cmd = runner._command_from_template(campaign_root=root, template_evaluation_dir=None,
                template_evaluation_index=0, galacticus_executable='/new/Galacticus.exe',
                changes_filename='map.xml', output_change_filename='output.xml',
                processed_parameters='params.xml', output_dir=root / 'map', output_filename='map.hdf5')
            self.assertEqual(cmd, ['/new/Galacticus.exe', '/base.xml', 'map.xml', 'output.xml',
                                   '--output-processed-parameters', 'params.xml'])


if __name__ == '__main__':
    unittest.main()
