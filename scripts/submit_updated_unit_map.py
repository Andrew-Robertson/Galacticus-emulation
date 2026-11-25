"""Reproduce the paper's ten UNIT tree-file batches at the updated joint MAP."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from run_best_fit_galacticus_model import _sha
from submit_preliminary_map_checks import RUN

REPO = Path(__file__).resolve().parents[1]
EXPECTED_EXE = 'ece5f40e1d3a203a2566bd92b4974a0b58b3eacd67c403338643b87bf09674c3'


def write_files(path, files):
    root = ET.Element('changes')
    ET.SubElement(root, 'change', type='update', path='mergerTreeConstructor/fileNames',
                  value='\n'.join(str(p) for p in files))
    ET.SubElement(root, 'change', type='update',
                  path="mergerTreeOperator[@value='scaleWeight']/scaleFactor", value='1000.0')
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='UTF-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--trees-dir', type=Path, default=Path(os.environ.get(
        'UNIT_MERGER_TREES_PATH', '/carnegie/nobackup/groups/dmtheory/mergerTrees/UNIT')))
    parser.add_argument('--output-dir', type=Path, help='Separate destination for a retry; must not exist.')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    view = campaign / 'preliminary_mixed_1020_20261118T211707Z'
    mcmc = view / 'gp_mcmc_preliminary' / RUN
    output = (args.output_dir.resolve() if args.output_dir else mcmc / 'bestFitModels_UNIT10_array')
    exe = campaign / 'diagnostics/eval0043_pr1785_20261113T234018Z/source/Galacticus.exe'
    source_map = mcmc / 'maximum_a_posteriori_model_changes.xml'
    tree_files = [args.trees_dir.resolve() / f'Tree_UNIT_001_SF10000_d{i}.hdf5' for i in range(1,101)]
    for path in [exe, source_map, view / 'assembly.json',
                 view / 'evaluations/sobol_1024_20p_simpleSizes_moreHalos-eval-0000/input.xml', *tree_files]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha(exe) != EXPECTED_EXE:
        raise ValueError('Executable checksum differs from tested PR1785 build')
    output.mkdir(exist_ok=False)
    frozen_map = output / source_map.name
    shutil.copy2(source_map, frozen_map)
    converter = output / 'EPStoUNIT.xml'
    shutil.copy2(REPO / 'configs/unit/paper_EPStoUNIT.xml', converter)
    config = output / 'pipeline.yaml'
    shutil.copy2(REPO / 'configs/pipelines/standard_observables_plus_halpha.yaml', config)
    proof = {
        'source_mcmc': str(mcmc), 'map_sha256': _sha(frozen_map),
        'executable': str(exe), 'executable_sha256': EXPECTED_EXE,
        'conversion_sha256': _sha(converter), 'pipeline_sha256': _sha(config),
        'selection': 'Paper batches d1-d100, ten files per run; weight scaleFactor=1000 per run',
        'combination': 'Paper figure averages the ten batch predictions; do not sum their normalized densities.',
        'tree_files': [{'path': str(p), 'size_bytes': p.stat().st_size,
                        'mtime_ns': p.stat().st_mtime_ns} for p in tree_files],
        'note': 'Tree-file records are size/mtime identities, not content checksums.',
    }
    (output / 'unit_provenance.json').write_text(json.dumps(proof, indent=2) + '\n')
    # Freeze the dataset environment as well as the input XML and MAP changes.
    snapshot = output / 'campaign_snapshot'
    (snapshot / 'evaluations/model-eval-0000').mkdir(parents=True)
    shutil.copy2(view / 'assembly.json', snapshot / 'assembly.json')
    shutil.copy2(view / 'evaluations/sobol_1024_20p_simpleSizes_moreHalos-eval-0000/input.xml',
                 snapshot / 'evaluations/model-eval-0000/input.xml')
    for index in range(10):
        start = 10*index + 1
        run = output / f'unit_{start:03d}_{start+9:03d}'
        run.mkdir()
        files_xml = run / 'UNIT_10files.xml'
        write_files(files_xml, tree_files[index*10:(index+1)*10])
        command = [sys.executable, str(REPO / 'scripts/submit_campaign_pipeline_slurm.py'),
                   '--config', str(config), '--stage', 'run_combined_best_fit_galacticus',
                   '--no-include-dependencies', '--profile', 'paper_unit_map',
                   '--set', f'ROOT={snapshot}', '--set', f'PIPELINE_OUTPUT_ROOT={output}',
                   '--set', f'COMBINED_MCMC_ROOT={output}',
                   '--set', f'COMBINED_BEST_FIT_RUN_DIR={run}',
                   '--set', f'BEST_FIT_GALACTICUS_EXECUTABLE={exe}',
                   '--set', f'UNIT_CONVERSION={converter}', '--set', f'UNIT_FILES={files_xml}',
                   '--script-dir', str(run / 'submission'), '--logs-dir', str(run / 'logs'),
                   '--provenance-path', str(run / 'SUBMISSION_PROVENANCE.md'),
                   '--python-command', sys.executable,
                   '--partition', 'expansion', '--qos', 'normal', '--job-prefix', f'newMAP-UNIT{start:03d}']
        if args.submit:
            command.append('--submit')
        subprocess.run(command, check=True)
    print('UNIT MAP batches:', output)
    print('Full outputs, reduced analyses, MAP dust sidecars and provenance retained in each unit_* directory.')


if __name__ == '__main__':
    main()
