#!/usr/bin/env python3
"""Freeze, generate, and stage nested equilibrium-energy Sobol campaigns."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy
from scipy.stats import qmc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from galacticus_emu.lhs import UniformPrior, TruncatedLogNormalPrior

FAMILY = REPO / 'runs/campaigns/equilibriumSizesEnergy'
BAR = "nodeOperator/nodeOperator[@value='barInstability']"
DYNAMICS = BAR + '/galacticDynamicsBarInstability'
ENERGY = BAR + '/barInstabilitySpheroidAngularMomentum'
ACCRETION = "nodeOperator/nodeOperator[@value='haloAngularMomentumInterpolate']"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def inverse(prior, u):
    name, p = prior['class'], prior['parameters']
    if name == 'ReflectedPrior':
        return p['constant'] - inverse(p['base'], 1-u)
    if name == 'LogUniformPrior':
        return np.exp(np.log(p['lower']) + u*np.log(p['upper']/p['lower']))
    classes = {'UniformPrior': UniformPrior, 'TruncatedLogNormalPrior': TruncatedLogNormalPrior}
    return classes[name](**p).inverse_cdf(u)


def freeze(family, source, seed=42):
    definition = family / 'family_definition.json'
    master = family / 'design_1024.csv'
    if definition.exists() or master.exists():
        raise FileExistsError('Family design already exists; do not overwrite a frozen design.')
    old = json.loads(source.read_text())
    specs = json.loads(json.dumps(old['free_parameters']))
    if len(specs) != 20:
        raise ValueError('Expected the historical 20-parameter campaign')
    old_retention = next(s for s in specs if s['short_name'] == 'barFractionAngularMomentumRetainedSpheroid')
    old_retention['short_name'] = 'barFractionAngularMomentumRetainedDisk'
    old_retention['path'] = DYNAMICS + '/fractionAngularMomentumRetainedDisk'
    old_retention['prior'] = {'class': 'ReflectedPrior', 'parameters': {
        'constant': 1., 'base': old_retention['prior']}}
    specs.append({'short_name': 'barSpheroidFormFactor', 'path': ENERGY + '/formFactorSpheroid',
                  'prior': {'class': 'LogUniformPrior', 'parameters': {'lower': .25, 'upper': 1.}}})
    quantiles = qmc.Sobol(d=len(specs), scramble=True, seed=seed).random_base2(10)
    values = np.column_stack([inverse(s['prior'], quantiles[:, i]) for i, s in enumerate(specs)])
    if not np.isfinite(values).all():
        raise ValueError('Non-finite design values')
    family.mkdir(parents=True, exist_ok=True)
    with master.open('w', newline='') as handle:
        writer = csv.writer(handle)
        names = [s['short_name'] for s in specs]
        writer.writerow(['design_index', *[n+'_quantile' for n in names], *names])
        for i, (u, v) in enumerate(zip(quantiles, values)):
            writer.writerow([i, *[format(x, '.17g') for x in u], *[format(x, '.17g') for x in v]])
    write_json(definition, {
        'schema_version': 1, 'family': 'equilibriumSizesEnergy', 'seed': seed,
        'master_size': 1024, 'ndim': len(specs), 'free_parameters': specs,
        'source_campaign': old['campaign_name'], 'source_design_sha256': digest(source),
        'design_sha256': digest(master),
        'generation_versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'fixed': {'galacticStructureSolver': 'equilibrium', 'bar_model': 'bindingEnergy',
                  'fractionAngularMomentumRetainedSpheroid': 1., 'formFactorDisk': .5,
                  'interactionEnergyFactor': 1., 'transition': [.003, .010],
                  'finiteDifferenceStepRelative': .01,
                  'angularMomentumSpecificGrowthFromAccretionRate': False,
                  'Blitz2006PressureCharacteristic_K_cm3': 35000., 'Blitz2006PressureExponent': .92},
        'notes': ['New 21-dimensional design, not the same points as the old 20-dimensional Sobol design.',
                  'Pilot and production use exact prefixes of the stored master table.',
                  'Disk prior is the full reflected old spheroid prior, not merely the same bounds.',
                  'c_s range is an exploratory modelling prior, not an observational constraint.'],
    })


def replace(root, path, value):
    change = ET.SubElement(root, 'change', type='replaceOrAppend', path=path)
    return ET.SubElement(change, path.rsplit('/', 1)[-1], value=str(value))


def model_changes(row, definition):
    root = ET.Element('changes')
    replace(root, 'galacticStructureSolver', 'equilibrium')
    replace(root, ACCRETION + '/angularMomentumSpecificGrowthFromAccretionRate', 'false')
    replace(root, DYNAMICS + '/fractionAngularMomentumRetainedSpheroid', '1.0')
    energy = replace(root, ENERGY, 'bindingEnergy')
    for name, value in [('formFactorDisk', '.5'), ('interactionEnergyFactor', '1.0'),
                        ('spheroidBaryonFractionTransitionMinimum', '.003'),
                        ('spheroidBaryonFractionTransitionMaximum', '.010'),
                        ('finiteDifferenceStepRelative', '.01')]:
        ET.SubElement(energy, name, value=value)
    for name, value in [('pressureCharacteristic', '35000.0'), ('pressureExponent', '.92')]:
        replace(root, 'starFormationRateSurfaceDensityDisks/' + name, value)
    for spec in definition['free_parameters']:
        replace(root, spec['path'], row[spec['short_name']])
    return ET.ElementTree(root)


def write_xml(tree, path):
    ET.indent(tree, space='  ')
    tree.write(path, encoding='UTF-8', xml_declaration=True)


def generate(family, name, n):
    if n < 1 or n > 1024 or n & (n-1):
        raise ValueError('n-eval must be a power of two between 1 and 1024')
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Campaign name must be a single directory name')
    definition = json.loads((family/'family_definition.json').read_text())
    if digest(family/'design_1024.csv') != definition['design_sha256']:
        raise ValueError('Frozen master design checksum mismatch')
    target = family/name
    if target.exists():
        raise FileExistsError(target)
    with (family/'design_1024.csv').open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)[:n]
        fields = ['evaluation_id', *reader.fieldnames]
    target.mkdir()
    with (target/'samples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row['evaluation_id'] = f'{name}-eval-{int(row["design_index"]):04d}'
            writer.writerow(row)
            evaluation = target/'evaluations'/row['evaluation_id']
            evaluation.mkdir(parents=True)
            write_xml(model_changes(row, definition), evaluation/'model_changes.xml')
            write_json(evaluation/'galacticus_input_values.json', {
                k: (v if k == 'evaluation_id' else float(v)) for k,v in row.items()})
    write_json(target/'campaign_design.json', {
        'schema_version': 2, 'campaign_name': name, 'n_eval': n, 'seed': definition['seed'],
        'free_parameters': definition['free_parameters'],
        'free_parameter_short_names': [s['short_name'] for s in definition['free_parameters']],
        'family_definition_sha256': digest(family/'family_definition.json'),
        'master_design_sha256': definition['design_sha256'],
        'samples_sha256': digest(target/'samples.csv'), 'fixed_physics': definition['fixed'],
        'status': 'design_only; prepare on HPC with a migrated base XML before submission',
    })
    (target/'README.md').write_text(
        f'# {name}\n\n{n} equilibrium-energy parameter combinations, taken from the first {n} rows '
        'of `../design_1024.csv`. See the family README for the physics and prior definitions.\n\n'
        f'Generated with:\n\n```bash\npython scripts/generate_equilibrium_energy_campaign.py '
        f'generate --name {name} --n-eval {n}\n```\n\n'
        'No Galacticus runs have been performed by the generator. `evaluations/` contains only '
        'parameter changes and input metadata. Use `prepare` on the HPC to create an executable run set.\n')
    return target


def prepare(campaign, base, executable, destination, trees_per_decade, partition, concurrency, cpus):
    campaign, base, executable = campaign.resolve(), base.resolve(), executable.resolve()
    destination = destination.absolute()
    definition = json.loads((campaign/'campaign_design.json').read_text())
    if digest(campaign/'samples.csv') != definition['samples_sha256']:
        raise ValueError('Campaign samples checksum mismatch')
    parameters = ET.parse(base).getroot()
    if parameters.find(".//mergerTreeConstructor[@value='read']") is not None:
        raise ValueError('Initial pilot preparation supports EPS only, not UNIT')
    if parameters.find('mergerTreeBuildMasses/treesPerDecade') is None:
        raise ValueError('Expected the migrated EPS base with mergerTreeBuildMasses/treesPerDecade')
    if parameters.find("starFormationRateSurfaceDensityDisks[@value='blitz2006']") is None:
        raise ValueError('Expected the Blitz2006 disk star-formation model')
    required = [ACCRETION, DYNAMICS, "mergerTreeOutputter/mergerTreeOutputter[@value='standard']"]
    required += [s['path'].rsplit('/', 1)[0] for s in definition['free_parameters']
                 if not s['path'].startswith(ENERGY)]
    for path in required:
        if parameters.find(path) is None:
            raise ValueError(f'Required parent is missing from the migrated base: {path}')
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Use a fresh run directory; refusing to overwrite existing runs')
    with (campaign/'samples.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    destination.mkdir(parents=True)
    shutil.copy2(base, destination/'base_parameters.xml')
    shutil.copy2(campaign/'samples.csv', destination/'samples.csv')
    shutil.copy2(campaign/'campaign_design.json', destination/'campaign_design.json')
    expected = digest(executable)
    write_json(destination/'run_manifest.json', {
        'executable': str(executable), 'executable_sha256': expected,
        'base_parameters_sha256': digest(base), 'trees_per_decade': trees_per_decade,
        'campaign': str(campaign), 'campaign_definition_sha256': digest(campaign/'campaign_design.json'),
        'generator_sha256': digest(Path(__file__)),
        'catalogs': 'standard outputter enabled; base output epochs and analyses preserved',
        'tree_note': 'Shared base and seed, but EPS trees are regenerated; not pre-generated matched trees.',
    })
    scripts = []
    for row in rows:
        name = row['evaluation_id']
        folder = destination/'evaluations'/name
        folder.mkdir(parents=True)
        # Reconstruct from the frozen sample, rather than trust an edited change file.
        write_xml(model_changes(row, definition), folder/'model_changes.xml')
        shutil.copy2(campaign/'evaluations'/name/'galacticus_input_values.json', folder/'galacticus_input_values.json')
        root = ET.Element('changes')
        replace(root, 'mergerTreeBuildMasses/treesPerDecade', trees_per_decade)
        replace(root, 'outputFileName', 'galacticus.hdf5')
        ET.SubElement(root, 'change', type='update',
                      path="mergerTreeOutputter/mergerTreeOutputter[@value='standard']", value='standard')
        write_xml(ET.ElementTree(root), folder/'run_changes.xml')
        lines = ['#!/bin/bash', 'set -euo pipefail', 'cd "$(dirname "$0")"',
                 ': "${GALACTICUS_DATA_PATH:?Set GALACTICUS_DATA_PATH before submission}"',
                 ': "${SHMR_PATH:?Set SHMR_PATH before submission}"',
                 f'export GALACTICUS_EXEC_PATH={shlex.quote(str(executable.parent))}',
                 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                 'export OMP_PLACES=cores OMP_PROC_BIND=spread',
                 f'echo {shlex.quote(expected + "  " + str(executable))} | sha256sum --check --status',
                 'if [[ -e galacticus.hdf5 || -e RUNNING || -e SUCCESS ]]; then',
                 '  echo "Existing output or run marker: refusing to overwrite" >&2; exit 1; fi',
                 'touch RUNNING', "trap 'rm -f RUNNING' EXIT",
                 f'{shlex.quote(str(executable))} ../../base_parameters.xml model_changes.xml run_changes.xml '
                 '--output-processed-parameters params_used.xml >galacticus.out 2>galacticus.err',
                 'touch SUCCESS', '']
        (folder/'run_eval.sh').write_text('\n'.join(lines))
        scripts.append(f'evaluations/{name}/run_eval.sh')
    (destination/'run_scripts.txt').write_text('\n'.join(scripts)+'\n')
    (destination/'logs').mkdir()
    (destination/'submit.slurm').write_text('\n'.join([
        '#!/bin/bash', '#SBATCH --job-name=equilibrium-E-pilot', f'#SBATCH --partition={partition}',
        '#SBATCH --nodes=1', '#SBATCH --ntasks=1', f'#SBATCH --cpus-per-task={cpus}',
        '#SBATCH --mem-per-cpu=10G', '#SBATCH --time=20:00:00',
        f'#SBATCH --array=0-{len(rows)-1}%{concurrency}',
        '#SBATCH --output=logs/slurm-%A_%a.out', '#SBATCH --error=logs/slurm-%A_%a.err',
        'set -euo pipefail', 'cd "$SLURM_SUBMIT_DIR"',
        'script=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" run_scripts.txt)',
        '[[ -n "$script" ]]', 'bash "$script"', '']))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', type=Path, default=FAMILY)
    sub = parser.add_subparsers(dest='action', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--source-design', type=Path, required=True)
    freeze_parser.add_argument('--seed', type=int, default=42)
    gen = sub.add_parser('generate')
    gen.add_argument('--name', default='pilot_128')
    gen.add_argument('--n-eval', type=int, default=128)
    stage = sub.add_parser('prepare')
    stage.add_argument('--name', default='pilot_128')
    stage.add_argument('--base-parameters', type=Path, required=True)
    stage.add_argument('--executable', type=Path, required=True)
    stage.add_argument('--runs-directory', type=Path, required=True)
    stage.add_argument('--trees-per-decade', type=int, default=500)
    stage.add_argument('--partition', choices=['obs','dmtheory'], default='obs')
    stage.add_argument('--concurrency', type=int, default=3)
    stage.add_argument('--cpus', type=int, default=24)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.family, args.source_design, args.seed)
        print(args.family)
    elif args.action == 'generate':
        print(generate(args.family, args.name, args.n_eval))
    else:
        if min(args.trees_per_decade, args.concurrency, args.cpus) < 1:
            parser.error('Sampling and resource counts must be positive')
        print(prepare(args.family/args.name, args.base_parameters, args.executable,
                      args.runs_directory, args.trees_per_decade, args.partition, args.concurrency, args.cpus))


if __name__ == '__main__':
    main()
