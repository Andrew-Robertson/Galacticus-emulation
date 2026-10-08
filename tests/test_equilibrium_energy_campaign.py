import csv
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('energy_campaign', REPO/'scripts/generate_equilibrium_energy_campaign.py')
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


@pytest.fixture
def family(tmp_path):
    family = tmp_path/'family'
    family.mkdir()
    for name in ('family_definition.json', 'design_1024.csv'):
        shutil.copy2(campaign.FAMILY/name, family/name)
    return family


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_disk_prior_is_reflected_and_energy_prior_is_log_uniform(family):
    definition = json.loads((family/'family_definition.json').read_text())
    specs = {s['short_name']: s for s in definition['free_parameters']}
    assert len(specs) == 21
    assert 'barFractionAngularMomentumRetainedSpheroid' not in specs
    p = specs['barFractionAngularMomentumRetainedDisk']['prior']
    u = np.array([0., .16, .5, .84, 1.])
    v = campaign.inverse(p, u)
    assert np.all(np.diff(v) > 0)
    np.testing.assert_allclose(v, 1-campaign.inverse(p['parameters']['base'], 1-u))
    np.testing.assert_allclose(v[[0,-1]], [.05,.95])
    np.testing.assert_allclose(campaign.inverse(specs['barSpheroidFormFactor']['prior'],
                                              np.array([0.,.5,1.])), [.25,.5,1.])


def test_generated_campaigns_have_exact_nested_values_and_no_overwrite(family):
    small = campaign.generate(family, 'small', 4)
    large = campaign.generate(family, 'large', 8)
    for a,b in zip(rows(small/'samples.csv'), rows(large/'samples.csv')):
        del a['evaluation_id']; del b['evaluation_id']
        assert a == b
    with pytest.raises(FileExistsError):
        campaign.generate(family, 'small', 4)
    with pytest.raises(ValueError):
        campaign.generate(family, '../elsewhere', 4)
    with pytest.raises(ValueError):
        campaign.generate(family, 'not_sobol_power_two', 5)


def test_changes_set_new_physics_without_disk_spheroid_coupling(family):
    target = campaign.generate(family, 'small', 4)
    row = rows(target/'samples.csv')[0]
    tree = ET.parse(target/'evaluations'/row['evaluation_id']/'model_changes.xml')
    changes = {c.get('path'): c for c in tree.getroot()}
    assert changes[campaign.DYNAMICS+'/fractionAngularMomentumRetainedSpheroid'][0].get('value') == '1.0'
    assert changes[campaign.DYNAMICS+'/fractionAngularMomentumRetainedDisk'][0].get('value') == row['barFractionAngularMomentumRetainedDisk']
    assert changes[campaign.ACCRETION+'/angularMomentumSpecificGrowthFromAccretionRate'][0].get('value') == 'false'
    assert changes['starFormationRateSurfaceDensityDisks/pressureCharacteristic'][0].get('value') == '35000.0'
    assert changes[campaign.ENERGY][0].get('value') == 'bindingEnergy'
    assert float(changes[campaign.ENERGY+'/formFactorSpheroid'][0].get('value')) > .25


def fixture_base(path, definition):
    root = ET.Element('parameters')
    paths = [campaign.ACCRETION, campaign.DYNAMICS,
             "starFormationRateSurfaceDensityDisks[@value='blitz2006']",
             "mergerTreeOutputter/mergerTreeOutputter[@value='standard']",
             'mergerTreeBuildMasses/treesPerDecade']
    paths += [s['path'].rsplit('/', 1)[0] for s in definition['free_parameters']
              if not s['path'].startswith(campaign.ENERGY)]
    for item in paths:
        node = root
        for step in item.split('/'):
            child = node.find(step)
            if child is None:
                match = re.fullmatch(r"(\w+)(?:\[@value='([^']+)'\])?", step)
                assert match
                child = ET.SubElement(node, match[1], value=match[2] or 'fixture')
            node = child
    ET.ElementTree(root).write(path)


def test_prepare_writes_separate_runs_and_valid_shell_scripts(family, tmp_path):
    target = campaign.generate(family, 'small', 4)
    base = tmp_path/'base.xml'
    definition = json.loads((target/'campaign_design.json').read_text())
    fixture_base(base, definition)
    exe = tmp_path/'Galacticus.exe'
    exe.write_text('#!/bin/bash\nexit 0\n')
    output = tmp_path/'cluster'/'runs'
    campaign.prepare(target, base, exe, output, 500, 'obs', 3, 24)
    assert (output/'base_parameters.xml').read_bytes() == base.read_bytes()
    assert '#SBATCH --array=0-3%3' in (output/'submit.slurm').read_text()
    assert not list(output.rglob('galacticus.hdf5'))
    for script in [output/'submit.slurm', *output.glob('evaluations/*/run_eval.sh')]:
        subprocess.run(['bash','-n',str(script)], check=True)
    for script in output.glob('evaluations/*/run_eval.sh'):
        text = script.read_text()
        assert 'sha256sum --check' in text
        assert '>galacticus.out 2>galacticus.err' in text
    with pytest.raises(FileExistsError):
        campaign.prepare(target, base, exe, output, 500, 'obs', 3, 24)


def test_modified_master_is_rejected(family):
    with (family/'design_1024.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='checksum'):
        campaign.generate(family, 'modified', 4)
