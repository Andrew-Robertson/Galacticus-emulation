import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'compact_tail', Path(__file__).parents[1]/'scripts/analyze_campaign_compact_tail.py')
tail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tail)


def test_weighted_quantile():
    assert tail.quantile(np.array([0., 10.]), np.array([9., 1.]), .5) == pytest.approx(1.)
    assert tail.neff(np.ones(20)) == 20


def test_cdf_strict_threshold_and_histogram_tails():
    pdf, cdf = tail.distribution(np.array([-8., 0., 8.]), np.ones(3))
    assert np.sum(pdf * np.diff(tail.PDF_EDGES)) == pytest.approx(1/3)
    assert cdf[np.argmin(abs(tail.CDF_X))] == pytest.approx(1/3)
    assert cdf[-1] == pytest.approx(2/3)


def test_uniform_size_shift_does_not_change_tail():
    rng = np.random.default_rng(42)
    x = rng.uniform(7,10,4000)
    y = .3*x + rng.normal(0,.3,len(x))
    w = rng.uniform(.1,1,len(x))
    a = tail.residuals(x,y,w)[0]
    b = tail.residuals(x,y+3,w)[0]
    np.testing.assert_allclose(a,b,atol=1e-12,equal_nan=True)


def test_no_interpolation_across_unsupported_bins_or_extrapolation():
    x = np.r_[np.full(25,8.1),np.full(25,8.6),8.4,7.9]
    r = tail.residuals(x,np.ones(len(x)),np.ones(len(x)))[0]
    assert np.isnan(r).all()


def test_missing_distribution_is_not_zero():
    pdf,cdf = tail.distribution(np.array([]),np.array([]))
    assert np.isnan(pdf).all() and np.isnan(cdf).all()


def test_weights_reject_overlap_and_incomplete_coverage(tmp_path):
    with h5py.File(tmp_path/'weights.hdf5','w') as f:
        f['mergerTreeStartIndex'] = [0,2]
        f['mergerTreeCount'] = [2,2]
        f['mergerTreeWeight'] = [2.,3.]
        f['nodeData/nodeSubsamplingWeight'] = [1.,.5,1.,2.]
        np.testing.assert_allclose(tail.expand_weights(f,4),[2,1,3,6])
        f['mergerTreeStartIndex'][1] = 1
        with pytest.raises(ValueError,match='Overlapping'):
            tail.expand_weights(f,4)
        f['mergerTreeStartIndex'][1] = 3
        f['mergerTreeCount'][1] = 1
        with pytest.raises(ValueError,match='incomplete'):
            tail.expand_weights(f,4)


def test_reduced_and_failed_catalogs_are_rejected(tmp_path):
    p = tmp_path/'catalog.hdf5'
    with h5py.File(p,'w') as f:
        f.attrs['statusCompletion'] = 1
    with pytest.raises(RuntimeError,match='Incomplete'):
        tail.read_catalog(p,reader='h5py')
    with h5py.File(p,'a') as f:
        f.attrs['statusCompletion'] = 0
    with pytest.raises(ValueError,match='full catalog required'):
        tail.read_catalog(p,reader='h5py')


def test_snapshot_selection_and_conversion(tmp_path):
    p = tmp_path/'catalog.hdf5'
    unit_type = [('scale','f8'),('description','S16'),('unit','S16'),('h','i4')]
    with h5py.File(p,'w') as f:
        f.attrs['statusCompletion'] = 0
        for name, a in [('Output2',.5),('Output1',1.)]:
            g = f.create_group('Outputs/'+name)
            g.attrs['outputExpansionFactor'] = a
            g['mergerTreeStartIndex'] = [0]
            g['mergerTreeCount'] = [4]
            g['mergerTreeWeight'] = [2.]
            g['nodeData/nodeSubsamplingWeight'] = np.ones(4)
            for key in ['spheroidMassStellar','diskMassStellar','spheroidRadius']:
                radius = key == 'spheroidRadius'
                d = g.create_dataset('nodeData/'+key,data=np.full(4,.001 if radius else 1e9))
                d.attrs['units'] = np.array((3.08567758135e22 if radius else 1.98892e30,b'',b'',0),dtype=unit_type)
    data,meta = tail.read_catalog(p,reader='h5py')
    assert meta['output'] == 'Output1' and meta['redshift'] == 0
    np.testing.assert_allclose(data['spheroidRadius'],1.)
    np.testing.assert_allclose(data['spheroidMassStellar'],1e9)
    with pytest.raises(ValueError,match='tolerance'):
        tail.read_catalog(p,redshift=3,reader='h5py')


def test_units_require_explicit_scale_and_no_little_h():
    class Dataset:
        name = 'radius'
        attrs = {'units': (3.08567758135e22,b'Mpc',b'Mpc',0)}
    assert tail.unit_factor(Dataset(),3.08567758135e19) == pytest.approx(1000.)
    Dataset.attrs['units'] = (3.08567758135e22,b'Mpc/h',b'Mpc',-1)
    with pytest.raises(ValueError,match='h scaling'):
        tail.unit_factor(Dataset(),3.08567758135e19)


def test_sigma_diagnostic_excludes_compact_not_missing_residuals():
    rng = np.random.default_rng(3)
    n = 4000
    mass = 10**rng.uniform(8,10,n)
    radius = 10**rng.normal(0,.1,n)
    compact = np.arange(n)%10 == 0
    radius[compact] *= .001
    data = dict(spheroidMassStellar=mass,diskMassStellar=mass,spheroidRadius=radius,
                weight=np.ones(n),velocityDispersion=np.full(n,250.),
                blackHoleMass=np.where(compact,1e5,1e8))
    result,rows,sigma,worst,info = tail.analyze(data)
    s = next(row for row in sigma if row['count'])
    assert .08 < s['compact_fraction'] < .12
    assert s['log_bh_mean_noncompact'] == pytest.approx(8.)
    assert s['log_bh_mean_supported'] < s['log_bh_mean_noncompact']
    assert len(worst) == 100
    assert info['residual_rows'] < info['selected_rows']
