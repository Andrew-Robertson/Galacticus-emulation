import importlib.util
from pathlib import Path
import numpy as np

p = Path(__file__).resolve().parents[1]/'scripts/pilot_observable_priors.py'
spec = importlib.util.spec_from_file_location('priors', p)
priors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(priors)


def test_transform_does_not_floor_empty_bins():
    result = priors.transform([0., -1., 10., np.inf], True)
    assert np.isnan(result[[0, 1, 3]]).all()
    assert result[2] == 1.
    result = priors.transform([0., -2., 3.], False, missing_zero=True)
    assert np.isnan(result[0])
    np.testing.assert_equal(result[1:], [-2., 3.])


def test_nonlog_fraction_zero_retained():
    np.testing.assert_equal(priors.transform([0., .5, 1.], False), [0., .5, 1.])
