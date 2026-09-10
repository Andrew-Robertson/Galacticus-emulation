from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.stats import beta

from galacticus_emu.agn_demographics import (
    DEFAULT_LEDD_COEFFICIENT,
    FractionBin,
    SnapshotCatalog,
    SwitchedDiskParameters,
    WeightedFraction,
    accretion_quantities,
    activity_mask,
    combine_host_properties,
    expand_tree_values,
    first_vlen_element,
    fit_trend_metrics,
    load_snapshot_catalog,
    pool_snapshot_catalogs,
    select_outputs,
    summarize_fractions,
    switched_disk_adaf_fraction,
    switched_disk_parameters_from_xml,
    weighted_jeffreys_fraction,
)


def test_combine_host_properties_clips_negative_sfr_and_maps_zero_to_minus_infinity() -> None:
    host = combine_host_properties(
        disk_stellar_mass_msun=[8.0, 10.0, 0.0],
        spheroid_stellar_mass_msun=[2.0, 0.0, 0.0],
        disk_sfr_msun_per_gyr=[2.0e9, 1.0, 0.0],
        spheroid_sfr_msun_per_gyr=[-1.0e9, -2.0, 0.0],
    )
    np.testing.assert_allclose(host.stellar_mass_msun, [10.0, 10.0, 0.0])
    np.testing.assert_allclose(host.sfr_msun_per_year, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(host.log10_ssfr_per_year[0], -1.0)
    assert np.isneginf(host.log10_ssfr_per_year[1])
    assert np.isnan(host.log10_ssfr_per_year[2])
    assert host.negative_sfr_count == 1
    assert host.nonpositive_sfr_count == 2


def test_accretion_quantities_use_catalog_units_and_keep_native_x_separate() -> None:
    units_in_si = 6.302397367723697e13
    result = accretion_quantities(
        black_hole_mass_msun=[1.0],
        mdot_rest_msun_per_gyr=[1.0],
        radiative_efficiency_catalog=[0.1],
        mdot_units_in_si=units_in_si,
        galacticus_mdot_edd_per_mbh=2.2206176144141274,
    )
    expected_power = units_in_si * 299792458.0**2 * 1.0e7
    np.testing.assert_allclose(result.rest_mass_power_erg_s, [expected_power], rtol=1.0e-14)
    np.testing.assert_allclose(result.eddington_ratio, [0.1 * expected_power / 1.26e38])
    np.testing.assert_allclose(result.accretion_rate_eddington_paper, [expected_power / 1.26e38])
    np.testing.assert_allclose(
        result.accretion_rate_eddington_galacticus, [1.0 / 2.2206176144141274]
    )
    assert not np.isclose(
        result.accretion_rate_eddington_paper[0],
        result.accretion_rate_eddington_galacticus[0],
        rtol=1.0e-5,
    )


def test_fixed_efficiency_changes_selected_lambda_but_preserves_catalog_lambda() -> None:
    result = accretion_quantities(
        [1.0e8], [2.0e8], [0.2], fixed_radiative_efficiency=0.1
    )
    np.testing.assert_allclose(result.radiative_efficiency_used, [0.1])
    np.testing.assert_allclose(result.eddington_ratio_catalog, 2.0 * result.eddington_ratio)


def test_switched_disk_fraction_has_low_and_high_adaf_branches() -> None:
    xmin, xmax = 0.01, 20.0
    x = np.asarray([1.0e-8, xmin, 1.0, xmax, 1.0e8, 0.0])
    fraction = switched_disk_adaf_fraction(
        x,
        thin_disk_minimum=xmin,
        thin_disk_maximum=xmax,
        transition_width=0.1,
        valid_accreting_black_hole=[True, True, True, True, True, False],
    )
    assert fraction[0] > 0.999
    np.testing.assert_allclose(fraction[1], 0.5, atol=1.0e-12)
    assert fraction[2] < 1.0e-10
    np.testing.assert_allclose(fraction[3], 0.5, atol=1.0e-12)
    assert fraction[4] > 0.999
    # This reproduces Galacticus' no-BH/non-accreting return value, but is not
    # itself sufficient to classify an object as thin-disk dominated.
    assert fraction[5] == 0.0


def test_thin_mode_requires_valid_accreting_bh_and_is_separate_from_lambda_cut() -> None:
    ratio = [0.1, 0.1, 1.0e-4]
    inclusive = activity_mask(ratio, 1.0e-3)
    thin = activity_mask(
        ratio,
        1.0e-3,
        mode_criterion="thin_disk_dominated",
        fraction_adaf=[0.1, 0.1, 0.1],
        valid_accreting_black_hole=[True, False, True],
    )
    np.testing.assert_array_equal(inclusive, [True, True, False])
    np.testing.assert_array_equal(thin, [True, False, False])


def test_weighted_jeffreys_fraction_reduces_to_binomial_for_unit_weights() -> None:
    estimate = weighted_jeffreys_fraction([True, False, False, False], np.ones(4))
    alpha = 0.5 * (1.0 - 0.682689492137)
    np.testing.assert_allclose(estimate.fraction, 0.25)
    np.testing.assert_allclose(estimate.effective_n, 4.0)
    np.testing.assert_allclose(estimate.lower, beta.ppf(alpha, 1.5, 3.5))
    np.testing.assert_allclose(estimate.upper, beta.ppf(1.0 - alpha, 1.5, 3.5))
    assert not estimate.is_upper_limit


def test_zero_weighted_numerator_is_flagged_as_upper_limit() -> None:
    estimate = weighted_jeffreys_fraction([False, False], [1.0, 3.0])
    np.testing.assert_allclose(estimate.effective_n, 1.6)
    assert estimate.fraction == 0.0
    assert 0.0 < estimate.upper < 1.0
    assert estimate.is_upper_limit


def test_expand_tree_values_validates_complete_nonoverlapping_ranges() -> None:
    expanded = expand_tree_values(5, [0, 2], [2, 3], [0.2, 0.7])
    np.testing.assert_allclose(expanded, [0.2, 0.2, 0.7, 0.7, 0.7])
    try:
        expand_tree_values(5, [0, 1], [2, 4], [0.2, 0.7])
    except ValueError as error:
        assert "Overlapping" in str(error) or "invalid" in str(error)
    else:
        raise AssertionError("overlapping/invalid tree ranges were accepted")


def test_first_vlen_element_uses_index_zero_and_marks_empty() -> None:
    values = np.empty(3, dtype=object)
    values[:] = [np.asarray([3.0, 30.0]), np.asarray([]), np.asarray([5.0])]
    result = first_vlen_element(values)
    np.testing.assert_allclose(result[[0, 2]], [3.0, 5.0])
    assert np.isnan(result[1])


def test_select_outputs_honors_redshift_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "outputs.hdf5"
    with h5py.File(path, "w") as handle:
        outputs = handle.create_group("Outputs")
        outputs.create_group("Output1").attrs["outputExpansionFactor"] = 1.0
        outputs.create_group("Output2").attrs["outputExpansionFactor"] = 0.5
        assert select_outputs(outputs, [0.01], tolerance=0.02)[0][1] == "Output1"
        try:
            select_outputs(outputs, [0.1], tolerance=0.02)
        except ValueError as error:
            assert "outside tolerance" in str(error)
        else:
            raise AssertionError("redshift outside tolerance was accepted")


def test_switched_disk_parameters_are_read_from_params_xml(tmp_path: Path) -> None:
    path = tmp_path / "params.xml"
    path.write_text(
        "<parameters><accretionDisks value='switched'>"
        "<accretionRateThinDiskMinimum value='0.02'/>"
        "<accretionRateThinDiskMaximum value='none'/>"
        "<accretionRateTransitionWidth value='0.3'/>"
        "</accretionDisks></parameters>"
    )
    parameters = switched_disk_parameters_from_xml(path)
    assert parameters == SwitchedDiskParameters(0.02, None, 0.3)


def test_summary_keeps_threshold_and_mode_dimensions_separate() -> None:
    snapshot = SnapshotCatalog(
        target_redshift=0.0,
        redshift=0.0,
        output_name="Output1",
        columns={
            "log10_stellar_mass_msun": np.asarray([10.2, 10.2, 10.2, 10.2]),
            "log10_ssfr_per_year": np.asarray([-10.0, -10.0, -12.0, -12.0]),
            "weight": np.ones(4),
            "eddington_ratio": np.asarray([0.2, 0.02, 0.2, 0.02]),
            "fraction_adaf": np.asarray([0.1, 0.9, 0.1, 0.9]),
            "valid_accreting_black_hole": np.ones(4, dtype=bool),
        },
        diagnostics={},
    )
    rows = summarize_fractions(
        snapshot,
        mass_edges=[10.0, 10.4],
        thresholds=[0.1],
        mode_criteria=["inclusive", "thin_disk_dominated"],
    )
    assert len(rows) == 4
    for population in ("star_forming", "quiescent"):
        inclusive = next(row for row in rows if row.population == population and row.mode_criterion == "inclusive")
        thin = next(row for row in rows if row.population == population and row.mode_criterion == "thin_disk_dominated")
        assert inclusive.estimate.raw_successes == 1
        assert thin.estimate.raw_successes == 1


def test_pool_snapshot_catalogs_concatenates_without_losing_columns() -> None:
    first = SnapshotCatalog(0.0, 0.0, "Output1", {"weight": np.asarray([1.0, 2.0])}, {})
    second = SnapshotCatalog(0.1, 0.1, "Output2", {"weight": np.asarray([3.0])}, {})
    pooled = pool_snapshot_catalogs([first, second])
    np.testing.assert_allclose(pooled.columns["weight"], [1.0, 2.0, 3.0])
    assert pooled.diagnostics["snapshots_pooled"] == 2
    assert pooled.diagnostics["redshift_min"] == 0.0
    assert pooled.diagnostics["redshift_max"] == 0.1


def test_fit_trend_metrics_operates_on_single_preselected_definition() -> None:
    rows: list[FractionBin] = []
    for population, slope in (("star_forming", 0.2), ("quiescent", -1.0)):
        for x in (10.2, 10.6, 11.0):
            fraction = 10.0 ** (slope * (x - 11.0) - (1.0 if population == "star_forming" else 2.0))
            rows.append(
                FractionBin(
                    0.0,
                    0.0,
                    "Output1",
                    "inclusive",
                    1.0e-3,
                    population,  # type: ignore[arg-type]
                    x - 0.2,
                    x + 0.2,
                    x,
                    WeightedFraction(fraction, fraction, fraction, 10.0, 1.0, 1.0, fraction, 10, 1, False),
                )
            )
    metrics = fit_trend_metrics(rows)
    np.testing.assert_allclose(metrics.star_forming_slope, 0.2)
    np.testing.assert_allclose(metrics.quiescent_slope, -1.0)
    np.testing.assert_allclose(metrics.separation_at_log10_mass_11, 1.0)


def test_loader_uses_tree_times_node_weights_vlen_zero_and_catalog_units(tmp_path: Path) -> None:
    path = tmp_path / "mini.hdf5"
    vlen = h5py.vlen_dtype(np.dtype("float64"))
    with h5py.File(path, "w") as handle:
        output = handle.create_group("Outputs/Output1")
        output.attrs["outputExpansionFactor"] = 1.0
        output.create_dataset("mergerTreeStartIndex", data=[0, 2])
        output.create_dataset("mergerTreeCount", data=[2, 1])
        output.create_dataset("mergerTreeWeight", data=[2.0, 3.0])
        output.create_dataset("mergerTreeIndex", data=[10, 11])
        data = output.create_group("nodeData")
        data.create_dataset("nodeIndex", data=[0, 1, 2])
        data.create_dataset("nodeIsIsolated", data=[1, 0, 1])
        data.create_dataset("nodeSubsamplingWeight", data=[1.0, 4.0, 2.0])
        data.create_dataset("diskMassStellar", data=[1.0e10, 2.0e10, 3.0e10])
        data.create_dataset("spheroidMassStellar", data=np.zeros(3))
        data.create_dataset("diskStarFormationRate", data=[1.0e9, 0.0, 0.0])
        data.create_dataset("spheroidStarFormationRate", data=[0.0, -1.0, 0.0])
        data.create_dataset("blackHoleMass", data=[1.0e7, 2.0e7, 3.0e7])
        for name, rows in (
            ("massAccretionRateBlackHoles", [[2.0, 99.0], [0.0], [3.0]]),
            ("radiativeEfficiencyBlackHoles", [[0.1, 0.9], [0.2], [0.3]]),
            ("powerJetBlackHoles", [[4.0, 40.0], [5.0], [6.0]]),
        ):
            dataset = data.create_dataset(name, (3,), dtype=vlen)
            for index, row in enumerate(rows):
                dataset[index] = row
            dataset.attrs["unitsInSI"] = (
                6.302397367723697e19 if name == "powerJetBlackHoles" else 6.302397367723697e13
            )
    with h5py.File(path, "r") as handle:
        catalog = load_snapshot_catalog(
            handle["Outputs/Output1"],
            target_redshift=0.0,
            switched_disk_parameters=SwitchedDiskParameters(),
        )
    np.testing.assert_allclose(catalog.columns["weight"], [2.0, 8.0, 6.0])
    np.testing.assert_allclose(catalog.columns["mdot_rest_msun_per_gyr"], [2.0, 0.0, 3.0])
    np.testing.assert_allclose(catalog.columns["radiative_efficiency_catalog"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(catalog.columns["power_jet_erg_s"][0], 4.0 * 6.302397367723697e26)
    assert catalog.diagnostics["negative_total_sfr_count_retained"] == 1
    assert catalog.diagnostics["nonpositive_total_sfr_count_retained"] == 2
    assert catalog.diagnostics["massAccretionRateBlackHoles_multiple_rows"] == 1


def test_paper_default_eddington_coefficient_is_exact() -> None:
    assert DEFAULT_LEDD_COEFFICIENT == 1.26e38
