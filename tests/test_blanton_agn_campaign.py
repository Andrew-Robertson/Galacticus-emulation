from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from galacticus_emu.blanton_agn_campaign import (
    DEFAULT_AGN_CASES,
    aggregate_campaign_shards,
    check_uniform_weights,
    config_hash,
    extraction_config,
    halpha_sfr_msun_per_year,
    log10_ssfr_from_sfr,
    observable_contract_hash,
    read_evaluation_shard,
    summarize_snapshot,
    write_evaluation_shard,
)


def _minimal_config() -> dict[str, object]:
    return extraction_config(
        redshifts=[0.0],
        mass_edges=[10.0, 10.4, 10.8],
        ssfr_cuts=[-11.0],
        lambda_thresholds=[1.0e-3],
        modes=["inclusive"],
    )


def _minimal_rows(evaluation_id: str):
    ratios = np.asarray([1.0e-4, 2.0e-4, 1.0e-2])
    arrays = {
        "log10_stellar_mass_msun": np.asarray([10.2, 10.2, 10.6]),
        "log10_ssfr_instantaneous_per_year": np.asarray([-10.0, -10.0, -12.0]),
        "log10_ssfr_halpha_ke12_per_year": np.asarray([-10.0, -10.0, -12.0]),
        "fraction_adaf": np.zeros(3),
        "valid_accreting_black_hole": np.ones(3, dtype=bool),
        "valid_sigma60": np.asarray([True, False, True]),
        **{case.lambda_key: ratios for case in DEFAULT_AGN_CASES},
    }
    return summarize_snapshot(
        arrays,
        evaluation_id=evaluation_id,
        output_name="Output1",
        target_redshift=0.0,
        redshift=0.0,
        config=_minimal_config(),
    )


def test_uniform_weight_check_uses_ten_percent_default() -> None:
    diagnostics = check_uniform_weights([0.91, 1.0, 1.09])
    np.testing.assert_allclose(diagnostics["tolerance"], 0.10)
    with pytest.raises(ValueError, match="not uniform"):
        check_uniform_weights([1.0, 100.0])


def test_uniform_weight_check_honors_explicit_tolerance() -> None:
    diagnostics = check_uniform_weights([0.95, 1.0, 1.05], tolerance=0.05)
    np.testing.assert_allclose(diagnostics["maximum_fractional_deviation_from_median"], 0.05)
    with pytest.raises(ValueError, match="not uniform"):
        check_uniform_weights([0.94, 1.0, 1.06], tolerance=0.05)


def test_halpha_conversion_and_zero_ssfr_semantics() -> None:
    sfr = halpha_sfr_msun_per_year([10.0**41.27, 0.0, np.nan])
    np.testing.assert_allclose(sfr[:2], [1.0, 0.0])
    ssfr = log10_ssfr_from_sfr(sfr, [1.0e10, 1.0e10, 1.0e10])
    np.testing.assert_allclose(ssfr[0], -10.0)
    assert np.isneginf(ssfr[1])
    assert np.isnan(ssfr[2])


def test_summary_retains_literal_zero_and_distinguishes_empty_denominator() -> None:
    rows, _ = _minimal_rows("test-eval-0000")
    zero = next(
        row
        for row in rows
        if row["agn_definition"] == "raw_catalog_eta_true_bh"
        and row["parent_selection"] == "unrestricted"
        and row["sfr_definition"] == "instantaneous"
        and row["population"] == "star_forming"
        and row["mass_low"] == 10.0
    )
    assert zero["n_galaxies"] == 2
    assert zero["n_agn"] == 0
    assert zero["f_agn"] == 0.0
    assert zero["is_zero"]
    assert not zero["is_missing"]

    missing = next(
        row
        for row in rows
        if row["agn_definition"] == "raw_catalog_eta_true_bh"
        and row["parent_selection"] == "unrestricted"
        and row["sfr_definition"] == "instantaneous"
        and row["population"] == "quiescent"
        and row["mass_low"] == 10.0
    )
    assert missing["n_galaxies"] == 0
    assert np.isnan(missing["f_agn"])
    assert not missing["is_zero"]
    assert missing["is_missing"]


def test_shards_round_trip_and_aggregate_with_samples(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    output_root = campaign / "additional_observables" / "blanton-agn" / "v1"
    output_root.mkdir(parents=True)
    evaluation_ids = ["campaign-eval-0000", "campaign-eval-0001"]
    pd.DataFrame(
        {"evaluation_id": evaluation_ids, "parameter_quantile": [0.25, 0.75]}
    ).to_csv(campaign / "samples.csv", index=False)
    configs = [_minimal_config(), _minimal_config()]
    configs[0]["weight_check"]["maximum_fractional_deviation_from_median"] = 0.05
    assert config_hash(configs[0]) != config_hash(configs[1])
    assert observable_contract_hash(configs[0]) == observable_contract_hash(configs[1])
    for evaluation_id, config in zip(evaluation_ids, configs, strict=True):
        fraction_rows, quiescent_rows = _minimal_rows(evaluation_id)
        metadata = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "config": config,
            "config_hash": config_hash(config),
            "snapshots": [],
        }
        write_evaluation_shard(
            output_root / "shards" / f"{evaluation_id}.hdf5",
            fraction_rows,
            quiescent_rows,
            metadata,
        )

    shard = read_evaluation_shard(output_root / "shards" / f"{evaluation_ids[0]}.hdf5")
    assert shard["metadata"]["evaluation_id"] == evaluation_ids[0]
    assert np.any(shard["fagn"]["f_agn"] == 0.0)
    assert np.any(np.isnan(shard["fagn"]["f_agn"]))

    outputs = aggregate_campaign_shards(campaign, overwrite=True)
    with h5py.File(outputs["hdf5"], "r") as handle:
        assert handle["evaluation_id"].asstr()[:].tolist() == evaluation_ids
        np.testing.assert_allclose(handle["weight_check_tolerance"][:], [0.05, 0.10])
        assert len(set(handle["shard_config_hash"].asstr()[:])) == 2
        assert handle["fagn/f_agn"].shape[0] == 2
        assert "agn_definition" in handle["fagn"]
        assert np.any(handle["fagn/f_agn"][:] == 0.0)
        assert np.any(np.isnan(handle["fagn/f_agn"][:]))
    wide = pd.read_csv(outputs["fagn_csv"])
    assert wide["evaluation_id"].tolist() == evaluation_ids
    assert "parameter_quantile" in wide
