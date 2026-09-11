# Blanton AGN campaign extraction

`scripts/extract_blanton_agn_campaign.py` calculates after-the-fact AGN and
quiescent fractions from the **full**, unreduced Galacticus HDF5 outputs.  It
does not alter those outputs.  The default sidecar location is

```text
CAMPAIGN/additional_observables/blanton-agn/v1/
├── shards/EVALUATION_ID.hdf5
├── blanton_agn_observables.hdf5
├── blanton_agn_fractions_with_samples.csv.gz
├── blanton_quiescent_fractions_with_samples.csv.gz
└── definitions.json
```

The per-evaluation shards make extraction safe for a SLURM array.  A completed
shard is skipped when its configuration hash matches, so an interrupted array
can be resubmitted.  Shards are written atomically.  Pass `--overwrite` only
when intentionally replacing an existing result.

## Default observable grid

Each selected Galacticus output is treated independently; the defaults are
`z=0` and `z=0.1`, with no pooling.  The default grid contains:

- stellar-mass edges from Suresh & Blanton Paper II:
  `10.0, 10.4, 10.8, 11.2, 11.6, 12.0` in
  `log10(Mstar/Msun)`;
- sSFR cuts at `-11.5`, `-11.0`, and `-10.5` in
  `log10(sSFR/yr^-1)`;
- two sSFR estimates:
  - `instantaneous`: Galacticus disk plus spheroid instantaneous SFR;
  - `halpha_ke12`: intrinsic star-formation H-alpha luminosity from the disk
    plus spheroid, excluding the AGN line component, converted using
    `log10(SFR/Msun yr^-1) = log10(L_Halpha/erg s^-1) - 41.27`;
- Eddington-ratio thresholds `0.001`, `0.01`, and `0.1`, using the strict
  condition `lambda > threshold`;
- star-forming and quiescent populations;
- an inclusive AGN definition and a Galacticus thin-disk-dominated diagnostic.

All galaxies, including both centrals and satellites, enter the applicable
stellar-mass and sSFR bin.

The primary raw definition is `raw_catalog_eta_true_bh` with parent selection
`unrestricted`.  It uses the Galacticus black-hole mass and saved radiative
efficiency:

```text
Lbol = eta_Galacticus Mdot c^2
lambda = Lbol / (1.38e38 erg/s × MBH/Msun)
```

It deliberately has no velocity-dispersion cut.  The additional definitions
are diagnostics:

- `raw_fixed_eta_0p1_true_bh`: same true black-hole mass and accretion rate,
  but fixed `eta=0.1`;
- `hbeta_true_bh`: infer bolometric luminosity from intrinsic Galacticus AGN
  H-beta, but retain the true Galacticus black-hole mass;
- `raw_catalog_eta_sigma_bh`: retain raw Galacticus luminosity but infer the
  black-hole mass from the observed M-sigma relation;
- `hbeta_sigma_bh`: infer both luminosity and black-hole mass in the
  Blanton-like way.

The last two require `sigma_e > 60 km/s`.  Sigma-restricted versions of the
first and third definitions are also stored to isolate the effect of the
parent-sample selection.

For every snapshot the extractor forms
`mergerTreeWeight × nodeSubsamplingWeight` and checks that every value is
within 10% of the median.  This is a guard against accidentally processing a
qualitatively different, strongly weighted halo sample; it is not used to
model small numerical differences between otherwise equal weights.  The
extractor raises an error if the check fails.  After that
check, weights are discarded and every stored fraction uses integer counts.

For each AGN observable the shard stores `n_galaxies`, `n_agn`, and their
literal ratio `f_agn`.  A populated bin with zero AGN is exactly `0.0`.  An
empty population/mass bin is `NaN`.  No floor, logarithm, replacement, sampling
uncertainty, PCA, or emulator transformation is applied at extraction time.
This supports a later bin-by-bin transformation such as
`log10(F_AGN + F_floor)` without conflating true zeros with missing bins.

## HPC extraction

For the current full campaign:

```bash
REPO=/home/arobert2/GalacticusEmu/Galacticus-emulation
CAMPAIGN=/home/arobert2/GalacticusEmu/Galacticus-emulation/runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos
```

A minimal SLURM submission script is:

```bash
#!/bin/bash
#SBATCH --job-name=blanton-agn
#SBATCH --array=0-1023
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/blanton-agn-%A-%a.out
#SBATCH --error=logs/blanton-agn-%A-%a.err

set -euo pipefail

REPO=/home/arobert2/GalacticusEmu/Galacticus-emulation
CAMPAIGN=/home/arobert2/GalacticusEmu/Galacticus-emulation/runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos

python "$REPO/scripts/extract_blanton_agn_campaign.py" evaluation "$CAMPAIGN" \
  --evaluation-index "$SLURM_ARRAY_TASK_ID"
```

Activate the appropriate Python environment or load modules above the `python`
line if the batch environment does not already provide the repository's
dependencies.  Submit from a directory containing `logs/`, or change the two
SLURM log paths.

After all array jobs finish, aggregate the shards:

```bash
python "$REPO/scripts/extract_blanton_agn_campaign.py" aggregate "$CAMPAIGN"
```

Aggregation is strict by default and reports missing evaluation IDs.  During a
partial test, add `--allow-missing`.  Add `--overwrite` when deliberately
regenerating existing aggregate files.

The aggregate HDF5 keeps the fraction and both integer counts as matrices with
shape `(n_evaluations, n_observables)`, plus a descriptor column for every
observable dimension.  The two CSV files prepend the original `samples.csv`
columns to wide fraction matrices.  Counts remain in HDF5; they are useful for
choosing mass bins and floors later without making the CSVs three times wider.

## Standalone smoke test

The same code can be exercised on a single full MAP output:

```bash
python scripts/extract_blanton_agn_campaign.py file /path/to/galacticus.hdf5 \
  --params-xml /path/to/params.xml \
  --evaluation-id MAP \
  --output-shard /tmp/MAP-blanton-agn.hdf5
```

Run `python scripts/extract_blanton_agn_campaign.py evaluation --help` to see
the command-line overrides for redshifts, mass edges, sSFR cuts, lambda cuts,
modes, and the weight tolerance.  Each shard records its exact configuration
hash.  Aggregation permits different weight-check tolerances because they are
validation metadata, while still requiring an exact match for every choice
that affects the observable values.
