<p align="center">
  <img src="assets/galacticemu_logo.png" alt="GalacticEmu logo" width="520">
</p>

# GalacticEmu

GalacticEmu is a set of Python tools for building emulators of
[Galacticus](https://github.com/galacticusorg/galacticus) outputs and using
them to calibrate galaxy-formation model parameters. The current workflows can
construct training campaigns, reduce Galacticus outputs, train and
cross-validate Gaussian-process emulators, and sample emulator-based
likelihoods with MCMC.

Semi-analytic models follow the formation and evolution of galaxy populations
by combining dark-matter halo assembly histories with simplified,
physically motivated descriptions of processes such as gas cooling, star
formation, feedback, chemical enrichment, and galaxy mergers. They are much
less computationally expensive than full hydrodynamical simulations and are
used both to investigate the physics of galaxy formation and to predict galaxy
observables for cosmological surveys.

The parameters controlling these physical prescriptions cannot all be fixed
from first principles and must be calibrated against observations. A careful
calibration involves comparing several observables while exploring a
high-dimensional parameter space, which can require far more model evaluations
than it is practical to run directly with Galacticus. An emulator learns the
mapping between model parameters and predicted observables from a finite set of
training runs, allowing that mapping to be evaluated rapidly during validation,
sensitivity studies, and MCMC calibration.

**[Explore the interactive emulator](https://galacticus-emulation.onrender.com/observables)**
to see how changing Galacticus parameters affects several predicted
observables. The hosted demonstration uses pre-trained emulators and does not
require a local installation.

The repository is under active development. It began as the workflow used for
*Emulator-Assisted Calibration of a Semi-Analytic Galaxy Formation Model for
the Roman Galaxy Redshift Survey*, and also provides a basis for future
Galacticus calibration work. The manuscript, figure data, and instructions for
inspecting or reproducing that particular analysis are in the
[paper directory](paper/README.md). A tagged release will preserve the version
associated with the paper while development continues on the main branch.

Worked example notebooks are planned during peer review. They will give a
guided account of the method by following a low-redshift stellar mass function
from archived Galacticus training outputs through emulator validation,
construction, and MCMC calibration.

## Workflow

A typical GalacticEmu calculation has the following stages:

1. Define the parameters to vary and choose the training points.
2. Generate the Galacticus parameter changes and commands for each point.
3. Run Galacticus locally or on an HPC system.
4. Extract the outputs needed for emulation from the full Galacticus files.
5. Train and cross-validate emulators for the chosen observables.
6. Use the emulators in likelihood analyses, including MCMC parameter
   calibration.

The main workflow used for the paper is described by
[`configs/pipelines/standard_observables_plus_halpha.yaml`](configs/pipelines/standard_observables_plus_halpha.yaml).
It is also a useful concrete example of how these stages fit together.

## Repository Layout

- `src/galacticus_emu/` contains the Python package code.
- `scripts/` contains command-line programs for campaign construction, output
  reduction, emulator training, validation, MCMC, and plotting.
- `configs/pipelines/` contains YAML descriptions of multi-stage workflows.
- `paper/` contains the manuscript and the material specific to reproducing
  its figures and tables.
- `demo_emulators/` contains small pre-trained emulator bundles, allowing the
  interactive emulator interface to run without first training new models.
- `data/` contains observational data used by some of the comparison scripts.
- `runs/` is the default location for generated campaigns and results, and is
  excluded from git.

Older or superseded programs are kept in `scripts/legacy/`; exploratory
science diagnostics are in `scripts/playground/`. The local interactive
emulator is described in
[`docs/interactive_emulator_app.md`](docs/interactive_emulator_app.md).

## Installation

GalacticEmu requires Python 3.11 or later. One way to set up a dedicated Conda
environment is:

```bash
conda create -n galacticemu python=3.11 pip
conda activate galacticemu
python -m pip install -e .
```

The final command reads the dependencies from `pyproject.toml` and installs
them into the active Conda environment. The `-e` option installs GalacticEmu
in editable mode, so changes made in the repository are immediately available
without reinstalling it.

Some paper figures require `getdist` for posterior plots and `pypdf` for PDF
composition. If you want to reproduce those figures, install the `paper`
extra:

```bash
python -m pip install -e ".[paper]"
```

The dependency versions are not currently locked, so these commands create a
working environment rather than reproducing the exact software environment
used for the paper.

## External Requirements And Paths

Running a new training campaign requires a Galacticus executable, its data
files, and the Galacticus parameter files used to define the calculation. Some
of the emission-line workflows also use a separate set of dust-processing
utilities. These are external to GalacticEmu.

Several scripts refer to these locations through the following environment
variables. The names are conventions used by this project; a given workflow
may need only a subset of them.

- `GALACTICUS_EXEC_PATH`: directory containing `Galacticus.exe`
- `GALACTICUS_DATA_PATH`: Galacticus datasets directory
- `GALACTICUS_PARAMETER_FILES`: repository containing the run-definition and
  parameter-change files
- `GALACTICUS_DUST_ROOT`: repository containing the dust-processing code
- `SHMR_PATH`: external stellar-to-halo mass relation data used by some older
  workflows

Machine-specific values can instead be recorded in `config/paths.toml`. Start
from [`config/paths.example.toml`](config/paths.example.toml) and edit the copy
for the machine on which the calculation will run.

The exact parameter-file and dust-processing versions needed for the paper are
discussed in [`paper/README.md`](paper/README.md). Public, versioned copies of
those two dependencies still need to be prepared before the complete paper
workflow can be reproduced from public sources alone.

## Inspecting The Workflow

The following command reads the main pipeline YAML, resolves its structure,
and lists the available stages. It does not execute any stage:

```bash
python scripts/run_campaign_pipeline.py \
  --config configs/pipelines/standard_observables_plus_halpha.yaml \
  --list
```

## Running A Pipeline

To inspect the commands for one stage of the main example pipeline, run:

```bash
python scripts/run_campaign_pipeline.py \
  --config configs/pipelines/standard_observables_plus_halpha.yaml \
  --stage train_standard_emulators
```

The pipeline runner is a dry run by default. Add `--execute` only after
checking the rendered commands and configuring the required paths and input
data. Campaigns can be generated and run on the same system or generated on
one machine and moved to another; the appropriate choice depends on the local
HPC setup.

Full Galacticus outputs contain galaxy catalogues and can be much larger than
the data needed to emulate observables that Galacticus calculated during the
training runs. To retain the analysis outputs and lightweight provenance while
excluding the large `/Outputs` catalogue group, make a reduced copy of a
campaign with:

```bash
python scripts/extract_campaign_hdf5_groups.py \
  runs/campaigns/<campaign-name> \
  --output-root runs/campaigns/<campaign-name>_reduced \
  --copy-sidecars
```

Use `python <script> --help` for the options accepted by an individual command.

## License

The GalacticEmu source code is available under the
[BSD 3-Clause License](LICENSE). The planned archival data release will state
its own license. Bundled third-party manuscript style files under `paper/`
retain the copyright and licensing notices in those files.
