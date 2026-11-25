"""Submit the extended-host MAP test and old/new posterior comparison on Caltech."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
RUN = ('MCMCs_production_from_exploratory_MAP/standard_observables_plus_emission_line_lfs/'
       'all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr')
RESULT = 'mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr_mcmc_results.hdf5'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--old-campaign', type=Path)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    campaign = args.campaign_root.resolve()
    view = campaign / 'preliminary_mixed_1020_20261118T211707Z'
    pipeline = view / 'gp_mcmc_preliminary'
    new = pipeline / RUN / RESULT
    candidates = ([args.old_campaign] if args.old_campaign else [
        campaign.parent / 'sobol_1024_20p_simpleSizes_moreHalos_reduced',
        campaign.parent / 'sobol_1024_20p_simpleSizes_moreHalos'])
    old = next((root / 'automatedPipeline_transformedParams_finalPaper_definitive' / RUN / RESULT
                for root in candidates if (root / 'automatedPipeline_transformedParams_finalPaper_definitive' / RUN / RESULT).is_file()), None)
    if old is None:
        raise FileNotFoundError('Final paper posterior not found; specify --old-campaign with its campaign directory')
    exe = campaign / 'diagnostics/eval0043_pr1785_20261113T234018Z/source/Galacticus.exe'
    for path in (new, exe, view / 'assembly.json', new.parent / 'maximum_a_posteriori_model_changes.xml'):
        if not path.is_file():
            raise FileNotFoundError(path)
    destination = new.parent / 'bestFitModel_GalacticusRun_hosts2e10'
    if destination.exists():
        raise FileExistsError(f'Inspect existing host-extension attempt: {destination}')
    # Reuse the exact executable already verified for the successful PR1785 retry.
    from run_best_fit_galacticus_model import _sha
    if _sha(exe) != 'ece5f40e1d3a203a2566bd92b4974a0b58b3eacd67c403338643b87bf09674c3':
        raise ValueError('PR1785 executable checksum differs from the tested build')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    submission = pipeline / f'referee_checks_{stamp}'
    submission.mkdir(parents=True)
    config = submission / 'pipeline.yaml'
    shutil.copy2(REPO / 'configs/pipelines/standard_observables_plus_halpha.yaml', config)
    command = [sys.executable, str(REPO / 'scripts/submit_campaign_pipeline_slurm.py'),
               '--config', str(config), '--stage', 'run_combined_best_fit_galacticus',
               '--no-include-dependencies', '--profile', 'mcmc_transformed_parameters',
               '--profile', 'paper_sobral_log_errors', '--profile', 'map_hosts_to_resolution',
               '--set', f'ROOT={view}', '--set', f'PIPELINE_OUTPUT_ROOT={pipeline}',
               '--set', f'BEST_FIT_GALACTICUS_EXECUTABLE={exe}',
               '--script-dir', str(submission / 'scripts'), '--logs-dir', str(submission / 'logs'),
               '--provenance-path', str(submission / 'SUBMISSION_PROVENANCE.md'),
               '--python-command', sys.executable,
               '--partition', 'expansion', '--qos', 'normal', '--job-prefix', 'MAP-hosts2e10']
    if args.submit:
        command.append('--submit')
    subprocess.run(command, check=True)
    compare = [sys.executable, str(REPO / 'scripts/compare_paper_updated_posteriors.py'),
               '--old-results', str(old), '--new-results', str(new),
               '--output-dir', str(submission / 'posterior_comparison')]
    script = submission / 'compare.sbatch'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport MPLBACKEND=Agg\nexport OPENBLAS_NUM_THREADS=1\n'
                      + 'cd ' + shlex.quote(str(REPO)) + '\n' + shlex.join(compare) + '\n')
    command = ['sbatch', '--job-name=paper-new-posteriors', '--partition=expansion', '--qos=normal',
               '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=16G', '--time=04:00:00',
               '--output=' + str(submission / 'comparison-%j.out'),
               '--error=' + str(submission / 'comparison-%j.err'), str(script)]
    if args.submit:
        subprocess.run(command, check=True)
    else:
        print(shlex.join(command))
    print('Old posterior:', old)
    print('New posterior:', new)
    print('Host-extension simulation:', destination)
    print('Comparison outputs:', submission / 'posterior_comparison')


if __name__ == '__main__':
    main()
