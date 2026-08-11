# Demo Emulators

This directory contains small checked-in emulator artifacts used by legacy
interactive slider demos. The matching browser UI files live under
`assets/interactive_*_demo/`.

The recommended campaign-backed app now loads trained products directly from:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/pipeline/emulators
```

Run it with:

```bash
python scripts/serve_pipeline_emulator_app.py
```

Use `playing/` for scratch training outputs, smoke-test artifacts, and candidate
emulators that are not meant to be loaded by default.
