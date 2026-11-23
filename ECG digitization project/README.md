# ECG Digitization Project

Geometry-guided ECG image-to-waveform digitization for a 3x4 short-lead layout with a full lead II rhythm strip.

The shared implementation is `geometry_based_digitizer (3x4 panels).py`. Its header attributes the color masking and dynamic-programming centerline extraction to the custom MDACC v2e engine. The engine is included in this file; the separate MDACC script is not required to run it.

## Install the Python packages

Use Python 3.10 or newer. In this project folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the virtual environment with `.venv\Scripts\activate`.

## Prepare the inputs

This script expects supplied lead geometry and calibration. It does not automatically identify the layout of an arbitrary ECG PDF.

Choose one input structure:

1. `panel_geometry_manifest.csv` plus a `crops/` directory of prepared lead images; or
2. a `png/` directory of full-page images plus corresponding `geometry/*.json` files.

The geometry specifies each lead's printed time window, pixel boundaries, baseline, pixels per millimeter, paper speed and voltage scale. The adapter expects the twelve short-lead panels and a full lead II rhythm strip. See the script's opening docstring for the detailed schema.

## Run the digitizer

```bash
python "geometry_based_digitizer (3x4 panels).py" --render-root "/path/to/prepared-input" --out-dir "digitized_output" --no-native --no-eval-bundle
```

Use `--native-root` with corresponding WFDB records when reference-waveform QC is wanted. `wfdb` is used for reading those records. The original modeling module and evaluation bundle are not required for the basic digitization command above.

## Outputs

- Numerical voltage waveforms sampled at 500 Hz, stored in NumPy arrays.
- Observation masks and sample-status arrays.
- Per-panel and per-record CSV QC tables, a summary JSON, and a preview figure.

Lead II uses the 10-second rhythm strip. The other eleven channels contain their printed 2.5-second windows; unprinted positions are zero-filled in the array and must be interpreted using the supplied observation/status masks. The short lead II panel is retained for QC only.

Correlation and SNR require a corresponding reference waveform. Without one, coverage alone does not establish reconstruction accuracy.

