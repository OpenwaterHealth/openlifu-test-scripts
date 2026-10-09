# openLIFU-test-scripts

## Disclaimer

CAUTION - The system described here has not been evaluated by the FDA and is not designed for the treatment or diagnosis of any disease. It is provided AS-IS, with no warranties. User assumes all liability and responsibility for identifying and mitigating risks associated with using this software.

Utility and verification scripts for OpenLIFU hardware validation, burn-in, and data analysis.

## Overview

This repository contains:
- Thermal and burn-in verification scripts for transmitter and console behavior
- Self-test and hardware sanity scripts
- Data processing helpers for logs and report generation
- Historical and legacy test content kept for reference

The main active package is `verification`, which contains most test runners and shared test logic.

## Repository Layout

- `verification/`: Primary test package and production requirement test scripts
- `tools/`: Helper utilities and support scripts
- `tutorials/`: Example and tutorial content
- `logs/`: Output logs produced by test runs
- `plots/`: Generated plots and visualizations
- `legacy/`: Older scripts retained for backwards reference
- `deprecated/`: Retired scripts that should not be used for new runs
- `build/`, `dist/`: Build artifacts

## Requirements

- Python 3.10 to 3.12
- OpenLIFU SDK installed and accessible
- Device access permissions for serial/USB interfaces when running against hardware

From `pyproject.toml`:
- Package name: `verification`
- Runtime dependency: `openlifu-sdk`

## Setup

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the package in editable mode from repository root:

```powershell
pip install -e .
```

## Common Test Scripts

### In `verification/`

- `prodreqs_tx_short_verification_test.py`
  - Short transmitter verification path
  - Useful for communication and functional checks

- `prodreqs_tx_long_verification_test.py`
  - Long-duration heating/burn-in workflow scaffold

- `prodreqs_base_class.py`
  - Shared base class, telemetry monitoring, limits, and CLI argument parser

## Running Tests

Because several scripts use package-relative imports, run them from the repository root with module syntax.

Examples:

```powershell
python -m verification.prodreqs_tx_short_verification_test --frequency 400 --num-modules 2 --test-runthrough
```

```powershell
python -m verification.prodreqs_tx_long_verification_test --frequency 400 --num-modules 2
```

## Frequently Used CLI Options

Most production requirement scripts consume argument parsing from `verification/prodreqs_base_class.py`.

Useful options include:
- `--frequency KHZ`
- `--num-modules N`
- `--test-case N`
- `--test-runthrough`
- `--simulate`
- `--external-power`
- `--console-shutoff-temp C`
- `--tx-shutoff-temp C`
- `--ambient-shutoff-temp C`
- `--temperature-check-interval S`
- `--temperature-log-interval S`
- `--log-dir DIR`
- `--verbose` or `--quiet`
- `--skip-logfile`

## Logs and Outputs

- Runtime logs are typically written under `logs/`
- Additional outputs may be created in:
  - `plots/`
  - `verification/*.csv`
  - project root files like `combined_logs.xlsx`

Use `--log-dir` to direct log output to a custom path.

## Safety Notes

When running against physical hardware:
- Ensure emergency stop and power cutoff procedures are known before starting
- Confirm shutoff temperatures and monitoring intervals are appropriate
- Prefer running with conservative thresholds during initial validation

## Troubleshooting

- Import errors:
  - Ensure you run via `python -m verification.<module>` from repo root
  - Confirm `pip install -e .` completed successfully

- Device connection failures:
  - Verify hardware is connected and powered
  - Confirm SDK/device drivers are installed
  - Re-run with `--verbose` for more detailed logs

- No log file generated:
  - Check `--skip-logfile` was not set
  - Confirm write access to the selected log directory

## Development Notes

- Some files in `legacy/` and `deprecated/` are intentionally preserved and may not reflect current behavior
- Keep new work in `verification/` unless intentionally adding migration content
- Maintain script compatibility with Python 3.10-3.12
