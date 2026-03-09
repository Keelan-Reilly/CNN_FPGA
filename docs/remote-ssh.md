# Remote SSH Development Notes

Use this repository in a headless workflow and keep generated outputs under `results/`.

## Typical CLI flow
- Build and run full sim: `make run`
- Batch sim output: `make run_batch`
- Experiment wrapper: `./experiments/run_batch_experiment.sh [run_name]`
- Vivado batch flow: `./fpga/vivado/run_batch.sh`
- FPGA experiment runner (config-driven): `python3 experiments/run_fpga_experiments.py --config experiments/configs/baseline_fpga.json`

## Output policy
- Canonical output root for new runs: `results/`
- Legacy `artifacts/experiments/` is retained only for historical data; do not write new outputs there.

## FPGA metrics workflow
- Configs live in `experiments/configs/`
- Supported top-level generic sweep parameters:
  - `DATA_WIDTH`, `FRAC_BITS`, `CLK_FREQ_HZ`, `BAUD_RATE`
- Per-run outputs:
  - `results/fpga/runs/<experiment_id>/<run_id>/`
- Aggregates:
  - `results/fpga/aggregates/<experiment_id>.csv`
  - `results/fpga/aggregates/<experiment_id>.json`

## Analysis / plotting (CLI only)
- Summary table:
  - `python3 analysis/fpga_summary.py --experiment-id <experiment_id> --include-failed`
- Plot generation:
  - `python3 analysis/fpga_plot.py --experiment-id <experiment_id>`
  - Optional:
    - `--x-param <param>`
    - `--group-by <param>`
    - `--filter KEY=VALUE` (repeatable)
  - Requires `matplotlib` in the Python environment.
- Plot output:
  - `results/fpga/plots/<experiment_id>/`

Current plots are based only on available aggregate metrics (`lut`, `ff`, `dsp`, `bram`, `wns_ns`, `fmax_mhz_est`).
Throughput/latency/model-vs-measurement plots require additional collected metrics and are intentionally not fabricated.
