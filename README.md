# Quantised CNN Inference Accelerator

This repository implements a small MNIST digit classifier as a fully RTL, BRAM-backed FPGA inference pipeline in SystemVerilog. It includes Verilator-based simulation, PyTorch training and weight export scripts, and a config-driven Vivado experiment flow for area, timing, and performance studies.

The current design accepts one 28x28 grayscale frame over UART, runs inference entirely in hardware, and transmits a single ASCII digit result.

## Project Overview

The accelerator datapath is:

`UART RX -> IFMAP BRAM -> conv2d -> CONV BRAM -> ReLU -> maxpool -> POOL BRAM -> dense -> argmax -> UART TX`

Key characteristics:

- The controller runs stages sequentially with explicit start/done handshakes.
- Intermediate tensors are stored in on-chip BRAM rather than streamed between stages.
- Arithmetic is fixed-point throughout.
- The main exposed architecture knobs are `DATA_WIDTH`, `FRAC_BITS`, `CLK_FREQ_HZ`, `BAUD_RATE`, and `DENSE_OUT_PAR`.

This repository is both an implementation repo and a small FPGA architecture-study repo. The README stays focused on how to run it; the full study write-up lives in `report.md`.

## Prerequisites

- `verilator` for RTL simulation
- a C++ toolchain and `make`
- Python 3
- PyTorch and torchvision for `python/train.py` and `python/quantise.py`
- Vivado for the FPGA implementation flow under `fpga/vivado/`
- `matplotlib` for plot generation in `analysis/fpga_plot.py`

## Quick Start

### Full-pipeline simulation

Build and run the default full inference testbench:

```bash
make run
```

Equivalent explicit target:

```bash
make run_full
```

This builds the top-level RTL with Verilator and runs the C++ full-pipeline testbench in [`tb/tb_full_pipeline.cpp`](/home/keelan/CNN_FPGA/tb/tb_full_pipeline.cpp).

### Batch simulation on MNIST files

Run the batch testbench using the default MNIST raw dataset paths:

```bash
make run_batch
```

Generate per-failure VCDs during a larger batch run:

```bash
make run_batch_vcd
```

Batch outputs are written under `results/verilator/batch/`.

### Train and export weights

Train the reference PyTorch model:

```bash
python3 python/train.py --data-dir ./data --out-dir ./output
```

Quantise the trained model and emit `.mem` files for RTL use:

```bash
python3 python/quantise.py --model-dir ./output --out-dir ./weights --weight-frac 7
```

## Reproducible Experiment and Analysis Flow

### Vivado experiment runner

Run the baseline FPGA experiment:

```bash
make fpga_experiments
```

Run the bit-width sweep wrapper:

```bash
make fpga_experiments_sweep
```

Or invoke the config-driven runner directly:

```bash
python3 experiments/run_fpga_experiments.py --config experiments/configs/baseline_fpga.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_dense_parallel_scaling.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_precision_resource.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_timing_target.json
```

Supported sweep/generic parameters in the current flow:

- `DATA_WIDTH`
- `FRAC_BITS`
- `CLK_FREQ_HZ`
- `BAUD_RATE`
- `DENSE_OUT_PAR`

### Summaries, plots, and latency model

Print an aggregate summary table:

```bash
make fpga_summary EXP=study_dense_parallel_scaling
python3 analysis/fpga_summary.py --experiment-id study_dense_parallel_scaling --include-failed
```

Generate plots from an aggregate dataset:

```bash
make fpga_plots EXP=study_dense_parallel_scaling
python3 analysis/fpga_plot.py --experiment-id study_dense_parallel_scaling --x-param DENSE_OUT_PAR
python3 analysis/fpga_plot.py --experiment-id study_precision_resource --x-param DATA_WIDTH
python3 analysis/fpga_plot.py --experiment-id study_timing_target --x-param CLK_FREQ_HZ
```

Build the dense-parallel analytical model dataset:

```bash
python3 experiments/analyze_latency_model.py --input results/fpga/aggregates/study_dense_parallel_scaling.csv
python3 analysis/fpga_plot.py --aggregate results/fpga/aggregates/study_dense_parallel_scaling_model.csv --x-param DENSE_OUT_PAR
```

### MAC-array framework v2

Run the workload-aware, constraint-aware MAC-array framework:

```bash
make fpga_framework_v2
python3 analysis/run_mac_array_framework.py --config experiments/configs/mac_array_framework_v2.json
```

Run the deterministic unit tests for the new logic:

```bash
make test
```

The v2 flow writes a reproducible results pack under `results/fpga/framework_v2/`, including:

- `framework_report.md`
- `evidence_registry.csv/json`
- `provenance_summary.csv/json`
- `workload_manifest.csv/json`
- `static_architectures.csv/json`
- `workload_evaluations.csv/json`
- `adaptive_evaluations.csv/json`
- `adaptive_constraint_evaluations.csv/json`
- `adaptive_phase_decisions.csv/json`
- `break_even.csv/json`
- `policy_recommendations.csv/json`
- `policy_diagnostics.csv/json`
- `regime_map.csv/json`
- `regime_summary.csv/json`
- `regime_rejection_summary.csv/json`
- `adaptive_rejection_surface.csv/json`
- `regime_insights.csv/json`
- `regime_insights.md`
- `plots/regime_winner_heatmaps.png`
- `plots/adaptive_rejection_surface.png`

Measured vs modelled status in the v2 pack:

- Measured: the checked-in CNN baseline and dense-parallel study aggregates.
- Anchored: the prior MAC-array static evidence carried in `experiments/configs/mac_array_architecture_evidence.json`.
- Modelled: workload-aware metrics, adaptive switching, break-even thresholds, and architecture recommendations.

### Output policy

- New generated outputs should go under `results/`.
- Batch simulation defaults to `results/verilator/batch/`.
- FPGA experiment runs go under `results/fpga/runs/<experiment_id>/<run_id>/`.
- Aggregate CSV/JSON outputs go under `results/fpga/aggregates/`.
- Legacy `artifacts/experiments/` is historical only and should not be used for new runs.

For a headless workflow and CLI notes, see [`docs/remote-ssh.md`](/home/keelan/CNN_FPGA/docs/remote-ssh.md).

## Current Checked-In Results

These numbers are from the current checked-in study aggregates under `results/fpga/aggregates/`.

### Baseline characterization

Baseline point: `DATA_WIDTH=16`, `FRAC_BITS=7`, `DENSE_OUT_PAR=1`, `CLK_FREQ_HZ=100 MHz`.

- Compute latency: `465,732` cycles, `4.65732 ms`
- Throughput: `214.72` inferences/s
- Area: `2720 LUT`, `1004 FF`, `5 DSP`, `6 BRAM`
- Timing: `WNS = +0.017 ns`, estimated `Fmax = 100.17 MHz`
- Stage breakdown: conv `344,962`, ReLU `25,090`, maxpool `32,930`, dense `62,732`, argmax `11`, bubble `7`

The baseline is clearly convolution-bound: conv accounts for about 74% of compute cycles.

### Dense parallel scaling takeaway

The main architecture sweep varies `DENSE_OUT_PAR = 1, 2, 5, 10`.

- Dense cycles scale from `62,732` at `P=1` to `6,275` at `P=10`
- End-to-end compute latency improves only from `465,732` to `409,275` cycles
- Throughput rises from `214.72` to `244.33` inferences/s
- LUT usage grows from `2720` to `19,061`
- DSP usage grows from `5` to `23`
- WNS becomes negative for every point above `P=1`

The checked-in result is the main study conclusion: dense parallelism scales locally, but the whole accelerator remains conv-bound.

### Precision/resource control result

The checked-in precision sweep covers `Q12.5`, `Q14.6`, and `Q16.7` style points:

- Latency stays fixed at `465,732` cycles across the sweep
- LUT usage moves from `2649` to `2720`
- FF usage moves from `900` to `1004`
- BRAM usage moves from `5.0` to `6.0`
- Estimated `Fmax` stays close to `100 MHz`

In the current architecture, precision changes implementation cost more than schedule.

### Timing-target takeaway

The timing-target sweep checks `80 MHz`, `100 MHz`, and `125 MHz` targets.

- `100 MHz` is the only clean timing point in the checked-in data: `WNS = +0.017 ns`
- `80 MHz` shows `WNS = -0.511 ns`
- `125 MHz` shows `WNS = -0.096 ns`
- The `125 MHz` run has no latency/throughput fields in the current aggregate

This makes `100 MHz` the practical reference point for the current checked-in study.

### Framework v2 takeaway

The repo now also includes a reusable MAC-array decision layer under `results/fpga/framework_v2/`. Its current checked-in narrative is:

- `shared` is preferred when DSP/LUT budgets are tight and the 8x8 shared anchor can halve DSP demand from `64` to `32`.
- `baseline` is preferred when throughput targets matter more than raw resource efficiency.
- `replicated` can become the right fixed mode for phase-changing demand on smaller grids, but 8x8 replicated remains ruled out by the preserved Artix-7 implementation-failure evidence.
- the bounded regime map currently finds no adaptive win region, which is reported explicitly rather than hidden.
- adaptive analysis is now constraint-filtered and provenance-tagged; under the tighter pair-aware switching model it is currently conservative rather than over-eager.
- the presentation pack now includes winner heatmaps and an adaptive rejection surface so the no-win story is inspectable instead of implicit.

### Direct MAC-array slice

The repo now also has a smallest-possible directly measurable MAC-array foothold: a standalone baseline-only spatial slice in `hdl/mac_array_direct_top.sv` with a dedicated config at `experiments/configs/study_mac_array_direct_baseline.json`. It is intentionally narrow, but it rides the same Vivado and aggregation pipeline as the rest of the repo.

```bash
make fpga_mac_direct_preview
make fpga_mac_direct_4x4
python3 analysis/run_mac_array_direct_slice.py
```

This path is explicitly separate from the CNN top and from the proxy-only refresh flow:

- it is a direct MAC-array RTL slice,
- it currently measures `baseline` only,
- it feeds comparison artifacts into `results/fpga/framework_v2/direct_slice/`,
- it does not claim direct shared/replicated/adaptive hardware support yet.

The checked-in direct baseline calibration set now covers `4x4`, `8x4`, and `8x8` for `K_DEPTH=32`:

- `4x4`: `16 DSP`, `1061 LUT`, `524 FF`, `WNS = +1.942 ns`, `33` cycles, `15.515 ops/cycle`
- `8x4`: `32 DSP`, `2134 LUT`, `1036 FF`, `WNS = +2.019 ns`, `33` cycles, `31.030 ops/cycle`
- `8x8`: `64 DSP`, `4287 LUT`, `2060 FF`, `WNS = +0.634 ns`, `33` cycles, `62.061 ops/cycle`

The direct comparison pack now shows:

- exact DSP agreement across all three measured baseline points,
- exact direct-slice latency and throughput agreement across all three points,
- lightweight framework LUT underprediction growing from `401` to `2187` LUT across the tested sizes,
- a direct-slice-calibrated linear LUT aid in `results/fpga/framework_v2/direct_slice/` that is reported as a baseline-only caution aid rather than a silent framework replacement.

Useful direct-slice commands:

```bash
make fpga_mac_direct_4x4
make fpga_mac_direct_8x4
make fpga_mac_direct_8x8
make fpga_mac_direct_report
```

### Selective measured refresh

Framework-v2 stays analytical by default, but the repo now also supports a small selective measured-refresh loop grounded in the regime outputs:

```bash
make fpga_refresh_preview
python3 experiments/run_measured_refresh.py --preview-scheduler --scheduler resource-aware --max-concurrent-jobs 2 --cpu-threshold-pct 85 --min-free-mem-gb 4 --per-job-mem-gb 8 --vivado-jobs-override 2
```

This writes `results/fpga/framework_v2/measured_refresh/` with:

- `measured_refresh_manifest.csv/json`: representative regime points chosen for refresh.
- `measured_refresh_queue.csv/json`: single-run generated configs that can be previewed or executed through the scheduler.
- `measured_model_comparison.csv/json`: honest measured-vs-modelled proxy comparison rows.
- `comparison_summary.md`: short narrative of what agrees, what is only proxy evidence, and what is not directly measurable yet.

Current scope is intentionally conservative:

- baseline/shared/replicated refreshes are proxy checks against the existing CNN Vivado study family, not direct MAC-array RTL measurements,
- adaptive has no direct measured-refresh path yet and is marked `not_directly_measurable_with_current_rtl`,
- the generated preview queue is selective and only contains the chosen representative runs, not the full original study configs.

### Optional Vivado queueing

Vivado remains optional. The default fast path is still the analytical/modelled framework.

When you do want measured refreshes, the existing experiment runner now supports an optional conservative resource-aware queue:

```bash
make fpga_queue_preview CFG=experiments/configs/study_dense_parallel_scaling.json
make fpga_experiments_parallel CFG=experiments/configs/study_dense_parallel_scaling.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_dense_parallel_scaling.json --scheduler resource-aware --dry-run --max-concurrent-jobs 2 --cpu-threshold-pct 85 --min-free-mem-gb 4 --per-job-mem-gb 8 --vivado-jobs-override 2
```

The scheduler is conservative and optional:

- `--scheduler resource-aware` enables resource gating
- `--dry-run` previews the deterministic queue without launching Vivado
- `--max-concurrent-jobs` caps concurrent runs
- `--cpu-threshold-pct`, `--min-free-mem-gb`, and `--per-job-mem-gb` control launch gating
- `--vivado-jobs-override` limits threads per Vivado run when desired

Scheduler artifacts are written beside the run directories, including `scheduler_queue.*`, `scheduler.log`, and `scheduler_summary.json`.

## Repository Layout

```text
.
├─ hdl/                 RTL modules
├─ tb/                  SystemVerilog and C++ testbenches
├─ python/              training, quantisation, and utility scripts
├─ weights/             fixed-point weight and input memory files
├─ fpga/vivado/         Vivado batch scripts and report parsing
├─ experiments/         experiment configs and orchestration scripts
├─ analysis/            summary, plotting, and framework-v2 analysis tools
├─ docs/                auxiliary documentation
├─ results/             canonical generated-output root
├─ report.md            full study write-up
├─ Makefile             primary CLI entrypoints
└─ README.md
```

## Notes and Limitations

- The current top-level design is single-frame and sequential; stages are not overlapped.
- Reported study latency is compute-only (`frame_loaded -> tx_start`), not UART-dominated wall-clock end-to-end time.
- The framework-v2 adaptive outputs are analytical and clearly labelled as modelled, not hardware-measured reconfiguration data.
- Batch simulation assumes MNIST raw files at `data/MNIST/raw/` unless you override `ARGS`.
- The repo includes UART-connected hardware logic, but it does not currently document a full board programming or flashing workflow.
- Fixed-point scaling and exported `.mem` files must stay aligned with the RTL parameters.

For the complete study narrative, use [`report.md`](/home/keelan/CNN_FPGA/report.md). For the recommended experiment set and plotting sequence, use [`experiments/FPGA_STUDY_SUITE.md`](/home/keelan/CNN_FPGA/experiments/FPGA_STUDY_SUITE.md).
