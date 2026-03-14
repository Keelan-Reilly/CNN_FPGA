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

## Repository Layout

```text
.
├─ hdl/                 RTL modules
├─ tb/                  SystemVerilog and C++ testbenches
├─ python/              training, quantisation, and utility scripts
├─ weights/             fixed-point weight and input memory files
├─ fpga/vivado/         Vivado batch scripts and report parsing
├─ experiments/         experiment configs and orchestration scripts
├─ analysis/            summary and plotting tools for aggregate results
├─ docs/                auxiliary documentation
├─ results/             canonical generated-output root
├─ report.md            full study write-up
├─ Makefile             primary CLI entrypoints
└─ README.md
```

## Notes and Limitations

- The current top-level design is single-frame and sequential; stages are not overlapped.
- Reported study latency is compute-only (`frame_loaded -> tx_start`), not UART-dominated wall-clock end-to-end time.
- Batch simulation assumes MNIST raw files at `data/MNIST/raw/` unless you override `ARGS`.
- The repo includes UART-connected hardware logic, but it does not currently document a full board programming or flashing workflow.
- Fixed-point scaling and exported `.mem` files must stay aligned with the RTL parameters.

For the complete study narrative, use [`report.md`](/home/keelan/CNN_FPGA/report.md). For the recommended experiment set and plotting sequence, use [`experiments/FPGA_STUDY_SUITE.md`](/home/keelan/CNN_FPGA/experiments/FPGA_STUDY_SUITE.md).
