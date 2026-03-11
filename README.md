# Quantised CNN Inference Accelerator (FPGA / RTL)

This project implements a digit-classification CNN entirely in **SystemVerilog RTL**.  
Inference is executed as a **staged, BRAM-backed hardware pipeline**:

conv → ReLU → maxpool → dense → argmax

An input **28×28 grayscale image (784 bytes)** is received over UART, written into on-chip memory, and processed fully in hardware.  
After inference, a single **ASCII digit (`'0'–'9'`)** corresponding to the argmax output is transmitted over UART TX.

Cycle-accurate simulation is performed using **Verilator**, with a C++ testbench and Python reference model used to verify correctness against a quantised software implementation.

This project demonstrates how fixed-point neural network inference can be implemented **entirely in RTL**, with explicit control over dataflow, memory access, and latency.

---

### Key Results

- **Compute latency (frame_loaded → tx_start):** ~465,732 cycles  
  → **~4.66 ms @ 100 MHz** (pipeline only)

- **End-to-end latency (UART RX image + compute + UART TX):** ~7,281,850 cycles @ 115,200 baud  
  → **~72.8 ms @ 100 MHz** (I/O dominates)

- **Accuracy (MNIST test):**
  - **Float PyTorch model (early convergence, 1 Epoch):** 94.26%
  - **Quantised RTL (Verilator batch, 1,000 images):** 92.20%

- **Top-level module:** `hdl/top_level.sv`  
- **Verification:** Cycle-accurate Verilator + fixed-point Python golden model

---
### Architecture Pipeline

UART RX
↓
IFMAP BRAM (28×28)
↓
Conv2D
↓
CONV BRAM
↓
ReLU
↓
MaxPool
↓
POOL BRAM
↓
Dense
↓
Argmax
↓
UART TX

---


## Simulation and Verification

### Verilator

- End-to-end C++ testbench driving UART input and observing UART output.
- Per-stage cycle counts are reported during simulation.
- Tested with **Verilator 5.018+**.

### Python Reference

- PyTorch model is trained and quantised.
- Fixed-point math is mirrored in software.
- Hardware logits and predictions are compared against software results.

---


# Quick Start

## 1) Build and run full inference

```bash
make run
```
Expected output:

```bash
---- Performance Report ----
Frame cycles: 435939
 conv  = 322963
 relu  = 18819
 pool  = 9411
 flat  = 1
 dense = 31373
 argmx = 12
----------------------------
TX byte: 0x37  (7)
Predicted digit (numeric): 7
```

## 2) FPGA deployment and UART test
1.	Build/flash your bitstream.
2.	Send an image and read back the prediction:

Output(Example)
``` bash
[SW] predicted 7
[HW] trial 1: 7
[HW] trial 2: 7
Agreement: 5/5
```

## FPGA Experiment Automation (Vivado Batch)

- Baseline run:
  - `python3 experiments/run_fpga_experiments.py --config experiments/configs/baseline_fpga.json`
- Sweep run:
  - `python3 experiments/run_fpga_experiments.py --config experiments/configs/sweep_bitwidth.json`
- Makefile wrappers:
  - `make fpga_experiments`
  - `make fpga_experiments_sweep`

Outputs are written only under `results/`:
- per-run: `results/fpga/runs/<experiment_id>/<run_id>/`
- aggregates: `results/fpga/aggregates/<experiment_id>.json` and `.csv`

Each run writes:
- `config_resolved.json`
- `run_meta.json`
- `vivado.log`
- `reports/`
- `metrics.json`

Supported sweep/build parameters (top-level generics):
- `DATA_WIDTH`
- `FRAC_BITS`
- `DENSE_OUT_PAR`
- `CLK_FREQ_HZ`
- `BAUD_RATE`

Metric notes:
- `WNS` is parsed from Vivado post-route timing summary.
- `fmax_mhz_est` is computed from `clock_period_target_ns` and parsed `WNS`:
  `fmax ≈ 1000 / (target_period - WNS)` when defined.

## FPGA Analysis and Plotting

CLI summary (includes failed runs):
- `python3 analysis/fpga_summary.py --experiment-id baseline_fpga --include-failed`
  - Default experiment id is `baseline_fpga` if `--experiment-id` is omitted.

Plot generation:
- `python3 analysis/fpga_plot.py --experiment-id sweep_bitwidth`
  - Default experiment id is `baseline_fpga` if `--experiment-id` is omitted.
- Optional controls:
  - `--x-param DATA_WIDTH`
  - `--group-by FRAC_BITS`
  - `--filter FRAC_BITS=7`
  - Requires `matplotlib` in the Python environment.

Makefile wrappers:
- `make fpga_summary EXP=sweep_bitwidth`
- `make fpga_plots EXP=sweep_bitwidth`

Plot output path:
- `results/fpga/plots/<experiment_id>/`

Currently generated architecture-study plots (when data exists in aggregate):
- `Fmax vs swept parameter`
- `LUT vs swept parameter`
- `FF vs swept parameter`
- `DSP vs swept parameter`
- `BRAM vs swept parameter`
- `WNS vs swept parameter`
- area-performance scatters: `Fmax vs LUT/FF/DSP/BRAM`

Not currently generated (missing metrics in aggregate dataset):
- throughput plots
- end-to-end latency plots
- model-vs-measurement plots requiring measured accuracy, cycle-level stage latency, or power/energy metrics

# Generated Output Policy

- Canonical generated-output root for new runs: `results/`
- Verilator batch default output: `results/verilator/batch/`
- Legacy `artifacts/experiments/` is retained for history only (deprecated for new runs)

# Repository Layout

```python

.
├─ hdl/                     # RTL modules
├─ tb/                      # SystemVerilog + C++ testbenches (assets in tb/assets/mem/)
├─ python/                  # training/quantisation/util scripts
├─ weights/                 # model/input memory images
├─ fpga/
│  └─ vivado/               # Vivado batch entrypoint scripts
├─ experiments/             # experiment orchestration wrappers
├─ analysis/                # architecture-study analysis assets
├─ docs/                    # project docs (including remote SSH notes)
├─ results/                 # canonical generated-output root
├─ artifacts/               # legacy outputs (deprecated for new runs)
├─ Makefile
└─ README.md     

```

# Parameters and Configuration

| Parameter      | Value                        |
|----------------|----------------------------- |
| Input size     | 28×28                        |
| Conv kernel    | 3×3                          |
| Pooling size   | 2×2                          |
| Dense outputs  | 10                           |
| UART Baudrate  | 115200                       |
| Clock freq     | 100 MHz                      |

---


# Notes and Limitations

- Single-frame execution (no batching or overlap).
- Fixed-point arithmetic throughout (no floating point).
- UART transfer time dominates end-to-end latency in hardware.
- Bit-widths and scaling must match the exported quantised model.
- Designed for clarity and determinism rather than maximum throughput.
