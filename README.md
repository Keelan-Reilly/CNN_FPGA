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

- **Total latency:** ~379491 cycles  
  → **~3.79 ms @ 100 MHz**
- **Prediction:** Matches software reference (e.g. digit `7`, UART byte `0x37`)
- **Top-level module:** `hdl/top_level.sv`
- **Verification:** Cycle-accurate Verilator simulation with per-stage timing

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
Frame cycles: 379491
 conv  = 288515
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

# Repository Layout

```python

.
├─ hdl/                     # RTL modules
│  ├─ top_level.sv          # top-level integration
│  ├─ conv2d.sv
│  ├─ relu.sv
│  ├─ maxpool.sv
│  ├─ dense.sv
│  ├─ argmax.sv
│  ├─ uart.sv               # UART RX/TX (8N1)
│  ├─ bram_sdp.sv
│  └─ bram_tdp.sv
├─ tb/
│  ├─ tb_testbench.sv       # unit-level testbenches
│  └─ tb_full_pipeline.cpp  # Verilator harness
├─ python/
│  ├─ train.py              # PyTorch model definition/training
│  ├─ quantise.py           # fixed-point export
│  ├─ make_image_mem.py     # PNG → .mem / .bin (784 bytes)
│  ├─ verify_fixed_point.py # fixed-point sanity checks
│  └─ fpga_infer_uart.py    # host-side UART inference
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