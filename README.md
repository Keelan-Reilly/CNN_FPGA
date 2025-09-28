# Quantised CNN Inference Accelerator

This project implements a digit-classification CNN in SystemVerilog. The pipeline runs conv → ReLU → maxpool → dense → argmax entirely in RTL using BRAM-backed buffers. A UART RX feeds a 28×28 grayscale image (784 bytes) into IFMAP memory; after inference, a single ASCII digit (‘0’–‘9’) is transmitted over UART TX.

Cycle-accurate simulation is done with Verilator; a small Python harness compares software vs hardware predictions over a serial port.

It demonstrates how neural network inference can be run entirely in RTL using pipelined hardware with streaming interfaces, and can be tested using cycle-accurate simulation via Verilator.

---

### Key Results
- **Latency:** ~227395 cycles total (~2.27 at 1000 MHz)
- **Prediction:** Matches ground truth (e.g., predicted digit: 7, UART byte: 0x37)
- **Top module:** `top_level.sv`
- **Simulation:** Verilator harness with timestamped stage output

---

### Architecture Pipeline
- UART RX → IFMAP BRAM (28×28) → conv2d → CONV BRAM → relu → maxpool → dense → argmax → UART TX
- BRAM topology
  	- IFMAP: single write port (UART) + read port (conv2d)
	- CONV buffer: true dual-port (A=read for ReLU/Pool, B=write from conv/ReLU)
	- POOL buffer: single write (from pool) + read (for dense)
- Pooling FSM: strictly linear CHW order (no address math in the critical path).
- Inference begins after 256 bytes (16×16) are received via UART.

---

## What this project contains

### RTL Modules (SystemVerilog)
- **hdl/top_level.sv** — Top-level module for simulation; instantiates all pipeline stages and manages sequencing.
- **hdl/conv2d.sv** — Performs sliding window convolution with quantised weights and zero-padding.
- **hdl/relu.sv** — Implements element-wise ReLU activation.
- **hdl/maxpool.sv** — Executes 2×2 max pooling for downsampling.
- **hdl/dense.sv** — Fully connected layer using integer-weight MAC operations.
- **hdl/argmax.sv** — Selects the neuron with the highest activation.
- **hdl/uart.sv** — UART receiver (8N1 format) for input data. UART transmitter for sending 
- **hdl/bram_sdp.sv, hdl/bram_tdp.sv** — simple/true dual-port RAMs
prediction results (digits 0–9).

---

## Simulation (Verilator)
- tb/tb_full_pipeline.cpp — C++ testbench harness for Verilator; simulates full pipeline from input to UART output.
- tb_testbench.sv  – Unit Testbenches for each HDL module
- obj_dir/ — Verilator build output.

---

## Python (helpers & host)
	•	python/train.py — defines/trains the PyTorch reference model
	•	python/quantise.py — exports fixed-point weights/biases to .mem
	•	python/make_image_mem.py — converts PNG → .mem / .bin (784 bytes)
	•	python/verify_fixed_point.py — sanity checks on quantised math
	•	python/fpga_infer_uart.py — sends image bytes to the FPGA and reads prediction. Runs PyTorch model on the same image and compares to hardware result.


# Quick Start

## 1) Build and run full inference

```bash
make run
```
Expected output (example):

```bash
---- Performance Report ----
Frame cycles: 227395
 conv  = 175619
 relu  = 12547
 pool  = 7843
 flat  = 1
 dense = 31373
 argmx = 12
----------------------------
TX byte: 0x37  (7)
Predicted digit (numeric): 7
```

## 2) Program FPGA & test over UART
1.	Build/flash your bitstream.
2.	Send an image and read back the prediction:

Output(Example)
``` bash
[SW] predicted 7  (conf 1.000)
[HW] trial 1: 7
...
Agreement: 5/5 trials equal to SW (7).
```

# CNN Architecture

This CNN classifies 16×16 grayscale digit images using the following sequence of layers:

1. **Conv2D**  
    - 1 filter, 3×3 kernel, stride 1, zero padding  
    - Output: 14×14 feature map  
    - Quantised integer weights

2. **ReLU Activation**  
    - Applies element-wise nonlinearity

3. **MaxPooling**  
    - 2×2 window, stride 2  
    - Output: 7×7 feature map

4. (*Flattening is handled implicitly*)  
    - The 7×7 pooled feature map is read linearly and fed directly into the dense layer.

5. **Dense (Fully Connected) Layer**  
    - 10 outputs (one per digit 0–9)  
    - Input size: 49  
    - Output: 10 logits

6. **Argmax**  
    - Selects the digit with the highest logit

---

# Parameters and Configuration

| Parameter      | Value                        |
|----------------|----------------------------- |
| Input size     | 28×28                        |
| Conv kernel    | 3×3                          |
| Pooling size   | 2×2                          |
| Dense outputs  | 10 (digits)                  |
| UART Baudrate  | 115200                       |
| Clock freq     | 100 MHz (sim)                |

---

# Repository Layout

```python

.
├─ hdl/                        # RTL modules
│  ├─ argmax.sv
│  ├─ conv2d.sv
│  ├─ dense.sv
│  ├─ fsm_controller.sv
│  ├─ maxpool.sv
│  ├─ relu.sv
│  ├─ top_level.sv            # top module
│  └─ uart.sv                 # combined UART RX/TX
├─ python/                    # Training and helper scripts
│  ├─ make_image_mem.py      # converts image → memory format
│  ├─ quantise.py            # quantises weights/biases to fixed-point
│  ├─ train.py               # defines and trains the model
│  ├─ uart_receive_and_infer.py # send image to FPGA, get prediction
│  ├─ uart_sim.py            # simulates UART in Verilator
│  └─ verify_fixed_point.py  # validate fixed-point correctness
├─ tb/
|  ├─ tb_testbench.sv   # unit testbenches for each module
│  └─ tb_full_pipeline.cpp   # C++ Verilator testbench
├─ Makefile
├─ .gitignore
└─ README.md                  

```

---

# How Inference Works

1.	UART_RX collects exactly 784 bytes and writes them into IFMAP BRAM; when the last byte arrives, it asserts frame_loaded.
2.	FSM controller sequences stages: conv → relu → pool → dense → argmax.
3.	UART_TX sends the ASCII digit of the final argmax result (unless debug streams are enabled, in which case those packets go first).

---

# Requirements

- **Verilator** (tested on 5.018+)
- **C++ compiler** (e.g., `g++`, `clang++`)
- **Python 3.8+** (optional, for output analysis)
- **Make**

---

# Notes and Limitations

- Images must be preprocessed to 28×28 grayscale and supplied as 784 bytes (.mem hex lines or raw .bin).
- The design uses sequential stage execution (no multi-frame batching).
- Quantised fixed-point (no floating point); ensure POST_SHIFT matches your export.
- UART assumes 8N1 at the configured BAUD_RATE, clocked by CLK_FREQ_HZ.
- The pooling module and dense address sequencer are designed for timing-clean, linear access into BRAM.
- Verilator sim reports per-stage cycle counts; hardware timing will depend on your target and constraints.
