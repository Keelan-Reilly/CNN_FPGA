# Quantised CNN Inference Accelerator

This project implements a simple digit-classification convolutional neural network in SystemVerilog. The CNN processes a downsampled 16×16 grayscale image through conv–ReLU–pool–dense–argmax stages. It receives input via UART, performs inference on FPGA, and transmits the predicted digit as a single byte over UART.

It demonstrates how neural network inference can be run entirely in RTL using pipelined hardware with streaming interfaces, and can be tested using cycle-accurate simulation via Verilator.

---

### Key Results
- **Latency:** ~88,743 cycles total (~354 µs at 250 MHz)
- **Prediction:** Matches ground truth (e.g., predicted digit: 4, UART byte: 0x34)
- **Top module:** `top_level.sv`
- **Simulation:** Verilator harness with timestamped stage output

---

### Architecture Pipeline
- UART input → conv2d → relu → maxpool → flatten → dense → argmax → UART output
- Stages are pipelined with valid/ready handshakes and run sequentially.
- Inference begins after 256 bytes (16×16) are received via UART.

---

## What this project contains

### RTL Modules (SystemVerilog)
- **hdl/top_level.sv** — Top-level module for simulation; instantiates all pipeline stages and manages sequencing.
- **hdl/conv2d.sv** — Performs sliding window convolution with quantised weights and zero-padding.
- **hdl/relu.sv** — Implements element-wise ReLU activation.
- **hdl/maxpool.sv** — Executes 2×2 max pooling for downsampling.
- **hdl/flatten.sv** — Flattens 2D feature maps into a 1D vector.
- **hdl/dense.sv** — Fully connected layer using integer-weight MAC operations.
- **hdl/argmax.sv** — Selects the neuron with the highest activation.
- **hdl/uart_rx.sv** — UART receiver (8N1 format) for input data.
- **hdl/uart_tx.sv** — UART transmitter for sending prediction results (digits 0–9).

---

## Simulation (Verilator)
- tb/tb_full_pipeline.cpp — C++ testbench harness for Verilator; simulates full pipeline from input to UART output.
- obj_dir/ — Verilator build output.

---

## Makefile targets

```bash
# Default: build + run full inference pipeline
make

# Build only
make build

# Run only (assumes build done)
make run

# Unit test for conv2d
make run_conv2d

# Clean output files
make clean
```

# Quick Start

## 1) Build and run full inference

```bash
make run
```
Expected output (example):

```bash
---- Performance Report ----
Frame cycles: 87843
 conv  = 62723
 relu  = 6275
 pool  = 1571
 flat  = 1569
 dense = 15693
 argmx = 12
----------------------------
TX byte: 0x34  (4)
Predicted digit (numeric): 4
```
Each stage runs sequentially. While the argmax module itself is combinational, the observed delay may span several cycles due to FSM handoff latency and synchronisation with the TX start trigger, especially at higher clock frequencies.

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

4. **Flatten**  
    - Converts 2D feature map to 1D vector (size 49)

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
| Input size     | 16×16                        |
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
│  └─ tb_full_pipeline.cpp   # C++ Verilator testbench
├─ uart_digit.txt             # sample UART input
├─ uart_out.txt               # UART output prediction
├─ wave.vcd                   # VCD waveform output
├─ Makefile
├─ .gitignore
└─ README.md                  

```

---

# How Inference Works

1. **UART_RX**
    - Receives 256 bytes (16×16 grayscale image)
    - Signals `frame_loaded` to start the pipeline

2. **Pipeline Control**
    - Finite State Machine (FSM) activates each stage in sequence: conv → relu → pool → flat → dense → argmax

3. **TX Output**
    - Argmax result (digit 0–9) is transmitted via UART

---

# Requirements

- **Verilator** (tested on 5.018+)
- **C++ compiler** (e.g., `g++`, `clang++`)
- **Python 3.8+** (optional, for output analysis)
- **Make**

---

# Notes and Limitations

- Input images must be preprocessed to 16×16 grayscale before transmission.
- All weights are quantised integers; MAC operations use fixed-point arithmetic (no floating point).
- UART TX/RX modules assume 8N1 format, 115200 baud rate, and 100 MHz input clock by default.
- The pipeline is designed for sequential stage execution; parallelism or multi-frame batching is not supported.
- Only single-digit classification is implemented (digits 0–9).
- For custom datasets or clock rates, RTL parameters and UART settings may require modification.
- No hardware support for image normalization or advanced preprocessing.
- Verilator simulation is cycle-accurate but does not model UART timing or external hardware delays.
