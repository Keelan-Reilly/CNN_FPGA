# Repository Design Document

## Purpose of This Document

This document is the detailed implementation/reference map for the repository. It is intended to answer a different question from the README and the report:

- [`README.md`](/home/keelan/CNN_FPGA/README.md) explains how to run the project and what the current checked-in results are.
- [`report.md`](/home/keelan/CNN_FPGA/report.md) explains the architecture-study narrative and findings.
- This document explains what exists in the repository, how the pieces fit together, and how the implementation behaves at a file-by-file level.

The goal is comprehensive understanding of the repository as it exists today, including the checked-in study artifacts under `results/`.

## Repository in One Sentence

This repository implements a small fixed-point CNN inference accelerator in SystemVerilog, verifies it with Verilator and stage-level testbenches, trains and exports weights from PyTorch, and wraps the implementation in a Vivado-based experiment and analysis flow for FPGA architecture studies.

## System-Level Architecture

The top-level hardware datapath is:

`UART RX -> IFMAP BRAM -> conv2d -> CONV BRAM -> ReLU -> maxpool -> POOL BRAM -> dense -> argmax -> UART TX`

Important architectural characteristics:

- The design is sequential, not overlapped. A controller starts one stage at a time and waits for its `done` signal before moving on.
- Intermediate tensors are stored in BRAM between stages. The repo is intentionally explicit about memory ownership and latency rather than hiding those details behind generated IP or HLS abstractions.
- The design processes one frame at a time. There is no batching or multi-frame overlap in the RTL datapath.
- Arithmetic is fixed-point throughout. PyTorch training exists to produce floating-point weights, but the hardware and hardware-matching software flows operate on quantised values.
- The repo is both an implementation project and an architecture-study project. That is why there is both low-level RTL/testbench code and higher-level experiment/plotting infrastructure.

## Top-Level Parameters and What They Mean

The repo currently treats these as the main public hardware knobs:

- `DATA_WIDTH`: activation/weight storage width in bits.
- `FRAC_BITS`: fixed-point fractional precision.
- `CLK_FREQ_HZ`: clock rate used both in the design and in derived timing/performance calculations.
- `BAUD_RATE`: UART line rate used for RX/TX timing.
- `DENSE_OUT_PAR`: degree of output-neuron parallelism inside the dense layer.

The experiment runner explicitly supports sweeping exactly these parameters. Other constants, such as image size, channel count, kernel size, and number of classes, are baked much more deeply into the current implementation and its exported weight files.

## How the Repo Is Organized

### Source and logic directories

- `hdl/`: the actual hardware implementation.
- `tb/`: unit-style and integration-style testbenches plus support assets used by those testbenches.
- `python/`: model training, quantisation, input preparation, UART sanity helpers, and a software golden model.
- `fpga/vivado/`: scripts that run Vivado in batch mode and parse implementation reports.
- `experiments/`: orchestration layer for repeated Vivado runs, Verilator performance collection, and analytical model generation.
- `analysis/`: loaders, CLI summaries, plotting tools, and the workload-aware MAC-array framework v2 analysis layer.

### Documentation directories

- `README.md`: entrypoint and current usage guide.
- `report.md`: study write-up focused on the dense-parallelism question and related sweeps.
- `docs/remote-ssh.md`: short notes for headless usage.

### Data and output directories

- `weights/`: current model memory images and sample input image files used by simulation and synthesis flows.
- `results/`: canonical output root for new experiment and analysis products.
- `batch_out/`: older or ad hoc batch output location still present in the repo.
- `.Xil/`, `obj_dir/`, `vivado.log`, `vivado.jou`, and related files: build/tool byproducts, not core source.

### Historical/quirky paths

The repo mostly standardizes on `results/`, but there are still multiple conventions present:

- `make run_batch` defaults to `results/verilator/batch/`.
- FPGA study runs and aggregates go under `results/fpga/`.
- `experiments/run_batch_experiment.sh` still writes to `results/experiments/`.
- `batch_out/` still exists, but it is not the main current path used by the newer README flow.

That is not a documentation mistake; it is an actual repo inconsistency worth knowing about.

## Top-Level Build and Workflow Entry Points

### `Makefile`

The `Makefile` is the main operator-facing entrypoint:

- `make run` / `make run_full`: compile the top-level RTL with Verilator and run the full-pipeline C++ testbench.
- `make run_batch`: run the batch C++ testbench over MNIST raw files and write batch artifacts.
- `make run_batch_vcd`: same idea, but with per-failure waveform generation enabled.
- `make fpga_experiments`: run the baseline FPGA experiment config.
- `make fpga_experiments_sweep`: run the bit-width sweep config.
- `make fpga_summary`: render a concise aggregate summary table.
- `make fpga_plots`: generate plots from an aggregate dataset.
- `make fpga_framework_v2`: generate the workload-aware MAC-array framework v2 results pack.
- `make fpga_refresh_preview`: generate the selective measured-refresh manifest and preview only the runnable queue.
- `make fpga_refresh_execute`: run the same selective refresh queue through the optional scheduler.
- `make test`: run the deterministic Python unit tests for the newer analysis logic.

The `Makefile` encodes an important design assumption: Verilator integration is first-class, not just an afterthought for unit tests. The full top-level design is expected to be simulation-friendly.

## RTL Design

### Overview

The RTL is written as explicit control-dominated SystemVerilog. The modules are not highly parameter-generic in the abstract-library sense; they are parameterized enough to support the current accelerator and the current sweeps, while remaining understandable and synthesis-friendly.

The most important design theme across the RTL is explicit accommodation of synchronous BRAM latency:

- stage FSMs often have `READ`, `WAIT`, `CAP`, and writeback phases,
- data-valid alignment is done with explicit pipeline registers,
- shared memories are multiplexed only when stage ownership is known,
- the top level tracks stage-active flags to prevent accidental concurrent access.

### `hdl/top_level.sv`

This is the central integration file and the best single place to understand the complete accelerator.

It is responsible for:

- exposing the system-level ports: `clk`, `reset`, `uart_rx_i`, `uart_tx_o`, and `predicted_digit`,
- defining the top-level configuration parameters,
- loading weight memory files differently for synthesis vs simulation,
- instantiating all BRAMs and compute stages,
- mapping UART RX bytes into fixed-point pixels through a LUT,
- asserting `frame_loaded` after 784 received bytes,
- wiring the stage controller, active flags, and BRAM ownership muxes,
- formatting the argmax result as an ASCII digit for UART TX,
- exposing simulation-only timing counters and debug prints.

The module is opinionated about memory topology:

- `ifmap_mem` is a simple dual-port BRAM where UART writes on port A and `conv2d` reads on port B.
- `conv_buf` is a true dual-port BRAM because ReLU and maxpool need to read from the same scratch tensor while conv and ReLU need write access at different times.
- `pool_mem` is another simple dual-port BRAM used by maxpool and dense.

The pixel conversion LUT is important because it defines the hardware-side interpretation of input bytes. Every received 8-bit pixel is mapped to a fixed-point value approximating `k / 255 * 2^FRAC_BITS`, rounded with integer arithmetic. The software golden model reproduces that exact transform.

The top-level controller does not directly manage argmax as a stage with its own start signal from the main FSM. Instead:

- the dense module writes logits into an array,
- `argmax` starts automatically on `dense_done`,
- UART TX readiness is handled via a `tx_pending` flag and the main controller’s `tx_start`.

Simulation-only instrumentation in this file is a major part of the study flow. Under `ifndef SYNTHESIS`, the module:

- tracks total cycles from `frame_loaded` to `tx_start`,
- tracks stage-local cycle counts,
- tracks bubble cycles and time waiting for TX,
- prints `PERF_METRIC ...` lines that `collect_verilator_perf.py` later parses,
- dumps logits and a software argmax cross-check at `dense_done`.

In other words, this file is both the deployed top level and the built-in measurement source for the architecture study.

### `hdl/fsm_controller.sv`

This module is the high-level sequential scheduler for the pipeline.

It has a simple role:

- wait for `frame_loaded`,
- pulse `conv_start`,
- then chain `relu_start`, `pool_start`, `dense_start`, and `tx_start` based on corresponding `done` or readiness signals,
- assert `busy` while a frame is in flight.

The controller includes a `FLAT` pseudo-stage because flattening is conceptually part of the pipeline even though it is effectively free in this implementation. At the top level, `flat_done` is just a one-cycle delayed marker derived from `pool_done`.

This file matters more than its size suggests because it encodes the repository’s main architectural assumption: stages do not overlap. Many study conclusions, especially the dense-parallelism analysis, depend on that fact.

### `hdl/conv2d.sv`

This is the most complex stage-level RTL block.

Conceptually it performs:

- SAME-padded convolution,
- over CHW-formatted input memory,
- one tap at a time,
- with a carefully staged MAC pipeline.

Notable implementation details:

- weights and biases are stored in internal ROM-style arrays loaded from `.mem` files,
- input feature-map reads come from a synchronous BRAM interface,
- padding is implemented by validity gating rather than separate padded storage,
- the accumulator uses a widened signed type `ACCW`,
- a dedicated multiply pipeline stage exists to shorten the DSP/adder timing path,
- the final scaled and saturated result is written through a staged `WRITE` state.

The FSM sequence is intentionally verbose:

- `READ`: compute coordinates, drive BRAM address, fetch weight,
- `WAIT`, `PROD`, `CAP`: align memory data, weight, and validity,
- `ACCUM_MUL`, `ACCUM_ADD`: pipeline the multiply and accumulation,
- `WRITE`: emit one clean BRAM write strobe,
- `FINISH`: pulse `done`.

The explicit staging is not accidental boilerplate. It is the design’s answer to the realities of synchronous BRAM and FPGA timing closure.

### `hdl/relu.sv`

This module performs an in-place pass over the convolution scratch buffer.

Its design is straightforward:

- sequentially read each activation,
- wait out BRAM latency using two explicit wait states,
- apply a signed ReLU clamp,
- write the result back to the same address.

The file is intentionally conservative about timing semantics:

- two wait states exist to tolerate BRAM latency plus nonblocking assignment behavior,
- reads are issued combinationally based on the FSM state,
- writes are tied directly to the `WRITE` state.

This stage is simple mathematically but illustrative architecturally: even trivial elementwise work becomes memory-schedule-dominated when implemented in explicit RTL over synchronous BRAM.

### `hdl/maxpool.sv`

This module implements 2x2 maxpool with stride 2, one pooled output at a time.

Its behavior is:

- calculate the top-left address of the current 2x2 window,
- fetch four values serially from the convolution buffer,
- store them in `a0..a3`,
- compute the maximum,
- write the pooled result into the output BRAM,
- advance spatial/channel coordinates.

Notable details:

- it tolerates up to a fixed internal `MAX_BRAM_LAT` delay by explicitly counting wait cycles,
- the output write address is tracked as a linear sequential pointer,
- the source layout and destination layout are both CHW-linearized,
- the FSM is contract-oriented: `ISSUE`, `WAIT`, `CAP`, `WRITE`, `FINISH`.

Like the other stage modules, maxpool emphasizes interface discipline more than raw cleverness.

### `hdl/dense.sv`

This module computes the fully connected layer.

Its model is:

- read one input activation,
- broadcast it to up to `DENSE_OUT_PAR` output lanes,
- multiply by the corresponding row weights,
- accumulate into one accumulator per active lane,
- after all input elements have been consumed, scale/saturate and commit each lane’s result into `out_vec`.

This file is the main microarchitectural tuning point in the study:

- `DENSE_OUT_PAR` is the exposed scaling knob,
- weights are stored flattened by output row,
- biases seed each accumulator in fixed-point space,
- `LAT` represents the read latency of the input source and must match the source driving `in_q`.

The top level instantiates dense with `LAT=2`, matching the pooled BRAM access timing. The analytical latency model in `experiments/analyze_latency_model.py` is based directly on this FSM structure.

The module also contains simulation-only debug infrastructure:

- per-cycle debug counters,
- optional `+quiet` suppression,
- bounded debug print volume.

This makes dense not just a compute block but also an observable object of study.

### `hdl/argmax.sv`

This module scans a vector of signed logits and returns the index of the first maximum value.

Behaviorally it is simple:

- on `start`, seed the best-so-far from `vec[0]`,
- compare one vector entry per cycle,
- update `bestv` and `besti` when a larger value appears,
- pulse `done` in `FINISH`.

The “first maximum wins” behavior matters because ties are resolved by not updating on equality.

### `hdl/uart.sv`

This file contains both `uart_rx` and `uart_tx`.

They are minimal 8N1 UART implementations intended for:

- FPGA board bring-up,
- Verilator-driven integration tests,
- simple host-side UART experiments.

`uart_rx`:

- detects a low start bit,
- waits half a bit to confirm it,
- samples each data bit on a fixed integer schedule,
- pulses `rx_dv` for one cycle when a byte has been received.

`uart_tx`:

- waits for `tx_dv`,
- emits start, data, and stop bits with integer bit timing,
- exposes `tx_busy` so the top-level controller avoids launching mid-frame.

No parity, framing-error handling, or oversampling sophistication is included. That is deliberate; the UART exists to support the accelerator flow, not to be a feature-complete serial core.

### `hdl/bram_sdp.sv`

This is a simple dual-port BRAM model:

- port A is write-only,
- port B is read-only,
- reads are synchronous with one-cycle latency.

It is used where the ownership pattern is inherently asymmetric, such as UART write plus compute read.

### `hdl/bram_tdp.sv`

This is a true dual-port BRAM model with two independent ports.

It is used for the convolution scratch buffer because that buffer has the most complex access-sharing requirements in the design.

The comments explicitly warn that read-during-write behavior is device dependent, which is an important clue about the level of hardware realism the repository aims for.

## Verification and Testbenches

### Philosophy

The repo uses two complementary verification styles:

- stage-level SystemVerilog testbenches that validate individual modules against local golden behavior and timing contracts,
- C++ Verilator testbenches that exercise the integrated top level through the same UART interface used in the actual design.

This split is useful:

- the SV benches catch contract/timing mistakes close to the module,
- the C++ benches verify that the complete end-to-end system still behaves correctly as assembled.

### `tb/tb_full_pipeline.cpp`

This is the simplest top-level integration testbench.

It:

- loads a raw binary file containing one or more 784-byte images,
- optionally loads labels,
- resets the DUT,
- serializes each image byte through the UART RX pin,
- waits for one UART TX byte back,
- decodes that byte as an ASCII digit prediction,
- reports predictions and, if labels are supplied, accuracy.

It uses a bit-accurate UART stimulus model rather than touching internal DUT memories. That is important: this testbench validates both inference logic and the UART-based framing path.

This testbench is also what the top-level performance collection helper uses indirectly for single-image measurement.

### `tb/tb_batch_pipeline.cpp`

This is the more featureful integration bench used for MNIST sweeps and accuracy logging.

It adds:

- native parsing of MNIST IDX image and label files,
- output directory creation,
- CSV/log/failure recording behavior,
- optional per-failure VCD generation,
- configurable start/count/stride/progress arguments,
- optional reset-between-images behavior.

It is more “workflow tooling” than a minimal testbench. In practice, it is the bridge between unit verification and reproducible evaluation.

### `tb/tb_conv2d.sv`

This testbench creates its own weight and bias memory images, builds a synthetic input feature map, models a one-cycle BRAM for input reads, captures output writes, and checks the convolution result.

Its purpose is not just mathematical correctness. It also validates:

- the stage’s memory-read contract,
- that outputs are produced at the expected addresses,
- saturation behavior on extreme-weight cases.

### `tb/tb_dense.sv`

This is a contract-plus-golden bench for the dense layer.

It includes:

- a parameterized BRAM-like vector source with configurable latency,
- multiple DUT instances or cases covering different latency/weight combinations,
- local golden computation with saturation handling,
- generated `.mem` files under `tb/assets/mem/`.

This bench is especially valuable because dense behavior depends on both arithmetic correctness and latency alignment between `in_en`, `in_addr`, and `in_q`.

### `tb/tb_maxpool.sv`

This bench verifies:

- maxpool handshake behavior,
- expected conv-read address sequencing,
- pooled output correctness,
- output write contract.

It uses a BRAM-like feature map source and multiple data patterns, including signed cases, so that both address sequencing and signed max behavior are exercised.

### `tb/tb_relu.sv`

This bench models a dual-port BRAM with configurable read latency and monitors the ReLU module’s read/write ordering.

It is notable because it checks more than value correctness:

- sequential address order,
- writeback timing relative to reads,
- one-cycle `done` pulse expectations.

That matches the repo’s general verification style: interface timing is treated as part of correctness, not a secondary concern.

### `tb/tb_uart.sv`

This is a loopback-style UART test:

- instantiate `uart_tx`,
- feed its output into `uart_rx`,
- send a handful of bytes,
- ensure the receiver reconstructs the same bytes.

It exists to validate the isolated serial building blocks without dragging the CNN pipeline into the test.

### `tb/tb_argmax.sv`

This is a compact directed test for argmax.

It verifies:

- the module finds the correct maximum index,
- ties resolve to the earliest index,
- the `done` pulse arrives before a timeout.

### `tb/assets/mem/`

This directory contains static or generated memory images used by testbenches.

These are support assets, not production model weights. Their role is to let the stage-level SV benches load deterministic weight/bias fixtures without depending on the main `weights/` directory.

### `tb/make_dense_mems.sh`

This is a helper for generating dense-related test memory assets. It is part of the test support tooling rather than the main operator workflow.

## Python Tooling

### `python/train.py`

This script trains the small reference CNN in PyTorch on MNIST.

Its model architecture mirrors the RTL accelerator structure:

- one convolution layer from 1 channel to 8 channels,
- ReLU,
- 2x2 maxpool,
- one fully connected layer to 10 classes.

The script:

- downloads or loads MNIST,
- trains with Adam and cross-entropy loss,
- evaluates test accuracy after each epoch,
- supports early stop by target accuracy,
- saves `small_cnn.pth`,
- saves a small sample batch for downstream use.

This is the software origin of the model parameters; it is not intended as a highly configurable research training framework.

### `python/quantise.py`

This script loads `small_cnn.pth`, quantises the weights and biases into fixed-point integer form, and writes:

- `conv1_weights.mem`
- `conv1_biases.mem`
- `fc1_weights.mem`
- `fc1_biases.mem`

The output format is hexadecimal lines suitable for `$readmemh`.

It supports:

- configurable fractional bits,
- configurable bit width for saturation,
- two’s complement encoding for negative values.

This file is the handoff point between PyTorch-space and RTL-space.

### `python/compare_sw_hw_fixed.py`

This is the most important software-side verification script.

It implements a hardware-matching software model for:

- byte-to-fixed input conversion,
- SAME-padded conv,
- ReLU,
- 2x2 maxpool,
- dense,
- argmax.

It can be used in two modes:

- software-only, to validate the fixed-point golden path,
- hardware-assisted over UART, to compare FPGA predictions against the golden model.

This script is effectively the algorithmic reference for the RTL. If there is a mismatch between high-level intuition and actual hardware behavior, this script is closer to the hardware than the original floating-point PyTorch model.

### `python/make_batch_idx.py`

This script extracts a fixed number of MNIST samples from IDX files and writes:

- `images.bin`
- `labels.bin`
- `input_image.bin`

It is a data-preparation helper for simulation workflows.

### `python/make_image_mem.py`

This script loads the saved sample batch, converts the first image into:

- `weights/input_image.mem`
- `weights/input_image.bin`

It is the easiest path from trained sample data to a deterministic single-image hardware test input.

### `python/echo.py`

This is a small UART sanity helper, apparently intended for quick hardware connectivity checks. It is not central to the repo’s main workflow, but it is useful for bring-up/debug.

## FPGA Automation

### Overview

The FPGA automation layer is intentionally split:

- shell/Tcl scripts perform the actual Vivado batch run,
- Python scripts parse reports and coordinate experiment bookkeeping.

### `fpga/vivado/run_batch.sh`

This shell script is the CLI wrapper around one Vivado batch implementation run.

It:

- parses arguments like run directory, part, top, XDC, clock period, jobs, and generics,
- resolves paths,
- creates the run directory and reports directory,
- supports either native `vivado` or a WSL-to-Windows `vivado.bat` path,
- passes all state into the Tcl script.

This script is the narrow waist between Python experiment orchestration and Vivado itself.

### `fpga/vivado/run_batch.tcl`

This Tcl script performs the actual implementation flow.

It:

- validates the repo and file layout,
- gathers HDL source files,
- copies required weight `.mem` files into the project directory for synthesis-time lookup,
- optionally applies a clock period,
- applies top-level `-generic` overrides for supported sweep parameters,
- runs synthesis, opt, placement, physical opt, and routing,
- emits utilization, timing summary, timing path, and DRC reports.

An important detail is the synthesis-time memory-file handling: `top_level.sv` expects synthesis-visible weight files without the `weights/` prefix, so the Tcl flow copies them into the Vivado project working directory.

### `fpga/vivado/parse_reports.py`

This Python parser converts post-route report text into machine-readable metrics.

It extracts:

- LUT,
- FF,
- DSP,
- BRAM,
- WNS,
- estimated Fmax from target period and slack,
- a compact timing-bottleneck summary.

It is deliberately lightweight and regex-driven rather than coupled to a heavy Vivado database interface.

## Experiment Framework

### `experiments/run_fpga_experiments.py`

This is the main orchestration script for study runs.

Its responsibilities are:

- load a JSON experiment config,
- expand explicit runs and Cartesian sweeps,
- invoke `fpga/vivado/run_batch.sh`,
- parse Vivado reports into metrics,
- run Verilator-based performance collection for the same parameter point,
- write per-run metadata and aggregate CSV/JSON outputs.

The script defines the current supported sweep parameter set explicitly:

- `DATA_WIDTH`
- `FRAC_BITS`
- `CLK_FREQ_HZ`
- `BAUD_RATE`
- `DENSE_OUT_PAR`

That explicit allowlist is important because it prevents configs from silently passing unsupported generic overrides.

### `experiments/collect_verilator_perf.py`

This helper measures compute performance for one architecture point using the full top-level Verilator flow.

It:

- converts `weights/input_image.mem` into a binary input if needed,
- recompiles the top level with the requested generics,
- runs one full-pipeline simulation,
- parses the performance printout from `top_level.sv`,
- computes latency time and throughput from cycle counts and `clock_hz`,
- writes `performance.json`.

This script is the link between the hardware implementation study and the cycle-level performance study.

### `experiments/analyze_latency_model.py`

This script takes an aggregate dataset and appends analytical model fields.

Its model is intentionally simple:

- calibrate a fixed non-dense term from a reference run,
- model dense cycles as a function of `DENSE_OUT_PAR`, `IN_DIM`, number of classes, and assumed dense latency,
- predict total latency, latency time, throughput, and model error.

The script encodes the architectural assumption that the pipeline is sequential and that dense is the only varying stage in the main scaling study.

### `experiments/run_batch_experiment.sh`

This is an older or more ad hoc wrapper around `make run_batch`.

It:

- reads MNIST paths and run settings from environment variables,
- creates an output directory under `results/experiments/`,
- invokes the batch run,
- appends a record to `results/experiments/index.csv`.

This script still works as a convenience wrapper, but it does not follow the newer `results/fpga` and `results/verilator` conventions.

### `experiments/configs/*.json`

These files define reproducible FPGA study runs.

`baseline_fpga.json`:

- single-point baseline implementation.

`study_baseline_characterization.json`:

- explicit baseline study point used in the report and README.

`study_dense_parallel_scaling.json`:

- main architectural sweep over `DENSE_OUT_PAR = [1, 2, 5, 10]`.

`study_precision_resource.json`:

- control sweep over fixed-point width/fraction combinations.

`study_timing_target.json`:

- sweep over implementation clock targets, with `CLK_FREQ_HZ` aligned to each timing point.

`sweep_bitwidth.json`:

- older or more generic bit-width sweep.

`sweep_dense_parallel.json`:

- older or more generic dense-parallel sweep.

`mac_array_framework_v2.json`:

- workload-aware MAC-array framework config,
- carries the canonical `4x4`, `8x4`, `8x8` grid set,
- records measured static anchors such as the 8x8 shared `64 -> 32 DSP` reduction and the 8x8 replicated Artix-7 implementation failure,
- defines switching-cost assumptions, workload classes, and policy-evaluation constraint presets.

There are effectively two tiers of configs:

- the `study_*` configs used by the current report structure,
- the older `baseline_` / `sweep_` configs that still exist and still produce valid aggregates.

### `experiments/FPGA_STUDY_SUITE.md`

This document is not executable code, but it matters operationally. It describes the intended sequence of study runs and the interpretation focus of each experiment family.

## Analysis Tooling

### `analysis/fpga_results.py`

This file provides shared aggregate-loading utilities.

Its role is to:

- locate aggregate datasets,
- load either CSV or JSON format,
- normalize numeric fields,
- parse serialized `params` back into dictionaries.

This module is the small but important compatibility layer that lets the rest of the analysis scripts treat datasets uniformly.

### `analysis/fpga_summary.py`

This script prints a concise table view of an aggregate dataset.

It shows:

- run ID,
- status,
- return code,
- compact parameter summary,
- LUT/FF/DSP/BRAM,
- WNS,
- estimated Fmax.

It is intentionally lightweight and human-oriented rather than a plotting or notebook tool.

### `analysis/fpga_plot.py`

This is the main plotting utility for the study.

It supports:

- selecting aggregates by experiment ID or explicit path,
- filtering rows by parameter values,
- automatic or explicit x-axis parameter selection,
- grouping by another parameter,
- plotting implementation metrics such as LUT/FF/DSP/BRAM/Fmax/WNS,
- plotting performance metrics such as latency or throughput when present,
- plotting derived metrics like speedup and throughput-per-resource,
- plotting model-vs-measurement overlays when model fields exist.

This script is general enough to support both current and future sweep datasets, but it still reflects the metrics schema produced by the current experiment framework.

### `analysis/mac_array_*.py` and `analysis/run_mac_array_framework.py`

These files are the v2 extension layer that turns the repo from a static comparison into a reusable decision framework.

Their division of responsibility is intentionally small and explicit:

- `mac_array_types.py`: typed data models for grids, architectures, workloads, and summaries.
- `mac_array_evidence.py`: validated evidence loading plus per-field provenance and switching-cost helpers.
- `mac_array_workloads.py`: workload parsing and validation.
- `mac_array_metrics.py`: static-resource derivation plus workload-aware throughput/utilization/efficiency calculations.
- `mac_array_adaptive.py`: adaptive mode switching and break-even analysis.
- `mac_array_policy.py`: rules-based architecture recommendation logic.
- `mac_array_regime.py`: bounded winner-map generation plus adaptive rejection surfaces.
- `mac_array_refresh.py`: selective measured-refresh manifesting and honest measured-vs-modelled proxy comparison.
- `mac_array_report.py`: report, CSV/JSON, plot, and progress-log generation.
- `run_mac_array_framework.py`: top-level CLI that ties the whole v2 flow together.

This layer intentionally does not invoke Vivado. It reuses the checked-in aggregates as measured evidence, combines them with explicit static MAC-array anchors, and writes deterministic outputs under `results/fpga/framework_v2/`.

The main non-code evidence input for this layer is `experiments/configs/mac_array_architecture_evidence.json`, which acts as a small evidence registry rather than burying static anchors and switching assumptions inside the top-level framework config.

The v2 layer also now emits a bounded regime map. This is not an exhaustive search; it is a disciplined sweep over grid, workload, budget class, and throughput class so the repo can show where `baseline`, `shared`, `replicated`, or `adaptive_mode_switching` wins without exploding into unreadable parameter space.

The newer measured-refresh helper under `experiments/run_measured_refresh.py` deliberately sits beside this layer rather than inside it. That is because the fast path stays analytical by default; measured refresh is a selective escalation path that maps a few representative regime points onto what the current CNN Vivado flow can honestly proxy.

## Results and Generated Artifacts

### `weights/`

This directory holds the currently active memory images for the hardware:

- convolution weights and biases,
- dense weights and biases,
- a deterministic input image in `.mem` and `.bin` forms.

These files are part of the practical working state of the repo, even though they are generated rather than handwritten.

### `results/fpga/aggregates/`

This is the most important checked-in results directory.

Each aggregate exists in both CSV and JSON form. These are the main datasets present:

- `baseline_fpga`
- `baseline_with_perf`
- `study_baseline_characterization`
- `study_dense_parallel_scaling`
- `study_dense_parallel_scaling_model`
- `study_precision_resource`
- `study_timing_target`
- `sweep_bitwidth`
- `sweep_dense_parallel`
- `sweep_dense_parallel_perf`
- `sweep_dense_parallel_perf_model`

The schema is centered around:

- implementation metrics: LUT, FF, DSP, BRAM, WNS, estimated Fmax,
- performance metrics: latency cycles, latency time, throughput,
- stage-local cycle counts,
- run metadata: status, duration, run path, params,
- optional model fields in the `_model` datasets.

The `study_*` files are the current canonical datasets for the documented architecture study.

### `results/fpga/runs/`

This directory is where per-run Vivado and Verilator artifacts are expected to live when experiments are executed locally.

Typical contents include:

- resolved config,
- run metadata,
- Vivado logs,
- report files,
- parsed metrics,
- `verilator_perf/` subdirectories when compute metrics are collected.

The checked-in repo snapshot mostly includes aggregates rather than all per-run directories, which is normal for a source repository.

The Vivado experiment runner can now also emit queue-management artifacts here when resource-aware scheduling is enabled:

- `scheduler_queue.*`: deterministic launch queue preview,
- `scheduler.log`: launch/block/complete events,
- `scheduler_summary.json`: concise summary of the queue execution settings and outcomes.

### `results/fpga/framework_v2/`

This is the generated output root for the newer workload-aware MAC-array framework.

Important artifacts include:

- `framework_report.md`: the top-level v2 narrative,
- `evidence_registry.*`: source ids, derivations, and evidence kinds for the architecture/switching layer,
- `provenance_summary.*`: compact breakdown of evidence kinds,
- `workload_manifest.*`: explicit workload descriptors and notes,
- `static_architectures.*`: canonical architecture/grid evidence table,
- `workload_evaluations.*`: fixed-mode workload metrics,
- `adaptive_evaluations.*`: switching-adjusted workload results,
- `adaptive_constraint_evaluations.*`: constraint-filtered adaptive candidates used by the policy layer,
- `adaptive_phase_decisions.*`: per-phase chosen modes plus transition-cost details,
- `break_even.*`: minimum phase duration before switching pays off,
- `policy_recommendations.*`: explicit recommendation table,
- `policy_diagnostics.*`: candidate-by-candidate feasibility and rejection reasons,
- `legacy_measured_summary.json`: extracted summary of the older checked-in measured FPGA study,
- `regime_map.*`, `regime_summary.*`, `regime_rejection_summary.*`, and `adaptive_rejection_surface.*`: bounded winner/rejection products for the regime analysis layer,
- `regime_insights.*`: presentation-oriented summaries derived directly from the generated regime outputs,
- `plots/regime_winner_heatmaps.png`: compact winner surface by grid/workload/budget/throughput,
- `plots/adaptive_rejection_surface.png`: dominant adaptive rejection reason by grid/workload,
- `measured_refresh/`: optional selective measured-refresh manifest, generated one-run configs, queue preview data, and measured-vs-modelled comparison outputs,
- `progress_log.md`: concise rationale for what the v2 layer added.

### `results/verilator/batch/`

This is the default target for batch simulation outputs from the newer Make-based flow.

It is intended to hold:

- batch logs,
- result CSVs,
- failure lists,
- optional VCDs.

### Legacy and side directories

There are older or secondary output conventions still visible:

- `results/experiments/` from `run_batch_experiment.sh`,
- `batch_out/`,
- various local tool outputs at the repo root.

These should be understood as part of the repo’s history, not necessarily the preferred path for new work.

## Typical Operational Flows

### 1. Train a model and export hardware weights

1. Run `python3 python/train.py --data-dir ./data --out-dir ./output`.
2. Run `python3 python/quantise.py --model-dir ./output --out-dir ./weights --weight-frac 7`.
3. Optionally run `python3 python/make_image_mem.py` to refresh deterministic single-image inputs.

This produces the `.mem` files the RTL expects.

### 2. Run a single-image or full-pipeline simulation

1. Use `make run` or `make run_full`.
2. The top-level Verilator build compiles all `hdl/*.sv` files and links `tb/tb_full_pipeline.cpp`.
3. The testbench pushes bytes through UART RX and waits for an ASCII digit on UART TX.

This validates end-to-end top-level behavior using the real external interface.

### 3. Run batch evaluation on MNIST

1. Ensure MNIST raw files exist under `data/MNIST/raw/` or pass overrides.
2. Run `make run_batch` or `make run_batch_vcd`.
3. Inspect outputs under `results/verilator/batch/`.

This is the main quick path for simulation-based accuracy checks over many images.

### 4. Run FPGA architecture studies

1. Choose a config under `experiments/configs/`.
2. Run `python3 experiments/run_fpga_experiments.py --config <config>`.
3. The script launches Vivado runs, parses reports, then collects Verilator performance for each point.
4. Aggregates are written under `results/fpga/aggregates/`.

This is the main reproducible study workflow.

### 5. Generate summaries and plots

1. Run `python3 analysis/fpga_summary.py --experiment-id <id> --include-failed`.
2. Run `python3 analysis/fpga_plot.py --experiment-id <id>`, with optional `--x-param`, `--group-by`, or `--filter`.
3. For dense-model comparisons, first run `python3 experiments/analyze_latency_model.py --input <aggregate.csv>`.

This is the path from collected runs to interpreted figures.

## Known Limitations and Important Repo Quirks

- The accelerator is sequential. Performance improvements in one stage do not automatically produce proportional end-to-end speedups.
- The study’s latency metrics are compute-only, measured from `frame_loaded` to `tx_start`, not full UART wall-clock latency.
- Input format assumptions are rigid: the top level expects exactly one 28x28 grayscale frame serialized as 784 bytes.
- The design depends on matching fixed-point conventions across training/export/software-golden/RTL. A mismatch in `FRAC_BITS` or weight files will produce incorrect behavior that can look superficially plausible.
- The repo has more than one generated-output convention. `results/` is the canonical root, but not every helper script follows the same subdirectory policy.
- Some files are more central than others. The most important implementation truths live in `hdl/top_level.sv`, `hdl/conv2d.sv`, `hdl/dense.sv`, `experiments/run_fpga_experiments.py`, `experiments/collect_verilator_perf.py`, and `analysis/fpga_plot.py`.
- The checked-in results are meaningful artifacts, not disposable noise. They are part of the repository’s current state and are referenced by both the README and the report.

## Reading Order for a New Contributor

If the goal is to understand the repo efficiently, a good order is:

1. [`README.md`](/home/keelan/CNN_FPGA/README.md)
2. `hdl/top_level.sv`
3. `hdl/fsm_controller.sv`
4. `hdl/conv2d.sv`, `hdl/dense.sv`, `hdl/relu.sv`, `hdl/maxpool.sv`, `hdl/argmax.sv`, `hdl/uart.sv`
5. `tb/tb_full_pipeline.cpp` and `tb/tb_batch_pipeline.cpp`
6. `python/quantise.py` and `python/compare_sw_hw_fixed.py`
7. `experiments/run_fpga_experiments.py`
8. `analysis/fpga_results.py`, `analysis/fpga_summary.py`, `analysis/fpga_plot.py`
9. [`report.md`](/home/keelan/CNN_FPGA/report.md)

That order moves from implementation, to verification, to orchestration, to interpretation.
