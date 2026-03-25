# FPGA CNN Accelerator Study

## 1. Introduction

This repository implements a small FPGA CNN accelerator for single-image inference and a batch experiment flow that collects post-route area/timing data together with cycle-accurate performance metrics from Verilator. The main question in this study is whether adding dense-layer parallelism improves end-to-end performance in a meaningful way, or whether another stage already dominates the accelerator.

The central result is that dense-layer parallelism scales locally but not globally. In the current design, convolution dominates total runtime, so increasing `DENSE_OUT_PAR` reduces dense latency sharply while only modestly improving end-to-end latency.

## 2. Accelerator Architecture

The top-level architecture is a sequential controller-driven pipeline:

`conv -> ReLU -> maxpool -> dense -> argmax -> UART TX`

Three architectural properties matter for the study:

- The controller schedules one stage at a time rather than overlapping stages.
- Intermediate tensors are stored in BRAM, and the dense layer reads the pooled feature map from memory.
- The only currently exposed architectural parallelism knob is `DENSE_OUT_PAR`, which increases the number of dense outputs processed in parallel.

This means the measured latency is naturally decomposed into stage-local cycle counts rather than pipeline overlap. The study therefore focuses on stage bottlenecks, area/performance tradeoffs, and timing closure as parallelism increases.

## 3. Experimental Methodology

Implementation results come from the Vivado batch flow, which records LUT, FF, DSP, BRAM, WNS, and estimated Fmax in `results/fpga/aggregates/*.csv`. Compute performance is measured separately using Verilator on the same top-level RTL, using a fixed deterministic input image for every run.

The performance path records:

- `latency_cycles`
- `latency_time_ms`
- `throughput_inferences_per_sec`
- stage cycle counters for `conv`, `ReLU`, `maxpool`, `dense`, `argmax`
- `bubble_cycles`, `busy_cycles`, and `tx_wait_cycles`

The compute window is `frame_loaded -> tx_start`, so the reported latency is compute-only rather than UART-dominated end-to-end wall-clock time.

## 4. Baseline Characterization

The baseline configuration is `DATA_WIDTH=16`, `FRAC_BITS=7`, `DENSE_OUT_PAR=1`, and `CLK_FREQ_HZ=100 MHz`, using the dataset in `results/fpga/aggregates/study_baseline_characterization.csv`.

Measured baseline results:

- Latency: `465,732` cycles, `4.65732 ms`
- Throughput: `214.72` inferences/s
- Area: `2720 LUT`, `1004 FF`, `5 DSP`, `6 BRAM`
- Timing: `WNS = +0.017 ns`, `Fmax ≈ 100.17 MHz`

The stage breakdown shows a clear bottleneck:

- Conv: `344,962` cycles (`74.1%`)
- ReLU: `25,090` cycles (`5.4%`)
- Maxpool: `32,930` cycles (`7.1%`)
- Dense: `62,732` cycles (`13.5%`)
- Argmax: `11` cycles
- Bubble: `7` cycles

The most important architectural conclusion from the baseline is that convolution already consumes roughly three quarters of total compute latency. Dense is significant, but it is not the global bottleneck.

## 5. Dense Parallelism Scaling (main result)

The main experiment sweeps `DENSE_OUT_PAR = {1, 2, 5, 10}` using `results/fpga/aggregates/study_dense_parallel_scaling.csv`. The relevant plots are:

- `results/fpga/plots/study_dense_parallel_scaling/latency_cycles_vs_DENSE_OUT_PAR.png`
- `results/fpga/plots/study_dense_parallel_scaling/throughput_vs_DENSE_OUT_PAR.png`
- `results/fpga/plots/study_dense_parallel_scaling/stage_cycles_breakdown_vs_DENSE_OUT_PAR.png`
- `results/fpga/plots/study_dense_parallel_scaling/speedup_vs_lut.png`
- `results/fpga/plots/study_dense_parallel_scaling/speedup_vs_dsp.png`

Measured results:

| `DENSE_OUT_PAR` | Latency (cycles) | Latency (ms) | Throughput (inf/s) | Dense cycles | LUT | DSP | WNS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1  | 465,732 | 4.65732 | 214.72 | 62,732 | 2,720  | 5  | +0.017 |
| 2  | 434,367 | 4.34367 | 230.22 | 31,367 | 4,467  | 7  | -1.146 |
| 5  | 415,548 | 4.15548 | 240.65 | 12,548 | 10,243 | 13 | -1.623 |
| 10 | 409,275 | 4.09275 | 244.33 | 6,275  | 19,061 | 23 | -1.652 |

Two effects are clear.

First, the dense stage scales almost ideally in isolation. Dense cycles drop from `62,732` at `P=1` to `6,275` at `P=10`, which is effectively a `10x` local speedup. The stage breakdown plot shows that the conv stage remains fixed at `344,962` cycles across the entire sweep, exactly as expected for a sequential design where only the dense block changed.

Second, the end-to-end benefit is much smaller. Total latency improves from `465,732` cycles to `409,275` cycles, only a `1.138x` speedup. Throughput rises from `214.72` to `244.33` inferences/s, a `13.8%` increase. This gap between local dense scaling and global accelerator speedup is the central architectural result of the study.

The area and timing costs are also substantial:

- LUTs grow from `2,720` to `19,061` (`7.0x`)
- DSPs grow from `5` to `23` (`4.6x`)
- WNS becomes negative as soon as `DENSE_OUT_PAR > 1`
- Estimated Fmax drops from `100.17 MHz` to about `85.82 MHz`

In short, increasing dense parallelism makes the dense block faster, but the accelerator quickly becomes convolution-bound while also becoming larger and harder to close timing on.

## 6. Precision vs Resource Study

The precision study uses `results/fpga/aggregates/study_precision_resource.csv` and keeps the architecture fixed while varying arithmetic format:

- `Q12.5`
- `Q14.6`
- `Q16.7`

Relevant plots:

- `results/fpga/plots/study_precision_resource/lut_vs_DATA_WIDTH.png`
- `results/fpga/plots/study_precision_resource/fmax_vs_DATA_WIDTH.png`
- `results/fpga/plots/study_precision_resource/latency_cycles_vs_DATA_WIDTH.png`

Measured results:

| Format | LUT | FF | BRAM | Fmax (MHz) | Latency (cycles) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q12.5` | 2,649 | 900  | 5.0 | 99.12  | 465,732 |
| `Q14.6` | 2,687 | 948  | 5.5 | 101.25 | 465,732 |
| `Q16.7` | 2,720 | 1,004 | 6.0 | 100.17 | 465,732 |

This is a useful control experiment. The cycle schedule is unchanged across the sweep, which is exactly what should happen in the current design because precision changes arithmetic width but not the controller schedule. The primary effects are implementation cost:

- LUT and FF usage rise gradually with width
- BRAM usage also increases
- Fmax stays close to `100 MHz` across the tested points

The conclusion is that precision changes are valuable for resource budgeting, but they do not answer the main microarchitectural scaling question in this accelerator.

## 7. Timing Target Study

The timing-target sweep uses `results/fpga/aggregates/study_timing_target.csv` and aligns `CLK_FREQ_HZ` with the implementation target:

- `80 MHz` target (`12.5 ns`)
- `100 MHz` target (`10 ns`)
- `125 MHz` target (`8 ns`)

Relevant plots:

- `results/fpga/plots/study_timing_target/wns_vs_CLK_FREQ_HZ.png`
- `results/fpga/plots/study_timing_target/fmax_vs_CLK_FREQ_HZ.png`
- `results/fpga/plots/study_timing_target/throughput_vs_CLK_FREQ_HZ.png`

Measured implementation results:

| Target | WNS (ns) | Estimated Fmax (MHz) | Latency (ms) | Throughput (inf/s) |
| --- | ---: | ---: | ---: | ---: |
| `80 MHz`  | -0.511 | 76.86  | 5.82165 | 171.77 |
| `100 MHz` | +0.017 | 100.17 | 4.65732 | 214.72 |
| `125 MHz` | -0.096 | 123.52 | unavailable | unavailable |

In this run set, `100 MHz` is the only clean timing point. Both `80 MHz` and `125 MHz` show negative slack in the aggregate dataset. The `125 MHz` run also lacks performance fields in the aggregate, so the timing-target study is best interpreted primarily through WNS and estimated Fmax rather than complete latency/throughput reporting.

The main value of this sweep is not architectural scaling but implementation realism: it shows that the baseline point is close to a practical operating point, and that more aggressive timing targets cannot simply be assumed to translate into usable throughput.

## 8. Analytical Latency Model

The repository also includes a small additive latency model in `results/fpga/aggregates/study_dense_parallel_scaling_model.json`. The key assumption is that the accelerator is sequential and does not overlap stages. The model therefore splits total latency into:

- a fixed term
- a dense-stage term that depends on `DENSE_OUT_PAR`

The fixed term is calibrated from the measured `P=1` run:

- `model_fixed_cycles = conv + ReLU + maxpool + argmax + bubble`
- `= 344,962 + 25,090 + 32,930 + 11 + 7 = 403,000 cycles`

The dense term is derived from the actual dense FSM:

`model_dense_cycles = ceil(NUM_CLASSES / DENSE_OUT_PAR) * (IN_DIM * (LAT + 2) + 1) + 2`

For this design:

- `NUM_CLASSES = 10`
- `IN_DIM = 1568`
- `LAT = 2`

The model-aware plots are:

- `results/fpga/plots/study_dense_parallel_scaling_model/measured_vs_predicted_latency_vs_DENSE_OUT_PAR.png`
- `results/fpga/plots/study_dense_parallel_scaling_model/measured_vs_predicted_throughput_vs_DENSE_OUT_PAR.png`
- `results/fpga/plots/study_dense_parallel_scaling_model/model_latency_decomposition_vs_DENSE_OUT_PAR.png`

For the current dense-parallel sweep, the model matches the measured data exactly, with `0` cycle error at `P = 1, 2, 5, 10`. That is possible because the sweep changes only the dense architecture while the rest of the sequential pipeline remains fixed.

## 9. Key Findings

1. The accelerator is convolution-bound in its current form. In the baseline, convolution accounts for `74.1%` of compute cycles.
2. Dense parallelism scales locally but not globally. Increasing `DENSE_OUT_PAR` from `1` to `10` gives almost a `10x` reduction in dense cycles, but only a `1.138x` end-to-end speedup.
3. The area/performance tradeoff is unfavorable beyond small dense parallelism. `P=10` requires `7.0x` more LUTs and `4.6x` more DSPs than the baseline while only improving throughput by `13.8%`.
4. Timing closure gets harder as dense parallelism increases. Every point above `P=1` shows negative WNS in the measured sweep.
5. Precision mostly affects implementation cost rather than schedule. In the tested range, latency cycles stay fixed while LUT, FF, and BRAM usage rise with bit width.
6. A small analytical model is sufficient for this architecture. Because the datapath is sequential and the dense sweep isolates one stage, a fixed-plus-dense model is enough to explain the measured behavior.

## 10. Framework V2 Extension

The repository now includes a second analysis layer aimed at a different question:

> Given workload characteristics and hardware/resource constraints, when should a designer choose baseline, shared, replicated, or adaptive mode switching?

This extension lives under `analysis/run_mac_array_framework.py` and emits a checked-in results pack under `results/fpga/framework_v2/`.

What it preserves:

- the measured CNN baseline characterization,
- the measured dense-parallel scaling sweep,
- the timing-closure realism already visible in the checked-in Artix-7-oriented results.

What it adds:

- a first-class architecture evidence registry with explicit record ids, value kinds, derivations, and source paths,
- explicit workload classes: `dense_steady`, `short_burst`, `underfilled`, `phase_changing`,
- workload descriptors such as phase count, burstiness, and utilization variance,
- richer normalized metrics such as throughput per cycle, DSP efficiency, LUT efficiency, and utilization estimates,
- a lightweight pair-aware adaptive switching-cost model,
- break-even analysis for when switching is worth paying for,
- a rules-based architecture selection policy under DSP/LUT and throughput constraints, with runner-ups and rejection reasons.
- a bounded regime-analysis layer with compact winner surfaces and adaptive rejection summaries.
- a selective measured-refresh loop that maps a few representative regime points onto what the current Vivado flow can honestly proxy.

What remains modelled rather than measured:

- workload-aware latency/throughput estimates for the MAC-array architecture family,
- switching-adjusted throughput for adaptive mode changes,
- break-even thresholds and recommendation tables.

The checked-in `results/fpga/framework_v2/framework_report.md` captures the strongest current v2 conclusions. In the current pack:

- the framework's `shared` wins now refer to the modelled shared family, whose 8x8 resource story still leans on the prior `64 -> 32 DSP` anchor rather than on the directly measured 4x4 and 8x4 shared implementations,
- `baseline` wins when throughput/latency targets dominate and the larger fixed mode is still feasible,
- `replicated` is only recommended in a narrow high-demand 4x4 phase-changing window,
- adaptive candidates are now evaluated under the same DSP/LUT/timing constraints as fixed modes, which makes the current framework more conservative about recommending switching,
- the bounded regime map currently finds no adaptive win region under the current evidence-backed assumptions,
- the new regime presentation layer makes that no-win result explicit by showing adaptive rejection surfaces dominated by `adaptive_gain_too_small`,
- the selective measured-refresh path still exists for broader regime sampling, but the repo now also has a three-point direct baseline calibration set: measured `4x4`, `8x4`, and `8x8` standalone baseline slices with exact DSP and direct-slice latency/throughput agreement against the simple model,
- the repo now also has a two-scale, three-way directly measured architecture bridge on the isolated direct slice: `4x4` and `8x4`, each measured for baseline, `shared_lut_saving`, and `shared_dsp_reducing`,
- at `4x4`, `shared_lut_saving` reduces LUT from `1061` to `679` while staying DSP-flat at `16`, and `shared_dsp_reducing` reduces DSP from `16` to `0` while landing at `910 LUT`,
- at `8x4`, `shared_lut_saving` reduces LUT from `2134` to `1351` while staying DSP-flat at `32`, and `shared_dsp_reducing` reduces DSP from `32` to `0` while landing at `1817 LUT`,
- those direct points now support a stronger measured decision layer under `results/fpga/framework_v2/direct_slice/`: a baseline-only LUT calibration aid, a direct shared-implementation comparison, a direct shared scaling summary, an explicit decision table, measured design rules, and a measured trust boundary for what is directly supported versus extrapolated,
- 8x8 replicated remains excluded on Artix-7 because the framework preserves that implementation failure as evidence,
- the measured scaling rule is now explicit: the 4x4 three-way role split survives at 8x4, so `shared_lut_saving` remains the LUT-relief option, `shared_dsp_reducing` remains the DSP-relief option, and `baseline` remains the performance-first option across the first measured scale step,
- the new trust overlay now marks which of those conclusions are directly supported by measured implementation evidence, which are only directionally or partially supported at the family level, and which still remain anchored/modelled extrapolations such as 8x8 shared DSP reduction.
- the framework now also carries a measured calibration aid and caution band: baseline LUT is numerically optimistic in the lightweight model, shared-family latency/throughput direction is well aligned with measured direct-slice behavior, and shared-family DSP/LUT projections remain implementation-dependent rather than silently treated as direct measured replacements.
- the repo now also carries a measured utility layer: `shared_lut_saving` is only worthwhile when LUT pressure is the real bottleneck, `shared_dsp_reducing` is only worthwhile when DSP pressure is the real bottleneck, and both lose their utility when performance matters more than the relieved resource.
- the repo now also carries a measured design-rule extraction layer: flexibility introduces fixed latency/throughput overhead, so it is justified only when the relieved bottleneck dominates that cost; otherwise baseline remains preferable.

## 11. Reproducibility

The full study can be reproduced from the experiment configs already in the repository:

```bash
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_baseline_characterization.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_dense_parallel_scaling.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_precision_resource.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_timing_target.json
python3 experiments/analyze_latency_model.py --input results/fpga/aggregates/study_dense_parallel_scaling.csv
python3 analysis/fpga_plot.py --experiment-id study_dense_parallel_scaling --x-param DENSE_OUT_PAR
python3 analysis/fpga_plot.py --aggregate results/fpga/aggregates/study_dense_parallel_scaling_model.csv --x-param DENSE_OUT_PAR
python3 analysis/fpga_plot.py --experiment-id study_precision_resource --x-param DATA_WIDTH
python3 analysis/fpga_plot.py --experiment-id study_timing_target --x-param CLK_FREQ_HZ
python3 analysis/run_mac_array_framework.py --config experiments/configs/mac_array_framework_v2.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_baseline_4x4.json --fail-fast
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_baseline_8x4.json --fail-fast
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_baseline_8x8.json --fail-fast
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_tradeoff_4x4.json --fail-fast
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_4x4.json --fail-fast
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_lut_8x4.json --fail-fast
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_8x4.json --fail-fast
python3 analysis/run_mac_array_direct_slice.py
python3 experiments/run_measured_refresh.py --preview-scheduler --scheduler resource-aware --max-concurrent-jobs 2 --cpu-threshold-pct 85 --min-free-mem-gb 4 --per-job-mem-gb 8 --vivado-jobs-override 2
```

The main datasets used in this report are:

- `results/fpga/aggregates/study_baseline_characterization.csv`
- `results/fpga/aggregates/study_dense_parallel_scaling.csv`
- `results/fpga/aggregates/study_dense_parallel_scaling_model.csv`
- `results/fpga/aggregates/study_precision_resource.csv`
- `results/fpga/aggregates/study_timing_target.csv`
- `results/fpga/framework_v2/framework_report.md`
- `results/fpga/framework_v2/shared_family_calibration_summary.md`
- `results/fpga/framework_v2/regime_insights.md`
- `results/fpga/framework_v2/direct_slice/direct_calibration_summary.md`
- `results/fpga/framework_v2/direct_slice/direct_tradeoff_summary.md`
- `results/fpga/framework_v2/measured_refresh/comparison_summary.md`
