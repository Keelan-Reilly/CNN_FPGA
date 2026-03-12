# FPGA Microarchitecture Study Suite

This study suite is intentionally small. It uses only the currently supported RTL and experiment knobs, and it is organized to answer a short list of architecture questions cleanly.

## 1. Baseline characterization

- Config: `experiments/configs/study_baseline_characterization.json`
- Question: what is the reference implementation point for area, timing, end-to-end latency, and stage bottlenecks?
- Use this as the anchor for every later comparison.

Generate:

- summary table
- stage-cycle breakdown
- latency and throughput reference values

Recommended plots:

- `stage_cycles_breakdown_vs_*`
- area/timing plots are not the focus here because this config is a single point

What to look for:

- conv should clearly dominate total latency
- dense should be a secondary contributor
- bubble cycles should remain negligible

## 2. Primary parallelism/scaling sweep

- Config: `experiments/configs/study_dense_parallel_scaling.json`
- Question: how does dense-stage parallelism scale locally, and why does end-to-end speedup saturate?
- This is the centerpiece experiment because it directly answers the current architecture question: local dense scaling versus global conv bottleneck.

Generate:

- the experiment aggregate
- the sibling model dataset with `experiments/analyze_latency_model.py`
- measured and model-aware plots

Recommended plots:

- measured latency vs `DENSE_OUT_PAR`
- throughput vs `DENSE_OUT_PAR`
- stage-cycle breakdown vs `DENSE_OUT_PAR`
- measured vs predicted latency
- measured vs predicted throughput
- model fixed-vs-dense decomposition
- speedup vs LUT
- speedup vs DSP

What to look for:

- dense cycles should shrink roughly with `1 / DENSE_OUT_PAR`
- conv cycles should remain flat and dominate total latency
- end-to-end speedup should flatten quickly
- LUT/DSP cost should rise faster than end-to-end speedup

## 3. Precision/resource sweep

- Config: `experiments/configs/study_precision_resource.json`
- Question: how much implementation cost changes with arithmetic precision when the architecture is otherwise fixed?
- This is useful as a control experiment for the report: it shows numeric format sensitivity without changing the microarchitectural schedule.

Recommended plots:

- LUT vs precision point
- FF vs precision point
- DSP vs precision point
- BRAM vs precision point
- Fmax vs precision point
- latency vs precision point
- throughput-per-LUT vs precision point

What to look for:

- cycle counts should stay nearly unchanged
- area and timing should move more than latency cycles
- this helps separate arithmetic-cost effects from architectural-parallelism effects

## 4. Timing-target sweep

- Config: `experiments/configs/study_timing_target.json`
- Question: how aggressively can the current baseline architecture be targeted in implementation, and what happens to timing closure and throughput-at-target?
- `CLK_FREQ_HZ` is aligned with the implementation target so the reported time/throughput fields remain meaningful for this sweep.

Recommended plots:

- WNS vs target
- Fmax vs target
- latency time vs target
- throughput vs target
- area-performance scatter

What to look for:

- where timing transitions from clean closure to negative slack
- whether higher requested clock targets produce meaningful throughput gains on paper
- whether timing target stress changes the reported bottleneck summary

## Report structure suggestion

- Start with the baseline characterization.
- Use the dense parallel sweep as the main architectural result.
- Use the precision sweep as a control showing that arithmetic-width changes mostly affect cost, not schedule.
- Use the timing-target sweep to discuss implementation limits and realistic operating points.

## Commands

Run experiments:

```bash
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_baseline_characterization.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_dense_parallel_scaling.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_precision_resource.json
python3 experiments/run_fpga_experiments.py --config experiments/configs/study_timing_target.json
```

Build the model dataset for the primary sweep:

```bash
python3 experiments/analyze_latency_model.py --input results/fpga/aggregates/study_dense_parallel_scaling.csv
```

Generate plots:

```bash
python3 analysis/fpga_plot.py --experiment-id study_baseline_characterization
python3 analysis/fpga_plot.py --experiment-id study_dense_parallel_scaling --x-param DENSE_OUT_PAR
python3 analysis/fpga_plot.py --aggregate results/fpga/aggregates/study_dense_parallel_scaling_model.csv --x-param DENSE_OUT_PAR
python3 analysis/fpga_plot.py --experiment-id study_precision_resource --x-param DATA_WIDTH
python3 analysis/fpga_plot.py --experiment-id study_timing_target --x-param CLK_FREQ_HZ
```
