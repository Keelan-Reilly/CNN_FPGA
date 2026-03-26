# -------- Config --------
VERILATOR       := verilator
TOP             := top_level
BUILD_DIR       := obj_dir
HDL_SRCS        := $(wildcard hdl/*.sv)
BATCH_OUT_DIR   := results/verilator/batch

# Testbenches
TB_FULL         := tb/tb_full_pipeline.cpp
TB_BATCH        := tb/tb_batch_pipeline.cpp
TB              ?= $(TB_FULL)

# Verilator flags
VERILATOR_FLAGS := -sv -Wall -Wno-fatal --trace \
                   -CFLAGS -std=c++17

# Runtime args to pass to the sim (override: make run ARGS="--count 1000 ...")
ARGS            ?=

# -------- Phony targets --------
.PHONY: all build run run_full run_batch run_batch_vcd fpga_experiments fpga_experiments_sweep fpga_experiments_parallel fpga_queue_preview fpga_summary fpga_plots fpga_framework_v2 fpga_refresh_preview fpga_refresh_execute fpga_mac_direct_preview fpga_mac_direct_tradeoff_preview fpga_mac_direct_shared_dsp_preview fpga_mac_direct_shared_lut_8x4_preview fpga_mac_direct_shared_dsp_8x4_preview fpga_mac_direct_shared_lut_8x8_preview fpga_mac_direct_shared_dsp_8x8_preview fpga_mac_direct_4x4 fpga_mac_direct_tradeoff_4x4 fpga_mac_direct_shared_dsp_4x4 fpga_mac_direct_8x4 fpga_mac_direct_shared_lut_8x4 fpga_mac_direct_shared_dsp_8x4 fpga_mac_direct_8x8 fpga_mac_direct_shared_lut_8x8 fpga_mac_direct_shared_dsp_8x8 fpga_mac_direct_report test clean realclean help

all: run

build: $(BUILD_DIR)/V$(TOP)

$(BUILD_DIR)/V$(TOP): $(HDL_SRCS) $(TB)
	$(VERILATOR) $(VERILATOR_FLAGS) \
	  --cc $(HDL_SRCS) --top-module $(TOP) \
	  --exe $(TB)
	$(MAKE) -C $(BUILD_DIR) -f V$(TOP).mk -j

run: build
	$(BUILD_DIR)/V$(TOP) $(ARGS)

# Convenience wrappers
run_full: TB=$(TB_FULL)
run_full: ARGS?=
run_full: run

run_batch: TB=$(TB_BATCH)
run_batch: ARGS= --images data/MNIST/raw/t10k-images-idx3-ubyte --labels data/MNIST/raw/t10k-labels-idx1-ubyte --outdir $(BATCH_OUT_DIR) --count 100 --progress 50 --quiet +quiet
run_batch: run

run_batch_vcd: TB=$(TB_BATCH)
run_batch_vcd: ARGS= --images data/MNIST/raw/t10k-images-idx3-ubyte --labels data/MNIST/raw/t10k-labels-idx1-ubyte --outdir $(BATCH_OUT_DIR) --count 1000 --progress 50 --vcd-on-fail
run_batch_vcd: run

fpga_experiments:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/baseline_fpga.json

fpga_experiments_sweep:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/sweep_bitwidth.json

CFG ?= experiments/configs/study_dense_parallel_scaling.json
SCHED_ARGS ?= --scheduler resource-aware --max-concurrent-jobs 2 --cpu-threshold-pct 85 --min-free-mem-gb 4 --per-job-mem-gb 8 --vivado-jobs-override 2

fpga_experiments_parallel:
	python3 experiments/run_fpga_experiments.py --config $(CFG) $(SCHED_ARGS)

fpga_queue_preview:
	python3 experiments/run_fpga_experiments.py --config $(CFG) $(SCHED_ARGS) --dry-run

EXP ?= baseline_fpga

fpga_summary:
	python3 analysis/fpga_summary.py --experiment-id $(EXP) --include-failed

fpga_plots:
	python3 analysis/fpga_plot.py --experiment-id $(EXP)

fpga_framework_v2:
	python3 analysis/run_mac_array_framework.py --config experiments/configs/mac_array_framework_v2.json

fpga_mac_direct_preview:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_baseline.json --dry-run

fpga_mac_direct_4x4:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_baseline_4x4.json --fail-fast

fpga_mac_direct_tradeoff_preview:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_tradeoff_4x4.json --dry-run

fpga_mac_direct_tradeoff_4x4:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_tradeoff_4x4.json --fail-fast

fpga_mac_direct_shared_dsp_preview:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_4x4.json --dry-run

fpga_mac_direct_shared_dsp_4x4:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_4x4.json --fail-fast

fpga_mac_direct_8x4:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_baseline_8x4.json --fail-fast

fpga_mac_direct_shared_lut_8x4_preview:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_lut_8x4.json --dry-run

fpga_mac_direct_shared_lut_8x4:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_lut_8x4.json --fail-fast

fpga_mac_direct_shared_dsp_8x4_preview:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_8x4.json --dry-run

fpga_mac_direct_shared_dsp_8x4:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_8x4.json --fail-fast

fpga_mac_direct_8x8:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_baseline_8x8.json --fail-fast

fpga_mac_direct_shared_lut_8x8_preview:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_lut_8x8.json --dry-run

fpga_mac_direct_shared_lut_8x8:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_lut_8x8.json --fail-fast

fpga_mac_direct_shared_dsp_8x8_preview:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_8x8.json --dry-run

fpga_mac_direct_shared_dsp_8x8:
	python3 experiments/run_fpga_experiments.py --config experiments/configs/study_mac_array_direct_shared_dsp_8x8.json --fail-fast

fpga_mac_direct_report:
	python3 analysis/run_mac_array_direct_slice.py

fpga_refresh_preview:
	python3 experiments/run_measured_refresh.py --preview-scheduler $(SCHED_ARGS)

fpga_refresh_execute:
	python3 experiments/run_measured_refresh.py --execute $(SCHED_ARGS)

test:
	python3 -m unittest discover -s tests -v

clean:
	rm -rf $(BUILD_DIR) *.vcd $(BATCH_OUT_DIR) batch_out

realclean: clean
	@echo "Real clean done."

help:
	@echo "Targets:"
	@echo "  make / make run           - build+run (default TB=$(TB_FULL))"
	@echo "  make build                - build only"
	@echo "  make run_full             - run full pipeline TB"
	@echo "  make run_batch            - run batch TB (writes $(BATCH_OUT_DIR)/*)"
	@echo "  make run_batch_vcd        - batch TB + per-failure VCDs"
	@echo "  make fpga_experiments     - run baseline Vivado experiment config"
	@echo "  make fpga_experiments_sweep - run bitwidth sweep Vivado config"
	@echo "  make fpga_experiments_parallel CFG=<config> - run optional resource-aware Vivado queue"
	@echo "  make fpga_queue_preview CFG=<config>        - preview the optional Vivado queue without launching jobs"
	@echo "  make fpga_summary EXP=<experiment_id> - print aggregate summary table (default EXP=baseline_fpga)"
	@echo "  make fpga_plots EXP=<experiment_id>   - generate architecture-study plots (default EXP=baseline_fpga)"
	@echo "  make fpga_framework_v2      - run workload-aware MAC-array framework v2 analysis"
	@echo "  make fpga_mac_direct_preview - preview the direct MAC-array baseline sweep queue"
	@echo "  make fpga_mac_direct_tradeoff_preview - preview the directly measurable 4x4 baseline-vs-shared tradeoff queue"
	@echo "  make fpga_mac_direct_shared_dsp_preview - preview the directly measurable 4x4 DSP-oriented shared point"
	@echo "  make fpga_mac_direct_4x4     - run one scoped directly measurable 4x4 MAC-array baseline point"
	@echo "  make fpga_mac_direct_tradeoff_4x4 - run the smallest directly measurable 4x4 baseline-vs-shared tradeoff"
	@echo "  make fpga_mac_direct_shared_dsp_4x4 - run the smallest directly measurable 4x4 DSP-oriented shared point"
	@echo "  make fpga_mac_direct_8x4     - run one scoped directly measurable 8x4 MAC-array baseline point"
	@echo "  make fpga_mac_direct_shared_lut_8x4_preview - preview the directly measurable 8x4 LUT-oriented shared point"
	@echo "  make fpga_mac_direct_shared_lut_8x4 - run the directly measurable 8x4 LUT-oriented shared point"
	@echo "  make fpga_mac_direct_shared_dsp_8x4_preview - preview the directly measurable 8x4 DSP-oriented shared point"
	@echo "  make fpga_mac_direct_shared_dsp_8x4 - run the directly measurable 8x4 DSP-oriented shared point"
	@echo "  make fpga_mac_direct_8x8     - run one scoped directly measurable 8x8 MAC-array baseline point"
	@echo "  make fpga_mac_direct_shared_lut_8x8_preview - preview the directly measurable 8x8 LUT-oriented shared point"
	@echo "  make fpga_mac_direct_shared_lut_8x8 - run the directly measurable 8x8 LUT-oriented shared point"
	@echo "  make fpga_mac_direct_shared_dsp_8x8_preview - preview the directly measurable 8x8 DSP-oriented shared point"
	@echo "  make fpga_mac_direct_shared_dsp_8x8 - run the directly measurable 8x8 DSP-oriented shared point"
	@echo "  make fpga_mac_direct_report  - generate direct measured-vs-modelled slice artifacts"
	@echo "  make fpga_refresh_preview   - build selective measured-refresh artifacts and preview the runnable queue"
	@echo "  make fpga_refresh_execute   - run the selective measured-refresh queue with the same scheduler knobs"
	@echo "  make test                   - run deterministic Python unit tests"
	@echo ""
	@echo "Overrides:"
	@echo "  make run TB=tb/tb_batch_pipeline.cpp ARGS='--count 200 --start 0'"
	
