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
.PHONY: all build run run_full run_batch run_batch_vcd fpga_experiments fpga_experiments_sweep fpga_summary fpga_plots clean realclean help

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

EXP ?= baseline_fpga

fpga_summary:
	python3 analysis/fpga_summary.py --experiment-id $(EXP) --include-failed

fpga_plots:
	python3 analysis/fpga_plot.py --experiment-id $(EXP)

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
	@echo "  make fpga_summary EXP=<experiment_id> - print aggregate summary table (default EXP=baseline_fpga)"
	@echo "  make fpga_plots EXP=<experiment_id>   - generate architecture-study plots (default EXP=baseline_fpga)"
	@echo ""
	@echo "Overrides:"
	@echo "  make run TB=tb/tb_batch_pipeline.cpp ARGS='--count 200 --start 0'"
	
