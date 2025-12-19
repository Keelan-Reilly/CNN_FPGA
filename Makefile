# -------- Config --------
VERILATOR       := verilator
TOP             := top_level
BUILD_DIR       := obj_dir
HDL_SRCS        := $(wildcard hdl/*.sv)

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
.PHONY: all build run run_full run_batch run_batch_vcd clean realclean help

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
run_batch: ARGS= --images data/MNIST/raw/t10k-images-idx3-ubyte --labels data/MNIST/raw/t10k-labels-idx1-ubyte --outdir batch_out --count 1000 --progress 50 --quiet +quiet
run_batch: run

run_batch_vcd: TB=$(TB_BATCH)
run_batch_vcd: ARGS= --images data/MNIST/raw/t10k-images-idx3-ubyte --labels data/MNIST/raw/t10k-labels-idx1-ubyte --outdir batch_out --count 1000 --progress 50 --vcd-on-fail
run_batch_vcd: run

clean:
	rm -rf $(BUILD_DIR) *.vcd batch_out

realclean: clean
	@echo "Real clean done."

help:
	@echo "Targets:"
	@echo "  make / make run           - build+run (default TB=$(TB_FULL))"
	@echo "  make build                - build only"
	@echo "  make run_full             - run full pipeline TB"
	@echo "  make run_batch            - run batch TB (writes batch_out/*)"
	@echo "  make run_batch_vcd        - batch TB + per-failure VCDs"
	@echo ""
	@echo "Overrides:"
	@echo "  make run TB=tb/tb_batch_pipeline.cpp ARGS='--count 200 --start 0'"
	