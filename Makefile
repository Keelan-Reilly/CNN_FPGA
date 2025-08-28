# -------- Config --------
VERILATOR       := verilator
TOP             := top_level
BUILD_DIR       := obj_dir
HDL_SRCS        := $(wildcard hdl/*.sv)

# Testbench
TB_FULL         := tb/tb_full_pipeline.cpp

# Verilator flags
VERILATOR_FLAGS := -Wall -Wno-fatal --trace

# -------- Phony targets --------
.PHONY: all build run clean realclean help

# Default target: build + run full pipeline
all: run

# -------- Full pipeline (top_level) --------
build: $(BUILD_DIR)/V$(TOP)

$(BUILD_DIR)/V$(TOP): $(HDL_SRCS) $(TB_FULL)
	$(VERILATOR) $(VERILATOR_FLAGS) \
	  --cc $(HDL_SRCS) --top-module $(TOP) \
	  --exe $(TB_FULL)
	$(MAKE) -C $(BUILD_DIR) -f V$(TOP).mk -j

run: build
	$(BUILD_DIR)/V$(TOP)

# -------- Utilities --------
clean:
	rm -rf $(BUILD_DIR) *.vcd

realclean: clean
	@echo "Real clean done."

help:
	@echo "Targets:"
	@echo "  make            - build and run full pipeline (top_level)"
	@echo "  make build      - build full pipeline only"
	@echo "  make run        - run full pipeline"
	@echo "  make clean      - remove obj_dir and VCDs"