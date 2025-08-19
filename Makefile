# -------- Config --------
VERILATOR       := verilator
TOP             := top_level
BUILD_DIR       := obj_dir
HDL_SRCS        := $(wildcard hdl/*.sv)

# Testbenches
TB_FULL         := tb/tb_full_pipeline.cpp
TB_CONV2D       := tb/tb_conv2d.cpp

# Verilator flags (ASCII hyphens only)
VERILATOR_FLAGS := -Wall -Wno-fatal --trace

# -------- Phony targets --------
.PHONY: all build run clean realclean build_conv2d run_conv2d help

# Default: build + run full pipeline
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

# -------- conv2d unit test --------
CONV_TOP        := conv2d
build_conv2d: $(BUILD_DIR)/V$(CONV_TOP)

$(BUILD_DIR)/V$(CONV_TOP): $(HDL_SRCS) $(TB_CONV2D)
	$(VERILATOR) $(VERILATOR_FLAGS) \
	  --cc $(HDL_SRCS) --top-module $(CONV_TOP) \
	  --exe $(TB_CONV2D)
	$(MAKE) -C $(BUILD_DIR) -f V$(CONV_TOP).mk -j

run_conv2d: build_conv2d
	$(BUILD_DIR)/V$(CONV_TOP)

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
	@echo "  make build_conv2d / run_conv2d - conv2d unit test"
	@echo "  make clean      - remove obj_dir and VCDs"