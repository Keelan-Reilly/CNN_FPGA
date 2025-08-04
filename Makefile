VERILATOR = verilator
TOP = top_level
BUILD_DIR = obj_dir

all: sim

sim:
	$(VERILATOR) --cc hdl/$(TOP).v --exe tb/tb_full_pipeline.cpp \
	  -Wall -Wno-fatal --trace
	make -C $(BUILD_DIR) -j --quiet
	$(BUILD_DIR)/V$(TOP) +vcd

clean:
	rm -rf $(BUILD_DIR) sim/*.vcd