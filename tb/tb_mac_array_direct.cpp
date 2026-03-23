#include "Vmac_array_direct_top.h"
#include "verilated.h"
#include <cstdint>
#include <iostream>

static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static inline void tick(Vmac_array_direct_top* tb) {
    tb->clk = 0;
    tb->eval();
    ++main_time;
    tb->clk = 1;
    tb->eval();
    ++main_time;
}

static void reset_dut(Vmac_array_direct_top* tb) {
    tb->reset = 1;
    tb->start_i = 0;
    for (int i = 0; i < 5; ++i) tick(tb);
    tb->reset = 0;
    for (int i = 0; i < 5; ++i) tick(tb);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vmac_array_direct_top* tb = new Vmac_array_direct_top;

    tb->clk = 0;
    tb->reset = 0;
    tb->start_i = 0;
    reset_dut(tb);

    tb->start_i = 1;
    tick(tb);
    tb->start_i = 0;

    int frame_cycles = 0;
    const int safety_cycles = 100000;
    while (!tb->done_o && frame_cycles < safety_cycles) {
        tick(tb);
        ++frame_cycles;
    }

    if (!tb->done_o) {
        std::cerr << "ERROR: direct MAC-array slice did not finish within safety window\n";
        delete tb;
        return 1;
    }

    std::cout << "Frame cycles: " << frame_cycles << "\n";
    std::cout << "Signature: " << static_cast<unsigned>(tb->signature_o) << "\n";

    delete tb;
    return 0;
}
