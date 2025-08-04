// tb_conv2d.cpp
#include "Vconv2d.h"            // generated from conv2d.v
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>
#include <iomanip>

vluint64_t main_time = 0;
// Called by $time in Verilog
double sc_time_stamp() { return main_time; }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);

    // Instantiate DUT
    Vconv2d* tb = new Vconv2d;

    // Setup VCD waveform tracing
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    tb->trace(tfp, 99);
    tfp->open("wave.vcd");

    // Constants must match the Verilog parameters
    const int FRAC_BITS   = 7;
    const int IMG_SIZE    = 5;   // small test image
    const int KERNEL_SIZE = 3;   // 3x3 kernel

    // --- Reset sequence ---
    tb->clk   = 0;
    tb->reset = 1;
    tb->start = 0;
    for (int i = 0; i < 4; i++) {
        // Toggle clock twice per iteration for a few cycles
        tb->clk = !tb->clk; tb->eval(); tfp->dump(main_time++);
        tb->clk = !tb->clk; tb->eval(); tfp->dump(main_time++);
    }
    tb->reset = 0;  // release reset

    // --- Load input_feature: values 1..25 scaled as fixed-point ---
    int val = 1;
    for (int r = 0; r < IMG_SIZE; r++) {
        for (int c = 0; c < IMG_SIZE; c++) {
            int fixed = val << FRAC_BITS;     // multiply by 2^FRAC_BITS
            tb->input_feature[0][r][c] = fixed;
            val++;
        }
    }

    // --- Load weights: all ones (fixed-point 1.0) ---
    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
        for (int kc = 0; kc < KERNEL_SIZE; kc++) {
            tb->weights[0][0][kr][kc] = (1 << FRAC_BITS);
        }
    }

    // --- Load biases: zero ---
    tb->biases[0] = 0;

    // --- Start convolution ---
    tb->start = 1;
    tb->clk   = !tb->clk; tb->eval(); tfp->dump(main_time++);
    tb->start = 0;

    // --- Run until done ---
    while (!tb->done) {
        tb->clk = !tb->clk; tb->eval(); tfp->dump(main_time++);
    }
    // One extra cycle to settle outputs
    tb->clk = !tb->clk; tb->eval(); tfp->dump(main_time++);

    // --- Print output feature map (integer form) ---
    std::cout << "Output feature map (integers):\n";
    for (int r = 0; r < IMG_SIZE; r++) {
        for (int c = 0; c < IMG_SIZE; c++) {
            int raw = tb->out_feature[0][r][c];
            int val_int = raw >> FRAC_BITS;  // convert back from fixed-point
            std::cout << std::setw(3) << val_int << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Simulation complete.\n";

    // Cleanup
    tfp->close();
    delete tb;
    return 0;
}