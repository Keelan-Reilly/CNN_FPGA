//======================================================================
// tb_full_pipeline.cpp — Verilator testbench for full inference pipeline
//----------------------------------------------------------------------
// This testbench simulates the full digit-classification pipeline in Verilator.
// It reads a raw 28×28 image from file, sends it over UART (8N1 format) to the DUT,
// waits for a UART response byte (ASCII-encoded digit), and prints/logs the result.
//
// Waveforms are dumped to wave.vcd for inspection in WaveTrace.
// UART timing is cycle-accurate to match DUT’s CLKS_PER_BIT.
//======================================================================

#include "Vtop_level.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>

// Global time variable for Verilator
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

// Image size: 28×28 grayscale pixels
static const int BYTES_PER_IMAGE = 28 * 28;

//----------------------------------------------------------------------
// tick() — Advance simulation by one full clock cycle (posedge + negedge)
//----------------------------------------------------------------------
inline void tick(Vtop_level* tb, VerilatedVcdC* tfp) {
    tb->clk = 0; tb->eval(); if (tfp) tfp->dump(main_time++);
    tb->clk = 1; tb->eval(); if (tfp) tfp->dump(main_time++);
}

//----------------------------------------------------------------------
// uart_send_byte() — Send 1 byte over UART to DUT
//   Format: 8 data bits, no parity, 1 stop bit (8N1)
//   Bit timing is controlled by CLKS_PER_BIT, matching DUT UART clock divisor
//----------------------------------------------------------------------
void uart_send_byte(Vtop_level* tb, VerilatedVcdC* tfp, uint8_t b, int clks_per_bit) {
    // Start bit (logic 0)
    tb->uart_rx_i = 0;
    for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);

    // Data bits (LSB first)
    for (int i = 0; i < 8; ++i) {
        tb->uart_rx_i = (b >> i) & 1;
        for (int k = 0; k < clks_per_bit; ++k) tick(tb, tfp);
    }

    // Stop bit (logic 1)
    tb->uart_rx_i = 1;
    for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);
}

//----------------------------------------------------------------------
// main() — Simulation entry point
//----------------------------------------------------------------------
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    // Load raw image input (28×28 grayscale, 784 bytes)
    std::ifstream fbin("weights/input_image.bin", std::ios::binary);
    if (!fbin) {
        std::cerr << "ERROR: input_image.bin not found (expected 784 bytes)\n";
        return 1;
    }

    std::vector<uint8_t> img((std::istreambuf_iterator<char>(fbin)), {});
    if ((int)img.size() != BYTES_PER_IMAGE) {
        std::cerr << "ERROR: input_image.bin has " << img.size()
                  << " bytes; expected " << BYTES_PER_IMAGE << "\n";
        return 1;
    }

    // Instantiate DUT (Design Under Test)
    Vtop_level* tb = new Vtop_level;

    // Enable waveform tracing
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    tb->trace(tfp, 99);
    tfp->open("wave.vcd");

    // Initial reset
    tb->clk = 0;
    tb->reset = 1;
    tb->uart_rx_i = 1; // idle line high
    for (int i = 0; i < 10; ++i) tick(tb, tfp);
    tb->reset = 0;
    for (int i = 0; i < 10; ++i) tick(tb, tfp);

    // UART bit timing: matches DUT's CLKS_PER_BIT
    const int BAUD_RATE = 115200;
    const int CLK_FREQ = 100000000; // 100 MHz
    const int CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    // Send all 784 bytes over UART
    for (int i = 0; i < BYTES_PER_IMAGE; ++i) {
        uart_send_byte(tb, tfp, img[i], CLKS_PER_BIT);
        // Optional idle time between bytes (helps prevent RX FIFO overflow)
        for (int k = 0; k < 4; ++k) tick(tb, tfp);
    }

    //----------------------------------------------------------------------  
    // Wait for UART TX response (1 byte): digit prediction (ASCII '0'–'9')
    //----------------------------------------------------------------------  

    uint8_t rx = 0;
    bool got = false;
    int safety = 0; // infinite-loop guard

    while (!got && safety < 2'000'000) {
        if (tb->uart_tx_o == 0) {
            // Align to middle of start bit
            for (int i = 0; i < CLKS_PER_BIT / 2; ++i) tick(tb, tfp);
            if (tb->uart_tx_o != 0) { tick(tb, tfp); continue; }

            // Sample each of the 8 data bits (LSB first)
            rx = 0;
            for (int bit = 0; bit < 8; ++bit) {
                for (int i = 0; i < CLKS_PER_BIT; ++i) tick(tb, tfp);
                int bitval = tb->uart_tx_o ? 1 : 0;
                rx |= (bitval << bit);
            }

            // Wait one stop bit (optional)
            for (int i = 0; i < CLKS_PER_BIT; ++i) tick(tb, tfp);
            got = true;
            break;
        }
        tick(tb, tfp);
        ++safety;
    }

    //----------------------------------------------------------------------  
    // Print prediction result and write to output file
    //----------------------------------------------------------------------  
    if (got) {
        char digit = static_cast<char>(rx);
        unsigned u = static_cast<unsigned>(rx);

        std::cout << "TX byte: 0x" << std::hex << std::uppercase << u
                  << std::dec << "  (" << (std::isprint(digit) ? digit : '?') << ")\n";

        if (digit >= '0' && digit <= '9') {
            std::cout << "Predicted digit (numeric): " << (digit - '0') << "\n";
        } else {
            std::cout << "Predicted digit (ASCII): " << digit << "  [non-digit]\n";
        }

        // Dump to file
        std::ofstream out("uart_digit.txt");
        out << digit << "\n";
    } else {
        std::cerr << "ERROR: Did not see TX output within safety window\n";
    }

    // Run a few extra cycles to drain logic
    for (int i = 0; i < 50; ++i) tick(tb, tfp);

    // Cleanup
    tfp->close();
    delete tfp;
    delete tb;

    return got ? 0 : 2;
}