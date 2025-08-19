#include "Vtop_level.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>

static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

// Sim parameters
static const int CLK_HALF = 1;   // 1 time unit per half-cycle
static const int BYTES_PER_IMAGE = 28*28;

// Helper: tick once (posedge + negedge)
inline void tick(Vtop_level* tb, VerilatedVcdC* tfp) {
    tb->clk = 0; tb->eval(); if (tfp) tfp->dump(main_time++);
    tb->clk = 1; tb->eval(); if (tfp) tfp->dump(main_time++);
}

// Drive one UART byte (8N1) at CLKS_PER_BIT (cycle-accurate to DUT clock)
void uart_send_byte(Vtop_level* tb, VerilatedVcdC* tfp, uint8_t b, int clks_per_bit) {
    // Start bit (low)
    tb->uart_rx_i = 0;
    for (int i=0;i<clks_per_bit;i++) tick(tb, tfp);

    // 8 data bits, LSB first
    for (int i=0;i<8;i++) {
        tb->uart_rx_i = (b >> i) & 1;
        for (int k=0;k<clks_per_bit;k++) tick(tb, tfp);
    }

    // Stop bit (high)
    tb->uart_rx_i = 1;
    for (int i=0;i<clks_per_bit;i++) tick(tb, tfp);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    // Load image bytes
    std::ifstream fbin("weights/input_image.bin", std::ios::binary);
    if (!fbin) {
        std::cerr << "ERROR: input_image.bin not found (784 bytes expected)\n";
        return 1;
    }
    std::vector<uint8_t> img((std::istreambuf_iterator<char>(fbin)), {});
    if ((int)img.size() != BYTES_PER_IMAGE) {
        std::cerr << "ERROR: input_image.bin has " << img.size()
                  << " bytes; expected " << BYTES_PER_IMAGE << "\n";
        return 1;
    }

    // Instantiate DUT
    Vtop_level* tb = new Vtop_level;

    // Enable VCD
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    tb->trace(tfp, 99);
    tfp->open("wave.vcd");

    // Reset
    tb->clk = 0;
    tb->reset = 1;
    tb->uart_rx_i = 1; // idle high
    for (int i=0;i<10;i++) tick(tb, tfp);
    tb->reset = 0;
    for (int i=0;i<10;i++) tick(tb, tfp);

    // CLKS_PER_BIT must match top_level parameter (default 434)
    const int CLKS_PER_BIT = 434;

    // Send all 784 bytes
    for (int i=0;i<BYTES_PER_IMAGE; ++i) {
        uart_send_byte(tb, tfp, img[i], CLKS_PER_BIT);
        // a few idle cycles between bytes
        for (int k=0;k<4;k++) tick(tb, tfp);
    }

    // Capture one TX byte (poll line; simple edge detect)
    // We’ll oversample by full cycles to keep it simple.
    // Wait for start bit (falling edge)
    uint8_t rx = 0;
    bool got = false;
    int safety = 0;

    while (!got && safety < 2000000) {
        // look for falling edge (start bit)
        if (tb->uart_tx_o == 0) {
            // move to the middle of the start bit
            for (int i = 0; i < CLKS_PER_BIT/2; ++i) tick(tb, tfp);

            // confirm it's still a valid start bit
            if (tb->uart_tx_o != 0) {
                // false start (glitch) – keep searching
                tick(tb, tfp);
                continue;
            }

            // now sample each data bit in the center
            rx = 0;
            for (int bit = 0; bit < 8; ++bit) {
                for (int i = 0; i < CLKS_PER_BIT; ++i) tick(tb, tfp);
                int bitval = tb->uart_tx_o ? 1 : 0;
                rx |= (bitval & 1) << bit;   // LSB-first
            }

            // (optional) check stop bit center
            for (int i = 0; i < CLKS_PER_BIT; ++i) tick(tb, tfp);

            got = true;
            break;
        }
        tick(tb, tfp);
        safety++;
    }

    // Dump result
    if (got) {
        char digit = static_cast<char>(rx);
        unsigned u = static_cast<unsigned>(rx);

        std::cout << "TX byte: 0x" << std::hex << std::uppercase << u
                << std::dec << "  (" << (std::isprint(static_cast<unsigned char>(digit)) ? digit : '?') << ")\n";

        // If it's a decimal digit, also print its numeric value
        if (digit >= '0' && digit <= '9') {
            std::cout << "Predicted digit (numeric): " << (digit - '0') << "\n";
        } else {
            std::cout << "Predicted digit (ASCII): " << digit << "  [non-digit]\n";
        }

        std::ofstream out("uart_digit.txt");
        out << digit << "\n";
    } else {
        std::cerr << "ERROR: Did not see TX output within safety window\n";
    }

    // a few extra ticks
    for (int i=0;i<50;i++) tick(tb, tfp);

    tfp->close();
    delete tfp;
    delete tb;
    return got ? 0 : 2;
}