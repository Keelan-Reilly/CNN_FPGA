#include "Vtop_level.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>

// Global time variable for Verilator
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const int BYTES_PER_IMAGE = 28 * 28;

inline void tick(Vtop_level* tb, VerilatedVcdC* tfp) {
    tb->clk = 0; tb->eval(); if (tfp) tfp->dump(main_time++);
    tb->clk = 1; tb->eval(); if (tfp) tfp->dump(main_time++);
}

void uart_send_byte(Vtop_level* tb, VerilatedVcdC* tfp, uint8_t b, int clks_per_bit) {
    tb->uart_rx_i = 0;
    for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);

    for (int i = 0; i < 8; ++i) {
        tb->uart_rx_i = (b >> i) & 1;
        for (int k = 0; k < clks_per_bit; ++k) tick(tb, tfp);
    }

    tb->uart_rx_i = 1;
    for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);
}

static bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def="") {
    for (int i = 1; i+1 < argc; ++i) {
        if (argv[i] == key) return argv[i+1];
    }
    return def;
}

static int get_int_arg(int argc, char** argv, const std::string& key, int def) {
    for (int i = 1; i+1 < argc; ++i) {
        if (argv[i] == key) return std::stoi(argv[i+1]);
    }
    return def;
}

static void reset_dut(Vtop_level* tb, VerilatedVcdC* tfp) {
    tb->reset = 1;
    tb->uart_rx_i = 1; // idle high
    for (int i = 0; i < 10; ++i) tick(tb, tfp);
    tb->reset = 0;
    for (int i = 0; i < 10; ++i) tick(tb, tfp);
}

static bool recv_uart_byte(Vtop_level* tb, VerilatedVcdC* tfp, int clks_per_bit, uint8_t& rx, int safety_cycles) {
    rx = 0;
    bool got = false;
    int safety = 0;

    while (!got && safety < safety_cycles) {
        if (tb->uart_tx_o == 0) {
            // middle of start bit
            for (int i = 0; i < clks_per_bit / 2; ++i) tick(tb, tfp);
            if (tb->uart_tx_o != 0) { tick(tb, tfp); ++safety; continue; }

            rx = 0;
            for (int bit = 0; bit < 8; ++bit) {
                for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);
                int bitval = tb->uart_tx_o ? 1 : 0;
                rx |= (bitval << bit);
            }
            for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp); // stop bit
            got = true;
            break;
        }
        tick(tb, tfp);
        ++safety;
    }
    return got;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    // UART timing: must match DUT
    const int BAUD_RATE = 115200;
    const int CLK_FREQ  = 100000000;
    const int CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    // Inputs
    std::string images_path = get_arg(argc, argv, "--images", "weights/input_image.bin");
    std::string labels_path = get_arg(argc, argv, "--labels", "");
    int n_req = get_int_arg(argc, argv, "--n", -1);
    bool waves = (get_int_arg(argc, argv, "--waves", 0) != 0); // 1 to enable

    std::vector<uint8_t> images;
    if (!read_file(images_path, images)) {
        std::cerr << "ERROR: images file not found: " << images_path << "\n";
        return 1;
    }
    if (images.size() % BYTES_PER_IMAGE != 0) {
        std::cerr << "ERROR: images file size " << images.size()
                  << " not a multiple of " << BYTES_PER_IMAGE << "\n";
        return 1;
    }

    int n_infer = static_cast<int>(images.size() / BYTES_PER_IMAGE);
    int N = (n_req > 0) ? std::min(n_req, n_infer) : n_infer;

    std::vector<uint8_t> labels;
    bool have_labels = false;
    if (!labels_path.empty()) {
        if (!read_file(labels_path, labels)) {
            std::cerr << "ERROR: labels file not found: " << labels_path << "\n";
            return 1;
        }
        if ((int)labels.size() < N) {
            std::cerr << "ERROR: labels file has " << labels.size()
                      << " bytes, but need at least " << N << "\n";
            return 1;
        }
        have_labels = true;
    }

    Vtop_level* tb = new Vtop_level;

    VerilatedVcdC* tfp = nullptr;
    if (waves) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        tb->trace(tfp, 99);
        tfp->open("wave.vcd");
    }

    tb->clk = 0;
    tb->uart_rx_i = 1;
    reset_dut(tb, tfp);

    int correct = 0;
    int total   = 0;

    for (int idx = 0; idx < N; ++idx) {
        // Reset between frames for a clean batch run
        reset_dut(tb, tfp);

        const uint8_t* img = &images[idx * BYTES_PER_IMAGE];

        // Send 784 bytes
        for (int i = 0; i < BYTES_PER_IMAGE; ++i) {
            uart_send_byte(tb, tfp, img[i], CLKS_PER_BIT);
            for (int k = 0; k < 4; ++k) tick(tb, tfp); // small idle gap
        }

        // Receive prediction
        uint8_t rx = 0;
        bool got = recv_uart_byte(tb, tfp, CLKS_PER_BIT, rx, 2'000'000);

        if (!got) {
            std::cerr << "[idx " << idx << "] ERROR: no TX within safety window\n";
            continue;
        }

        char digit = static_cast<char>(rx);
        int pred = (digit >= '0' && digit <= '9') ? (digit - '0') : -1;

        if (have_labels) {
            int gold = labels[idx] % 10;
            bool ok = (pred == gold);
            correct += ok ? 1 : 0;
            total   += 1;

            std::cout << "[idx " << idx << "] pred=" << pred << " gold=" << gold
                      << (ok ? "  OK\n" : "  FAIL\n");
        } else {
            std::cout << "[idx " << idx << "] pred=" << pred
                      << " (TX=0x" << std::hex << std::uppercase << (unsigned)rx
                      << std::dec << ")\n";
        }

        // optional: stop early on first failure
        // if (have_labels && pred != (labels[idx]%10)) break;
    }

    if (have_labels && total > 0) {
        double acc = 100.0 * (double)correct / (double)total;
        std::cout << "Batch accuracy: " << correct << "/" << total
                  << " = " << std::fixed << std::setprecision(2) << acc << "%\n";
    }

    if (tfp) {
        tfp->close();
        delete tfp;
    }
    delete tb;
    return 0;
}