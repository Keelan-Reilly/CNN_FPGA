//======================================================================
// tb_batch_pipeline.cpp — Verilator batch testbench (MNIST IDX format)
//----------------------------------------------------------------------
// Runs a batch of MNIST test images through the UART-fed DUT, records
// pass/fail against labels, writes results.csv + batch.log + failures.txt.
// Optionally re-runs failing indices with a per-failure VCD.
//
// Usage example:
//   ./obj_dir/Vtop_level --images data/t10k-images-idx3-ubyte --labels data/t10k-labels-idx1-ubyte --count 1000
//
// Notes:
// - Assumes DUT accepts 784 bytes over UART RX, then outputs 1 byte over UART TX ('0'..'9').
// - Keeps terminal output sane; everything is written to log files.
//======================================================================

#include "Vtop_level.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static constexpr int IMG_W = 28;
static constexpr int IMG_H = 28;
static constexpr int BYTES_PER_IMAGE = IMG_W * IMG_H;

static inline uint32_t read_be_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

struct MnistImages {
    uint32_t count = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint8_t> data; // count * rows * cols
};

struct MnistLabels {
    uint32_t count = 0;
    std::vector<uint8_t> data; // count
};

static bool load_mnist_images(const std::string& path, MnistImages& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "Could not open images file: " + path; return false; }

    uint32_t magic = read_be_u32(f);
    if (magic != 2051) { err = "Bad images magic (expected 2051): got " + std::to_string(magic); return false; }

    out.count = read_be_u32(f);
    out.rows  = read_be_u32(f);
    out.cols  = read_be_u32(f);

    if (out.rows != IMG_H || out.cols != IMG_W) {
        std::ostringstream oss;
        oss << "Unexpected image dims: " << out.rows << "x" << out.cols << " (expected 28x28)";
        err = oss.str();
        return false;
    }

    const size_t bytes = size_t(out.count) * size_t(out.rows) * size_t(out.cols);
    out.data.resize(bytes);
    f.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(bytes));
    if (f.gcount() != static_cast<std::streamsize>(bytes)) {
        err = "Images file truncated (read " + std::to_string(f.gcount()) + " of " + std::to_string(bytes) + ")";
        return false;
    }
    return true;
}

static bool load_mnist_labels(const std::string& path, MnistLabels& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "Could not open labels file: " + path; return false; }

    uint32_t magic = read_be_u32(f);
    if (magic != 2049) { err = "Bad labels magic (expected 2049): got " + std::to_string(magic); return false; }

    out.count = read_be_u32(f);
    out.data.resize(out.count);
    f.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(out.count));
    if (f.gcount() != static_cast<std::streamsize>(out.count)) {
        err = "Labels file truncated (read " + std::to_string(f.gcount()) + " of " + std::to_string(out.count) + ")";
        return false;
    }
    return true;
}

//----------------------------------------------------------------------
// tick() — one full clock cycle
//----------------------------------------------------------------------
static inline void tick(Vtop_level* tb, VerilatedVcdC* tfp) {
    tb->clk = 0; tb->eval(); if (tfp) tfp->dump(main_time); main_time++;
    tb->clk = 1; tb->eval(); if (tfp) tfp->dump(main_time); main_time++;
}

//----------------------------------------------------------------------
// UART send byte (8N1)
//----------------------------------------------------------------------
static void uart_send_byte(Vtop_level* tb, VerilatedVcdC* tfp, uint8_t b, int clks_per_bit) {
    tb->uart_rx_i = 0;
    for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);

    for (int i = 0; i < 8; ++i) {
        tb->uart_rx_i = (b >> i) & 1;
        for (int k = 0; k < clks_per_bit; ++k) tick(tb, tfp);
    }

    tb->uart_rx_i = 1;
    for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);
}

//----------------------------------------------------------------------
// UART receive one byte (8N1), returns got=false on timeout
//----------------------------------------------------------------------
static bool uart_recv_byte(Vtop_level* tb, VerilatedVcdC* tfp, int clks_per_bit, uint8_t& out_byte, int max_cycles) {
    int safety = 0;
    while (safety < max_cycles) {
        if (tb->uart_tx_o == 0) {
            for (int i = 0; i < clks_per_bit / 2; ++i) tick(tb, tfp);
            if (tb->uart_tx_o != 0) { tick(tb, tfp); ++safety; continue; }

            uint8_t rx = 0;
            for (int bit = 0; bit < 8; ++bit) {
                for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);
                rx |= (tb->uart_tx_o ? 1 : 0) << bit;
            }

            for (int i = 0; i < clks_per_bit; ++i) tick(tb, tfp);
            out_byte = rx;
            return true;
        }
        tick(tb, tfp);
        ++safety;
    }
    return false;
}

static void apply_reset(Vtop_level* tb, VerilatedVcdC* tfp, int cycles) {
    tb->reset = 1;
    for (int i = 0; i < cycles; ++i) tick(tb, tfp);
    tb->reset = 0;
    for (int i = 0; i < cycles; ++i) tick(tb, tfp);
}

static std::string arg_value(int argc, char** argv, const std::string& key, const std::string& def = "") {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key && i + 1 < argc) return argv[i + 1];
    }
    return def;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == key) return true;
    return false;
}

static int arg_int(int argc, char** argv, const std::string& key, int def) {
    const std::string v = arg_value(argc, argv, key, "");
    if (v.empty()) return def;
    return std::stoi(v);
}

static void ensure_dir(const std::string& p) {
    std::filesystem::create_directories(std::filesystem::path(p));
}

static std::string path_join(const std::string& a, const std::string& b) {
    return (std::filesystem::path(a) / std::filesystem::path(b)).string();
}

static std::string vcd_name_for_fail(const std::string& outdir, int idx) {
    std::ostringstream oss;
    oss << "wave_fail_idx" << idx << ".vcd";
    return path_join(outdir, oss.str());
}

struct RunResult {
    int idx = -1;
    int gold = -1;
    int pred = -1;
    bool ok = false;
    bool got_uart = false;
    uint64_t cycles = 0;
};

static RunResult run_one(
    Vtop_level* tb,
    VerilatedVcdC* tfp,
    const uint8_t* img784,
    int idx,
    int gold,
    int clks_per_bit,
    int idle_ticks_between_bytes,
    int uart_timeout_cycles,
    bool do_reset_each
) {
    const vluint64_t start_time = main_time;

    if (do_reset_each) apply_reset(tb, tfp, 10);

    // Ensure idle UART RX
    tb->uart_rx_i = 1;

    // Send 784 bytes
    for (int i = 0; i < BYTES_PER_IMAGE; ++i) {
        uart_send_byte(tb, tfp, img784[i], clks_per_bit);
        for (int k = 0; k < idle_ticks_between_bytes; ++k) tick(tb, tfp);
    }

    // Wait for prediction
    uint8_t rx = 0;
    bool got = uart_recv_byte(tb, tfp, clks_per_bit, rx, uart_timeout_cycles);

    // Drain a few cycles
    for (int i = 0; i < 50; ++i) tick(tb, tfp);

    RunResult rr;
    rr.idx = idx;
    rr.gold = gold;
    rr.got_uart = got;
    rr.cycles = uint64_t(main_time - start_time) / 2; // convert ticks to cycles

    if (got && rx >= '0' && rx <= '9') rr.pred = int(rx - '0');
    else rr.pred = -1;

    rr.ok = (rr.gold >= 0 && rr.pred == rr.gold && rr.got_uart);
    return rr;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string images_path = arg_value(argc, argv, "--images", "");
    const std::string labels_path = arg_value(argc, argv, "--labels", "");
    const std::string outdir      = arg_value(argc, argv, "--outdir", "batch_out");

    const int start_idx = arg_int(argc, argv, "--start", 0);
    const int count     = arg_int(argc, argv, "--count", 100);
    const int stride    = arg_int(argc, argv, "--stride", 1);
    const int progressN = arg_int(argc, argv, "--progress", 50);

    const bool vcd_on_fail   = has_flag(argc, argv, "--vcd-on-fail");
    const bool reset_each    = has_flag(argc, argv, "--reset-each");
    const bool quiet_console = has_flag(argc, argv, "--quiet"); // suppress even progress

    // UART parameters
    const int clk_hz = arg_int(argc, argv, "--clk-hz", 100000000);
    const int baud   = arg_int(argc, argv, "--baud", 115200);
    const int clks_per_bit = arg_int(argc, argv, "--clks-per-bit", clk_hz / baud);

    // Small idle between bytes (prevents FIFO overflow in some designs)
    const int idle_ticks_between_bytes = arg_int(argc, argv, "--idle", 4);

    // Timeout while waiting for DUT TX (in *cycles*, not ticks)
    const int uart_timeout_cycles = arg_int(argc, argv, "--timeout", 2000000);

    if (images_path.empty()) {
        std::cerr << "ERROR: --images is required (MNIST t10k-images-idx3-ubyte)\n";
        return 2;
    }
    if (labels_path.empty()) {
        std::cerr << "ERROR: --labels is required (MNIST t10k-labels-idx1-ubyte)\n";
        return 2;
    }

    ensure_dir(outdir);

    // Load MNIST
    MnistImages imgs;
    MnistLabels lbls;
    std::string err;

    if (!load_mnist_images(images_path, imgs, err)) {
        std::cerr << "ERROR loading images: " << err << "\n";
        return 2;
    }
    if (!load_mnist_labels(labels_path, lbls, err)) {
        std::cerr << "ERROR loading labels: " << err << "\n";
        return 2;
    }
    if (lbls.count != imgs.count) {
        std::cerr << "ERROR: images count (" << imgs.count << ") != labels count (" << lbls.count << ")\n";
        return 2;
    }

    const int end_idx = start_idx + (count - 1) * stride;
    if (start_idx < 0 || end_idx >= int(imgs.count)) {
        std::cerr << "ERROR: requested indices out of range. dataset count=" << imgs.count
                  << " start=" << start_idx << " end=" << end_idx << "\n";
        return 2;
    }

    // Open logs
    std::ofstream csv(path_join(outdir, "results.csv"));
    std::ofstream blog(path_join(outdir, "batch.log"));
    std::ofstream fails(path_join(outdir, "failures.txt"));

    csv  << "idx,gold,pred,ok,got_uart,cycles\n";
    blog << "Batch run\n"
         << " images: " << images_path << "\n"
         << " labels: " << labels_path << "\n"
         << " start=" << start_idx << " count=" << count << " stride=" << stride << "\n"
         << " clk_hz=" << clk_hz << " baud=" << baud << " clks_per_bit=" << clks_per_bit << "\n"
         << " idle_ticks_between_bytes=" << idle_ticks_between_bytes
         << " timeout_cycles=" << uart_timeout_cycles << "\n"
         << " vcd_on_fail=" << (vcd_on_fail ? "true" : "false")
         << " reset_each=" << (reset_each ? "true" : "false") << "\n\n";

    // Instantiate DUT
    Vtop_level* tb = new Vtop_level;
    tb->clk = 0;
    tb->reset = 0;
    tb->uart_rx_i = 1; // idle high

    // Tracing is compiled-in via --trace; we only open VCD when needed.
    Verilated::traceEverOn(true);

    // Initial reset once
    apply_reset(tb, nullptr, 10);

    int pass = 0, fail_cnt = 0, uart_fail = 0;
    uint64_t total_cycles = 0;

    for (int n = 0; n < count; ++n) {
        const int idx = start_idx + n * stride;
        const int gold = int(lbls.data[idx]);
        const uint8_t* img = imgs.data.data() + size_t(idx) * BYTES_PER_IMAGE;

        RunResult rr = run_one(tb, nullptr, img, idx, gold, clks_per_bit, idle_ticks_between_bytes, uart_timeout_cycles, reset_each);

        total_cycles += rr.cycles;

        if (!rr.got_uart) uart_fail++;

        csv << rr.idx << "," << rr.gold << "," << rr.pred << "," << (rr.ok ? 1 : 0) << ","
            << (rr.got_uart ? 1 : 0) << "," << rr.cycles << "\n";

        if (rr.ok) {
            pass++;
        } else {
            fail_cnt++;
            fails << rr.idx << "\n";

            blog << "[FAIL] idx=" << rr.idx << " gold=" << rr.gold << " pred=" << rr.pred
                 << " got_uart=" << (rr.got_uart ? "true" : "false")
                 << " cycles=" << rr.cycles << "\n";

            if (!quiet_console) {
                std::cout << "[FAIL] idx " << rr.idx << " gold=" << rr.gold << " pred=" << rr.pred
                          << " got_uart=" << (rr.got_uart ? "true" : "false") << "\n";
            }

            if (vcd_on_fail) {
                // Re-run failing index with VCD enabled
                VerilatedVcdC* tfp = new VerilatedVcdC;
                tb->trace(tfp, 99);
                const std::string vcd_path = vcd_name_for_fail(outdir, rr.idx);
                tfp->open(vcd_path.c_str());

                blog << "  Re-running idx=" << rr.idx << " with VCD: " << vcd_path << "\n";

                // Re-run (fresh reset for clean wave)
                RunResult rr2 = run_one(tb, tfp, img, idx, gold, clks_per_bit, idle_ticks_between_bytes, uart_timeout_cycles, true);

                blog << "  [VCD-RERUN] idx=" << rr2.idx << " gold=" << rr2.gold << " pred=" << rr2.pred
                     << " got_uart=" << (rr2.got_uart ? "true" : "false")
                     << " cycles=" << rr2.cycles << " ok=" << (rr2.ok ? "true" : "false") << "\n";

                tfp->close();
                delete tfp;
            }
        }

        if (!quiet_console && (progressN > 0) && ((n + 1) % progressN == 0)) {
            const double acc = 100.0 * double(pass) / double(n + 1);
            std::cout << "[PROGRESS] " << (n + 1) << "/" << count
                      << "  acc=" << std::fixed << std::setprecision(2) << acc << "%\n";
        }
    }

    const double acc = (count > 0) ? (100.0 * double(pass) / double(count)) : 0.0;
    const double avg_cycles = (count > 0) ? (double(total_cycles) / double(count)) : 0.0;

    blog << "\nSummary:\n"
         << " total=" << count
         << " pass=" << pass
         << " fail=" << fail_cnt
         << " uart_fail=" << uart_fail
         << " acc=" << std::fixed << std::setprecision(2) << acc << "%\n"
         << " avg_cycles=" << std::fixed << std::setprecision(1) << avg_cycles << "\n";

    if (!quiet_console) {
        std::cout << "\nSummary: total=" << count
                  << " pass=" << pass
                  << " fail=" << fail_cnt
                  << " uart_fail=" << uart_fail
                  << " acc=" << std::fixed << std::setprecision(2) << acc << "%\n";
        std::cout << "Wrote: " << path_join(outdir, "results.csv") << "\n";
        std::cout << "       " << path_join(outdir, "batch.log") << "\n";
        std::cout << "       " << path_join(outdir, "failures.txt") << "\n";
        if (vcd_on_fail) std::cout << "VCDs:  " << outdir << "/wave_fail_idx*.vcd\n";
    }

    delete tb;
    return (fail_cnt == 0) ? 0 : 1;
}