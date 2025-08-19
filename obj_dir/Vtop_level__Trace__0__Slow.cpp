// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtop_level__Syms.h"


VL_ATTR_COLD void Vtop_level___024root__trace_init_sub__TOP__0(Vtop_level___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_init_sub__TOP__0\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("$rootio", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+98,0,"uart_rx_i",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+99,0,"uart_tx_o",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+100,0,"predicted_digit",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->popPrefix();
    tracep->pushPrefix("top_level", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+102,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+103,0,"FRAC_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+104,0,"IMG_SIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+105,0,"IN_CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+106,0,"OUT_CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+107,0,"NUM_CLASSES",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+108,0,"CLKS_PER_BIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+98,0,"uart_rx_i",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+99,0,"uart_tx_o",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+100,0,"predicted_digit",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+109,0,"POOLED",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+110,0,"FLAT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("logits", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declBus(c+19+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("conv_b", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("dense_b", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declBus(c+9+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBit(c+29,0,"rx_dv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+30,0,"rx_byte",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+31,0,"r",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+32,0,"c",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBit(c+33,0,"frame_loaded",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+34,0,"conv_start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+35,0,"relu_start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+36,0,"pool_start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+37,0,"dense_start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+38,0,"tx_start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+39,0,"conv_done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+40,0,"relu_done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+41,0,"pool_done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+42,0,"dense_done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+43,0,"flattening",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+44,0,"flat_done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+45,0,"fi_c",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+46,0,"fi_r",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+47,0,"fi_q",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+48,0,"fi_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBit(c+49,0,"argmax_done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+49,0,"tx_ready",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+50,0,"tx_busy",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+38,0,"tx_dv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+101,0,"tx_byte",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+111,0,"ASCII_0",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->pushPrefix("RX", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+108,0,"CLKS_PER_BIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+98,0,"rx",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+29,0,"rx_dv",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+30,0,"rx_byte",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+51,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 2,0);
    tracep->declBus(c+52,0,"clk_cnt",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 8,0);
    tracep->declBus(c+53,0,"bit_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 2,0);
    tracep->declBus(c+54,0,"data",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->popPrefix();
    tracep->pushPrefix("TX", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+108,0,"CLKS_PER_BIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+38,0,"tx_dv",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+101,0,"tx_byte",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+99,0,"tx",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+50,0,"tx_busy",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+55,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 2,0);
    tracep->declBus(c+56,0,"clk_cnt",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 8,0);
    tracep->declBus(c+57,0,"bit_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 2,0);
    tracep->declBus(c+58,0,"data_reg",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->popPrefix();
    tracep->pushPrefix("ctrl", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+33,0,"frame_loaded",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+39,0,"conv_done",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+40,0,"relu_done",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+41,0,"pool_done",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+44,0,"flat_done",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+42,0,"dense_done",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+49,0,"tx_ready",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+34,0,"conv_start",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+35,0,"relu_start",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+36,0,"pool_start",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+37,0,"dense_start",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+38,0,"tx_start",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+59,0,"busy",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+60,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 2,0);
    tracep->popPrefix();
    tracep->pushPrefix("u_argmax", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+102,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+107,0,"DIM",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+42,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("vec", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declBus(c+19+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+100,0,"idx",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+49,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+61,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+62,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+63,0,"bestv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+64,0,"besti",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->popPrefix();
    tracep->pushPrefix("u_conv", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+102,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+103,0,"FRAC_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+105,0,"IN_CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+106,0,"OUT_CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+112,0,"KERNEL",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+104,0,"IMG_SIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+34,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("biases", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBit(c+39,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+113,0,"signed_accum_width",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+105,0,"PAD",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+65,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+66,0,"oc",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+67,0,"ic",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+114,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+115,0,"j",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+68,0,"ki",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+69,0,"kj",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declQuad(c+70,0,"accum",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 35,0);
    tracep->declBus(c+72,0,"out_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+73,0,"out_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+74,0,"mult",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declQuad(c+75,0,"scale_and_saturate__Vstatic__shifted",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 35,0);
    tracep->declBus(c+77,0,"scale_and_saturate__Vstatic__result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->pushPrefix("unnamedblk1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+78,0,"in_r",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+79,0,"in_c",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+80,0,"in_val",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("u_dense", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+102,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+103,0,"FRAC_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+110,0,"IN_DIM",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+107,0,"OUT_DIM",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+37,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("biases", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declBus(c+9+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("out_vec", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declBus(c+19+i*1,0,"",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBit(c+42,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+116,0,"ACCW",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+81,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+82,0,"o",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+83,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declQuad(c+84,0,"acc",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 42,0);
    tracep->declBus(c+86,0,"prod",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("u_pool", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+102,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+106,0,"CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+104,0,"IN_SIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+117,0,"POOL",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+36,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+41,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+109,0,"OUT_SIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+87,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+88,0,"c",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+89,0,"r",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+90,0,"q",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("u_relu", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+102,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+103,0,"FRAC_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+106,0,"CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+104,0,"IMG_SIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+96,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+97,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+35,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+40,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+91,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+92,0,"c",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+93,0,"r",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+94,0,"q",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->pushPrefix("unnamedblk1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+95,0,"v",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vtop_level___024root__trace_init_top(Vtop_level___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_init_top\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtop_level___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vtop_level___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vtop_level___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtop_level___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtop_level___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vtop_level___024root__trace_register(Vtop_level___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_register\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    tracep->addConstCb(&Vtop_level___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vtop_level___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vtop_level___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vtop_level___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vtop_level___024root__trace_const_0_sub_0(Vtop_level___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtop_level___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_const_0\n"); );
    // Init
    Vtop_level___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop_level___024root*>(voidSelf);
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtop_level___024root__trace_const_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtop_level___024root__trace_const_0_sub_0(Vtop_level___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_const_0_sub_0\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+102,(0x10U),32);
    bufp->fullIData(oldp+103,(7U),32);
    bufp->fullIData(oldp+104,(0x1cU),32);
    bufp->fullIData(oldp+105,(1U),32);
    bufp->fullIData(oldp+106,(8U),32);
    bufp->fullIData(oldp+107,(0xaU),32);
    bufp->fullIData(oldp+108,(0x1b2U),32);
    bufp->fullIData(oldp+109,(0xeU),32);
    bufp->fullIData(oldp+110,(0x620U),32);
    bufp->fullCData(oldp+111,(0x30U),8);
    bufp->fullIData(oldp+112,(3U),32);
    bufp->fullIData(oldp+113,(0x24U),32);
    bufp->fullIData(oldp+114,(vlSelfRef.top_level__DOT__u_conv__DOT__i),32);
    bufp->fullIData(oldp+115,(vlSelfRef.top_level__DOT__u_conv__DOT__j),32);
    bufp->fullIData(oldp+116,(0x2bU),32);
    bufp->fullIData(oldp+117,(2U),32);
}

VL_ATTR_COLD void Vtop_level___024root__trace_full_0_sub_0(Vtop_level___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtop_level___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_full_0\n"); );
    // Init
    Vtop_level___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop_level___024root*>(voidSelf);
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtop_level___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtop_level___024root__trace_full_0_sub_0(Vtop_level___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_full_0_sub_0\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullSData(oldp+1,(vlSelfRef.top_level__DOT__conv_b[0]),16);
    bufp->fullSData(oldp+2,(vlSelfRef.top_level__DOT__conv_b[1]),16);
    bufp->fullSData(oldp+3,(vlSelfRef.top_level__DOT__conv_b[2]),16);
    bufp->fullSData(oldp+4,(vlSelfRef.top_level__DOT__conv_b[3]),16);
    bufp->fullSData(oldp+5,(vlSelfRef.top_level__DOT__conv_b[4]),16);
    bufp->fullSData(oldp+6,(vlSelfRef.top_level__DOT__conv_b[5]),16);
    bufp->fullSData(oldp+7,(vlSelfRef.top_level__DOT__conv_b[6]),16);
    bufp->fullSData(oldp+8,(vlSelfRef.top_level__DOT__conv_b[7]),16);
    bufp->fullSData(oldp+9,(vlSelfRef.top_level__DOT__dense_b[0]),16);
    bufp->fullSData(oldp+10,(vlSelfRef.top_level__DOT__dense_b[1]),16);
    bufp->fullSData(oldp+11,(vlSelfRef.top_level__DOT__dense_b[2]),16);
    bufp->fullSData(oldp+12,(vlSelfRef.top_level__DOT__dense_b[3]),16);
    bufp->fullSData(oldp+13,(vlSelfRef.top_level__DOT__dense_b[4]),16);
    bufp->fullSData(oldp+14,(vlSelfRef.top_level__DOT__dense_b[5]),16);
    bufp->fullSData(oldp+15,(vlSelfRef.top_level__DOT__dense_b[6]),16);
    bufp->fullSData(oldp+16,(vlSelfRef.top_level__DOT__dense_b[7]),16);
    bufp->fullSData(oldp+17,(vlSelfRef.top_level__DOT__dense_b[8]),16);
    bufp->fullSData(oldp+18,(vlSelfRef.top_level__DOT__dense_b[9]),16);
    bufp->fullSData(oldp+19,(vlSelfRef.top_level__DOT__logits[0]),16);
    bufp->fullSData(oldp+20,(vlSelfRef.top_level__DOT__logits[1]),16);
    bufp->fullSData(oldp+21,(vlSelfRef.top_level__DOT__logits[2]),16);
    bufp->fullSData(oldp+22,(vlSelfRef.top_level__DOT__logits[3]),16);
    bufp->fullSData(oldp+23,(vlSelfRef.top_level__DOT__logits[4]),16);
    bufp->fullSData(oldp+24,(vlSelfRef.top_level__DOT__logits[5]),16);
    bufp->fullSData(oldp+25,(vlSelfRef.top_level__DOT__logits[6]),16);
    bufp->fullSData(oldp+26,(vlSelfRef.top_level__DOT__logits[7]),16);
    bufp->fullSData(oldp+27,(vlSelfRef.top_level__DOT__logits[8]),16);
    bufp->fullSData(oldp+28,(vlSelfRef.top_level__DOT__logits[9]),16);
    bufp->fullBit(oldp+29,(vlSelfRef.top_level__DOT__rx_dv));
    bufp->fullCData(oldp+30,(vlSelfRef.top_level__DOT__rx_byte),8);
    bufp->fullIData(oldp+31,(vlSelfRef.top_level__DOT__r),32);
    bufp->fullIData(oldp+32,(vlSelfRef.top_level__DOT__c),32);
    bufp->fullBit(oldp+33,(vlSelfRef.top_level__DOT__frame_loaded));
    bufp->fullBit(oldp+34,(vlSelfRef.top_level__DOT__conv_start));
    bufp->fullBit(oldp+35,(vlSelfRef.top_level__DOT__relu_start));
    bufp->fullBit(oldp+36,(vlSelfRef.top_level__DOT__pool_start));
    bufp->fullBit(oldp+37,(vlSelfRef.top_level__DOT__dense_start));
    bufp->fullBit(oldp+38,(vlSelfRef.top_level__DOT__tx_start));
    bufp->fullBit(oldp+39,(vlSelfRef.top_level__DOT__conv_done));
    bufp->fullBit(oldp+40,(vlSelfRef.top_level__DOT__relu_done));
    bufp->fullBit(oldp+41,(vlSelfRef.top_level__DOT__pool_done));
    bufp->fullBit(oldp+42,(vlSelfRef.top_level__DOT__dense_done));
    bufp->fullBit(oldp+43,(vlSelfRef.top_level__DOT__flattening));
    bufp->fullBit(oldp+44,(vlSelfRef.top_level__DOT__flat_done));
    bufp->fullIData(oldp+45,(vlSelfRef.top_level__DOT__fi_c),32);
    bufp->fullIData(oldp+46,(vlSelfRef.top_level__DOT__fi_r),32);
    bufp->fullIData(oldp+47,(vlSelfRef.top_level__DOT__fi_q),32);
    bufp->fullIData(oldp+48,(vlSelfRef.top_level__DOT__fi_idx),32);
    bufp->fullBit(oldp+49,(vlSelfRef.top_level__DOT__argmax_done));
    bufp->fullBit(oldp+50,(vlSelfRef.top_level__DOT__tx_busy));
    bufp->fullCData(oldp+51,(vlSelfRef.top_level__DOT__RX__DOT__state),3);
    bufp->fullSData(oldp+52,(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt),9);
    bufp->fullCData(oldp+53,(vlSelfRef.top_level__DOT__RX__DOT__bit_idx),3);
    bufp->fullCData(oldp+54,(vlSelfRef.top_level__DOT__RX__DOT__data),8);
    bufp->fullCData(oldp+55,(vlSelfRef.top_level__DOT__TX__DOT__state),3);
    bufp->fullSData(oldp+56,(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt),9);
    bufp->fullCData(oldp+57,(vlSelfRef.top_level__DOT__TX__DOT__bit_idx),3);
    bufp->fullCData(oldp+58,(vlSelfRef.top_level__DOT__TX__DOT__data_reg),8);
    bufp->fullBit(oldp+59,(vlSelfRef.top_level__DOT__ctrl__DOT__busy));
    bufp->fullCData(oldp+60,(vlSelfRef.top_level__DOT__ctrl__DOT__state),3);
    bufp->fullCData(oldp+61,(vlSelfRef.top_level__DOT__u_argmax__DOT__state),2);
    bufp->fullIData(oldp+62,(vlSelfRef.top_level__DOT__u_argmax__DOT__i),32);
    bufp->fullSData(oldp+63,(vlSelfRef.top_level__DOT__u_argmax__DOT__bestv),16);
    bufp->fullCData(oldp+64,(vlSelfRef.top_level__DOT__u_argmax__DOT__besti),4);
    bufp->fullCData(oldp+65,(vlSelfRef.top_level__DOT__u_conv__DOT__state),2);
    bufp->fullIData(oldp+66,(vlSelfRef.top_level__DOT__u_conv__DOT__oc),32);
    bufp->fullIData(oldp+67,(vlSelfRef.top_level__DOT__u_conv__DOT__ic),32);
    bufp->fullIData(oldp+68,(vlSelfRef.top_level__DOT__u_conv__DOT__ki),32);
    bufp->fullIData(oldp+69,(vlSelfRef.top_level__DOT__u_conv__DOT__kj),32);
    bufp->fullQData(oldp+70,(vlSelfRef.top_level__DOT__u_conv__DOT__accum),36);
    bufp->fullIData(oldp+72,(vlSelfRef.top_level__DOT__u_conv__DOT__out_row),32);
    bufp->fullIData(oldp+73,(vlSelfRef.top_level__DOT__u_conv__DOT__out_col),32);
    bufp->fullIData(oldp+74,(vlSelfRef.top_level__DOT__u_conv__DOT__mult),32);
    bufp->fullQData(oldp+75,(vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__shifted),36);
    bufp->fullSData(oldp+77,(vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__result),16);
    bufp->fullIData(oldp+78,(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r),32);
    bufp->fullIData(oldp+79,(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c),32);
    bufp->fullSData(oldp+80,(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val),16);
    bufp->fullCData(oldp+81,(vlSelfRef.top_level__DOT__u_dense__DOT__state),2);
    bufp->fullIData(oldp+82,(vlSelfRef.top_level__DOT__u_dense__DOT__o),32);
    bufp->fullIData(oldp+83,(vlSelfRef.top_level__DOT__u_dense__DOT__i),32);
    bufp->fullQData(oldp+84,(vlSelfRef.top_level__DOT__u_dense__DOT__acc),43);
    bufp->fullIData(oldp+86,(vlSelfRef.top_level__DOT__u_dense__DOT__prod),32);
    bufp->fullCData(oldp+87,(vlSelfRef.top_level__DOT__u_pool__DOT__state),2);
    bufp->fullIData(oldp+88,(vlSelfRef.top_level__DOT__u_pool__DOT__c),32);
    bufp->fullIData(oldp+89,(vlSelfRef.top_level__DOT__u_pool__DOT__r),32);
    bufp->fullIData(oldp+90,(vlSelfRef.top_level__DOT__u_pool__DOT__q),32);
    bufp->fullCData(oldp+91,(vlSelfRef.top_level__DOT__u_relu__DOT__state),2);
    bufp->fullIData(oldp+92,(vlSelfRef.top_level__DOT__u_relu__DOT__c),32);
    bufp->fullIData(oldp+93,(vlSelfRef.top_level__DOT__u_relu__DOT__r),32);
    bufp->fullIData(oldp+94,(vlSelfRef.top_level__DOT__u_relu__DOT__q),32);
    bufp->fullSData(oldp+95,(vlSelfRef.top_level__DOT__u_relu__DOT__unnamedblk1__DOT__v),16);
    bufp->fullBit(oldp+96,(vlSelfRef.clk));
    bufp->fullBit(oldp+97,(vlSelfRef.reset));
    bufp->fullBit(oldp+98,(vlSelfRef.uart_rx_i));
    bufp->fullBit(oldp+99,(vlSelfRef.uart_tx_o));
    bufp->fullCData(oldp+100,(vlSelfRef.predicted_digit),4);
    bufp->fullCData(oldp+101,((0xffU & ((IData)(0x30U) 
                                        + (IData)(vlSelfRef.predicted_digit)))),8);
}
