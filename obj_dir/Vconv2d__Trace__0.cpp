// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vconv2d__Syms.h"


void Vconv2d___024root__trace_chg_0_sub_0(Vconv2d___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vconv2d___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_chg_0\n"); );
    // Init
    Vconv2d___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vconv2d___024root*>(voidSelf);
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vconv2d___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vconv2d___024root__trace_chg_0_sub_0(Vconv2d___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_chg_0_sub_0\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[1U]))) {
        bufp->chgCData(oldp+0,(vlSelfRef.conv2d__DOT__state),2);
        bufp->chgIData(oldp+1,(vlSelfRef.conv2d__DOT__oc),32);
        bufp->chgIData(oldp+2,(vlSelfRef.conv2d__DOT__ic),32);
        bufp->chgIData(oldp+3,(vlSelfRef.conv2d__DOT__ki),32);
        bufp->chgIData(oldp+4,(vlSelfRef.conv2d__DOT__kj),32);
        bufp->chgQData(oldp+5,(vlSelfRef.conv2d__DOT__accum),36);
        bufp->chgIData(oldp+7,(vlSelfRef.conv2d__DOT__out_row),32);
        bufp->chgIData(oldp+8,(vlSelfRef.conv2d__DOT__out_col),32);
        bufp->chgIData(oldp+9,(vlSelfRef.conv2d__DOT__mult),32);
        bufp->chgQData(oldp+10,(vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__shifted),36);
        bufp->chgSData(oldp+12,(vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__result),16);
        bufp->chgIData(oldp+13,(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r),32);
        bufp->chgIData(oldp+14,(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c),32);
        bufp->chgSData(oldp+15,(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val),16);
    }
    bufp->chgBit(oldp+16,(vlSelfRef.clk));
    bufp->chgBit(oldp+17,(vlSelfRef.reset));
    bufp->chgBit(oldp+18,(vlSelfRef.start));
    bufp->chgSData(oldp+19,(vlSelfRef.biases[0]),16);
    bufp->chgSData(oldp+20,(vlSelfRef.biases[1]),16);
    bufp->chgSData(oldp+21,(vlSelfRef.biases[2]),16);
    bufp->chgSData(oldp+22,(vlSelfRef.biases[3]),16);
    bufp->chgSData(oldp+23,(vlSelfRef.biases[4]),16);
    bufp->chgSData(oldp+24,(vlSelfRef.biases[5]),16);
    bufp->chgSData(oldp+25,(vlSelfRef.biases[6]),16);
    bufp->chgSData(oldp+26,(vlSelfRef.biases[7]),16);
    bufp->chgBit(oldp+27,(vlSelfRef.done));
}

void Vconv2d___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_cleanup\n"); );
    // Init
    Vconv2d___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vconv2d___024root*>(voidSelf);
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
}
