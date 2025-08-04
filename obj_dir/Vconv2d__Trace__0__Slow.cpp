// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vconv2d__Syms.h"


VL_ATTR_COLD void Vconv2d___024root__trace_init_sub__TOP__0(Vconv2d___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_init_sub__TOP__0\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("$rootio", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+17,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+18,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+19,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("biases", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+20+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBit(c+28,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->popPrefix();
    tracep->pushPrefix("conv2d", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+29,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+30,0,"FRAC_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+31,0,"IN_CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+32,0,"OUT_CHANNELS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+33,0,"KERNEL",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+34,0,"IMG_SIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+17,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+18,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+19,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("biases", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+20+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBit(c+28,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+35,0,"signed_accum_width",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+31,0,"PAD",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+1,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+2,0,"oc",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+3,0,"ic",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+36,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+37,0,"j",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+4,0,"ki",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+5,0,"kj",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declQuad(c+6,0,"accum",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 35,0);
    tracep->declBus(c+8,0,"out_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+9,0,"out_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+10,0,"mult",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declQuad(c+11,0,"scale_and_saturate__Vstatic__shifted",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 35,0);
    tracep->declBus(c+13,0,"scale_and_saturate__Vstatic__result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->pushPrefix("unnamedblk1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+14,0,"in_r",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+15,0,"in_c",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+16,0,"in_val",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vconv2d___024root__trace_init_top(Vconv2d___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_init_top\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vconv2d___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vconv2d___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vconv2d___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vconv2d___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vconv2d___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vconv2d___024root__trace_register(Vconv2d___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_register\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    tracep->addConstCb(&Vconv2d___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vconv2d___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vconv2d___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vconv2d___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vconv2d___024root__trace_const_0_sub_0(Vconv2d___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vconv2d___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_const_0\n"); );
    // Init
    Vconv2d___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vconv2d___024root*>(voidSelf);
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vconv2d___024root__trace_const_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vconv2d___024root__trace_const_0_sub_0(Vconv2d___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_const_0_sub_0\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+29,(0x10U),32);
    bufp->fullIData(oldp+30,(7U),32);
    bufp->fullIData(oldp+31,(1U),32);
    bufp->fullIData(oldp+32,(8U),32);
    bufp->fullIData(oldp+33,(3U),32);
    bufp->fullIData(oldp+34,(0x1cU),32);
    bufp->fullIData(oldp+35,(0x24U),32);
    bufp->fullIData(oldp+36,(vlSelfRef.conv2d__DOT__i),32);
    bufp->fullIData(oldp+37,(vlSelfRef.conv2d__DOT__j),32);
}

VL_ATTR_COLD void Vconv2d___024root__trace_full_0_sub_0(Vconv2d___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vconv2d___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_full_0\n"); );
    // Init
    Vconv2d___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vconv2d___024root*>(voidSelf);
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vconv2d___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vconv2d___024root__trace_full_0_sub_0(Vconv2d___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root__trace_full_0_sub_0\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullCData(oldp+1,(vlSelfRef.conv2d__DOT__state),2);
    bufp->fullIData(oldp+2,(vlSelfRef.conv2d__DOT__oc),32);
    bufp->fullIData(oldp+3,(vlSelfRef.conv2d__DOT__ic),32);
    bufp->fullIData(oldp+4,(vlSelfRef.conv2d__DOT__ki),32);
    bufp->fullIData(oldp+5,(vlSelfRef.conv2d__DOT__kj),32);
    bufp->fullQData(oldp+6,(vlSelfRef.conv2d__DOT__accum),36);
    bufp->fullIData(oldp+8,(vlSelfRef.conv2d__DOT__out_row),32);
    bufp->fullIData(oldp+9,(vlSelfRef.conv2d__DOT__out_col),32);
    bufp->fullIData(oldp+10,(vlSelfRef.conv2d__DOT__mult),32);
    bufp->fullQData(oldp+11,(vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__shifted),36);
    bufp->fullSData(oldp+13,(vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__result),16);
    bufp->fullIData(oldp+14,(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r),32);
    bufp->fullIData(oldp+15,(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c),32);
    bufp->fullSData(oldp+16,(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val),16);
    bufp->fullBit(oldp+17,(vlSelfRef.clk));
    bufp->fullBit(oldp+18,(vlSelfRef.reset));
    bufp->fullBit(oldp+19,(vlSelfRef.start));
    bufp->fullSData(oldp+20,(vlSelfRef.biases[0]),16);
    bufp->fullSData(oldp+21,(vlSelfRef.biases[1]),16);
    bufp->fullSData(oldp+22,(vlSelfRef.biases[2]),16);
    bufp->fullSData(oldp+23,(vlSelfRef.biases[3]),16);
    bufp->fullSData(oldp+24,(vlSelfRef.biases[4]),16);
    bufp->fullSData(oldp+25,(vlSelfRef.biases[5]),16);
    bufp->fullSData(oldp+26,(vlSelfRef.biases[6]),16);
    bufp->fullSData(oldp+27,(vlSelfRef.biases[7]),16);
    bufp->fullBit(oldp+28,(vlSelfRef.done));
}
