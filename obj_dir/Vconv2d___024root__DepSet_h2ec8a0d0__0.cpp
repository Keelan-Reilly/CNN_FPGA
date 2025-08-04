// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vconv2d.h for the primary calling header

#include "Vconv2d__pch.h"
#include "Vconv2d___024root.h"

void Vconv2d___024root___eval_act(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_act\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

void Vconv2d___024root___nba_sequent__TOP__0(Vconv2d___024root* vlSelf);

void Vconv2d___024root___eval_nba(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_nba\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vconv2d___024root___nba_sequent__TOP__0(vlSelf);
        vlSelfRef.__Vm_traceActivity[1U] = 1U;
    }
}

VL_INLINE_OPT void Vconv2d___024root___nba_sequent__TOP__0(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___nba_sequent__TOP__0\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    SData/*15:0*/ __Vfunc_conv2d__DOT__scale_and_saturate__0__Vfuncout;
    __Vfunc_conv2d__DOT__scale_and_saturate__0__Vfuncout = 0;
    QData/*35:0*/ __Vfunc_conv2d__DOT__scale_and_saturate__0__val;
    __Vfunc_conv2d__DOT__scale_and_saturate__0__val = 0;
    CData/*1:0*/ __Vdly__conv2d__DOT__state;
    __Vdly__conv2d__DOT__state = 0;
    IData/*31:0*/ __Vdly__conv2d__DOT__out_row;
    __Vdly__conv2d__DOT__out_row = 0;
    IData/*31:0*/ __Vdly__conv2d__DOT__out_col;
    __Vdly__conv2d__DOT__out_col = 0;
    IData/*31:0*/ __Vdly__conv2d__DOT__oc;
    __Vdly__conv2d__DOT__oc = 0;
    IData/*31:0*/ __Vdly__conv2d__DOT__ic;
    __Vdly__conv2d__DOT__ic = 0;
    IData/*31:0*/ __VdlyMask__conv2d__DOT__ic;
    __VdlyMask__conv2d__DOT__ic = 0;
    SData/*15:0*/ __VdlyVal__out_feature__v0;
    __VdlyVal__out_feature__v0 = 0;
    CData/*4:0*/ __VdlyDim0__out_feature__v0;
    __VdlyDim0__out_feature__v0 = 0;
    CData/*4:0*/ __VdlyDim1__out_feature__v0;
    __VdlyDim1__out_feature__v0 = 0;
    CData/*2:0*/ __VdlyDim2__out_feature__v0;
    __VdlyDim2__out_feature__v0 = 0;
    CData/*0:0*/ __VdlySet__out_feature__v0;
    __VdlySet__out_feature__v0 = 0;
    // Body
    __Vdly__conv2d__DOT__state = vlSelfRef.conv2d__DOT__state;
    __Vdly__conv2d__DOT__out_row = vlSelfRef.conv2d__DOT__out_row;
    __Vdly__conv2d__DOT__out_col = vlSelfRef.conv2d__DOT__out_col;
    __Vdly__conv2d__DOT__oc = vlSelfRef.conv2d__DOT__oc;
    __VdlySet__out_feature__v0 = 0U;
    if (vlSelfRef.reset) {
        __Vdly__conv2d__DOT__ic = 0U;
        __VdlyMask__conv2d__DOT__ic = 0xffffffffU;
    }
    vlSelfRef.conv2d__DOT__ic = ((__Vdly__conv2d__DOT__ic 
                                  & __VdlyMask__conv2d__DOT__ic) 
                                 | (vlSelfRef.conv2d__DOT__ic 
                                    & (~ __VdlyMask__conv2d__DOT__ic)));
    if (vlSelfRef.reset) {
        __Vdly__conv2d__DOT__state = 0U;
        vlSelfRef.done = 0U;
        __Vdly__conv2d__DOT__out_row = 0U;
        __Vdly__conv2d__DOT__out_col = 0U;
        __Vdly__conv2d__DOT__oc = 0U;
    } else if ((0U == (IData)(vlSelfRef.conv2d__DOT__state))) {
        vlSelfRef.done = 0U;
        if (vlSelfRef.start) {
            __Vdly__conv2d__DOT__oc = 0U;
            __Vdly__conv2d__DOT__out_row = 0U;
            __Vdly__conv2d__DOT__out_col = 0U;
            __Vdly__conv2d__DOT__state = 1U;
        }
    } else if ((1U == (IData)(vlSelfRef.conv2d__DOT__state))) {
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = (vlSelfRef.conv2d__DOT__out_row - (IData)(1U));
        vlSelfRef.conv2d__DOT__kj = 3U;
        vlSelfRef.conv2d__DOT__ki = 3U;
        vlSelfRef.conv2d__DOT__ic = 1U;
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = (vlSelfRef.conv2d__DOT__out_col - (IData)(1U));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [0U]
                                                                [0U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = (vlSelfRef.conv2d__DOT__out_row - (IData)(1U));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = vlSelfRef.conv2d__DOT__out_col;
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [0U]
                                                                [1U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = (vlSelfRef.conv2d__DOT__out_row - (IData)(1U));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = ((IData)(1U) + vlSelfRef.conv2d__DOT__out_col);
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [0U]
                                                                [2U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = vlSelfRef.conv2d__DOT__out_row;
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = (vlSelfRef.conv2d__DOT__out_col - (IData)(1U));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [1U]
                                                                [0U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = vlSelfRef.conv2d__DOT__out_row;
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = vlSelfRef.conv2d__DOT__out_col;
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [1U]
                                                                [1U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = vlSelfRef.conv2d__DOT__out_row;
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = ((IData)(1U) + vlSelfRef.conv2d__DOT__out_col);
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [1U]
                                                                [2U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = ((IData)(1U) + vlSelfRef.conv2d__DOT__out_row);
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = (vlSelfRef.conv2d__DOT__out_col - (IData)(1U));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [2U]
                                                                [0U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = ((IData)(1U) + vlSelfRef.conv2d__DOT__out_row);
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = vlSelfRef.conv2d__DOT__out_col;
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [2U]
                                                                [1U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r 
            = ((IData)(1U) + vlSelfRef.conv2d__DOT__out_row);
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c 
            = ((IData)(1U) + vlSelfRef.conv2d__DOT__out_col);
        vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val 
            = ((((VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r) 
                  | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)) 
                 | VL_GTS_III(32, 0U, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)) 
                | VL_LTES_III(32, 0x1cU, vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c))
                         ? vlSelfRef.input_feature[0U]
                        [((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r))
                           ? (0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_r)
                           : 0U)][(0x1fU & vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_c)]
                         : 0U));
        vlSelfRef.conv2d__DOT__mult = VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, (IData)(vlSelfRef.conv2d__DOT__unnamedblk1__DOT__in_val)), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelfRef.weights
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)]
                                                                [0U]
                                                                [2U]
                                                                [2U]));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + VL_EXTENDS_QI(36,32, vlSelfRef.conv2d__DOT__mult)));
        vlSelfRef.conv2d__DOT__accum = (0xfffffffffULL 
                                        & (vlSelfRef.conv2d__DOT__accum 
                                           + (((QData)((IData)(
                                                               (0xfffffU 
                                                                & (- (IData)(
                                                                             (1U 
                                                                              & (vlSelfRef.biases
                                                                                [
                                                                                (7U 
                                                                                & vlSelfRef.conv2d__DOT__oc)] 
                                                                                >> 0xfU))))))) 
                                               << 0x10U) 
                                              | (QData)((IData)(
                                                                vlSelfRef.biases
                                                                [
                                                                (7U 
                                                                 & vlSelfRef.conv2d__DOT__oc)])))));
        __Vfunc_conv2d__DOT__scale_and_saturate__0__val 
            = vlSelfRef.conv2d__DOT__accum;
        vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__shifted 
            = (0xfffffffffULL & VL_SHIFTRS_QQI(36,36,32, __Vfunc_conv2d__DOT__scale_and_saturate__0__val, 7U));
        vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__result 
            = (VL_LTS_IQQ(36, 0x7fffULL, vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__shifted)
                ? 0x7fffU : (VL_GTS_IQQ(36, 0xfffff8000ULL, vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__shifted)
                              ? 0x8000U : (0xffffU 
                                           & (IData)(vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__shifted))));
        __Vfunc_conv2d__DOT__scale_and_saturate__0__Vfuncout 
            = vlSelfRef.conv2d__DOT__scale_and_saturate__Vstatic__result;
        vlSelfRef.conv2d__DOT____Vlvbound_hfb6f3c5f__0 
            = __Vfunc_conv2d__DOT__scale_and_saturate__0__Vfuncout;
        if (((0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__out_col)) 
             && (0x1bU >= (0x1fU & vlSelfRef.conv2d__DOT__out_row)))) {
            __VdlyVal__out_feature__v0 = vlSelfRef.conv2d__DOT____Vlvbound_hfb6f3c5f__0;
            __VdlyDim0__out_feature__v0 = (0x1fU & vlSelfRef.conv2d__DOT__out_col);
            __VdlyDim1__out_feature__v0 = (0x1fU & vlSelfRef.conv2d__DOT__out_row);
            __VdlyDim2__out_feature__v0 = (7U & vlSelfRef.conv2d__DOT__oc);
            __VdlySet__out_feature__v0 = 1U;
        }
        if ((0x1bU == vlSelfRef.conv2d__DOT__out_col)) {
            __Vdly__conv2d__DOT__out_col = 0U;
            if ((0x1bU == vlSelfRef.conv2d__DOT__out_row)) {
                __Vdly__conv2d__DOT__out_row = 0U;
                if ((7U == vlSelfRef.conv2d__DOT__oc)) {
                    __Vdly__conv2d__DOT__state = 3U;
                } else {
                    __Vdly__conv2d__DOT__oc = ((IData)(1U) 
                                               + vlSelfRef.conv2d__DOT__oc);
                }
            } else {
                __Vdly__conv2d__DOT__out_row = ((IData)(1U) 
                                                + vlSelfRef.conv2d__DOT__out_row);
            }
        } else {
            __Vdly__conv2d__DOT__out_col = ((IData)(1U) 
                                            + vlSelfRef.conv2d__DOT__out_col);
        }
    } else if ((3U == (IData)(vlSelfRef.conv2d__DOT__state))) {
        vlSelfRef.done = 1U;
        __Vdly__conv2d__DOT__state = 0U;
    } else {
        __Vdly__conv2d__DOT__state = 0U;
    }
    __VdlyMask__conv2d__DOT__ic = 0U;
    vlSelfRef.conv2d__DOT__state = __Vdly__conv2d__DOT__state;
    vlSelfRef.conv2d__DOT__out_row = __Vdly__conv2d__DOT__out_row;
    vlSelfRef.conv2d__DOT__out_col = __Vdly__conv2d__DOT__out_col;
    vlSelfRef.conv2d__DOT__oc = __Vdly__conv2d__DOT__oc;
    if (__VdlySet__out_feature__v0) {
        vlSelfRef.out_feature[__VdlyDim2__out_feature__v0][__VdlyDim1__out_feature__v0][__VdlyDim0__out_feature__v0] 
            = __VdlyVal__out_feature__v0;
    }
}

void Vconv2d___024root___eval_triggers__act(Vconv2d___024root* vlSelf);

bool Vconv2d___024root___eval_phase__act(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_phase__act\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<1> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vconv2d___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vconv2d___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vconv2d___024root___eval_phase__nba(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_phase__nba\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vconv2d___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vconv2d___024root___dump_triggers__nba(Vconv2d___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vconv2d___024root___dump_triggers__act(Vconv2d___024root* vlSelf);
#endif  // VL_DEBUG

void Vconv2d___024root___eval(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY(((0x64U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vconv2d___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("hdl/conv2d.sv", 6, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vconv2d___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("hdl/conv2d.sv", 6, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vconv2d___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vconv2d___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vconv2d___024root___eval_debug_assertions(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_debug_assertions\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((vlSelfRef.clk & 0xfeU)))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY(((vlSelfRef.reset & 0xfeU)))) {
        Verilated::overWidthError("reset");}
    if (VL_UNLIKELY(((vlSelfRef.start & 0xfeU)))) {
        Verilated::overWidthError("start");}
}
#endif  // VL_DEBUG
