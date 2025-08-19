// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop_level.h for the primary calling header

#include "Vtop_level__pch.h"
#include "Vtop_level___024root.h"

void Vtop_level___024root___eval_act(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_act\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

void Vtop_level___024root___nba_sequent__TOP__0(Vtop_level___024root* vlSelf);

void Vtop_level___024root___eval_nba(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_nba\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtop_level___024root___nba_sequent__TOP__0(vlSelf);
        vlSelfRef.__Vm_traceActivity[1U] = 1U;
    }
}

VL_INLINE_OPT void Vtop_level___024root___nba_sequent__TOP__0(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___nba_sequent__TOP__0\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    SData/*15:0*/ __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__Vfuncout;
    __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__Vfuncout = 0;
    QData/*35:0*/ __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__val;
    __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__val = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__1__Vfuncout;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__1__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__1__a;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__1__a = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__1__b;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__1__b = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__1__c;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__1__c = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__1__d;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__1__d = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__1__m1;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__1__m1 = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__1__m2;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__1__m2 = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__Vfuncout;
    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__Vfuncout = 0;
    QData/*42:0*/ __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__v;
    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__v = 0;
    QData/*42:0*/ __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__s;
    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__s = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mx;
    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mx = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mn;
    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mn = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__r;
    __Vdly__top_level__DOT__r = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__c;
    __Vdly__top_level__DOT__c = 0;
    CData/*0:0*/ __Vdly__top_level__DOT__flattening;
    __Vdly__top_level__DOT__flattening = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__fi_c;
    __Vdly__top_level__DOT__fi_c = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__fi_r;
    __Vdly__top_level__DOT__fi_r = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__fi_q;
    __Vdly__top_level__DOT__fi_q = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__fi_idx;
    __Vdly__top_level__DOT__fi_idx = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__RX__DOT__state;
    __Vdly__top_level__DOT__RX__DOT__state = 0;
    SData/*8:0*/ __Vdly__top_level__DOT__RX__DOT__clk_cnt;
    __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__RX__DOT__bit_idx;
    __Vdly__top_level__DOT__RX__DOT__bit_idx = 0;
    CData/*7:0*/ __Vdly__top_level__DOT__RX__DOT__data;
    __Vdly__top_level__DOT__RX__DOT__data = 0;
    CData/*1:0*/ __Vdly__top_level__DOT__u_conv__DOT__state;
    __Vdly__top_level__DOT__u_conv__DOT__state = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__out_row;
    __Vdly__top_level__DOT__u_conv__DOT__out_row = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__out_col;
    __Vdly__top_level__DOT__u_conv__DOT__out_col = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__oc;
    __Vdly__top_level__DOT__u_conv__DOT__oc = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__ic;
    __Vdly__top_level__DOT__u_conv__DOT__ic = 0;
    IData/*31:0*/ __VdlyMask__top_level__DOT__u_conv__DOT__ic;
    __VdlyMask__top_level__DOT__u_conv__DOT__ic = 0;
    CData/*1:0*/ __Vdly__top_level__DOT__u_relu__DOT__state;
    __Vdly__top_level__DOT__u_relu__DOT__state = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_relu__DOT__c;
    __Vdly__top_level__DOT__u_relu__DOT__c = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_relu__DOT__r;
    __Vdly__top_level__DOT__u_relu__DOT__r = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_relu__DOT__q;
    __Vdly__top_level__DOT__u_relu__DOT__q = 0;
    CData/*1:0*/ __Vdly__top_level__DOT__u_pool__DOT__state;
    __Vdly__top_level__DOT__u_pool__DOT__state = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_pool__DOT__c;
    __Vdly__top_level__DOT__u_pool__DOT__c = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_pool__DOT__r;
    __Vdly__top_level__DOT__u_pool__DOT__r = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_pool__DOT__q;
    __Vdly__top_level__DOT__u_pool__DOT__q = 0;
    CData/*1:0*/ __Vdly__top_level__DOT__u_dense__DOT__state;
    __Vdly__top_level__DOT__u_dense__DOT__state = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_dense__DOT__o;
    __Vdly__top_level__DOT__u_dense__DOT__o = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_dense__DOT__i;
    __Vdly__top_level__DOT__u_dense__DOT__i = 0;
    QData/*42:0*/ __Vdly__top_level__DOT__u_dense__DOT__acc;
    __Vdly__top_level__DOT__u_dense__DOT__acc = 0;
    QData/*42:0*/ __VdlyMask__top_level__DOT__u_dense__DOT__acc;
    __VdlyMask__top_level__DOT__u_dense__DOT__acc = 0;
    CData/*1:0*/ __Vdly__top_level__DOT__u_argmax__DOT__state;
    __Vdly__top_level__DOT__u_argmax__DOT__state = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_argmax__DOT__i;
    __Vdly__top_level__DOT__u_argmax__DOT__i = 0;
    CData/*3:0*/ __Vdly__top_level__DOT__u_argmax__DOT__besti;
    __Vdly__top_level__DOT__u_argmax__DOT__besti = 0;
    SData/*15:0*/ __Vdly__top_level__DOT__u_argmax__DOT__bestv;
    __Vdly__top_level__DOT__u_argmax__DOT__bestv = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__ctrl__DOT__state;
    __Vdly__top_level__DOT__ctrl__DOT__state = 0;
    CData/*0:0*/ __Vdly__top_level__DOT__conv_start;
    __Vdly__top_level__DOT__conv_start = 0;
    CData/*0:0*/ __Vdly__top_level__DOT__relu_start;
    __Vdly__top_level__DOT__relu_start = 0;
    CData/*0:0*/ __Vdly__top_level__DOT__pool_start;
    __Vdly__top_level__DOT__pool_start = 0;
    CData/*0:0*/ __Vdly__top_level__DOT__dense_start;
    __Vdly__top_level__DOT__dense_start = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__TX__DOT__state;
    __Vdly__top_level__DOT__TX__DOT__state = 0;
    SData/*8:0*/ __Vdly__top_level__DOT__TX__DOT__clk_cnt;
    __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__TX__DOT__bit_idx;
    __Vdly__top_level__DOT__TX__DOT__bit_idx = 0;
    CData/*7:0*/ __Vdly__top_level__DOT__TX__DOT__data_reg;
    __Vdly__top_level__DOT__TX__DOT__data_reg = 0;
    SData/*15:0*/ __VdlyVal__top_level__DOT__ifmap__v0;
    __VdlyVal__top_level__DOT__ifmap__v0 = 0;
    CData/*4:0*/ __VdlyDim0__top_level__DOT__ifmap__v0;
    __VdlyDim0__top_level__DOT__ifmap__v0 = 0;
    CData/*4:0*/ __VdlyDim1__top_level__DOT__ifmap__v0;
    __VdlyDim1__top_level__DOT__ifmap__v0 = 0;
    CData/*0:0*/ __VdlySet__top_level__DOT__ifmap__v0;
    __VdlySet__top_level__DOT__ifmap__v0 = 0;
    SData/*15:0*/ __VdlyVal__top_level__DOT__flat_vec__v0;
    __VdlyVal__top_level__DOT__flat_vec__v0 = 0;
    SData/*10:0*/ __VdlyDim0__top_level__DOT__flat_vec__v0;
    __VdlyDim0__top_level__DOT__flat_vec__v0 = 0;
    CData/*0:0*/ __VdlySet__top_level__DOT__flat_vec__v0;
    __VdlySet__top_level__DOT__flat_vec__v0 = 0;
    SData/*15:0*/ __VdlyVal__top_level__DOT__conv_out__v0;
    __VdlyVal__top_level__DOT__conv_out__v0 = 0;
    CData/*4:0*/ __VdlyDim0__top_level__DOT__conv_out__v0;
    __VdlyDim0__top_level__DOT__conv_out__v0 = 0;
    CData/*4:0*/ __VdlyDim1__top_level__DOT__conv_out__v0;
    __VdlyDim1__top_level__DOT__conv_out__v0 = 0;
    CData/*2:0*/ __VdlyDim2__top_level__DOT__conv_out__v0;
    __VdlyDim2__top_level__DOT__conv_out__v0 = 0;
    CData/*0:0*/ __VdlySet__top_level__DOT__conv_out__v0;
    __VdlySet__top_level__DOT__conv_out__v0 = 0;
    SData/*15:0*/ __VdlyVal__top_level__DOT__relu_out__v0;
    __VdlyVal__top_level__DOT__relu_out__v0 = 0;
    CData/*4:0*/ __VdlyDim0__top_level__DOT__relu_out__v0;
    __VdlyDim0__top_level__DOT__relu_out__v0 = 0;
    CData/*4:0*/ __VdlyDim1__top_level__DOT__relu_out__v0;
    __VdlyDim1__top_level__DOT__relu_out__v0 = 0;
    CData/*2:0*/ __VdlyDim2__top_level__DOT__relu_out__v0;
    __VdlyDim2__top_level__DOT__relu_out__v0 = 0;
    CData/*0:0*/ __VdlySet__top_level__DOT__relu_out__v0;
    __VdlySet__top_level__DOT__relu_out__v0 = 0;
    SData/*15:0*/ __VdlyVal__top_level__DOT__pool_out__v0;
    __VdlyVal__top_level__DOT__pool_out__v0 = 0;
    CData/*3:0*/ __VdlyDim0__top_level__DOT__pool_out__v0;
    __VdlyDim0__top_level__DOT__pool_out__v0 = 0;
    CData/*3:0*/ __VdlyDim1__top_level__DOT__pool_out__v0;
    __VdlyDim1__top_level__DOT__pool_out__v0 = 0;
    CData/*2:0*/ __VdlyDim2__top_level__DOT__pool_out__v0;
    __VdlyDim2__top_level__DOT__pool_out__v0 = 0;
    CData/*0:0*/ __VdlySet__top_level__DOT__pool_out__v0;
    __VdlySet__top_level__DOT__pool_out__v0 = 0;
    SData/*15:0*/ __VdlyVal__top_level__DOT__logits__v0;
    __VdlyVal__top_level__DOT__logits__v0 = 0;
    CData/*3:0*/ __VdlyDim0__top_level__DOT__logits__v0;
    __VdlyDim0__top_level__DOT__logits__v0 = 0;
    CData/*0:0*/ __VdlySet__top_level__DOT__logits__v0;
    __VdlySet__top_level__DOT__logits__v0 = 0;
    // Body
    __Vdly__top_level__DOT__RX__DOT__state = vlSelfRef.top_level__DOT__RX__DOT__state;
    __Vdly__top_level__DOT__RX__DOT__clk_cnt = vlSelfRef.top_level__DOT__RX__DOT__clk_cnt;
    __Vdly__top_level__DOT__RX__DOT__bit_idx = vlSelfRef.top_level__DOT__RX__DOT__bit_idx;
    __Vdly__top_level__DOT__RX__DOT__data = vlSelfRef.top_level__DOT__RX__DOT__data;
    __Vdly__top_level__DOT__u_relu__DOT__state = vlSelfRef.top_level__DOT__u_relu__DOT__state;
    __Vdly__top_level__DOT__u_relu__DOT__c = vlSelfRef.top_level__DOT__u_relu__DOT__c;
    __Vdly__top_level__DOT__u_relu__DOT__r = vlSelfRef.top_level__DOT__u_relu__DOT__r;
    __Vdly__top_level__DOT__u_relu__DOT__q = vlSelfRef.top_level__DOT__u_relu__DOT__q;
    __VdlySet__top_level__DOT__relu_out__v0 = 0U;
    __Vdly__top_level__DOT__u_dense__DOT__state = vlSelfRef.top_level__DOT__u_dense__DOT__state;
    __Vdly__top_level__DOT__u_dense__DOT__o = vlSelfRef.top_level__DOT__u_dense__DOT__o;
    __Vdly__top_level__DOT__u_dense__DOT__i = vlSelfRef.top_level__DOT__u_dense__DOT__i;
    __VdlySet__top_level__DOT__logits__v0 = 0U;
    __Vdly__top_level__DOT__u_pool__DOT__state = vlSelfRef.top_level__DOT__u_pool__DOT__state;
    __Vdly__top_level__DOT__u_pool__DOT__c = vlSelfRef.top_level__DOT__u_pool__DOT__c;
    __Vdly__top_level__DOT__u_pool__DOT__r = vlSelfRef.top_level__DOT__u_pool__DOT__r;
    __Vdly__top_level__DOT__u_pool__DOT__q = vlSelfRef.top_level__DOT__u_pool__DOT__q;
    __VdlySet__top_level__DOT__pool_out__v0 = 0U;
    __Vdly__top_level__DOT__u_conv__DOT__state = vlSelfRef.top_level__DOT__u_conv__DOT__state;
    __Vdly__top_level__DOT__u_conv__DOT__out_row = vlSelfRef.top_level__DOT__u_conv__DOT__out_row;
    __Vdly__top_level__DOT__u_conv__DOT__out_col = vlSelfRef.top_level__DOT__u_conv__DOT__out_col;
    __Vdly__top_level__DOT__u_conv__DOT__oc = vlSelfRef.top_level__DOT__u_conv__DOT__oc;
    __VdlySet__top_level__DOT__conv_out__v0 = 0U;
    __Vdly__top_level__DOT__u_argmax__DOT__state = vlSelfRef.top_level__DOT__u_argmax__DOT__state;
    __Vdly__top_level__DOT__u_argmax__DOT__i = vlSelfRef.top_level__DOT__u_argmax__DOT__i;
    __Vdly__top_level__DOT__u_argmax__DOT__besti = vlSelfRef.top_level__DOT__u_argmax__DOT__besti;
    __Vdly__top_level__DOT__u_argmax__DOT__bestv = vlSelfRef.top_level__DOT__u_argmax__DOT__bestv;
    __Vdly__top_level__DOT__r = vlSelfRef.top_level__DOT__r;
    __Vdly__top_level__DOT__c = vlSelfRef.top_level__DOT__c;
    __VdlySet__top_level__DOT__ifmap__v0 = 0U;
    __Vdly__top_level__DOT__flattening = vlSelfRef.top_level__DOT__flattening;
    __Vdly__top_level__DOT__fi_c = vlSelfRef.top_level__DOT__fi_c;
    __Vdly__top_level__DOT__fi_r = vlSelfRef.top_level__DOT__fi_r;
    __Vdly__top_level__DOT__fi_q = vlSelfRef.top_level__DOT__fi_q;
    __Vdly__top_level__DOT__fi_idx = vlSelfRef.top_level__DOT__fi_idx;
    __VdlySet__top_level__DOT__flat_vec__v0 = 0U;
    __Vdly__top_level__DOT__ctrl__DOT__state = vlSelfRef.top_level__DOT__ctrl__DOT__state;
    __Vdly__top_level__DOT__conv_start = vlSelfRef.top_level__DOT__conv_start;
    __Vdly__top_level__DOT__relu_start = vlSelfRef.top_level__DOT__relu_start;
    __Vdly__top_level__DOT__pool_start = vlSelfRef.top_level__DOT__pool_start;
    __Vdly__top_level__DOT__dense_start = vlSelfRef.top_level__DOT__dense_start;
    __Vdly__top_level__DOT__TX__DOT__state = vlSelfRef.top_level__DOT__TX__DOT__state;
    __Vdly__top_level__DOT__TX__DOT__clk_cnt = vlSelfRef.top_level__DOT__TX__DOT__clk_cnt;
    __Vdly__top_level__DOT__TX__DOT__bit_idx = vlSelfRef.top_level__DOT__TX__DOT__bit_idx;
    __Vdly__top_level__DOT__TX__DOT__data_reg = vlSelfRef.top_level__DOT__TX__DOT__data_reg;
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__u_conv__DOT__ic = 0U;
        __VdlyMask__top_level__DOT__u_conv__DOT__ic = 0xffffffffU;
    }
    vlSelfRef.top_level__DOT__u_conv__DOT__ic = ((__Vdly__top_level__DOT__u_conv__DOT__ic 
                                                  & __VdlyMask__top_level__DOT__u_conv__DOT__ic) 
                                                 | (vlSelfRef.top_level__DOT__u_conv__DOT__ic 
                                                    & (~ __VdlyMask__top_level__DOT__u_conv__DOT__ic)));
    __VdlyMask__top_level__DOT__u_conv__DOT__ic = 0U;
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__TX__DOT__state = 0U;
        vlSelfRef.uart_tx_o = 1U;
        vlSelfRef.top_level__DOT__tx_busy = 0U;
        __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
        __Vdly__top_level__DOT__TX__DOT__bit_idx = 0U;
        __Vdly__top_level__DOT__TX__DOT__data_reg = 0U;
        vlSelfRef.top_level__DOT__TX__DOT__state = __Vdly__top_level__DOT__TX__DOT__state;
        vlSelfRef.top_level__DOT__TX__DOT__clk_cnt 
            = __Vdly__top_level__DOT__TX__DOT__clk_cnt;
        vlSelfRef.top_level__DOT__TX__DOT__bit_idx 
            = __Vdly__top_level__DOT__TX__DOT__bit_idx;
        vlSelfRef.top_level__DOT__TX__DOT__data_reg 
            = __Vdly__top_level__DOT__TX__DOT__data_reg;
        __Vdly__top_level__DOT__ctrl__DOT__state = 0U;
        __Vdly__top_level__DOT__conv_start = 0U;
        __Vdly__top_level__DOT__relu_start = 0U;
        __Vdly__top_level__DOT__pool_start = 0U;
        __Vdly__top_level__DOT__dense_start = 0U;
        vlSelfRef.top_level__DOT__tx_start = 0U;
        vlSelfRef.top_level__DOT__ctrl__DOT__busy = 0U;
        vlSelfRef.top_level__DOT__ctrl__DOT__state 
            = __Vdly__top_level__DOT__ctrl__DOT__state;
        __Vdly__top_level__DOT__u_relu__DOT__state = 0U;
        vlSelfRef.top_level__DOT__relu_done = 0U;
        __Vdly__top_level__DOT__u_relu__DOT__c = 0U;
        __Vdly__top_level__DOT__u_relu__DOT__r = 0U;
        __Vdly__top_level__DOT__u_relu__DOT__q = 0U;
        vlSelfRef.top_level__DOT__relu_start = __Vdly__top_level__DOT__relu_start;
        vlSelfRef.top_level__DOT__u_relu__DOT__state 
            = __Vdly__top_level__DOT__u_relu__DOT__state;
        vlSelfRef.top_level__DOT__u_relu__DOT__c = __Vdly__top_level__DOT__u_relu__DOT__c;
        vlSelfRef.top_level__DOT__u_relu__DOT__r = __Vdly__top_level__DOT__u_relu__DOT__r;
        vlSelfRef.top_level__DOT__u_relu__DOT__q = __Vdly__top_level__DOT__u_relu__DOT__q;
        __Vdly__top_level__DOT__u_conv__DOT__state = 0U;
        vlSelfRef.top_level__DOT__conv_done = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__out_row = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__out_col = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__oc = 0U;
    } else {
        if ((0U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = 1U;
            vlSelfRef.top_level__DOT__tx_busy = 0U;
            if (vlSelfRef.top_level__DOT__tx_start) {
                vlSelfRef.top_level__DOT__tx_busy = 1U;
                __Vdly__top_level__DOT__TX__DOT__data_reg 
                    = (0xffU & ((IData)(0x30U) + (IData)(vlSelfRef.predicted_digit)));
                __Vdly__top_level__DOT__TX__DOT__state = 1U;
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = 0U;
            if ((0x1b1U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
                __Vdly__top_level__DOT__TX__DOT__bit_idx = 0U;
                __Vdly__top_level__DOT__TX__DOT__state = 2U;
            } else {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt 
                    = (0x1ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt)));
            }
        } else if ((2U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = (1U & ((IData)(vlSelfRef.top_level__DOT__TX__DOT__data_reg) 
                                         >> (IData)(vlSelfRef.top_level__DOT__TX__DOT__bit_idx)));
            if ((0x1b1U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
                if ((7U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__bit_idx))) {
                    __Vdly__top_level__DOT__TX__DOT__state = 3U;
                } else {
                    __Vdly__top_level__DOT__TX__DOT__bit_idx 
                        = (7U & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__bit_idx)));
                }
            } else {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt 
                    = (0x1ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt)));
            }
        } else if ((3U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = 1U;
            if ((0x1b1U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__TX__DOT__state = 4U;
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
            } else {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt 
                    = (0x1ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt)));
            }
        } else if ((4U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            __Vdly__top_level__DOT__TX__DOT__state = 0U;
        }
        vlSelfRef.top_level__DOT__TX__DOT__state = __Vdly__top_level__DOT__TX__DOT__state;
        vlSelfRef.top_level__DOT__TX__DOT__clk_cnt 
            = __Vdly__top_level__DOT__TX__DOT__clk_cnt;
        vlSelfRef.top_level__DOT__TX__DOT__bit_idx 
            = __Vdly__top_level__DOT__TX__DOT__bit_idx;
        vlSelfRef.top_level__DOT__TX__DOT__data_reg 
            = __Vdly__top_level__DOT__TX__DOT__data_reg;
        __Vdly__top_level__DOT__conv_start = 0U;
        __Vdly__top_level__DOT__relu_start = 0U;
        __Vdly__top_level__DOT__pool_start = 0U;
        __Vdly__top_level__DOT__dense_start = 0U;
        vlSelfRef.top_level__DOT__tx_start = 0U;
        if ((4U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
                    __Vdly__top_level__DOT__ctrl__DOT__state = 0U;
                } else if (vlSelfRef.top_level__DOT__argmax_done) {
                    vlSelfRef.top_level__DOT__tx_start = 1U;
                    __Vdly__top_level__DOT__ctrl__DOT__state = 7U;
                }
            } else if ((1U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
                if (vlSelfRef.top_level__DOT__dense_done) {
                    __Vdly__top_level__DOT__ctrl__DOT__state = 6U;
                }
            } else if (vlSelfRef.top_level__DOT__flat_done) {
                __Vdly__top_level__DOT__dense_start = 1U;
                __Vdly__top_level__DOT__ctrl__DOT__state = 5U;
            }
        } else if ((2U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
                if (vlSelfRef.top_level__DOT__pool_done) {
                    __Vdly__top_level__DOT__ctrl__DOT__state = 4U;
                }
            } else if (vlSelfRef.top_level__DOT__relu_done) {
                __Vdly__top_level__DOT__pool_start = 1U;
                __Vdly__top_level__DOT__ctrl__DOT__state = 3U;
            }
        } else if ((1U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
            if (vlSelfRef.top_level__DOT__conv_done) {
                __Vdly__top_level__DOT__relu_start = 1U;
                __Vdly__top_level__DOT__ctrl__DOT__state = 2U;
            }
        } else {
            vlSelfRef.top_level__DOT__ctrl__DOT__busy = 0U;
            if (vlSelfRef.top_level__DOT__frame_loaded) {
                vlSelfRef.top_level__DOT__ctrl__DOT__busy = 1U;
                __Vdly__top_level__DOT__conv_start = 1U;
                __Vdly__top_level__DOT__ctrl__DOT__state = 1U;
            }
        }
        vlSelfRef.top_level__DOT__ctrl__DOT__state 
            = __Vdly__top_level__DOT__ctrl__DOT__state;
        if ((0U == (IData)(vlSelfRef.top_level__DOT__u_relu__DOT__state))) {
            vlSelfRef.top_level__DOT__relu_done = 0U;
            if (vlSelfRef.top_level__DOT__relu_start) {
                __Vdly__top_level__DOT__u_relu__DOT__c = 0U;
                __Vdly__top_level__DOT__u_relu__DOT__r = 0U;
                __Vdly__top_level__DOT__u_relu__DOT__q = 0U;
                __Vdly__top_level__DOT__u_relu__DOT__state = 1U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__u_relu__DOT__state))) {
            vlSelfRef.top_level__DOT__u_relu__DOT__unnamedblk1__DOT__v 
                = ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__q))
                    ? vlSelfRef.top_level__DOT__conv_out
                   [(7U & vlSelfRef.top_level__DOT__u_relu__DOT__c)]
                   [((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__r))
                      ? (0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__r)
                      : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__q)]
                    : 0U);
            vlSelfRef.top_level__DOT__u_relu__DOT____Vlvbound_he9448718__0 
                = ((0x8000U & (IData)(vlSelfRef.top_level__DOT__u_relu__DOT__unnamedblk1__DOT__v))
                    ? 0U : (IData)(vlSelfRef.top_level__DOT__u_relu__DOT__unnamedblk1__DOT__v));
            if (((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__q)) 
                 && (0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__r)))) {
                __VdlyVal__top_level__DOT__relu_out__v0 
                    = vlSelfRef.top_level__DOT__u_relu__DOT____Vlvbound_he9448718__0;
                __VdlyDim0__top_level__DOT__relu_out__v0 
                    = (0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__q);
                __VdlyDim1__top_level__DOT__relu_out__v0 
                    = (0x1fU & vlSelfRef.top_level__DOT__u_relu__DOT__r);
                __VdlyDim2__top_level__DOT__relu_out__v0 
                    = (7U & vlSelfRef.top_level__DOT__u_relu__DOT__c);
                __VdlySet__top_level__DOT__relu_out__v0 = 1U;
            }
            if ((0x1bU == vlSelfRef.top_level__DOT__u_relu__DOT__q)) {
                __Vdly__top_level__DOT__u_relu__DOT__q = 0U;
                if ((0x1bU == vlSelfRef.top_level__DOT__u_relu__DOT__r)) {
                    __Vdly__top_level__DOT__u_relu__DOT__r = 0U;
                    if ((7U == vlSelfRef.top_level__DOT__u_relu__DOT__c)) {
                        __Vdly__top_level__DOT__u_relu__DOT__state = 2U;
                    } else {
                        __Vdly__top_level__DOT__u_relu__DOT__c 
                            = ((IData)(1U) + vlSelfRef.top_level__DOT__u_relu__DOT__c);
                    }
                } else {
                    __Vdly__top_level__DOT__u_relu__DOT__r 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_relu__DOT__r);
                }
            } else {
                __Vdly__top_level__DOT__u_relu__DOT__q 
                    = ((IData)(1U) + vlSelfRef.top_level__DOT__u_relu__DOT__q);
            }
        } else if ((2U == (IData)(vlSelfRef.top_level__DOT__u_relu__DOT__state))) {
            vlSelfRef.top_level__DOT__relu_done = 1U;
            __Vdly__top_level__DOT__u_relu__DOT__state = 0U;
        }
        vlSelfRef.top_level__DOT__relu_start = __Vdly__top_level__DOT__relu_start;
        vlSelfRef.top_level__DOT__u_relu__DOT__state 
            = __Vdly__top_level__DOT__u_relu__DOT__state;
        vlSelfRef.top_level__DOT__u_relu__DOT__c = __Vdly__top_level__DOT__u_relu__DOT__c;
        vlSelfRef.top_level__DOT__u_relu__DOT__r = __Vdly__top_level__DOT__u_relu__DOT__r;
        vlSelfRef.top_level__DOT__u_relu__DOT__q = __Vdly__top_level__DOT__u_relu__DOT__q;
        if ((0U == (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__state))) {
            vlSelfRef.top_level__DOT__conv_done = 0U;
            if (vlSelfRef.top_level__DOT__conv_start) {
                __Vdly__top_level__DOT__u_conv__DOT__oc = 0U;
                __Vdly__top_level__DOT__u_conv__DOT__out_row = 0U;
                __Vdly__top_level__DOT__u_conv__DOT__out_col = 0U;
                __Vdly__top_level__DOT__u_conv__DOT__state = 1U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__state))) {
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = (vlSelfRef.top_level__DOT__u_conv__DOT__out_row 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__kj = 3U;
            vlSelfRef.top_level__DOT__u_conv__DOT__ki = 3U;
            vlSelfRef.top_level__DOT__u_conv__DOT__ic = 1U;
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = (vlSelfRef.top_level__DOT__u_conv__DOT__out_col 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][0U]
                                            [0U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = (vlSelfRef.top_level__DOT__u_conv__DOT__out_row 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = vlSelfRef.top_level__DOT__u_conv__DOT__out_col;
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][0U]
                                            [1U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = (vlSelfRef.top_level__DOT__u_conv__DOT__out_row 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_col);
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][0U]
                                            [2U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = vlSelfRef.top_level__DOT__u_conv__DOT__out_row;
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = (vlSelfRef.top_level__DOT__u_conv__DOT__out_col 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][1U]
                                            [0U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = vlSelfRef.top_level__DOT__u_conv__DOT__out_row;
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = vlSelfRef.top_level__DOT__u_conv__DOT__out_col;
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][1U]
                                            [1U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = vlSelfRef.top_level__DOT__u_conv__DOT__out_row;
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_col);
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][1U]
                                            [2U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_row);
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = (vlSelfRef.top_level__DOT__u_conv__DOT__out_col 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][2U]
                                            [0U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_row);
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = vlSelfRef.top_level__DOT__u_conv__DOT__out_col;
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][2U]
                                            [1U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r 
                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_row);
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c 
                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_col);
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val 
                = ((((VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r) 
                      | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)) 
                     | VL_GTS_III(32, 0U, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)) 
                    | VL_LTES_III(32, 0x1cU, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                    ? 0U : ((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c))
                             ? vlSelfRef.top_level__DOT__ifmap
                            [0U][((0x1bU >= (0x1fU 
                                             & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r))
                                   ? (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r)
                                   : 0U)][(0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c)]
                             : 0U));
            vlSelfRef.top_level__DOT__u_conv__DOT__mult 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val)), 
                              VL_EXTENDS_II(32,16, 
                                            vlSelfRef.top_level__DOT__conv_w
                                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                            [0U][2U]
                                            [2U]));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                     + VL_EXTENDS_QI(36,32, vlSelfRef.top_level__DOT__u_conv__DOT__mult)));
            vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                = (0xfffffffffULL & VL_SHIFTL_QQI(36,36,32, 
                                                  (vlSelfRef.top_level__DOT__u_conv__DOT__accum 
                                                   + 
                                                   (((QData)((IData)(
                                                                     (0xfffffU 
                                                                      & (- (IData)(
                                                                                (1U 
                                                                                & (vlSelfRef.top_level__DOT__conv_b
                                                                                [
                                                                                (7U 
                                                                                & vlSelfRef.top_level__DOT__u_conv__DOT__oc)] 
                                                                                >> 0xfU))))))) 
                                                     << 0x10U) 
                                                    | (QData)((IData)(
                                                                      vlSelfRef.top_level__DOT__conv_b
                                                                      [
                                                                      (7U 
                                                                       & vlSelfRef.top_level__DOT__u_conv__DOT__oc)])))), 7U));
            __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__val 
                = vlSelfRef.top_level__DOT__u_conv__DOT__accum;
            vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__shifted 
                = (0xfffffffffULL & VL_SHIFTRS_QQI(36,36,32, __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__val, 7U));
            vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__result 
                = (VL_LTS_IQQ(36, 0x7fffULL, vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__shifted)
                    ? 0x7fffU : (VL_GTS_IQQ(36, 0xfffff8000ULL, vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__shifted)
                                  ? 0x8000U : (0xffffU 
                                               & (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__shifted))));
            __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__Vfuncout 
                = vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__result;
            vlSelfRef.top_level__DOT__u_conv__DOT____Vlvbound_hfb6f3c5f__0 
                = __Vfunc_top_level__DOT__u_conv__DOT__scale_and_saturate__0__Vfuncout;
            if (((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__out_col)) 
                 && (0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__out_row)))) {
                __VdlyVal__top_level__DOT__conv_out__v0 
                    = vlSelfRef.top_level__DOT__u_conv__DOT____Vlvbound_hfb6f3c5f__0;
                __VdlyDim0__top_level__DOT__conv_out__v0 
                    = (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__out_col);
                __VdlyDim1__top_level__DOT__conv_out__v0 
                    = (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__out_row);
                __VdlyDim2__top_level__DOT__conv_out__v0 
                    = (7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc);
                __VdlySet__top_level__DOT__conv_out__v0 = 1U;
            }
            if ((0x1bU == vlSelfRef.top_level__DOT__u_conv__DOT__out_col)) {
                __Vdly__top_level__DOT__u_conv__DOT__out_col = 0U;
                if ((0x1bU == vlSelfRef.top_level__DOT__u_conv__DOT__out_row)) {
                    __Vdly__top_level__DOT__u_conv__DOT__out_row = 0U;
                    if ((7U == vlSelfRef.top_level__DOT__u_conv__DOT__oc)) {
                        __Vdly__top_level__DOT__u_conv__DOT__state = 3U;
                    } else {
                        __Vdly__top_level__DOT__u_conv__DOT__oc 
                            = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__oc);
                    }
                } else {
                    __Vdly__top_level__DOT__u_conv__DOT__out_row 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_row);
                }
            } else {
                __Vdly__top_level__DOT__u_conv__DOT__out_col 
                    = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__out_col);
            }
        } else if ((3U == (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__state))) {
            vlSelfRef.top_level__DOT__conv_done = 1U;
            __Vdly__top_level__DOT__u_conv__DOT__state = 0U;
        } else {
            __Vdly__top_level__DOT__u_conv__DOT__state = 0U;
        }
    }
    vlSelfRef.top_level__DOT__conv_start = __Vdly__top_level__DOT__conv_start;
    vlSelfRef.top_level__DOT__u_conv__DOT__state = __Vdly__top_level__DOT__u_conv__DOT__state;
    vlSelfRef.top_level__DOT__u_conv__DOT__out_row 
        = __Vdly__top_level__DOT__u_conv__DOT__out_row;
    vlSelfRef.top_level__DOT__u_conv__DOT__out_col 
        = __Vdly__top_level__DOT__u_conv__DOT__out_col;
    vlSelfRef.top_level__DOT__u_conv__DOT__oc = __Vdly__top_level__DOT__u_conv__DOT__oc;
    if (__VdlySet__top_level__DOT__conv_out__v0) {
        vlSelfRef.top_level__DOT__conv_out[__VdlyDim2__top_level__DOT__conv_out__v0][__VdlyDim1__top_level__DOT__conv_out__v0][__VdlyDim0__top_level__DOT__conv_out__v0] 
            = __VdlyVal__top_level__DOT__conv_out__v0;
    }
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__u_argmax__DOT__state = 0U;
        vlSelfRef.top_level__DOT__argmax_done = 0U;
        vlSelfRef.predicted_digit = 0U;
        __Vdly__top_level__DOT__u_argmax__DOT__i = 0U;
        __Vdly__top_level__DOT__u_argmax__DOT__besti = 0U;
        __Vdly__top_level__DOT__u_argmax__DOT__bestv = 0U;
        vlSelfRef.top_level__DOT__u_argmax__DOT__state 
            = __Vdly__top_level__DOT__u_argmax__DOT__state;
        vlSelfRef.top_level__DOT__u_argmax__DOT__i 
            = __Vdly__top_level__DOT__u_argmax__DOT__i;
        vlSelfRef.top_level__DOT__u_argmax__DOT__besti 
            = __Vdly__top_level__DOT__u_argmax__DOT__besti;
        vlSelfRef.top_level__DOT__u_argmax__DOT__bestv 
            = __Vdly__top_level__DOT__u_argmax__DOT__bestv;
        __Vdly__top_level__DOT__r = 0U;
        __Vdly__top_level__DOT__c = 0U;
        vlSelfRef.top_level__DOT__frame_loaded = 0U;
    } else {
        if ((0U == (IData)(vlSelfRef.top_level__DOT__u_argmax__DOT__state))) {
            vlSelfRef.top_level__DOT__argmax_done = 0U;
            if (vlSelfRef.top_level__DOT__dense_done) {
                __Vdly__top_level__DOT__u_argmax__DOT__i = 1U;
                __Vdly__top_level__DOT__u_argmax__DOT__besti = 0U;
                __Vdly__top_level__DOT__u_argmax__DOT__bestv 
                    = vlSelfRef.top_level__DOT__logits
                    [0U];
                __Vdly__top_level__DOT__u_argmax__DOT__state = 1U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__u_argmax__DOT__state))) {
            if (VL_GTS_III(16, ((9U >= (0xfU & vlSelfRef.top_level__DOT__u_argmax__DOT__i))
                                 ? vlSelfRef.top_level__DOT__logits
                                [(0xfU & vlSelfRef.top_level__DOT__u_argmax__DOT__i)]
                                 : 0U), (IData)(vlSelfRef.top_level__DOT__u_argmax__DOT__bestv))) {
                __Vdly__top_level__DOT__u_argmax__DOT__bestv 
                    = ((9U >= (0xfU & vlSelfRef.top_level__DOT__u_argmax__DOT__i))
                        ? vlSelfRef.top_level__DOT__logits
                       [(0xfU & vlSelfRef.top_level__DOT__u_argmax__DOT__i)]
                        : 0U);
                __Vdly__top_level__DOT__u_argmax__DOT__besti 
                    = (0xfU & vlSelfRef.top_level__DOT__u_argmax__DOT__i);
            }
            if ((9U == vlSelfRef.top_level__DOT__u_argmax__DOT__i)) {
                __Vdly__top_level__DOT__u_argmax__DOT__state = 2U;
            } else {
                __Vdly__top_level__DOT__u_argmax__DOT__i 
                    = ((IData)(1U) + vlSelfRef.top_level__DOT__u_argmax__DOT__i);
            }
        } else if ((2U == (IData)(vlSelfRef.top_level__DOT__u_argmax__DOT__state))) {
            vlSelfRef.predicted_digit = vlSelfRef.top_level__DOT__u_argmax__DOT__besti;
            vlSelfRef.top_level__DOT__argmax_done = 1U;
            __Vdly__top_level__DOT__u_argmax__DOT__state = 0U;
        }
        vlSelfRef.top_level__DOT__u_argmax__DOT__state 
            = __Vdly__top_level__DOT__u_argmax__DOT__state;
        vlSelfRef.top_level__DOT__u_argmax__DOT__i 
            = __Vdly__top_level__DOT__u_argmax__DOT__i;
        vlSelfRef.top_level__DOT__u_argmax__DOT__besti 
            = __Vdly__top_level__DOT__u_argmax__DOT__besti;
        vlSelfRef.top_level__DOT__u_argmax__DOT__bestv 
            = __Vdly__top_level__DOT__u_argmax__DOT__bestv;
        vlSelfRef.top_level__DOT__frame_loaded = 0U;
        if (vlSelfRef.top_level__DOT__rx_dv) {
            vlSelfRef.top_level__DOT____Vlvbound_h5be8a0cc__0 
                = (0xffffU & VL_SHIFTL_III(16,16,32, (IData)(vlSelfRef.top_level__DOT__rx_byte), 7U));
            if (((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__c)) 
                 && (0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__r)))) {
                __VdlyVal__top_level__DOT__ifmap__v0 
                    = vlSelfRef.top_level__DOT____Vlvbound_h5be8a0cc__0;
                __VdlyDim0__top_level__DOT__ifmap__v0 
                    = (0x1fU & vlSelfRef.top_level__DOT__c);
                __VdlyDim1__top_level__DOT__ifmap__v0 
                    = (0x1fU & vlSelfRef.top_level__DOT__r);
                __VdlySet__top_level__DOT__ifmap__v0 = 1U;
            }
            if ((0x1bU == vlSelfRef.top_level__DOT__c)) {
                __Vdly__top_level__DOT__c = 0U;
                if ((0x1bU == vlSelfRef.top_level__DOT__r)) {
                    __Vdly__top_level__DOT__r = 0U;
                    vlSelfRef.top_level__DOT__frame_loaded = 1U;
                } else {
                    __Vdly__top_level__DOT__r = ((IData)(1U) 
                                                 + vlSelfRef.top_level__DOT__r);
                }
            } else {
                __Vdly__top_level__DOT__c = ((IData)(1U) 
                                             + vlSelfRef.top_level__DOT__c);
            }
        }
    }
    vlSelfRef.top_level__DOT__r = __Vdly__top_level__DOT__r;
    vlSelfRef.top_level__DOT__c = __Vdly__top_level__DOT__c;
    if (__VdlySet__top_level__DOT__ifmap__v0) {
        vlSelfRef.top_level__DOT__ifmap[0U][__VdlyDim1__top_level__DOT__ifmap__v0][__VdlyDim0__top_level__DOT__ifmap__v0] 
            = __VdlyVal__top_level__DOT__ifmap__v0;
    }
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__flattening = 0U;
        vlSelfRef.top_level__DOT__flat_done = 0U;
        __Vdly__top_level__DOT__fi_c = 0U;
        __Vdly__top_level__DOT__fi_r = 0U;
        __Vdly__top_level__DOT__fi_q = 0U;
        __Vdly__top_level__DOT__fi_idx = 0U;
    } else {
        vlSelfRef.top_level__DOT__flat_done = 0U;
        if (((IData)(vlSelfRef.top_level__DOT__pool_done) 
             & (~ (IData)(vlSelfRef.top_level__DOT__flattening)))) {
            __Vdly__top_level__DOT__flattening = 1U;
            __Vdly__top_level__DOT__fi_c = 0U;
            __Vdly__top_level__DOT__fi_r = 0U;
            __Vdly__top_level__DOT__fi_q = 0U;
            __Vdly__top_level__DOT__fi_idx = 0U;
        }
        if (vlSelfRef.top_level__DOT__flattening) {
            vlSelfRef.top_level__DOT____Vlvbound_h92949796__0 
                = ((0xdU >= (0xfU & vlSelfRef.top_level__DOT__fi_q))
                    ? vlSelfRef.top_level__DOT__pool_out
                   [(7U & vlSelfRef.top_level__DOT__fi_c)]
                   [((0xdU >= (0xfU & vlSelfRef.top_level__DOT__fi_r))
                      ? (0xfU & vlSelfRef.top_level__DOT__fi_r)
                      : 0U)][(0xfU & vlSelfRef.top_level__DOT__fi_q)]
                    : 0U);
            if ((0x61fU >= (0x7ffU & vlSelfRef.top_level__DOT__fi_idx))) {
                __VdlyVal__top_level__DOT__flat_vec__v0 
                    = vlSelfRef.top_level__DOT____Vlvbound_h92949796__0;
                __VdlyDim0__top_level__DOT__flat_vec__v0 
                    = (0x7ffU & vlSelfRef.top_level__DOT__fi_idx);
                __VdlySet__top_level__DOT__flat_vec__v0 = 1U;
            }
            if ((0xdU == vlSelfRef.top_level__DOT__fi_q)) {
                __Vdly__top_level__DOT__fi_q = 0U;
                if ((0xdU == vlSelfRef.top_level__DOT__fi_r)) {
                    __Vdly__top_level__DOT__fi_r = 0U;
                    if ((7U == vlSelfRef.top_level__DOT__fi_c)) {
                        __Vdly__top_level__DOT__flattening = 0U;
                        vlSelfRef.top_level__DOT__flat_done = 1U;
                    } else {
                        __Vdly__top_level__DOT__fi_c 
                            = ((IData)(1U) + vlSelfRef.top_level__DOT__fi_c);
                    }
                } else {
                    __Vdly__top_level__DOT__fi_r = 
                        ((IData)(1U) + vlSelfRef.top_level__DOT__fi_r);
                }
            } else {
                __Vdly__top_level__DOT__fi_q = ((IData)(1U) 
                                                + vlSelfRef.top_level__DOT__fi_q);
            }
            __Vdly__top_level__DOT__fi_idx = ((IData)(1U) 
                                              + vlSelfRef.top_level__DOT__fi_idx);
        }
    }
    vlSelfRef.top_level__DOT__flattening = __Vdly__top_level__DOT__flattening;
    vlSelfRef.top_level__DOT__fi_c = __Vdly__top_level__DOT__fi_c;
    vlSelfRef.top_level__DOT__fi_r = __Vdly__top_level__DOT__fi_r;
    vlSelfRef.top_level__DOT__fi_q = __Vdly__top_level__DOT__fi_q;
    vlSelfRef.top_level__DOT__fi_idx = __Vdly__top_level__DOT__fi_idx;
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__u_dense__DOT__state = 0U;
        vlSelfRef.top_level__DOT__dense_done = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__o = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__i = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__acc = 0ULL;
        __VdlyMask__top_level__DOT__u_dense__DOT__acc = 0x7ffffffffffULL;
    } else if ((2U & (IData)(vlSelfRef.top_level__DOT__u_dense__DOT__state))) {
        if ((1U & (IData)(vlSelfRef.top_level__DOT__u_dense__DOT__state))) {
            vlSelfRef.top_level__DOT__dense_done = 1U;
            __Vdly__top_level__DOT__u_dense__DOT__state = 0U;
        } else {
            __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__v 
                = vlSelfRef.top_level__DOT__u_dense__DOT__acc;
            __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__s = 0;
            __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mx = 0;
            __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mn = 0;
            {
                __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__s 
                    = (0x7ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__v, 7U));
                __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mx = 0x7fffU;
                __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mn = 0x8000U;
                if (VL_GTS_IQQ(43, __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__s, 
                               (0x7ffffffffffULL & 
                                VL_EXTENDS_QI(43,16, (IData)(__Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mx))))) {
                    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__Vfuncout 
                        = __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mx;
                } else if (VL_LTS_IQQ(43, __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__s, 
                                      (0x7ffffffffffULL 
                                       & VL_EXTENDS_QI(43,16, (IData)(__Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mn))))) {
                    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__Vfuncout 
                        = __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__mn;
                    goto __Vlabel1;
                } else {
                    __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__Vfuncout 
                        = (0xffffU & (IData)(__Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__s));
                    goto __Vlabel1;
                }
                __Vlabel1: ;
            }
            vlSelfRef.top_level__DOT__u_dense__DOT____Vlvbound_h29ef7931__0 
                = __Vfunc_top_level__DOT__u_dense__DOT__scale_sat__2__Vfuncout;
            if (VL_LIKELY(((9U >= (0xfU & vlSelfRef.top_level__DOT__u_dense__DOT__o))))) {
                __VdlyVal__top_level__DOT__logits__v0 
                    = vlSelfRef.top_level__DOT__u_dense__DOT____Vlvbound_h29ef7931__0;
                __VdlyDim0__top_level__DOT__logits__v0 
                    = (0xfU & vlSelfRef.top_level__DOT__u_dense__DOT__o);
                __VdlySet__top_level__DOT__logits__v0 = 1U;
            }
            if ((9U == vlSelfRef.top_level__DOT__u_dense__DOT__o)) {
                __Vdly__top_level__DOT__u_dense__DOT__state = 3U;
            } else {
                __Vdly__top_level__DOT__u_dense__DOT__o 
                    = ((IData)(1U) + vlSelfRef.top_level__DOT__u_dense__DOT__o);
                __Vdly__top_level__DOT__u_dense__DOT__i = 0U;
                __Vdly__top_level__DOT__u_dense__DOT__acc 
                    = (0x7ffffffffffULL & VL_SHIFTL_QQI(43,43,32, 
                                                        (((QData)((IData)(
                                                                          (0x7ffffffU 
                                                                           & (- (IData)(
                                                                                (1U 
                                                                                & (((9U 
                                                                                >= 
                                                                                (0xfU 
                                                                                & ((IData)(1U) 
                                                                                + vlSelfRef.top_level__DOT__u_dense__DOT__o)))
                                                                                 ? 
                                                                                vlSelfRef.top_level__DOT__dense_b
                                                                                [
                                                                                (0xfU 
                                                                                & ((IData)(1U) 
                                                                                + vlSelfRef.top_level__DOT__u_dense__DOT__o))]
                                                                                 : 0U) 
                                                                                >> 0xfU))))))) 
                                                          << 0x10U) 
                                                         | (QData)((IData)(
                                                                           ((9U 
                                                                             >= 
                                                                             (0xfU 
                                                                              & ((IData)(1U) 
                                                                                + vlSelfRef.top_level__DOT__u_dense__DOT__o)))
                                                                             ? 
                                                                            vlSelfRef.top_level__DOT__dense_b
                                                                            [
                                                                            (0xfU 
                                                                             & ((IData)(1U) 
                                                                                + vlSelfRef.top_level__DOT__u_dense__DOT__o))]
                                                                             : 0U)))), 7U));
                __VdlyMask__top_level__DOT__u_dense__DOT__acc = 0x7ffffffffffULL;
                __Vdly__top_level__DOT__u_dense__DOT__state = 1U;
            }
        }
    } else if ((1U & (IData)(vlSelfRef.top_level__DOT__u_dense__DOT__state))) {
        vlSelfRef.top_level__DOT__u_dense__DOT__prod 
            = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                            ((0x61fU 
                                              >= (0x7ffU 
                                                  & vlSelfRef.top_level__DOT__u_dense__DOT__i))
                                              ? vlSelfRef.top_level__DOT__flat_vec
                                             [(0x7ffU 
                                               & vlSelfRef.top_level__DOT__u_dense__DOT__i)]
                                              : 0U)), 
                          VL_EXTENDS_II(32,16, ((0x61fU 
                                                 >= 
                                                 (0x7ffU 
                                                  & vlSelfRef.top_level__DOT__u_dense__DOT__i))
                                                 ? 
                                                vlSelfRef.top_level__DOT__dense_w
                                                [((9U 
                                                   >= 
                                                   (0xfU 
                                                    & vlSelfRef.top_level__DOT__u_dense__DOT__o))
                                                   ? 
                                                  (0xfU 
                                                   & vlSelfRef.top_level__DOT__u_dense__DOT__o)
                                                   : 0U)]
                                                [(0x7ffU 
                                                  & vlSelfRef.top_level__DOT__u_dense__DOT__i)]
                                                 : 0U)));
        vlSelfRef.top_level__DOT__u_dense__DOT__acc 
            = (0x7ffffffffffULL & (vlSelfRef.top_level__DOT__u_dense__DOT__acc 
                                   + VL_EXTENDS_QI(43,32, vlSelfRef.top_level__DOT__u_dense__DOT__prod)));
        if ((0x61fU == vlSelfRef.top_level__DOT__u_dense__DOT__i)) {
            __Vdly__top_level__DOT__u_dense__DOT__state = 2U;
        } else {
            __Vdly__top_level__DOT__u_dense__DOT__i 
                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_dense__DOT__i);
        }
    } else {
        vlSelfRef.top_level__DOT__dense_done = 0U;
        if (vlSelfRef.top_level__DOT__dense_start) {
            __Vdly__top_level__DOT__u_dense__DOT__o = 0U;
            __Vdly__top_level__DOT__u_dense__DOT__i = 0U;
            __Vdly__top_level__DOT__u_dense__DOT__acc 
                = (0x7ffffffffffULL & VL_SHIFTL_QQI(43,43,32, 
                                                    (((QData)((IData)(
                                                                      (0x7ffffffU 
                                                                       & (- (IData)(
                                                                                (1U 
                                                                                & (vlSelfRef.top_level__DOT__dense_b
                                                                                [0U] 
                                                                                >> 0xfU))))))) 
                                                      << 0x10U) 
                                                     | (QData)((IData)(
                                                                       vlSelfRef.top_level__DOT__dense_b
                                                                       [0U]))), 7U));
            __VdlyMask__top_level__DOT__u_dense__DOT__acc = 0x7ffffffffffULL;
            __Vdly__top_level__DOT__u_dense__DOT__state = 1U;
        }
    }
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__RX__DOT__state = 0U;
        vlSelfRef.top_level__DOT__rx_dv = 0U;
        __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
        __Vdly__top_level__DOT__RX__DOT__bit_idx = 0U;
        __Vdly__top_level__DOT__RX__DOT__data = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__state = 0U;
        vlSelfRef.top_level__DOT__pool_done = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__c = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__r = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__q = 0U;
    } else {
        vlSelfRef.top_level__DOT__rx_dv = 0U;
        if ((0U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((1U & (~ (IData)(vlSelfRef.uart_rx_i)))) {
                __Vdly__top_level__DOT__RX__DOT__state = 1U;
                __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((0xd9U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
                __Vdly__top_level__DOT__RX__DOT__state = 2U;
                __Vdly__top_level__DOT__RX__DOT__bit_idx = 0U;
            } else {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt 
                    = (0x1ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt)));
            }
        } else if ((2U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((0x1b1U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
                __Vdly__top_level__DOT__RX__DOT__data 
                    = (((~ ((IData)(1U) << (IData)(vlSelfRef.top_level__DOT__RX__DOT__bit_idx))) 
                        & (IData)(__Vdly__top_level__DOT__RX__DOT__data)) 
                       | (0xffU & ((IData)(vlSelfRef.uart_rx_i) 
                                   << (IData)(vlSelfRef.top_level__DOT__RX__DOT__bit_idx))));
                if ((7U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__bit_idx))) {
                    __Vdly__top_level__DOT__RX__DOT__state = 3U;
                } else {
                    __Vdly__top_level__DOT__RX__DOT__bit_idx 
                        = (7U & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__RX__DOT__bit_idx)));
                }
            } else {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt 
                    = (0x1ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt)));
            }
        } else if ((3U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((0x1b1U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt))) {
                vlSelfRef.top_level__DOT__rx_byte = vlSelfRef.top_level__DOT__RX__DOT__data;
                vlSelfRef.top_level__DOT__rx_dv = 1U;
                __Vdly__top_level__DOT__RX__DOT__state = 4U;
                __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
            } else {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt 
                    = (0x1ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt)));
            }
        } else if ((4U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            __Vdly__top_level__DOT__RX__DOT__state = 0U;
        }
        if ((0U == (IData)(vlSelfRef.top_level__DOT__u_pool__DOT__state))) {
            vlSelfRef.top_level__DOT__pool_done = 0U;
            if (vlSelfRef.top_level__DOT__pool_start) {
                __Vdly__top_level__DOT__u_pool__DOT__c = 0U;
                __Vdly__top_level__DOT__u_pool__DOT__r = 0U;
                __Vdly__top_level__DOT__u_pool__DOT__q = 0U;
                __Vdly__top_level__DOT__u_pool__DOT__state = 1U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__u_pool__DOT__state))) {
            if ((0x1bU >= (0x1fU & ((IData)(1U) + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q))))) {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__d 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__c)]
                    [((0x1bU >= (0x1fU & ((IData)(1U) 
                                          + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))))
                       ? (0x1fU & ((IData)(1U) + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       : 0U)][(0x1fU & ((IData)(1U) 
                                        + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q)))];
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__b 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__c)]
                    [((0x1bU >= (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       ? (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))
                       : 0U)][(0x1fU & ((IData)(1U) 
                                        + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q)))];
            } else {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__d = 0U;
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__b = 0U;
            }
            if ((0x1bU >= (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q)))) {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__c 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__c)]
                    [((0x1bU >= (0x1fU & ((IData)(1U) 
                                          + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))))
                       ? (0x1fU & ((IData)(1U) + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       : 0U)][(0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q))];
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__a 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__c)]
                    [((0x1bU >= (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       ? (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))
                       : 0U)][(0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q))];
            } else {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__c = 0U;
                __Vfunc_top_level__DOT__u_pool__DOT__max4__1__a = 0U;
            }
            __Vfunc_top_level__DOT__u_pool__DOT__max4__1__m1 
                = (VL_GTS_III(16, (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__a), (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__b))
                    ? (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__a)
                    : (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__b));
            __Vfunc_top_level__DOT__u_pool__DOT__max4__1__m2 
                = (VL_GTS_III(16, (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__c), (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__d))
                    ? (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__c)
                    : (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__d));
            __Vfunc_top_level__DOT__u_pool__DOT__max4__1__Vfuncout 
                = (VL_GTS_III(16, (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__m1), (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__m2))
                    ? (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__m1)
                    : (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__1__m2));
            vlSelfRef.top_level__DOT__u_pool__DOT____Vlvbound_h96184ae1__0 
                = __Vfunc_top_level__DOT__u_pool__DOT__max4__1__Vfuncout;
            if (((0xdU >= (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__q)) 
                 && (0xdU >= (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__r)))) {
                __VdlyVal__top_level__DOT__pool_out__v0 
                    = vlSelfRef.top_level__DOT__u_pool__DOT____Vlvbound_h96184ae1__0;
                __VdlyDim0__top_level__DOT__pool_out__v0 
                    = (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__q);
                __VdlyDim1__top_level__DOT__pool_out__v0 
                    = (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__r);
                __VdlyDim2__top_level__DOT__pool_out__v0 
                    = (7U & vlSelfRef.top_level__DOT__u_pool__DOT__c);
                __VdlySet__top_level__DOT__pool_out__v0 = 1U;
            }
            if ((0xdU == vlSelfRef.top_level__DOT__u_pool__DOT__q)) {
                __Vdly__top_level__DOT__u_pool__DOT__q = 0U;
                if ((0xdU == vlSelfRef.top_level__DOT__u_pool__DOT__r)) {
                    __Vdly__top_level__DOT__u_pool__DOT__r = 0U;
                    if ((7U == vlSelfRef.top_level__DOT__u_pool__DOT__c)) {
                        __Vdly__top_level__DOT__u_pool__DOT__state = 2U;
                    } else {
                        __Vdly__top_level__DOT__u_pool__DOT__c 
                            = ((IData)(1U) + vlSelfRef.top_level__DOT__u_pool__DOT__c);
                    }
                } else {
                    __Vdly__top_level__DOT__u_pool__DOT__r 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_pool__DOT__r);
                }
            } else {
                __Vdly__top_level__DOT__u_pool__DOT__q 
                    = ((IData)(1U) + vlSelfRef.top_level__DOT__u_pool__DOT__q);
            }
        } else if ((2U == (IData)(vlSelfRef.top_level__DOT__u_pool__DOT__state))) {
            vlSelfRef.top_level__DOT__pool_done = 1U;
            __Vdly__top_level__DOT__u_pool__DOT__state = 0U;
        }
    }
    vlSelfRef.top_level__DOT__dense_start = __Vdly__top_level__DOT__dense_start;
    vlSelfRef.top_level__DOT__u_dense__DOT__state = __Vdly__top_level__DOT__u_dense__DOT__state;
    vlSelfRef.top_level__DOT__u_dense__DOT__o = __Vdly__top_level__DOT__u_dense__DOT__o;
    vlSelfRef.top_level__DOT__u_dense__DOT__i = __Vdly__top_level__DOT__u_dense__DOT__i;
    if (__VdlySet__top_level__DOT__flat_vec__v0) {
        vlSelfRef.top_level__DOT__flat_vec[__VdlyDim0__top_level__DOT__flat_vec__v0] 
            = __VdlyVal__top_level__DOT__flat_vec__v0;
    }
    vlSelfRef.top_level__DOT__u_dense__DOT__acc = (
                                                   (__Vdly__top_level__DOT__u_dense__DOT__acc 
                                                    & __VdlyMask__top_level__DOT__u_dense__DOT__acc) 
                                                   | (vlSelfRef.top_level__DOT__u_dense__DOT__acc 
                                                      & (~ __VdlyMask__top_level__DOT__u_dense__DOT__acc)));
    __VdlyMask__top_level__DOT__u_dense__DOT__acc = 0ULL;
    if (__VdlySet__top_level__DOT__logits__v0) {
        vlSelfRef.top_level__DOT__logits[__VdlyDim0__top_level__DOT__logits__v0] 
            = __VdlyVal__top_level__DOT__logits__v0;
    }
    vlSelfRef.top_level__DOT__RX__DOT__state = __Vdly__top_level__DOT__RX__DOT__state;
    vlSelfRef.top_level__DOT__RX__DOT__clk_cnt = __Vdly__top_level__DOT__RX__DOT__clk_cnt;
    vlSelfRef.top_level__DOT__RX__DOT__bit_idx = __Vdly__top_level__DOT__RX__DOT__bit_idx;
    vlSelfRef.top_level__DOT__RX__DOT__data = __Vdly__top_level__DOT__RX__DOT__data;
    vlSelfRef.top_level__DOT__pool_start = __Vdly__top_level__DOT__pool_start;
    vlSelfRef.top_level__DOT__u_pool__DOT__state = __Vdly__top_level__DOT__u_pool__DOT__state;
    vlSelfRef.top_level__DOT__u_pool__DOT__c = __Vdly__top_level__DOT__u_pool__DOT__c;
    vlSelfRef.top_level__DOT__u_pool__DOT__r = __Vdly__top_level__DOT__u_pool__DOT__r;
    vlSelfRef.top_level__DOT__u_pool__DOT__q = __Vdly__top_level__DOT__u_pool__DOT__q;
    if (__VdlySet__top_level__DOT__relu_out__v0) {
        vlSelfRef.top_level__DOT__relu_out[__VdlyDim2__top_level__DOT__relu_out__v0][__VdlyDim1__top_level__DOT__relu_out__v0][__VdlyDim0__top_level__DOT__relu_out__v0] 
            = __VdlyVal__top_level__DOT__relu_out__v0;
    }
    if (__VdlySet__top_level__DOT__pool_out__v0) {
        vlSelfRef.top_level__DOT__pool_out[__VdlyDim2__top_level__DOT__pool_out__v0][__VdlyDim1__top_level__DOT__pool_out__v0][__VdlyDim0__top_level__DOT__pool_out__v0] 
            = __VdlyVal__top_level__DOT__pool_out__v0;
    }
}

void Vtop_level___024root___eval_triggers__act(Vtop_level___024root* vlSelf);

bool Vtop_level___024root___eval_phase__act(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_phase__act\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<1> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtop_level___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vtop_level___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtop_level___024root___eval_phase__nba(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_phase__nba\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtop_level___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop_level___024root___dump_triggers__nba(Vtop_level___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop_level___024root___dump_triggers__act(Vtop_level___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop_level___024root___eval(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
            Vtop_level___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("hdl/top_level.sv", 2, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vtop_level___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("hdl/top_level.sv", 2, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vtop_level___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vtop_level___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtop_level___024root___eval_debug_assertions(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_debug_assertions\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((vlSelfRef.clk & 0xfeU)))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY(((vlSelfRef.reset & 0xfeU)))) {
        Verilated::overWidthError("reset");}
    if (VL_UNLIKELY(((vlSelfRef.uart_rx_i & 0xfeU)))) {
        Verilated::overWidthError("uart_rx_i");}
}
#endif  // VL_DEBUG
