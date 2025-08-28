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
    QData/*37:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__Vfuncout;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__b;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__b = 0;
    QData/*37:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__Vfuncout;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__b;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__b = 0;
    QData/*37:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__Vfuncout;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__b;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__b = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__Vfuncout;
    __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__ch;
    __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__ch = 0;
    IData/*31:0*/ __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__r;
    __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__r = 0;
    IData/*31:0*/ __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__c;
    __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__c = 0;
    QData/*37:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__Vfuncout;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__b;
    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__b = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__5__Vfuncout;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__5__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__5__a;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__5__a = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__5__b;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__5__b = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__5__x;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__5__x = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__5__y;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__5__y = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__5__m1;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__5__m1 = 0;
    SData/*15:0*/ __Vfunc_top_level__DOT__u_pool__DOT__max4__5__m2;
    __Vfunc_top_level__DOT__u_pool__DOT__max4__5__m2 = 0;
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
    QData/*63:0*/ __Vdly__top_level__DOT__cycle_ctr;
    __Vdly__top_level__DOT__cycle_ctr = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__RX__DOT__state;
    __Vdly__top_level__DOT__RX__DOT__state = 0;
    SData/*9:0*/ __Vdly__top_level__DOT__RX__DOT__clk_cnt;
    __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__RX__DOT__bit_idx;
    __Vdly__top_level__DOT__RX__DOT__bit_idx = 0;
    CData/*7:0*/ __Vdly__top_level__DOT__RX__DOT__data;
    __Vdly__top_level__DOT__RX__DOT__data = 0;
    CData/*1:0*/ __Vdly__top_level__DOT__u_conv__DOT__state;
    __Vdly__top_level__DOT__u_conv__DOT__state = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__oc;
    __Vdly__top_level__DOT__u_conv__DOT__oc = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__orow;
    __Vdly__top_level__DOT__u_conv__DOT__orow = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__ocol;
    __Vdly__top_level__DOT__u_conv__DOT__ocol = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__ic;
    __Vdly__top_level__DOT__u_conv__DOT__ic = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__kr;
    __Vdly__top_level__DOT__u_conv__DOT__kr = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__kc;
    __Vdly__top_level__DOT__u_conv__DOT__kc = 0;
    QData/*37:0*/ __Vdly__top_level__DOT__u_conv__DOT__acc;
    __Vdly__top_level__DOT__u_conv__DOT__acc = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__prod;
    __Vdly__top_level__DOT__u_conv__DOT__prod = 0;
    IData/*31:0*/ __VdlyMask__top_level__DOT__u_conv__DOT__prod;
    __VdlyMask__top_level__DOT__u_conv__DOT__prod = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__sat_pos_cnt;
    __Vdly__top_level__DOT__u_conv__DOT__sat_pos_cnt = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_conv__DOT__sat_neg_cnt;
    __Vdly__top_level__DOT__u_conv__DOT__sat_neg_cnt = 0;
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
    IData/*31:0*/ __Vdly__top_level__DOT__u_pool__DOT__ch;
    __Vdly__top_level__DOT__u_pool__DOT__ch = 0;
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
    IData/*31:0*/ __Vdly__top_level__DOT__u_dense__DOT__sat_pos_cnt;
    __Vdly__top_level__DOT__u_dense__DOT__sat_pos_cnt = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_dense__DOT__sat_neg_cnt;
    __Vdly__top_level__DOT__u_dense__DOT__sat_neg_cnt = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_dense__DOT__prod;
    __Vdly__top_level__DOT__u_dense__DOT__prod = 0;
    IData/*31:0*/ __VdlyMask__top_level__DOT__u_dense__DOT__prod;
    __VdlyMask__top_level__DOT__u_dense__DOT__prod = 0;
    CData/*1:0*/ __Vdly__top_level__DOT__u_argmax__DOT__state;
    __Vdly__top_level__DOT__u_argmax__DOT__state = 0;
    IData/*31:0*/ __Vdly__top_level__DOT__u_argmax__DOT__i;
    __Vdly__top_level__DOT__u_argmax__DOT__i = 0;
    CData/*3:0*/ __Vdly__top_level__DOT__u_argmax__DOT__besti;
    __Vdly__top_level__DOT__u_argmax__DOT__besti = 0;
    SData/*15:0*/ __Vdly__top_level__DOT__u_argmax__DOT__bestv;
    __Vdly__top_level__DOT__u_argmax__DOT__bestv = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__TX__DOT__state;
    __Vdly__top_level__DOT__TX__DOT__state = 0;
    CData/*0:0*/ __Vdly__top_level__DOT__tx_busy;
    __Vdly__top_level__DOT__tx_busy = 0;
    SData/*9:0*/ __Vdly__top_level__DOT__TX__DOT__clk_cnt;
    __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0;
    CData/*2:0*/ __Vdly__top_level__DOT__TX__DOT__bit_idx;
    __Vdly__top_level__DOT__TX__DOT__bit_idx = 0;
    CData/*7:0*/ __Vdly__top_level__DOT__TX__DOT__data_reg;
    __Vdly__top_level__DOT__TX__DOT__data_reg = 0;
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
    __Vdly__top_level__DOT__cycle_ctr = vlSelfRef.top_level__DOT__cycle_ctr;
    __Vdly__top_level__DOT__RX__DOT__state = vlSelfRef.top_level__DOT__RX__DOT__state;
    __Vdly__top_level__DOT__RX__DOT__clk_cnt = vlSelfRef.top_level__DOT__RX__DOT__clk_cnt;
    __Vdly__top_level__DOT__RX__DOT__bit_idx = vlSelfRef.top_level__DOT__RX__DOT__bit_idx;
    __Vdly__top_level__DOT__RX__DOT__data = vlSelfRef.top_level__DOT__RX__DOT__data;
    __Vdly__top_level__DOT__u_relu__DOT__state = vlSelfRef.top_level__DOT__u_relu__DOT__state;
    __Vdly__top_level__DOT__u_relu__DOT__c = vlSelfRef.top_level__DOT__u_relu__DOT__c;
    __Vdly__top_level__DOT__u_relu__DOT__r = vlSelfRef.top_level__DOT__u_relu__DOT__r;
    __Vdly__top_level__DOT__u_relu__DOT__q = vlSelfRef.top_level__DOT__u_relu__DOT__q;
    __VdlySet__top_level__DOT__relu_out__v0 = 0U;
    __Vdly__top_level__DOT__u_pool__DOT__state = vlSelfRef.top_level__DOT__u_pool__DOT__state;
    __Vdly__top_level__DOT__u_pool__DOT__ch = vlSelfRef.top_level__DOT__u_pool__DOT__ch;
    __Vdly__top_level__DOT__u_pool__DOT__r = vlSelfRef.top_level__DOT__u_pool__DOT__r;
    __Vdly__top_level__DOT__u_pool__DOT__q = vlSelfRef.top_level__DOT__u_pool__DOT__q;
    __VdlySet__top_level__DOT__pool_out__v0 = 0U;
    __Vdly__top_level__DOT__u_dense__DOT__state = vlSelfRef.top_level__DOT__u_dense__DOT__state;
    __Vdly__top_level__DOT__u_dense__DOT__o = vlSelfRef.top_level__DOT__u_dense__DOT__o;
    __Vdly__top_level__DOT__u_dense__DOT__i = vlSelfRef.top_level__DOT__u_dense__DOT__i;
    __Vdly__top_level__DOT__u_dense__DOT__sat_pos_cnt 
        = vlSelfRef.top_level__DOT__u_dense__DOT__sat_pos_cnt;
    __Vdly__top_level__DOT__u_dense__DOT__sat_neg_cnt 
        = vlSelfRef.top_level__DOT__u_dense__DOT__sat_neg_cnt;
    __VdlySet__top_level__DOT__logits__v0 = 0U;
    __Vdly__top_level__DOT__u_argmax__DOT__state = vlSelfRef.top_level__DOT__u_argmax__DOT__state;
    __Vdly__top_level__DOT__u_argmax__DOT__i = vlSelfRef.top_level__DOT__u_argmax__DOT__i;
    __Vdly__top_level__DOT__u_argmax__DOT__besti = vlSelfRef.top_level__DOT__u_argmax__DOT__besti;
    __Vdly__top_level__DOT__u_argmax__DOT__bestv = vlSelfRef.top_level__DOT__u_argmax__DOT__bestv;
    __Vdly__top_level__DOT__u_conv__DOT__state = vlSelfRef.top_level__DOT__u_conv__DOT__state;
    __Vdly__top_level__DOT__u_conv__DOT__oc = vlSelfRef.top_level__DOT__u_conv__DOT__oc;
    __Vdly__top_level__DOT__u_conv__DOT__orow = vlSelfRef.top_level__DOT__u_conv__DOT__orow;
    __Vdly__top_level__DOT__u_conv__DOT__ocol = vlSelfRef.top_level__DOT__u_conv__DOT__ocol;
    __Vdly__top_level__DOT__u_conv__DOT__ic = vlSelfRef.top_level__DOT__u_conv__DOT__ic;
    __Vdly__top_level__DOT__u_conv__DOT__kr = vlSelfRef.top_level__DOT__u_conv__DOT__kr;
    __Vdly__top_level__DOT__u_conv__DOT__kc = vlSelfRef.top_level__DOT__u_conv__DOT__kc;
    __Vdly__top_level__DOT__u_conv__DOT__acc = vlSelfRef.top_level__DOT__u_conv__DOT__acc;
    __Vdly__top_level__DOT__u_conv__DOT__sat_pos_cnt 
        = vlSelfRef.top_level__DOT__u_conv__DOT__sat_pos_cnt;
    __Vdly__top_level__DOT__u_conv__DOT__sat_neg_cnt 
        = vlSelfRef.top_level__DOT__u_conv__DOT__sat_neg_cnt;
    __VdlySet__top_level__DOT__conv_out__v0 = 0U;
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
    __Vdly__top_level__DOT__tx_busy = vlSelfRef.top_level__DOT__tx_busy;
    __Vdly__top_level__DOT__TX__DOT__clk_cnt = vlSelfRef.top_level__DOT__TX__DOT__clk_cnt;
    __Vdly__top_level__DOT__TX__DOT__bit_idx = vlSelfRef.top_level__DOT__TX__DOT__bit_idx;
    __Vdly__top_level__DOT__TX__DOT__data_reg = vlSelfRef.top_level__DOT__TX__DOT__data_reg;
    if (VL_UNLIKELY((((~ (IData)(vlSelfRef.reset)) 
                      & (IData)(vlSelfRef.top_level__DOT__tx_start_q))))) {
        VL_WRITEF_NX("---- Performance Report ----\nFrame cycles: %0#\n conv  = %0#\n relu  = %0#\n pool  = %0#\n flat  = %0#\n dense = %0#\n argmx = %0#\n----------------------------\n",0,
                     64,(vlSelfRef.top_level__DOT__t_tx 
                         - vlSelfRef.top_level__DOT__t_start),
                     64,(vlSelfRef.top_level__DOT__t_conv 
                         - vlSelfRef.top_level__DOT__t_start),
                     64,(vlSelfRef.top_level__DOT__t_relu 
                         - vlSelfRef.top_level__DOT__t_conv),
                     64,(vlSelfRef.top_level__DOT__t_pool 
                         - vlSelfRef.top_level__DOT__t_relu),
                     64,(vlSelfRef.top_level__DOT__t_flat 
                         - vlSelfRef.top_level__DOT__t_pool),
                     64,(vlSelfRef.top_level__DOT__t_dense 
                         - vlSelfRef.top_level__DOT__t_flat),
                     64,(vlSelfRef.top_level__DOT__t_tx 
                         - vlSelfRef.top_level__DOT__t_dense));
    }
    vlSelfRef.top_level__DOT__tx_start_q = ((~ (IData)(vlSelfRef.reset)) 
                                            & (IData)(vlSelfRef.top_level__DOT__tx_start));
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__u_dense__DOT__prod = 0U;
        __VdlyMask__top_level__DOT__u_dense__DOT__prod = 0xffffffffU;
        __Vdly__top_level__DOT__cycle_ctr = 0ULL;
        vlSelfRef.top_level__DOT__t_argmax = 0ULL;
        __Vdly__top_level__DOT__TX__DOT__state = 0U;
        vlSelfRef.uart_tx_o = 1U;
        __Vdly__top_level__DOT__tx_busy = 0U;
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
        vlSelfRef.top_level__DOT__t_relu = 0ULL;
        vlSelfRef.top_level__DOT__t_pool = 0ULL;
        vlSelfRef.top_level__DOT__t_dense = 0ULL;
        vlSelfRef.top_level__DOT__t_conv = 0ULL;
        vlSelfRef.top_level__DOT__t_start = 0ULL;
        vlSelfRef.top_level__DOT__t_flat = 0ULL;
        vlSelfRef.top_level__DOT__t_tx = 0ULL;
        vlSelfRef.top_level__DOT__cycle_ctr = __Vdly__top_level__DOT__cycle_ctr;
        __Vdly__top_level__DOT__ctrl__DOT__state = 0U;
        __Vdly__top_level__DOT__conv_start = 0U;
        __Vdly__top_level__DOT__relu_start = 0U;
        __Vdly__top_level__DOT__pool_start = 0U;
        __Vdly__top_level__DOT__dense_start = 0U;
        vlSelfRef.top_level__DOT__tx_start = 0U;
        vlSelfRef.top_level__DOT__ctrl__DOT__busy = 0U;
        vlSelfRef.top_level__DOT__tx_busy = __Vdly__top_level__DOT__tx_busy;
        vlSelfRef.top_level__DOT__ctrl__DOT__state 
            = __Vdly__top_level__DOT__ctrl__DOT__state;
        __Vdly__top_level__DOT__u_relu__DOT__state = 0U;
        vlSelfRef.top_level__DOT__relu_done = 0U;
        __Vdly__top_level__DOT__u_relu__DOT__c = 0U;
        __Vdly__top_level__DOT__u_relu__DOT__r = 0U;
        __Vdly__top_level__DOT__u_relu__DOT__q = 0U;
        __Vdly__top_level__DOT__u_argmax__DOT__state = 0U;
        vlSelfRef.top_level__DOT__argmax_done = 0U;
        vlSelfRef.predicted_digit = 0U;
        __Vdly__top_level__DOT__u_argmax__DOT__i = 0U;
        __Vdly__top_level__DOT__u_argmax__DOT__besti = 0U;
        __Vdly__top_level__DOT__u_argmax__DOT__bestv = 0x8000U;
    } else {
        __Vdly__top_level__DOT__cycle_ctr = (1ULL + vlSelfRef.top_level__DOT__cycle_ctr);
        if (vlSelfRef.top_level__DOT__argmax_done) {
            vlSelfRef.top_level__DOT__t_argmax = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        if ((0U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = 1U;
            __Vdly__top_level__DOT__tx_busy = 0U;
            if (vlSelfRef.top_level__DOT__tx_start) {
                __Vdly__top_level__DOT__tx_busy = 1U;
                __Vdly__top_level__DOT__TX__DOT__data_reg 
                    = (0xffU & ((IData)(0x30U) + (IData)(vlSelfRef.predicted_digit)));
                __Vdly__top_level__DOT__TX__DOT__state = 1U;
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = 0U;
            if ((0x363U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
                __Vdly__top_level__DOT__TX__DOT__bit_idx = 0U;
                __Vdly__top_level__DOT__TX__DOT__state = 2U;
            } else {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt 
                    = (0x3ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt)));
            }
        } else if ((2U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = (1U & ((IData)(vlSelfRef.top_level__DOT__TX__DOT__data_reg) 
                                         >> (IData)(vlSelfRef.top_level__DOT__TX__DOT__bit_idx)));
            if ((0x363U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
                if ((7U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__bit_idx))) {
                    __Vdly__top_level__DOT__TX__DOT__state = 3U;
                } else {
                    __Vdly__top_level__DOT__TX__DOT__bit_idx 
                        = (7U & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__bit_idx)));
                }
            } else {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt 
                    = (0x3ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt)));
            }
        } else if ((3U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__state))) {
            vlSelfRef.uart_tx_o = 1U;
            if ((0x363U == (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__TX__DOT__state = 4U;
                __Vdly__top_level__DOT__TX__DOT__clk_cnt = 0U;
            } else {
                __Vdly__top_level__DOT__TX__DOT__clk_cnt 
                    = (0x3ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt)));
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
        if (vlSelfRef.top_level__DOT__relu_done) {
            vlSelfRef.top_level__DOT__t_relu = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        if (vlSelfRef.top_level__DOT__pool_done) {
            vlSelfRef.top_level__DOT__t_pool = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        if (vlSelfRef.top_level__DOT__dense_done) {
            vlSelfRef.top_level__DOT__t_dense = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        if (vlSelfRef.top_level__DOT__conv_done) {
            vlSelfRef.top_level__DOT__t_conv = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        if (vlSelfRef.top_level__DOT__frame_loaded) {
            vlSelfRef.top_level__DOT__t_start = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        if (vlSelfRef.top_level__DOT__flat_done) {
            vlSelfRef.top_level__DOT__t_flat = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        if (vlSelfRef.top_level__DOT__tx_start) {
            vlSelfRef.top_level__DOT__t_tx = vlSelfRef.top_level__DOT__cycle_ctr;
        }
        vlSelfRef.top_level__DOT__cycle_ctr = __Vdly__top_level__DOT__cycle_ctr;
        __Vdly__top_level__DOT__conv_start = 0U;
        __Vdly__top_level__DOT__relu_start = 0U;
        __Vdly__top_level__DOT__pool_start = 0U;
        __Vdly__top_level__DOT__dense_start = 0U;
        vlSelfRef.top_level__DOT__tx_start = 0U;
        if ((4U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
            if ((2U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.top_level__DOT__ctrl__DOT__state))) {
                    __Vdly__top_level__DOT__ctrl__DOT__state = 0U;
                } else if (((IData)(vlSelfRef.top_level__DOT__argmax_done) 
                            & (~ (IData)(vlSelfRef.top_level__DOT__tx_busy)))) {
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
        vlSelfRef.top_level__DOT__tx_busy = __Vdly__top_level__DOT__tx_busy;
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
    }
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__u_conv__DOT__state = 0U;
        vlSelfRef.top_level__DOT__conv_done = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__oc = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__orow = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__ocol = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__ic = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__kr = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__kc = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__acc = 0ULL;
        __Vdly__top_level__DOT__u_conv__DOT__prod = 0U;
        __VdlyMask__top_level__DOT__u_conv__DOT__prod = 0xffffffffU;
        __Vdly__top_level__DOT__u_conv__DOT__sat_pos_cnt = 0U;
        __Vdly__top_level__DOT__u_conv__DOT__sat_neg_cnt = 0U;
    } else {
        vlSelfRef.top_level__DOT__conv_done = 0U;
        if ((2U & (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__state))) {
                vlSelfRef.top_level__DOT__conv_done = 1U;
                __Vdly__top_level__DOT__u_conv__DOT__state = 0U;
            } else {
                vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__shifted 
                    = (0x3fffffffffULL & VL_SHIFTRS_QQI(38,38,32, vlSelfRef.top_level__DOT__u_conv__DOT__acc, 7U));
                vlSelfRef.top_level__DOT__u_conv__DOT____Vlvbound_h3b513ba9__0 
                    = vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__res;
                if (((0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__ocol)) 
                     && (0x1bU >= (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__orow)))) {
                    __VdlyVal__top_level__DOT__conv_out__v0 
                        = vlSelfRef.top_level__DOT__u_conv__DOT____Vlvbound_h3b513ba9__0;
                    __VdlyDim0__top_level__DOT__conv_out__v0 
                        = (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__ocol);
                    __VdlyDim1__top_level__DOT__conv_out__v0 
                        = (0x1fU & vlSelfRef.top_level__DOT__u_conv__DOT__orow);
                    __VdlyDim2__top_level__DOT__conv_out__v0 
                        = (7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc);
                    __VdlySet__top_level__DOT__conv_out__v0 = 1U;
                }
                if (VL_LTS_IQQ(38, 0x7fffULL, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__shifted)) {
                    __Vdly__top_level__DOT__u_conv__DOT__sat_pos_cnt 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__sat_pos_cnt);
                    vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__res = 0x7fffU;
                } else if (VL_GTS_IQQ(38, 0x3fffff8000ULL, vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__shifted)) {
                    __Vdly__top_level__DOT__u_conv__DOT__sat_neg_cnt 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__sat_neg_cnt);
                    vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__res = 0x8000U;
                } else {
                    vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__res 
                        = (0xffffU & (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk2__DOT__shifted));
                }
                if ((0x1bU == vlSelfRef.top_level__DOT__u_conv__DOT__ocol)) {
                    __Vdly__top_level__DOT__u_conv__DOT__ocol = 0U;
                    if ((0x1bU == vlSelfRef.top_level__DOT__u_conv__DOT__orow)) {
                        __Vdly__top_level__DOT__u_conv__DOT__orow = 0U;
                        if ((7U == vlSelfRef.top_level__DOT__u_conv__DOT__oc)) {
                            __Vdly__top_level__DOT__u_conv__DOT__state = 3U;
                        } else {
                            __Vdly__top_level__DOT__u_conv__DOT__oc 
                                = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__oc);
                            __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__b 
                                = vlSelfRef.top_level__DOT__conv_b
                                [(7U & ((IData)(1U) 
                                        + vlSelfRef.top_level__DOT__u_conv__DOT__oc))];
                            __Vdly__top_level__DOT__u_conv__DOT__ic = 0U;
                            __Vdly__top_level__DOT__u_conv__DOT__kr = 0U;
                            __Vdly__top_level__DOT__u_conv__DOT__kc = 0U;
                            __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__Vfuncout 
                                = (0x3fffffffffULL 
                                   & VL_SHIFTL_QQI(38,38,32, 
                                                   (((QData)((IData)(
                                                                     (0x3fffffU 
                                                                      & (- (IData)(
                                                                                (1U 
                                                                                & ((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__b) 
                                                                                >> 0xfU))))))) 
                                                     << 0x10U) 
                                                    | (QData)((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__b))), 7U));
                            __Vdly__top_level__DOT__u_conv__DOT__state = 1U;
                            __Vdly__top_level__DOT__u_conv__DOT__acc 
                                = __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__0__Vfuncout;
                        }
                    } else {
                        __Vdly__top_level__DOT__u_conv__DOT__orow 
                            = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__orow);
                        __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__b 
                            = vlSelfRef.top_level__DOT__conv_b
                            [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)];
                        __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__Vfuncout 
                            = (0x3fffffffffULL & VL_SHIFTL_QQI(38,38,32, 
                                                               (((QData)((IData)(
                                                                                (0x3fffffU 
                                                                                & (- (IData)(
                                                                                (1U 
                                                                                & ((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__b) 
                                                                                >> 0xfU))))))) 
                                                                 << 0x10U) 
                                                                | (QData)((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__b))), 7U));
                        __Vdly__top_level__DOT__u_conv__DOT__ic = 0U;
                        __Vdly__top_level__DOT__u_conv__DOT__kr = 0U;
                        __Vdly__top_level__DOT__u_conv__DOT__kc = 0U;
                        __Vdly__top_level__DOT__u_conv__DOT__acc 
                            = __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__1__Vfuncout;
                        __Vdly__top_level__DOT__u_conv__DOT__state = 1U;
                    }
                } else {
                    __Vdly__top_level__DOT__u_conv__DOT__ocol 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__ocol);
                    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__b 
                        = vlSelfRef.top_level__DOT__conv_b
                        [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)];
                    __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__Vfuncout 
                        = (0x3fffffffffULL & VL_SHIFTL_QQI(38,38,32, 
                                                           (((QData)((IData)(
                                                                             (0x3fffffU 
                                                                              & (- (IData)(
                                                                                (1U 
                                                                                & ((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__b) 
                                                                                >> 0xfU))))))) 
                                                             << 0x10U) 
                                                            | (QData)((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__b))), 7U));
                    __Vdly__top_level__DOT__u_conv__DOT__ic = 0U;
                    __Vdly__top_level__DOT__u_conv__DOT__kr = 0U;
                    __Vdly__top_level__DOT__u_conv__DOT__kc = 0U;
                    __Vdly__top_level__DOT__u_conv__DOT__acc 
                        = __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__2__Vfuncout;
                    __Vdly__top_level__DOT__u_conv__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelfRef.top_level__DOT__u_conv__DOT__state))) {
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__ir 
                = ((vlSelfRef.top_level__DOT__u_conv__DOT__orow 
                    + vlSelfRef.top_level__DOT__u_conv__DOT__kr) 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__icc 
                = ((vlSelfRef.top_level__DOT__u_conv__DOT__ocol 
                    + vlSelfRef.top_level__DOT__u_conv__DOT__kc) 
                   - (IData)(1U));
            vlSelfRef.top_level__DOT__u_conv__DOT__prod 
                = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                ([&]() {
                            __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__c 
                                = vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__icc;
                            __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__r 
                                = vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__ir;
                            __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__ch 
                                = vlSelfRef.top_level__DOT__u_conv__DOT__ic;
                            __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__Vfuncout 
                                = ((((VL_GTS_III(32, 0U, __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__r) 
                                      | VL_LTES_III(32, 0x1cU, __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__r)) 
                                     | VL_GTS_III(32, 0U, __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__c)) 
                                    | VL_LTES_III(32, 0x1cU, __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__c))
                                    ? 0U : ((0x1bU 
                                             >= (0x1fU 
                                                 & __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__c))
                                             ? vlSelfRef.top_level__DOT__ifmap
                                            [((0U >= 
                                               (1U 
                                                & __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__ch)) 
                                              && (1U 
                                                  & __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__ch))]
                                            [((0x1bU 
                                               >= (0x1fU 
                                                   & __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__r))
                                               ? (0x1fU 
                                                  & __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__r)
                                               : 0U)]
                                            [(0x1fU 
                                              & __Vfunc_top_level__DOT__u_conv__DOT__in_at__3__c)]
                                             : 0U));
                        }(), (IData)(__Vfunc_top_level__DOT__u_conv__DOT__in_at__3__Vfuncout))), 
                              VL_EXTENDS_II(32,16, 
                                            ((2U >= 
                                              (3U & vlSelfRef.top_level__DOT__u_conv__DOT__kc))
                                              ? vlSelfRef.top_level__DOT__conv_w
                                             [(7U & vlSelfRef.top_level__DOT__u_conv__DOT__oc)]
                                             [((0U 
                                                >= 
                                                (1U 
                                                 & vlSelfRef.top_level__DOT__u_conv__DOT__ic)) 
                                               && (1U 
                                                   & vlSelfRef.top_level__DOT__u_conv__DOT__ic))]
                                             [((2U 
                                                >= 
                                                (3U 
                                                 & vlSelfRef.top_level__DOT__u_conv__DOT__kr))
                                                ? (3U 
                                                   & vlSelfRef.top_level__DOT__u_conv__DOT__kr)
                                                : 0U)]
                                             [(3U & vlSelfRef.top_level__DOT__u_conv__DOT__kc)]
                                              : 0U)));
            __Vdly__top_level__DOT__u_conv__DOT__acc 
                = (0x3fffffffffULL & (vlSelfRef.top_level__DOT__u_conv__DOT__acc 
                                      + VL_EXTENDS_QI(38,32, vlSelfRef.top_level__DOT__u_conv__DOT__prod)));
            if ((2U == vlSelfRef.top_level__DOT__u_conv__DOT__kc)) {
                __Vdly__top_level__DOT__u_conv__DOT__kc = 0U;
                if ((2U == vlSelfRef.top_level__DOT__u_conv__DOT__kr)) {
                    __Vdly__top_level__DOT__u_conv__DOT__kr = 0U;
                    if ((0U == vlSelfRef.top_level__DOT__u_conv__DOT__ic)) {
                        __Vdly__top_level__DOT__u_conv__DOT__state = 2U;
                    } else {
                        __Vdly__top_level__DOT__u_conv__DOT__ic 
                            = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__ic);
                    }
                } else {
                    __Vdly__top_level__DOT__u_conv__DOT__kr 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__kr);
                }
            } else {
                __Vdly__top_level__DOT__u_conv__DOT__kc 
                    = ((IData)(1U) + vlSelfRef.top_level__DOT__u_conv__DOT__kc);
            }
        } else if (vlSelfRef.top_level__DOT__conv_start) {
            __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__b 
                = vlSelfRef.top_level__DOT__conv_b[0U];
            __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__Vfuncout 
                = (0x3fffffffffULL & VL_SHIFTL_QQI(38,38,32, 
                                                   (((QData)((IData)(
                                                                     (0x3fffffU 
                                                                      & (- (IData)(
                                                                                (1U 
                                                                                & ((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__b) 
                                                                                >> 0xfU))))))) 
                                                     << 0x10U) 
                                                    | (QData)((IData)(__Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__b))), 7U));
            __Vdly__top_level__DOT__u_conv__DOT__oc = 0U;
            __Vdly__top_level__DOT__u_conv__DOT__orow = 0U;
            __Vdly__top_level__DOT__u_conv__DOT__ocol = 0U;
            __Vdly__top_level__DOT__u_conv__DOT__ic = 0U;
            __Vdly__top_level__DOT__u_conv__DOT__kr = 0U;
            __Vdly__top_level__DOT__u_conv__DOT__kc = 0U;
            __Vdly__top_level__DOT__u_conv__DOT__acc 
                = __Vfunc_top_level__DOT__u_conv__DOT__bias_to_accq2f__4__Vfuncout;
            __Vdly__top_level__DOT__u_conv__DOT__state = 1U;
        }
    }
    if (vlSelfRef.reset) {
        __Vdly__top_level__DOT__r = 0U;
        __Vdly__top_level__DOT__c = 0U;
        vlSelfRef.top_level__DOT__frame_loaded = 0U;
        __Vdly__top_level__DOT__flattening = 0U;
        vlSelfRef.top_level__DOT__flat_done = 0U;
        __Vdly__top_level__DOT__fi_c = 0U;
        __Vdly__top_level__DOT__fi_r = 0U;
        __Vdly__top_level__DOT__fi_q = 0U;
        __Vdly__top_level__DOT__fi_idx = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__state = 0U;
        vlSelfRef.top_level__DOT__dense_done = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__o = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__i = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__acc = 0ULL;
        __VdlyMask__top_level__DOT__u_dense__DOT__acc = 0x7ffffffffffULL;
        __Vdly__top_level__DOT__u_dense__DOT__sat_pos_cnt = 0U;
        __Vdly__top_level__DOT__u_dense__DOT__sat_neg_cnt = 0U;
        __Vdly__top_level__DOT__RX__DOT__state = 0U;
        vlSelfRef.top_level__DOT__rx_dv = 0U;
        __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
        __Vdly__top_level__DOT__RX__DOT__bit_idx = 0U;
        __Vdly__top_level__DOT__RX__DOT__data = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__state = 0U;
        vlSelfRef.top_level__DOT__pool_done = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__ch = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__r = 0U;
        __Vdly__top_level__DOT__u_pool__DOT__q = 0U;
    } else {
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
        vlSelfRef.top_level__DOT__dense_done = 0U;
        if ((2U & (IData)(vlSelfRef.top_level__DOT__u_dense__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.top_level__DOT__u_dense__DOT__state))) {
                vlSelfRef.top_level__DOT__dense_done = 1U;
                __Vdly__top_level__DOT__u_dense__DOT__state = 0U;
            } else {
                vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__shifted 
                    = (0x7ffffffffffULL & VL_SHIFTRS_QQI(43,43,32, vlSelfRef.top_level__DOT__u_dense__DOT__acc, 7U));
                vlSelfRef.top_level__DOT__u_dense__DOT____Vlvbound_h29ef7931__0 
                    = vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__res;
                if ((9U >= (0xfU & vlSelfRef.top_level__DOT__u_dense__DOT__o))) {
                    __VdlyVal__top_level__DOT__logits__v0 
                        = vlSelfRef.top_level__DOT__u_dense__DOT____Vlvbound_h29ef7931__0;
                    __VdlyDim0__top_level__DOT__logits__v0 
                        = (0xfU & vlSelfRef.top_level__DOT__u_dense__DOT__o);
                    __VdlySet__top_level__DOT__logits__v0 = 1U;
                }
                if (VL_LTS_IQQ(43, 0x7fffULL, vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__shifted)) {
                    __Vdly__top_level__DOT__u_dense__DOT__sat_pos_cnt 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_dense__DOT__sat_pos_cnt);
                    vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__res = 0x7fffU;
                } else if (VL_GTS_IQQ(43, 0x7ffffff8000ULL, vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__shifted)) {
                    __Vdly__top_level__DOT__u_dense__DOT__sat_neg_cnt 
                        = ((IData)(1U) + vlSelfRef.top_level__DOT__u_dense__DOT__sat_neg_cnt);
                    vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__res = 0x8000U;
                } else {
                    vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__res 
                        = (0xffffU & (IData)(vlSelfRef.top_level__DOT__u_dense__DOT__unnamedblk1__DOT__shifted));
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
                                                  >= 
                                                  (0x7ffU 
                                                   & vlSelfRef.top_level__DOT__u_dense__DOT__i))
                                                  ? 
                                                 vlSelfRef.top_level__DOT__flat_vec
                                                 [(0x7ffU 
                                                   & vlSelfRef.top_level__DOT__u_dense__DOT__i)]
                                                  : 0U)), 
                              VL_EXTENDS_II(32,16, 
                                            ((0x61fU 
                                              >= (0x7ffU 
                                                  & vlSelfRef.top_level__DOT__u_dense__DOT__i))
                                              ? vlSelfRef.top_level__DOT__dense_w
                                             [((9U 
                                                >= 
                                                (0xfU 
                                                 & vlSelfRef.top_level__DOT__u_dense__DOT__o))
                                                ? (0xfU 
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
        } else if (vlSelfRef.top_level__DOT__dense_start) {
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
        vlSelfRef.top_level__DOT__rx_dv = 0U;
        if ((0U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((1U & (~ (IData)(vlSelfRef.uart_rx_i)))) {
                __Vdly__top_level__DOT__RX__DOT__state = 1U;
                __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((0x1b2U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt))) {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
                __Vdly__top_level__DOT__RX__DOT__state = 2U;
                __Vdly__top_level__DOT__RX__DOT__bit_idx = 0U;
            } else {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt 
                    = (0x3ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt)));
            }
        } else if ((2U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((0x363U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt))) {
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
                    = (0x3ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt)));
            }
        } else if ((3U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            if ((0x363U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt))) {
                vlSelfRef.top_level__DOT__rx_byte = vlSelfRef.top_level__DOT__RX__DOT__data;
                vlSelfRef.top_level__DOT__rx_dv = 1U;
                __Vdly__top_level__DOT__RX__DOT__state = 4U;
                __Vdly__top_level__DOT__RX__DOT__clk_cnt = 0U;
            } else {
                __Vdly__top_level__DOT__RX__DOT__clk_cnt 
                    = (0x3ffU & ((IData)(1U) + (IData)(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt)));
            }
        } else if ((4U == (IData)(vlSelfRef.top_level__DOT__RX__DOT__state))) {
            __Vdly__top_level__DOT__RX__DOT__state = 0U;
        }
        if ((0U == (IData)(vlSelfRef.top_level__DOT__u_pool__DOT__state))) {
            vlSelfRef.top_level__DOT__pool_done = 0U;
            if (vlSelfRef.top_level__DOT__pool_start) {
                __Vdly__top_level__DOT__u_pool__DOT__ch = 0U;
                __Vdly__top_level__DOT__u_pool__DOT__r = 0U;
                __Vdly__top_level__DOT__u_pool__DOT__q = 0U;
                __Vdly__top_level__DOT__u_pool__DOT__state = 1U;
            }
        } else if ((1U == (IData)(vlSelfRef.top_level__DOT__u_pool__DOT__state))) {
            if ((0x1bU >= (0x1fU & ((IData)(1U) + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q))))) {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__y 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__ch)]
                    [((0x1bU >= (0x1fU & ((IData)(1U) 
                                          + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))))
                       ? (0x1fU & ((IData)(1U) + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       : 0U)][(0x1fU & ((IData)(1U) 
                                        + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q)))];
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__b 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__ch)]
                    [((0x1bU >= (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       ? (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))
                       : 0U)][(0x1fU & ((IData)(1U) 
                                        + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q)))];
            } else {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__y = 0U;
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__b = 0U;
            }
            if ((0x1bU >= (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q)))) {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__x 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__ch)]
                    [((0x1bU >= (0x1fU & ((IData)(1U) 
                                          + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))))
                       ? (0x1fU & ((IData)(1U) + VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       : 0U)][(0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q))];
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__a 
                    = vlSelfRef.top_level__DOT__relu_out
                    [(7U & vlSelfRef.top_level__DOT__u_pool__DOT__ch)]
                    [((0x1bU >= (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r)))
                       ? (0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__r))
                       : 0U)][(0x1fU & VL_MULS_III(32, (IData)(2U), vlSelfRef.top_level__DOT__u_pool__DOT__q))];
            } else {
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__x = 0U;
                __Vfunc_top_level__DOT__u_pool__DOT__max4__5__a = 0U;
            }
            __Vfunc_top_level__DOT__u_pool__DOT__max4__5__m1 
                = (VL_GTS_III(16, (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__a), (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__b))
                    ? (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__a)
                    : (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__b));
            __Vfunc_top_level__DOT__u_pool__DOT__max4__5__m2 
                = (VL_GTS_III(16, (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__x), (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__y))
                    ? (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__x)
                    : (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__y));
            __Vfunc_top_level__DOT__u_pool__DOT__max4__5__Vfuncout 
                = (VL_GTS_III(16, (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__m1), (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__m2))
                    ? (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__m1)
                    : (IData)(__Vfunc_top_level__DOT__u_pool__DOT__max4__5__m2));
            vlSelfRef.top_level__DOT__u_pool__DOT____Vlvbound_h816b22dc__0 
                = __Vfunc_top_level__DOT__u_pool__DOT__max4__5__Vfuncout;
            if (((0xdU >= (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__q)) 
                 && (0xdU >= (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__r)))) {
                __VdlyVal__top_level__DOT__pool_out__v0 
                    = vlSelfRef.top_level__DOT__u_pool__DOT____Vlvbound_h816b22dc__0;
                __VdlyDim0__top_level__DOT__pool_out__v0 
                    = (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__q);
                __VdlyDim1__top_level__DOT__pool_out__v0 
                    = (0xfU & vlSelfRef.top_level__DOT__u_pool__DOT__r);
                __VdlyDim2__top_level__DOT__pool_out__v0 
                    = (7U & vlSelfRef.top_level__DOT__u_pool__DOT__ch);
                __VdlySet__top_level__DOT__pool_out__v0 = 1U;
            }
            if ((0xdU == vlSelfRef.top_level__DOT__u_pool__DOT__q)) {
                __Vdly__top_level__DOT__u_pool__DOT__q = 0U;
                if ((0xdU == vlSelfRef.top_level__DOT__u_pool__DOT__r)) {
                    __Vdly__top_level__DOT__u_pool__DOT__r = 0U;
                    if ((7U == vlSelfRef.top_level__DOT__u_pool__DOT__ch)) {
                        __Vdly__top_level__DOT__u_pool__DOT__state = 2U;
                    } else {
                        __Vdly__top_level__DOT__u_pool__DOT__ch 
                            = ((IData)(1U) + vlSelfRef.top_level__DOT__u_pool__DOT__ch);
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
    vlSelfRef.top_level__DOT__relu_start = __Vdly__top_level__DOT__relu_start;
    vlSelfRef.top_level__DOT__u_relu__DOT__state = __Vdly__top_level__DOT__u_relu__DOT__state;
    vlSelfRef.top_level__DOT__u_relu__DOT__c = __Vdly__top_level__DOT__u_relu__DOT__c;
    vlSelfRef.top_level__DOT__u_relu__DOT__r = __Vdly__top_level__DOT__u_relu__DOT__r;
    vlSelfRef.top_level__DOT__u_relu__DOT__q = __Vdly__top_level__DOT__u_relu__DOT__q;
    vlSelfRef.top_level__DOT__u_argmax__DOT__state 
        = __Vdly__top_level__DOT__u_argmax__DOT__state;
    vlSelfRef.top_level__DOT__u_argmax__DOT__i = __Vdly__top_level__DOT__u_argmax__DOT__i;
    vlSelfRef.top_level__DOT__u_argmax__DOT__besti 
        = __Vdly__top_level__DOT__u_argmax__DOT__besti;
    vlSelfRef.top_level__DOT__u_argmax__DOT__bestv 
        = __Vdly__top_level__DOT__u_argmax__DOT__bestv;
    vlSelfRef.top_level__DOT__conv_start = __Vdly__top_level__DOT__conv_start;
    vlSelfRef.top_level__DOT__u_conv__DOT__state = __Vdly__top_level__DOT__u_conv__DOT__state;
    vlSelfRef.top_level__DOT__u_conv__DOT__oc = __Vdly__top_level__DOT__u_conv__DOT__oc;
    vlSelfRef.top_level__DOT__u_conv__DOT__orow = __Vdly__top_level__DOT__u_conv__DOT__orow;
    vlSelfRef.top_level__DOT__u_conv__DOT__ocol = __Vdly__top_level__DOT__u_conv__DOT__ocol;
    vlSelfRef.top_level__DOT__u_conv__DOT__ic = __Vdly__top_level__DOT__u_conv__DOT__ic;
    vlSelfRef.top_level__DOT__u_conv__DOT__kr = __Vdly__top_level__DOT__u_conv__DOT__kr;
    vlSelfRef.top_level__DOT__u_conv__DOT__kc = __Vdly__top_level__DOT__u_conv__DOT__kc;
    vlSelfRef.top_level__DOT__u_conv__DOT__acc = __Vdly__top_level__DOT__u_conv__DOT__acc;
    vlSelfRef.top_level__DOT__u_conv__DOT__sat_pos_cnt 
        = __Vdly__top_level__DOT__u_conv__DOT__sat_pos_cnt;
    vlSelfRef.top_level__DOT__u_conv__DOT__sat_neg_cnt 
        = __Vdly__top_level__DOT__u_conv__DOT__sat_neg_cnt;
    vlSelfRef.top_level__DOT__u_conv__DOT__prod = (
                                                   (__Vdly__top_level__DOT__u_conv__DOT__prod 
                                                    & __VdlyMask__top_level__DOT__u_conv__DOT__prod) 
                                                   | (vlSelfRef.top_level__DOT__u_conv__DOT__prod 
                                                      & (~ __VdlyMask__top_level__DOT__u_conv__DOT__prod)));
    __VdlyMask__top_level__DOT__u_conv__DOT__prod = 0U;
    if (__VdlySet__top_level__DOT__conv_out__v0) {
        vlSelfRef.top_level__DOT__conv_out[__VdlyDim2__top_level__DOT__conv_out__v0][__VdlyDim1__top_level__DOT__conv_out__v0][__VdlyDim0__top_level__DOT__conv_out__v0] 
            = __VdlyVal__top_level__DOT__conv_out__v0;
    }
    vlSelfRef.top_level__DOT__r = __Vdly__top_level__DOT__r;
    vlSelfRef.top_level__DOT__c = __Vdly__top_level__DOT__c;
    if (__VdlySet__top_level__DOT__ifmap__v0) {
        vlSelfRef.top_level__DOT__ifmap[0U][__VdlyDim1__top_level__DOT__ifmap__v0][__VdlyDim0__top_level__DOT__ifmap__v0] 
            = __VdlyVal__top_level__DOT__ifmap__v0;
    }
    vlSelfRef.top_level__DOT__flattening = __Vdly__top_level__DOT__flattening;
    vlSelfRef.top_level__DOT__fi_c = __Vdly__top_level__DOT__fi_c;
    vlSelfRef.top_level__DOT__fi_r = __Vdly__top_level__DOT__fi_r;
    vlSelfRef.top_level__DOT__fi_q = __Vdly__top_level__DOT__fi_q;
    vlSelfRef.top_level__DOT__fi_idx = __Vdly__top_level__DOT__fi_idx;
    vlSelfRef.top_level__DOT__dense_start = __Vdly__top_level__DOT__dense_start;
    vlSelfRef.top_level__DOT__u_dense__DOT__state = __Vdly__top_level__DOT__u_dense__DOT__state;
    vlSelfRef.top_level__DOT__u_dense__DOT__o = __Vdly__top_level__DOT__u_dense__DOT__o;
    vlSelfRef.top_level__DOT__u_dense__DOT__i = __Vdly__top_level__DOT__u_dense__DOT__i;
    vlSelfRef.top_level__DOT__u_dense__DOT__sat_pos_cnt 
        = __Vdly__top_level__DOT__u_dense__DOT__sat_pos_cnt;
    vlSelfRef.top_level__DOT__u_dense__DOT__sat_neg_cnt 
        = __Vdly__top_level__DOT__u_dense__DOT__sat_neg_cnt;
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
    vlSelfRef.top_level__DOT__u_dense__DOT__prod = 
        ((__Vdly__top_level__DOT__u_dense__DOT__prod 
          & __VdlyMask__top_level__DOT__u_dense__DOT__prod) 
         | (vlSelfRef.top_level__DOT__u_dense__DOT__prod 
            & (~ __VdlyMask__top_level__DOT__u_dense__DOT__prod)));
    __VdlyMask__top_level__DOT__u_dense__DOT__prod = 0U;
    vlSelfRef.top_level__DOT__RX__DOT__state = __Vdly__top_level__DOT__RX__DOT__state;
    vlSelfRef.top_level__DOT__RX__DOT__clk_cnt = __Vdly__top_level__DOT__RX__DOT__clk_cnt;
    vlSelfRef.top_level__DOT__RX__DOT__bit_idx = __Vdly__top_level__DOT__RX__DOT__bit_idx;
    vlSelfRef.top_level__DOT__RX__DOT__data = __Vdly__top_level__DOT__RX__DOT__data;
    vlSelfRef.top_level__DOT__pool_start = __Vdly__top_level__DOT__pool_start;
    vlSelfRef.top_level__DOT__u_pool__DOT__state = __Vdly__top_level__DOT__u_pool__DOT__state;
    vlSelfRef.top_level__DOT__u_pool__DOT__ch = __Vdly__top_level__DOT__u_pool__DOT__ch;
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
            VL_FATAL_MT("hdl/top_level.sv", 20, "", "NBA region did not converge.");
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
                VL_FATAL_MT("hdl/top_level.sv", 20, "", "Active region did not converge.");
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
