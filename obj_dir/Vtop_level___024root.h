// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop_level.h for the primary calling header

#ifndef VERILATED_VTOP_LEVEL___024ROOT_H_
#define VERILATED_VTOP_LEVEL___024ROOT_H_  // guard

#include "verilated.h"


class Vtop_level__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop_level___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        VL_IN8(clk,0,0);
        VL_IN8(reset,0,0);
        VL_IN8(uart_rx_i,0,0);
        VL_OUT8(uart_tx_o,0,0);
        VL_OUT8(predicted_digit,3,0);
        CData/*0:0*/ top_level__DOT__rx_dv;
        CData/*7:0*/ top_level__DOT__rx_byte;
        CData/*0:0*/ top_level__DOT__frame_loaded;
        CData/*0:0*/ top_level__DOT__conv_start;
        CData/*0:0*/ top_level__DOT__relu_start;
        CData/*0:0*/ top_level__DOT__pool_start;
        CData/*0:0*/ top_level__DOT__dense_start;
        CData/*0:0*/ top_level__DOT__tx_start;
        CData/*0:0*/ top_level__DOT__conv_done;
        CData/*0:0*/ top_level__DOT__relu_done;
        CData/*0:0*/ top_level__DOT__pool_done;
        CData/*0:0*/ top_level__DOT__dense_done;
        CData/*0:0*/ top_level__DOT__flattening;
        CData/*0:0*/ top_level__DOT__flat_done;
        CData/*0:0*/ top_level__DOT__argmax_done;
        CData/*0:0*/ top_level__DOT__tx_busy;
        CData/*0:0*/ top_level__DOT__tx_start_q;
        CData/*2:0*/ top_level__DOT__RX__DOT__state;
        CData/*2:0*/ top_level__DOT__RX__DOT__bit_idx;
        CData/*7:0*/ top_level__DOT__RX__DOT__data;
        CData/*1:0*/ top_level__DOT__u_conv__DOT__state;
        CData/*1:0*/ top_level__DOT__u_relu__DOT__state;
        CData/*1:0*/ top_level__DOT__u_pool__DOT__state;
        CData/*1:0*/ top_level__DOT__u_dense__DOT__state;
        CData/*1:0*/ top_level__DOT__u_argmax__DOT__state;
        CData/*3:0*/ top_level__DOT__u_argmax__DOT__besti;
        CData/*2:0*/ top_level__DOT__TX__DOT__state;
        CData/*2:0*/ top_level__DOT__TX__DOT__bit_idx;
        CData/*7:0*/ top_level__DOT__TX__DOT__data_reg;
        CData/*0:0*/ top_level__DOT__ctrl__DOT__busy;
        CData/*2:0*/ top_level__DOT__ctrl__DOT__state;
        CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
        CData/*0:0*/ __VactContinue;
        SData/*15:0*/ top_level__DOT____Vlvbound_h5be8a0cc__0;
        SData/*15:0*/ top_level__DOT____Vlvbound_h92949796__0;
        SData/*9:0*/ top_level__DOT__RX__DOT__clk_cnt;
        SData/*15:0*/ top_level__DOT__u_conv__DOT__unnamedblk2__DOT__res;
        SData/*15:0*/ top_level__DOT__u_conv__DOT____Vlvbound_h3b513ba9__0;
        SData/*15:0*/ top_level__DOT__u_relu__DOT__unnamedblk1__DOT__v;
        SData/*15:0*/ top_level__DOT__u_relu__DOT____Vlvbound_he9448718__0;
        SData/*15:0*/ top_level__DOT__u_pool__DOT____Vlvbound_h816b22dc__0;
        SData/*15:0*/ top_level__DOT__u_dense__DOT__unnamedblk1__DOT__res;
        SData/*15:0*/ top_level__DOT__u_dense__DOT____Vlvbound_h29ef7931__0;
        SData/*15:0*/ top_level__DOT__u_argmax__DOT__bestv;
        SData/*9:0*/ top_level__DOT__TX__DOT__clk_cnt;
        IData/*31:0*/ top_level__DOT__r;
        IData/*31:0*/ top_level__DOT__c;
        IData/*31:0*/ top_level__DOT__fi_c;
        IData/*31:0*/ top_level__DOT__fi_r;
        IData/*31:0*/ top_level__DOT__fi_q;
        IData/*31:0*/ top_level__DOT__fi_idx;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__oc;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__orow;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__ocol;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__ic;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__kr;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__kc;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__prod;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__sat_pos_cnt;
    };
    struct {
        IData/*31:0*/ top_level__DOT__u_conv__DOT__sat_neg_cnt;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__unnamedblk1__DOT__ir;
        IData/*31:0*/ top_level__DOT__u_conv__DOT__unnamedblk1__DOT__icc;
        IData/*31:0*/ top_level__DOT__u_relu__DOT__c;
        IData/*31:0*/ top_level__DOT__u_relu__DOT__r;
        IData/*31:0*/ top_level__DOT__u_relu__DOT__q;
        IData/*31:0*/ top_level__DOT__u_pool__DOT__ch;
        IData/*31:0*/ top_level__DOT__u_pool__DOT__r;
        IData/*31:0*/ top_level__DOT__u_pool__DOT__q;
        IData/*31:0*/ top_level__DOT__u_dense__DOT__o;
        IData/*31:0*/ top_level__DOT__u_dense__DOT__i;
        IData/*31:0*/ top_level__DOT__u_dense__DOT__prod;
        IData/*31:0*/ top_level__DOT__u_dense__DOT__sat_pos_cnt;
        IData/*31:0*/ top_level__DOT__u_dense__DOT__sat_neg_cnt;
        IData/*31:0*/ top_level__DOT__u_argmax__DOT__i;
        IData/*31:0*/ __VactIterCount;
        QData/*63:0*/ top_level__DOT__cycle_ctr;
        QData/*63:0*/ top_level__DOT__t_start;
        QData/*63:0*/ top_level__DOT__t_conv;
        QData/*63:0*/ top_level__DOT__t_relu;
        QData/*63:0*/ top_level__DOT__t_pool;
        QData/*63:0*/ top_level__DOT__t_flat;
        QData/*63:0*/ top_level__DOT__t_dense;
        QData/*63:0*/ top_level__DOT__t_argmax;
        QData/*63:0*/ top_level__DOT__t_tx;
        QData/*37:0*/ top_level__DOT__u_conv__DOT__acc;
        QData/*37:0*/ top_level__DOT__u_conv__DOT__unnamedblk2__DOT__shifted;
        QData/*42:0*/ top_level__DOT__u_dense__DOT__acc;
        QData/*42:0*/ top_level__DOT__u_dense__DOT__unnamedblk1__DOT__shifted;
        VlUnpacked<VlUnpacked<VlUnpacked<SData/*15:0*/, 28>, 28>, 1> top_level__DOT__ifmap;
        VlUnpacked<VlUnpacked<VlUnpacked<SData/*15:0*/, 28>, 28>, 8> top_level__DOT__conv_out;
        VlUnpacked<VlUnpacked<VlUnpacked<SData/*15:0*/, 28>, 28>, 8> top_level__DOT__relu_out;
        VlUnpacked<VlUnpacked<VlUnpacked<SData/*15:0*/, 14>, 14>, 8> top_level__DOT__pool_out;
        VlUnpacked<SData/*15:0*/, 1568> top_level__DOT__flat_vec;
        VlUnpacked<SData/*15:0*/, 10> top_level__DOT__logits;
        VlUnpacked<VlUnpacked<VlUnpacked<VlUnpacked<SData/*15:0*/, 3>, 3>, 1>, 8> top_level__DOT__conv_w;
        VlUnpacked<SData/*15:0*/, 8> top_level__DOT__conv_b;
        VlUnpacked<VlUnpacked<SData/*15:0*/, 1568>, 10> top_level__DOT__dense_w;
        VlUnpacked<SData/*15:0*/, 10> top_level__DOT__dense_b;
        VlUnpacked<CData/*0:0*/, 2> __Vm_traceActivity;
    };
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtop_level__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtop_level___024root(Vtop_level__Syms* symsp, const char* v__name);
    ~Vtop_level___024root();
    VL_UNCOPYABLE(Vtop_level___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
