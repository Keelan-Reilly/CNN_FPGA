// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vconv2d.h for the primary calling header

#ifndef VERILATED_VCONV2D___024ROOT_H_
#define VERILATED_VCONV2D___024ROOT_H_  // guard

#include "verilated.h"


class Vconv2d__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vconv2d___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_IN8(start,0,0);
    VL_OUT8(done,0,0);
    CData/*1:0*/ conv2d__DOT__state;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __VactContinue;
    SData/*15:0*/ conv2d__DOT__scale_and_saturate__Vstatic__result;
    SData/*15:0*/ conv2d__DOT__unnamedblk1__DOT__in_val;
    SData/*15:0*/ conv2d__DOT____Vlvbound_hfb6f3c5f__0;
    IData/*31:0*/ conv2d__DOT__oc;
    IData/*31:0*/ conv2d__DOT__ic;
    IData/*31:0*/ conv2d__DOT__i;
    IData/*31:0*/ conv2d__DOT__j;
    IData/*31:0*/ conv2d__DOT__ki;
    IData/*31:0*/ conv2d__DOT__kj;
    IData/*31:0*/ conv2d__DOT__out_row;
    IData/*31:0*/ conv2d__DOT__out_col;
    IData/*31:0*/ conv2d__DOT__mult;
    IData/*31:0*/ conv2d__DOT__unnamedblk1__DOT__in_r;
    IData/*31:0*/ conv2d__DOT__unnamedblk1__DOT__in_c;
    IData/*31:0*/ __VactIterCount;
    QData/*35:0*/ conv2d__DOT__accum;
    QData/*35:0*/ conv2d__DOT__scale_and_saturate__Vstatic__shifted;
    VL_IN16(input_feature[1][28][28],15,0);
    VL_IN16(weights[8][1][3][3],15,0);
    VL_IN16(biases[8],15,0);
    VL_OUT16(out_feature[8][28][28],15,0);
    VlUnpacked<CData/*0:0*/, 2> __Vm_traceActivity;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vconv2d__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vconv2d___024root(Vconv2d__Syms* symsp, const char* v__name);
    ~Vconv2d___024root();
    VL_UNCOPYABLE(Vconv2d___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
