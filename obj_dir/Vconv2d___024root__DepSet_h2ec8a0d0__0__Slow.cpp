// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vconv2d.h for the primary calling header

#include "Vconv2d__pch.h"
#include "Vconv2d___024root.h"

VL_ATTR_COLD void Vconv2d___024root___eval_static(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_static\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
}

VL_ATTR_COLD void Vconv2d___024root___eval_initial(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_initial\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vconv2d___024root___eval_final(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_final\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vconv2d___024root___eval_settle(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___eval_settle\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vconv2d___024root___dump_triggers__act(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___dump_triggers__act\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vconv2d___024root___dump_triggers__nba(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___dump_triggers__nba\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vconv2d___024root___ctor_var_reset(Vconv2d___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vconv2d___024root___ctor_var_reset\n"); );
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->reset = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9928399931838511862ull);
    vlSelf->start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9867861323841650631ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 28; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 28; ++__Vi2) {
                vlSelf->input_feature[__Vi0][__Vi1][__Vi2] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 12495436754274323411ull);
            }
        }
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 1; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 3; ++__Vi2) {
                for (int __Vi3 = 0; __Vi3 < 3; ++__Vi3) {
                    vlSelf->weights[__Vi0][__Vi1][__Vi2][__Vi3] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15258019614976312503ull);
                }
            }
        }
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->biases[__Vi0] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15469516661481283275ull);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 28; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 28; ++__Vi2) {
                vlSelf->out_feature[__Vi0][__Vi1][__Vi2] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 8096526658755244842ull);
            }
        }
    }
    vlSelf->done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10296494685231209730ull);
    vlSelf->conv2d__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15665625573853264505ull);
    vlSelf->conv2d__DOT__oc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6065006630322828002ull);
    vlSelf->conv2d__DOT__ic = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7762845923123043944ull);
    vlSelf->conv2d__DOT__i = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1446177197121380162ull);
    vlSelf->conv2d__DOT__j = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3470775546313984063ull);
    vlSelf->conv2d__DOT__ki = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16662314611094567428ull);
    vlSelf->conv2d__DOT__kj = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12802240707248738342ull);
    vlSelf->conv2d__DOT__accum = VL_SCOPED_RAND_RESET_Q(36, __VscopeHash, 11713764598465111676ull);
    vlSelf->conv2d__DOT__out_row = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5609170803939242514ull);
    vlSelf->conv2d__DOT__out_col = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17165614022220569309ull);
    vlSelf->conv2d__DOT__mult = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17959403901482262014ull);
    vlSelf->conv2d__DOT__scale_and_saturate__Vstatic__shifted = VL_SCOPED_RAND_RESET_Q(36, __VscopeHash, 8455904427687765424ull);
    vlSelf->conv2d__DOT__scale_and_saturate__Vstatic__result = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17378310655084106245ull);
    vlSelf->conv2d__DOT__unnamedblk1__DOT__in_r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5860301641025007868ull);
    vlSelf->conv2d__DOT__unnamedblk1__DOT__in_c = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14956251438101036499ull);
    vlSelf->conv2d__DOT__unnamedblk1__DOT__in_val = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 1243549558539542280ull);
    vlSelf->conv2d__DOT____Vlvbound_hfb6f3c5f__0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 1598373562064092324ull);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9526919608049418986ull);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
