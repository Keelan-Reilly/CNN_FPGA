// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop_level.h for the primary calling header

#include "Vtop_level__pch.h"
#include "Vtop_level___024root.h"

VL_ATTR_COLD void Vtop_level___024root___eval_static(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_static\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
}

VL_ATTR_COLD void Vtop_level___024root___eval_initial__TOP(Vtop_level___024root* vlSelf);
VL_ATTR_COLD void Vtop_level___024root____Vm_traceActivitySetAll(Vtop_level___024root* vlSelf);

VL_ATTR_COLD void Vtop_level___024root___eval_initial(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_initial\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtop_level___024root___eval_initial__TOP(vlSelf);
    Vtop_level___024root____Vm_traceActivitySetAll(vlSelf);
}

VL_ATTR_COLD void Vtop_level___024root___eval_initial__TOP(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_initial__TOP\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlWide<7>/*223:0*/ __Vtemp_1;
    VlWide<6>/*191:0*/ __Vtemp_2;
    VlWide<6>/*191:0*/ __Vtemp_3;
    VlWide<6>/*191:0*/ __Vtemp_4;
    // Body
    __Vtemp_1[0U] = 0x2e6d656dU;
    __Vtemp_1[1U] = 0x67687473U;
    __Vtemp_1[2U] = 0x5f776569U;
    __Vtemp_1[3U] = 0x6f6e7631U;
    __Vtemp_1[4U] = 0x74732f63U;
    __Vtemp_1[5U] = 0x65696768U;
    __Vtemp_1[6U] = 0x77U;
    VL_READMEM_N(true, 16, 72, 0, VL_CVT_PACK_STR_NW(7, __Vtemp_1)
                 ,  &(vlSelfRef.top_level__DOT__conv_w)
                 , 0, ~0ULL);
    __Vtemp_2[0U] = 0x2e6d656dU;
    __Vtemp_2[1U] = 0x61736573U;
    __Vtemp_2[2U] = 0x315f6269U;
    __Vtemp_2[3U] = 0x636f6e76U;
    __Vtemp_2[4U] = 0x6874732fU;
    __Vtemp_2[5U] = 0x77656967U;
    VL_READMEM_N(true, 16, 8, 0, VL_CVT_PACK_STR_NW(6, __Vtemp_2)
                 ,  &(vlSelfRef.top_level__DOT__conv_b)
                 , 0, ~0ULL);
    __Vtemp_3[0U] = 0x2e6d656dU;
    __Vtemp_3[1U] = 0x67687473U;
    __Vtemp_3[2U] = 0x5f776569U;
    __Vtemp_3[3U] = 0x2f666331U;
    __Vtemp_3[4U] = 0x67687473U;
    __Vtemp_3[5U] = 0x776569U;
    VL_READMEM_N(true, 16, 15680, 0, VL_CVT_PACK_STR_NW(6, __Vtemp_3)
                 ,  &(vlSelfRef.top_level__DOT__dense_w)
                 , 0, ~0ULL);
    __Vtemp_4[0U] = 0x2e6d656dU;
    __Vtemp_4[1U] = 0x61736573U;
    __Vtemp_4[2U] = 0x315f6269U;
    __Vtemp_4[3U] = 0x732f6663U;
    __Vtemp_4[4U] = 0x69676874U;
    __Vtemp_4[5U] = 0x7765U;
    VL_READMEM_N(true, 16, 10, 0, VL_CVT_PACK_STR_NW(6, __Vtemp_4)
                 ,  &(vlSelfRef.top_level__DOT__dense_b)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vtop_level___024root___eval_final(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_final\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vtop_level___024root___eval_settle(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___eval_settle\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop_level___024root___dump_triggers__act(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___dump_triggers__act\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
VL_ATTR_COLD void Vtop_level___024root___dump_triggers__nba(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___dump_triggers__nba\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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

VL_ATTR_COLD void Vtop_level___024root____Vm_traceActivitySetAll(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root____Vm_traceActivitySetAll\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vm_traceActivity[0U] = 1U;
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
}

VL_ATTR_COLD void Vtop_level___024root___ctor_var_reset(Vtop_level___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root___ctor_var_reset\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->reset = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9928399931838511862ull);
    vlSelf->uart_rx_i = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11432468475150732234ull);
    vlSelf->uart_tx_o = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1201968793088762859ull);
    vlSelf->predicted_digit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8889876055336614851ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 28; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 28; ++__Vi2) {
                vlSelf->top_level__DOT__ifmap[__Vi0][__Vi1][__Vi2] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9893952259924131846ull);
            }
        }
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 28; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 28; ++__Vi2) {
                vlSelf->top_level__DOT__conv_out[__Vi0][__Vi1][__Vi2] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10595758254149046011ull);
            }
        }
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 28; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 28; ++__Vi2) {
                vlSelf->top_level__DOT__relu_out[__Vi0][__Vi1][__Vi2] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2040009673793463297ull);
            }
        }
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 14; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 14; ++__Vi2) {
                vlSelf->top_level__DOT__pool_out[__Vi0][__Vi1][__Vi2] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17291921405562294774ull);
            }
        }
    }
    for (int __Vi0 = 0; __Vi0 < 1568; ++__Vi0) {
        vlSelf->top_level__DOT__flat_vec[__Vi0] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 13284303424514132050ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->top_level__DOT__logits[__Vi0] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 3925566954647844862ull);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 1; ++__Vi1) {
            for (int __Vi2 = 0; __Vi2 < 3; ++__Vi2) {
                for (int __Vi3 = 0; __Vi3 < 3; ++__Vi3) {
                    vlSelf->top_level__DOT__conv_w[__Vi0][__Vi1][__Vi2][__Vi3] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 12516140811449363944ull);
                }
            }
        }
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->top_level__DOT__conv_b[__Vi0] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 720044096339697943ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 1568; ++__Vi1) {
            vlSelf->top_level__DOT__dense_w[__Vi0][__Vi1] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 14282690292223452899ull);
        }
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->top_level__DOT__dense_b[__Vi0] = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16381725052644697568ull);
    }
    vlSelf->top_level__DOT__rx_dv = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13364741273847757910ull);
    vlSelf->top_level__DOT__rx_byte = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 18047919160606142194ull);
    vlSelf->top_level__DOT__r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8353212350101413921ull);
    vlSelf->top_level__DOT__c = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11789736766103002692ull);
    vlSelf->top_level__DOT__frame_loaded = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9767476922913319860ull);
    vlSelf->top_level__DOT__conv_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10904179980051743104ull);
    vlSelf->top_level__DOT__relu_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8591314330772093067ull);
    vlSelf->top_level__DOT__pool_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5205834157258719414ull);
    vlSelf->top_level__DOT__dense_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5431772360319549236ull);
    vlSelf->top_level__DOT__tx_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2762094843260905517ull);
    vlSelf->top_level__DOT__conv_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15308183051350162070ull);
    vlSelf->top_level__DOT__relu_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13335138739901982139ull);
    vlSelf->top_level__DOT__pool_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12297399787612500363ull);
    vlSelf->top_level__DOT__dense_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15050103595713584530ull);
    vlSelf->top_level__DOT__flattening = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4860683679269879698ull);
    vlSelf->top_level__DOT__flat_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9565391622756820452ull);
    vlSelf->top_level__DOT__fi_c = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8434606569274393253ull);
    vlSelf->top_level__DOT__fi_r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17330767714231272774ull);
    vlSelf->top_level__DOT__fi_q = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13399364973821778298ull);
    vlSelf->top_level__DOT__fi_idx = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15788449849384624510ull);
    vlSelf->top_level__DOT__argmax_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11725749989154061301ull);
    vlSelf->top_level__DOT__tx_busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6193814293124594576ull);
    vlSelf->top_level__DOT____Vlvbound_h5be8a0cc__0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6340117938229540196ull);
    vlSelf->top_level__DOT____Vlvbound_h92949796__0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16217872417084875257ull);
    vlSelf->top_level__DOT__RX__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 7221696853652436212ull);
    vlSelf->top_level__DOT__RX__DOT__clk_cnt = VL_SCOPED_RAND_RESET_I(9, __VscopeHash, 15400944635295775173ull);
    vlSelf->top_level__DOT__RX__DOT__bit_idx = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 6097306945984396480ull);
    vlSelf->top_level__DOT__RX__DOT__data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 14107454625259222004ull);
    vlSelf->top_level__DOT__u_conv__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 6701778664127891056ull);
    vlSelf->top_level__DOT__u_conv__DOT__oc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16961422446899589059ull);
    vlSelf->top_level__DOT__u_conv__DOT__ic = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10935224987804925919ull);
    vlSelf->top_level__DOT__u_conv__DOT__i = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17008222452859265190ull);
    vlSelf->top_level__DOT__u_conv__DOT__j = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6385747547801923652ull);
    vlSelf->top_level__DOT__u_conv__DOT__ki = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6909082892992529521ull);
    vlSelf->top_level__DOT__u_conv__DOT__kj = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15351851605155045606ull);
    vlSelf->top_level__DOT__u_conv__DOT__accum = VL_SCOPED_RAND_RESET_Q(36, __VscopeHash, 6933522508826442439ull);
    vlSelf->top_level__DOT__u_conv__DOT__out_row = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8933430247596473478ull);
    vlSelf->top_level__DOT__u_conv__DOT__out_col = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9166845003470717906ull);
    vlSelf->top_level__DOT__u_conv__DOT__mult = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3083233452272774497ull);
    vlSelf->top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__shifted = VL_SCOPED_RAND_RESET_Q(36, __VscopeHash, 12710761587933156368ull);
    vlSelf->top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__result = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9342591468933359971ull);
    vlSelf->top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10841738040801421798ull);
    vlSelf->top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15459593753315686845ull);
    vlSelf->top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9728277257306766714ull);
    vlSelf->top_level__DOT__u_conv__DOT____Vlvbound_hfb6f3c5f__0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9084392294782578205ull);
    vlSelf->top_level__DOT__u_relu__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12089837726726795802ull);
    vlSelf->top_level__DOT__u_relu__DOT__c = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 18007582987655669408ull);
    vlSelf->top_level__DOT__u_relu__DOT__r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1099223119854941636ull);
    vlSelf->top_level__DOT__u_relu__DOT__q = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17648336687239143574ull);
    vlSelf->top_level__DOT__u_relu__DOT__unnamedblk1__DOT__v = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 11502815217453372385ull);
    vlSelf->top_level__DOT__u_relu__DOT____Vlvbound_he9448718__0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 7068498588256972058ull);
    vlSelf->top_level__DOT__u_pool__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1689654150658979990ull);
    vlSelf->top_level__DOT__u_pool__DOT__c = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4698726645677242770ull);
    vlSelf->top_level__DOT__u_pool__DOT__r = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6575013539072073628ull);
    vlSelf->top_level__DOT__u_pool__DOT__q = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16749100629285247836ull);
    vlSelf->top_level__DOT__u_pool__DOT____Vlvbound_h96184ae1__0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17847930811665465096ull);
    vlSelf->top_level__DOT__u_dense__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3420882083368678629ull);
    vlSelf->top_level__DOT__u_dense__DOT__o = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 159448593253465362ull);
    vlSelf->top_level__DOT__u_dense__DOT__i = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5386448264387074695ull);
    vlSelf->top_level__DOT__u_dense__DOT__acc = VL_SCOPED_RAND_RESET_Q(43, __VscopeHash, 7320902222342914594ull);
    vlSelf->top_level__DOT__u_dense__DOT__prod = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1836073423990216421ull);
    vlSelf->top_level__DOT__u_dense__DOT____Vlvbound_h29ef7931__0 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9883787771992225080ull);
    vlSelf->top_level__DOT__u_argmax__DOT__state = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 18112026162798082849ull);
    vlSelf->top_level__DOT__u_argmax__DOT__i = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11376761372682210822ull);
    vlSelf->top_level__DOT__u_argmax__DOT__bestv = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6850478075538915012ull);
    vlSelf->top_level__DOT__u_argmax__DOT__besti = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 14795422083483549678ull);
    vlSelf->top_level__DOT__ctrl__DOT__busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15295149254818491477ull);
    vlSelf->top_level__DOT__ctrl__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 10884229784769337729ull);
    vlSelf->top_level__DOT__TX__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 6773949650726794591ull);
    vlSelf->top_level__DOT__TX__DOT__clk_cnt = VL_SCOPED_RAND_RESET_I(9, __VscopeHash, 3062438902759986204ull);
    vlSelf->top_level__DOT__TX__DOT__bit_idx = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 1108246983975144345ull);
    vlSelf->top_level__DOT__TX__DOT__data_reg = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15739227173206963418ull);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9526919608049418986ull);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
