// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtop_level__Syms.h"


void Vtop_level___024root__trace_chg_0_sub_0(Vtop_level___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtop_level___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_chg_0\n"); );
    // Init
    Vtop_level___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop_level___024root*>(voidSelf);
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtop_level___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vtop_level___024root__trace_chg_0_sub_0(Vtop_level___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_chg_0_sub_0\n"); );
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[0U]))) {
        bufp->chgSData(oldp+0,(vlSelfRef.top_level__DOT__conv_b[0]),16);
        bufp->chgSData(oldp+1,(vlSelfRef.top_level__DOT__conv_b[1]),16);
        bufp->chgSData(oldp+2,(vlSelfRef.top_level__DOT__conv_b[2]),16);
        bufp->chgSData(oldp+3,(vlSelfRef.top_level__DOT__conv_b[3]),16);
        bufp->chgSData(oldp+4,(vlSelfRef.top_level__DOT__conv_b[4]),16);
        bufp->chgSData(oldp+5,(vlSelfRef.top_level__DOT__conv_b[5]),16);
        bufp->chgSData(oldp+6,(vlSelfRef.top_level__DOT__conv_b[6]),16);
        bufp->chgSData(oldp+7,(vlSelfRef.top_level__DOT__conv_b[7]),16);
        bufp->chgSData(oldp+8,(vlSelfRef.top_level__DOT__dense_b[0]),16);
        bufp->chgSData(oldp+9,(vlSelfRef.top_level__DOT__dense_b[1]),16);
        bufp->chgSData(oldp+10,(vlSelfRef.top_level__DOT__dense_b[2]),16);
        bufp->chgSData(oldp+11,(vlSelfRef.top_level__DOT__dense_b[3]),16);
        bufp->chgSData(oldp+12,(vlSelfRef.top_level__DOT__dense_b[4]),16);
        bufp->chgSData(oldp+13,(vlSelfRef.top_level__DOT__dense_b[5]),16);
        bufp->chgSData(oldp+14,(vlSelfRef.top_level__DOT__dense_b[6]),16);
        bufp->chgSData(oldp+15,(vlSelfRef.top_level__DOT__dense_b[7]),16);
        bufp->chgSData(oldp+16,(vlSelfRef.top_level__DOT__dense_b[8]),16);
        bufp->chgSData(oldp+17,(vlSelfRef.top_level__DOT__dense_b[9]),16);
    }
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[1U]))) {
        bufp->chgSData(oldp+18,(vlSelfRef.top_level__DOT__logits[0]),16);
        bufp->chgSData(oldp+19,(vlSelfRef.top_level__DOT__logits[1]),16);
        bufp->chgSData(oldp+20,(vlSelfRef.top_level__DOT__logits[2]),16);
        bufp->chgSData(oldp+21,(vlSelfRef.top_level__DOT__logits[3]),16);
        bufp->chgSData(oldp+22,(vlSelfRef.top_level__DOT__logits[4]),16);
        bufp->chgSData(oldp+23,(vlSelfRef.top_level__DOT__logits[5]),16);
        bufp->chgSData(oldp+24,(vlSelfRef.top_level__DOT__logits[6]),16);
        bufp->chgSData(oldp+25,(vlSelfRef.top_level__DOT__logits[7]),16);
        bufp->chgSData(oldp+26,(vlSelfRef.top_level__DOT__logits[8]),16);
        bufp->chgSData(oldp+27,(vlSelfRef.top_level__DOT__logits[9]),16);
        bufp->chgBit(oldp+28,(vlSelfRef.top_level__DOT__rx_dv));
        bufp->chgCData(oldp+29,(vlSelfRef.top_level__DOT__rx_byte),8);
        bufp->chgIData(oldp+30,(vlSelfRef.top_level__DOT__r),32);
        bufp->chgIData(oldp+31,(vlSelfRef.top_level__DOT__c),32);
        bufp->chgBit(oldp+32,(vlSelfRef.top_level__DOT__frame_loaded));
        bufp->chgBit(oldp+33,(vlSelfRef.top_level__DOT__conv_start));
        bufp->chgBit(oldp+34,(vlSelfRef.top_level__DOT__relu_start));
        bufp->chgBit(oldp+35,(vlSelfRef.top_level__DOT__pool_start));
        bufp->chgBit(oldp+36,(vlSelfRef.top_level__DOT__dense_start));
        bufp->chgBit(oldp+37,(vlSelfRef.top_level__DOT__tx_start));
        bufp->chgBit(oldp+38,(vlSelfRef.top_level__DOT__conv_done));
        bufp->chgBit(oldp+39,(vlSelfRef.top_level__DOT__relu_done));
        bufp->chgBit(oldp+40,(vlSelfRef.top_level__DOT__pool_done));
        bufp->chgBit(oldp+41,(vlSelfRef.top_level__DOT__dense_done));
        bufp->chgBit(oldp+42,(vlSelfRef.top_level__DOT__flattening));
        bufp->chgBit(oldp+43,(vlSelfRef.top_level__DOT__flat_done));
        bufp->chgIData(oldp+44,(vlSelfRef.top_level__DOT__fi_c),32);
        bufp->chgIData(oldp+45,(vlSelfRef.top_level__DOT__fi_r),32);
        bufp->chgIData(oldp+46,(vlSelfRef.top_level__DOT__fi_q),32);
        bufp->chgIData(oldp+47,(vlSelfRef.top_level__DOT__fi_idx),32);
        bufp->chgBit(oldp+48,(vlSelfRef.top_level__DOT__argmax_done));
        bufp->chgBit(oldp+49,(vlSelfRef.top_level__DOT__tx_busy));
        bufp->chgCData(oldp+50,(vlSelfRef.top_level__DOT__RX__DOT__state),3);
        bufp->chgSData(oldp+51,(vlSelfRef.top_level__DOT__RX__DOT__clk_cnt),9);
        bufp->chgCData(oldp+52,(vlSelfRef.top_level__DOT__RX__DOT__bit_idx),3);
        bufp->chgCData(oldp+53,(vlSelfRef.top_level__DOT__RX__DOT__data),8);
        bufp->chgCData(oldp+54,(vlSelfRef.top_level__DOT__TX__DOT__state),3);
        bufp->chgSData(oldp+55,(vlSelfRef.top_level__DOT__TX__DOT__clk_cnt),9);
        bufp->chgCData(oldp+56,(vlSelfRef.top_level__DOT__TX__DOT__bit_idx),3);
        bufp->chgCData(oldp+57,(vlSelfRef.top_level__DOT__TX__DOT__data_reg),8);
        bufp->chgBit(oldp+58,(vlSelfRef.top_level__DOT__ctrl__DOT__busy));
        bufp->chgCData(oldp+59,(vlSelfRef.top_level__DOT__ctrl__DOT__state),3);
        bufp->chgCData(oldp+60,(vlSelfRef.top_level__DOT__u_argmax__DOT__state),2);
        bufp->chgIData(oldp+61,(vlSelfRef.top_level__DOT__u_argmax__DOT__i),32);
        bufp->chgSData(oldp+62,(vlSelfRef.top_level__DOT__u_argmax__DOT__bestv),16);
        bufp->chgCData(oldp+63,(vlSelfRef.top_level__DOT__u_argmax__DOT__besti),4);
        bufp->chgCData(oldp+64,(vlSelfRef.top_level__DOT__u_conv__DOT__state),2);
        bufp->chgIData(oldp+65,(vlSelfRef.top_level__DOT__u_conv__DOT__oc),32);
        bufp->chgIData(oldp+66,(vlSelfRef.top_level__DOT__u_conv__DOT__ic),32);
        bufp->chgIData(oldp+67,(vlSelfRef.top_level__DOT__u_conv__DOT__ki),32);
        bufp->chgIData(oldp+68,(vlSelfRef.top_level__DOT__u_conv__DOT__kj),32);
        bufp->chgQData(oldp+69,(vlSelfRef.top_level__DOT__u_conv__DOT__accum),36);
        bufp->chgIData(oldp+71,(vlSelfRef.top_level__DOT__u_conv__DOT__out_row),32);
        bufp->chgIData(oldp+72,(vlSelfRef.top_level__DOT__u_conv__DOT__out_col),32);
        bufp->chgIData(oldp+73,(vlSelfRef.top_level__DOT__u_conv__DOT__mult),32);
        bufp->chgQData(oldp+74,(vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__shifted),36);
        bufp->chgSData(oldp+76,(vlSelfRef.top_level__DOT__u_conv__DOT__scale_and_saturate__Vstatic__result),16);
        bufp->chgIData(oldp+77,(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_r),32);
        bufp->chgIData(oldp+78,(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_c),32);
        bufp->chgSData(oldp+79,(vlSelfRef.top_level__DOT__u_conv__DOT__unnamedblk1__DOT__in_val),16);
        bufp->chgCData(oldp+80,(vlSelfRef.top_level__DOT__u_dense__DOT__state),2);
        bufp->chgIData(oldp+81,(vlSelfRef.top_level__DOT__u_dense__DOT__o),32);
        bufp->chgIData(oldp+82,(vlSelfRef.top_level__DOT__u_dense__DOT__i),32);
        bufp->chgQData(oldp+83,(vlSelfRef.top_level__DOT__u_dense__DOT__acc),43);
        bufp->chgIData(oldp+85,(vlSelfRef.top_level__DOT__u_dense__DOT__prod),32);
        bufp->chgCData(oldp+86,(vlSelfRef.top_level__DOT__u_pool__DOT__state),2);
        bufp->chgIData(oldp+87,(vlSelfRef.top_level__DOT__u_pool__DOT__c),32);
        bufp->chgIData(oldp+88,(vlSelfRef.top_level__DOT__u_pool__DOT__r),32);
        bufp->chgIData(oldp+89,(vlSelfRef.top_level__DOT__u_pool__DOT__q),32);
        bufp->chgCData(oldp+90,(vlSelfRef.top_level__DOT__u_relu__DOT__state),2);
        bufp->chgIData(oldp+91,(vlSelfRef.top_level__DOT__u_relu__DOT__c),32);
        bufp->chgIData(oldp+92,(vlSelfRef.top_level__DOT__u_relu__DOT__r),32);
        bufp->chgIData(oldp+93,(vlSelfRef.top_level__DOT__u_relu__DOT__q),32);
        bufp->chgSData(oldp+94,(vlSelfRef.top_level__DOT__u_relu__DOT__unnamedblk1__DOT__v),16);
    }
    bufp->chgBit(oldp+95,(vlSelfRef.clk));
    bufp->chgBit(oldp+96,(vlSelfRef.reset));
    bufp->chgBit(oldp+97,(vlSelfRef.uart_rx_i));
    bufp->chgBit(oldp+98,(vlSelfRef.uart_tx_o));
    bufp->chgCData(oldp+99,(vlSelfRef.predicted_digit),4);
    bufp->chgCData(oldp+100,((0xffU & ((IData)(0x30U) 
                                       + (IData)(vlSelfRef.predicted_digit)))),8);
}

void Vtop_level___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_level___024root__trace_cleanup\n"); );
    // Init
    Vtop_level___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop_level___024root*>(voidSelf);
    Vtop_level__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
}
