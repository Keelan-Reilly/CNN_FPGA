// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vconv2d__pch.h"
#include "verilated_vcd_c.h"

//============================================================
// Constructors

Vconv2d::Vconv2d(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vconv2d__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , reset{vlSymsp->TOP.reset}
    , start{vlSymsp->TOP.start}
    , done{vlSymsp->TOP.done}
    , input_feature{vlSymsp->TOP.input_feature}
    , weights{vlSymsp->TOP.weights}
    , biases{vlSymsp->TOP.biases}
    , out_feature{vlSymsp->TOP.out_feature}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
    contextp()->traceBaseModelCbAdd(
        [this](VerilatedTraceBaseC* tfp, int levels, int options) { traceBaseModel(tfp, levels, options); });
}

Vconv2d::Vconv2d(const char* _vcname__)
    : Vconv2d(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vconv2d::~Vconv2d() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vconv2d___024root___eval_debug_assertions(Vconv2d___024root* vlSelf);
#endif  // VL_DEBUG
void Vconv2d___024root___eval_static(Vconv2d___024root* vlSelf);
void Vconv2d___024root___eval_initial(Vconv2d___024root* vlSelf);
void Vconv2d___024root___eval_settle(Vconv2d___024root* vlSelf);
void Vconv2d___024root___eval(Vconv2d___024root* vlSelf);

void Vconv2d::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vconv2d::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vconv2d___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_activity = true;
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vconv2d___024root___eval_static(&(vlSymsp->TOP));
        Vconv2d___024root___eval_initial(&(vlSymsp->TOP));
        Vconv2d___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vconv2d___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vconv2d::eventsPending() { return false; }

uint64_t Vconv2d::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vconv2d::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vconv2d___024root___eval_final(Vconv2d___024root* vlSelf);

VL_ATTR_COLD void Vconv2d::final() {
    Vconv2d___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vconv2d::hierName() const { return vlSymsp->name(); }
const char* Vconv2d::modelName() const { return "Vconv2d"; }
unsigned Vconv2d::threads() const { return 1; }
void Vconv2d::prepareClone() const { contextp()->prepareClone(); }
void Vconv2d::atClone() const {
    contextp()->threadPoolpOnClone();
}
std::unique_ptr<VerilatedTraceConfig> Vconv2d::traceConfig() const {
    return std::unique_ptr<VerilatedTraceConfig>{new VerilatedTraceConfig{false, false, false}};
};

//============================================================
// Trace configuration

void Vconv2d___024root__trace_decl_types(VerilatedVcd* tracep);

void Vconv2d___024root__trace_init_top(Vconv2d___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD static void trace_init(void* voidSelf, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vconv2d___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vconv2d___024root*>(voidSelf);
    Vconv2d__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (!vlSymsp->_vm_contextp__->calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
            "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->pushPrefix(std::string{vlSymsp->name()}, VerilatedTracePrefixType::SCOPE_MODULE);
    Vconv2d___024root__trace_decl_types(tracep);
    Vconv2d___024root__trace_init_top(vlSelf, tracep);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vconv2d___024root__trace_register(Vconv2d___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD void Vconv2d::traceBaseModel(VerilatedTraceBaseC* tfp, int levels, int options) {
    (void)levels; (void)options;
    VerilatedVcdC* const stfp = dynamic_cast<VerilatedVcdC*>(tfp);
    if (VL_UNLIKELY(!stfp)) {
        vl_fatal(__FILE__, __LINE__, __FILE__,"'Vconv2d::trace()' called on non-VerilatedVcdC object;"
            " use --trace-fst with VerilatedFst object, and --trace-vcd with VerilatedVcd object");
    }
    stfp->spTrace()->addModel(this);
    stfp->spTrace()->addInitCb(&trace_init, &(vlSymsp->TOP));
    Vconv2d___024root__trace_register(&(vlSymsp->TOP), stfp->spTrace());
}
