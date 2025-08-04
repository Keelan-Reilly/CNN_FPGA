// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vconv2d__pch.h"
#include "Vconv2d.h"
#include "Vconv2d___024root.h"

// FUNCTIONS
Vconv2d__Syms::~Vconv2d__Syms()
{
}

Vconv2d__Syms::Vconv2d__Syms(VerilatedContext* contextp, const char* namep, Vconv2d* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
        // Check resources
        Verilated::stackCheck(85);
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-12);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
}
