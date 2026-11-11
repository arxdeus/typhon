// Tiny shim for TargetMachine options not reachable through the LLVM C API.
#include "llvm-c/TargetMachine.h"
#include "llvm/Target/TargetMachine.h"

extern "C" void typhon_tm_set_emulated_tls(LLVMTargetMachineRef tm, int on) {
  reinterpret_cast<llvm::TargetMachine *>(tm)->Options.EmulatedTLS = on != 0;
}
