// Tiny shim for TargetMachine options not reachable through the LLVM C API.
#include "llvm-c/TargetMachine.h"
#include "llvm/Target/TargetMachine.h"

extern "C" void typhon_tm_set_emulated_tls(LLVMTargetMachineRef tm, int on) {
  reinterpret_cast<llvm::TargetMachine *>(tm)->Options.EmulatedTLS = on != 0;
}

// Port of the TargetParser part of codon `Aarch64::getFeatures`:
// `parseCpu` + `addCPUDefaults` + `addArchDefaults` + `toLLVMFeatureList`.
// Writes a comma-separated "+a,-b" list (NUL-terminated, truncated to cap) and
// returns its full length, or -1 if the CPU is unknown.
#include "llvm/TargetParser/AArch64TargetParser.h"
#include <cstring>
#include <string>
#include <vector>

extern "C" long typhon_aarch64_cpu_default_features(const char *cpu, char *buf, unsigned long cap) {
  const std::optional<llvm::AArch64::CpuInfo> cpuInfo = llvm::AArch64::parseCpu(cpu);
  if (!cpuInfo)
    return -1;
  const auto *archInfo = llvm::AArch64::getArchForCpu(cpu);
  llvm::AArch64::ExtensionSet extensions;
  extensions.addCPUDefaults(*cpuInfo);
  if (archInfo)
    extensions.addArchDefaults(*archInfo);
  std::vector<llvm::StringRef> features;
  extensions.toLLVMFeatureList(features);
  std::string out;
  for (auto &f : features) {
    if (!out.empty())
      out += ',';
    out += f.str();
  }
  if (cap > 0) {
    unsigned long n = out.size() < cap - 1 ? out.size() : cap - 1;
    std::memcpy(buf, out.data(), n);
    buf[n] = 0;
  }
  return (long)out.size();
}
