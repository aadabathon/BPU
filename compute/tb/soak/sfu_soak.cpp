// Compare bpu_sfu (combinational build) with reference records streamed on stdin
// by gen_sfu_soak.py: little-endian uint32 triples (func, a, expected y).
#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include "Vbpu_sfu.h"
#include "verilated.h"

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new Vbpu_sfu;
  dut->clk_i = 0;
  dut->rst_ni = 1;
  dut->valid_i = 1;
  static const char* names[] = {"rcp", "rsqrt", "exp2", "exp", "log2"};
  uint64_t count[5] = {0}, errors[5] = {0};
  uint32_t rec[3 * 4096];
  size_t n;
  while ((n = std::fread(rec, 12, 4096, stdin)) > 0) {
    for (size_t i = 0; i < n; ++i) {
      uint32_t func = rec[3 * i], a = rec[3 * i + 1], ref = rec[3 * i + 2];
      dut->func_i = func;
      dut->a_i = a;
      dut->eval();
      ++count[func];
      if (dut->y_o != ref && errors[func]++ < 5)
        std::printf("%s(%08x): rtl %08x ref %08x\n", names[func], a, dut->y_o, ref);
    }
  }
  uint64_t bad = 0;
  for (int f = 0; f < 5; ++f) {
    std::printf("bpu_sfu %-5s: %" PRIu64 " vectors, %" PRIu64 " mismatches\n", names[f], count[f], errors[f]);
    bad += errors[f];
  }
  delete dut;
  return bad ? 1 : 0;
}
