// High-volume check of the fp32 units against the host CPU's IEEE-754 binary32
// arithmetic (x86-64 SSE / AArch64: round-to-nearest-even with subnormals, the
// same contract as docs/numerics.md). Built by scripts/soak_fp32.sh with the unit
// under test combinational (PipeMask = 0), so one eval() is one operation.
//
//   soak_mul  [count] [seed]      random operand mix, default 50M
//   soak_add  [count] [seed]
//   soak_mul  bf16                every bf16 x bf16 pair (the QMV scale path), 2^32 ops
//   soak_i2f                      every 22-bit integer (the QMV group-sum path)
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "verilated.h"
#include BPU_TOP_HEADER

static inline float as_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t as_u32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint64_t rng_state = 1;
static inline uint64_t next_u64() {   // xorshift64*
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  return rng_state * 0x2545F4914F6CDD1DULL;
}

static inline uint32_t compose(uint32_t sign, uint32_t exp, uint32_t man) {
  return (sign << 31) | ((exp & 0xff) << 23) | (man & 0x7fffff);
}

// Operand mix aimed at rounding corners: raw bits, moderate range, sparse
// mantissas (exact ties), subnormals, and the overflow/underflow edges.
static uint32_t operand(uint32_t kind, uint32_t partner_exp) {
  uint64_t r = next_u64();
  uint32_t sign = r & 1, man = (r >> 1) & 0x7fffff, exp;
  switch (kind % 6) {
    case 0: return static_cast<uint32_t>(r >> 16);                     // any bit pattern
    case 1: exp = 100 + (r >> 40) % 55; break;                         // moderate
    case 2: exp = 100 + (r >> 40) % 55;                                // sparse mantissa
            man &= ~((1u << (23 - (r >> 50) % 12)) - 1); break;
    case 3: exp = (r >> 40) % 3; break;                                // subnormal edge
    case 4: exp = 250 + (r >> 40) % 5; break;                          // overflow edge
    default: exp = partner_exp; break;                                 // same exponent
  }
  return compose(sign, exp, man);
}

static int report(const char* what, uint64_t count, uint64_t errors) {
  std::printf("%s %s: %" PRIu64 " vectors, %" PRIu64 " mismatches\n", BPU_OP_NAME, what, count, errors);
  return errors ? 1 : 0;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new BPU_TOP_CLASS;
  dut->clk_i = 0;
  dut->rst_ni = 1;
  dut->valid_i = 1;
  uint64_t errors = 0, count = 0;

#if defined(BPU_OP_I2F)
  for (int64_t x = -(1 << 21); x < (1 << 21); ++x) {
    dut->x_i = static_cast<uint32_t>(x) & 0x3fffff;
    dut->eval();
    uint32_t ref = as_u32(static_cast<float>(x));
    if (dut->y_o != ref && errors++ < 10)
      std::printf("i2f %" PRId64 ": rtl %08x ref %08x\n", x, dut->y_o, ref);
    ++count;
  }
  return report("int2fp32 exhaustive", count, errors);
#else
  auto check = [&](uint32_t a, uint32_t b) {
    dut->a_i = a;
    dut->b_i = b;
    dut->eval();
#if defined(BPU_OP_MUL)
    float r = as_f32(a) * as_f32(b);
#else
    float r = as_f32(a) + as_f32(b);
#endif
    uint32_t y = dut->y_o;
    bool ok = std::isnan(r) ? (y == 0x7fc00000u) : (y == as_u32(r));
    if (!ok && errors++ < 10) std::printf("%08x %08x: rtl %08x ref %08x\n", a, b, y, as_u32(r));
    ++count;
  };

  if (argc > 1 && std::strcmp(argv[1], "bf16") == 0) {
    for (uint64_t a = 0; a < 0x10000; ++a)
      for (uint64_t b = 0; b < 0x10000; ++b)
        check(static_cast<uint32_t>(a << 16), static_cast<uint32_t>(b << 16));
    return report("bf16 x bf16 exhaustive", count, errors);
  }

  uint64_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 0) : 50000000ULL;
  rng_state = argc > 2 ? std::strtoull(argv[2], nullptr, 0) | 1 : 1;
  for (uint64_t i = 0; i < n; ++i) {
    uint32_t kind = static_cast<uint32_t>(next_u64() >> 58);
    uint32_t a = operand(kind, 0);
    uint32_t b = operand(kind >> 3, (a >> 23) & 0xff);
    check(a, b);
  }
  return report("random mix", count, errors);
#endif
}
