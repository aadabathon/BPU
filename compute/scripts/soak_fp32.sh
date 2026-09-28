#!/usr/bin/env bash
# Soak the fp32 units against the host CPU's IEEE arithmetic (tb/soak/fp_soak.cpp).
# Usage: compute/scripts/soak_fp32.sh [count] [seed]      (default 50M per unit)
#        BPU_SOAK_BF16=1 compute/scripts/soak_fp32.sh     (+ all 2^32 bf16 x bf16 products)
# Needs verilator, make and a C++ compiler.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compute="$here/.."
rtl="$compute/rtl"
build="${BPU_BUILD_ROOT:-$compute/sim_build}/soak"
mkdir -p "$build"
mapfile -t files < <(cd "$rtl" && grep -v '^//' compute.f | grep -v '^[[:space:]]*$' | sed "s#^#$rtl/#")
count="${1:-50000000}"
seed="${2:-1}"

build_one() {   # name top define params...
  local name="$1" top="$2" def="$3"; shift 3
  verilator --cc --exe --build -O3 -j 0 -Wno-fatal --top-module "$top" "$@" \
    -CFLAGS "-O2 -DBPU_OP_$def -DBPU_OP_NAME='\"$top\"' -DBPU_TOP_HEADER='\"V$top.h\"' -DBPU_TOP_CLASS=V$top" \
    --Mdir "$build/$name" -o "$name" "${files[@]}" "$compute/tb/soak/fp_soak.cpp" >/dev/null
  echo "$build/$name/$name"
}

mul=$(build_one soak_mul bpu_fp32_mul MUL "-GPipeMask=3'b000")
add=$(build_one soak_add bpu_fp32_add ADD "-GPipeMask=3'b000")
i2f=$(build_one soak_i2f bpu_int2fp32 I2F -GInW=22 "-GReg=1'b0")

status=0
"$mul" "$count" "$seed" || status=1
"$add" "$count" "$seed" || status=1
"$i2f" || status=1
if [ "${BPU_SOAK_BF16:-0}" = "1" ]; then "$mul" bf16 || status=1; fi
exit $status
