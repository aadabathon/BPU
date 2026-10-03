#!/usr/bin/env bash
# Soak bpu_sfu against bpuref.sfu: every mantissa of each function's critical
# binades plus random inputs (~110M vectors), streamed through a pipe.
# Usage: compute/scripts/soak_sfu.sh [seed]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compute="$here/.."
rtl="$compute/rtl"
build="${BPU_BUILD_ROOT:-$compute/sim_build}/soak/soak_sfu"
mkdir -p "$build"
mapfile -t files < <(cd "$rtl" && grep -v '^//' compute.f | grep -v '^[[:space:]]*$' | sed "s#^#$rtl/#")

verilator --cc --exe --build -O3 -j 0 -Wno-fatal --top-module bpu_sfu "-GPipeMask=5'b00000" \
  -CFLAGS -O2 --Mdir "$build" -o soak_sfu "${files[@]}" "$compute/tb/soak/sfu_soak.cpp" >/dev/null

PYTHONPATH="$compute/model" python3 "$compute/tb/soak/gen_sfu_soak.py" "${1:-1}" | "$build/soak_sfu"
