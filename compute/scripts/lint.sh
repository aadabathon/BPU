#!/usr/bin/env bash
# Verilator -Wall lint of every compute top, at every named configuration.
# Usage: compute/scripts/lint.sh            (from anywhere; needs verilator on PATH)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl="$here/../rtl"
cd "$rtl"
mapfile -t files < <(grep -v '^//' compute.f | grep -v '^[[:space:]]*$')

lint() {
  local top="$1"; shift
  echo "--- lint $top $*"
  # PINCONNECTEMPTY: explicitly unconnected outputs (.foo_o()) are intentional.
  verilator --lint-only -Wall -Wno-MULTITOP -Wno-PINCONNECTEMPTY --top-module "$top" "$@" "${files[@]}"
}

lint bpu_fp32_mul
lint bpu_fp32_mul "-GPipeMask=3'b000"
lint bpu_fp32_add
lint bpu_fp32_add "-GPipeMask=3'b000"
lint bpu_int2fp32
lint bpu_sfu
lint bpu_sfu "-GPipeMask=5'b00000"
lint bpu_sfu "-GPipeMask=5'b01010"

# Keep in sync with compute/model/bpuref/configs.py
lint bpu_qmv_slice -GLanes=64 -GRowInterleave=4 -GMaxK=6144                        # fpga
lint bpu_qmv_slice -GLanes=16 -GRowInterleave=1 -GMaxK=2048 -GTreeRegEvery=0 \
     "-GMulPipe=3'b010" "-GAddPipe=3'b010"                                         # asic
lint bpu_qmv_slice -GLanes=4 -GRowInterleave=2 -GMaxK=256 -GProdReg=0 -GTreeRegEvery=1 \
     -GI2fReg=0 "-GMulPipe=3'b000" "-GAddPipe=3'b000"                             # tiny

lint bpu_qmv_array -GNSlice=32 -GLanes=64 -GRowInterleave=4 -GMaxK=6144                 # fpga
lint bpu_qmv_array -GNSlice=1 -GLanes=16 -GRowInterleave=1 -GMaxK=2048 -GTreeRegEvery=0      "-GMulPipe=3'b010" "-GAddPipe=3'b010"                                         # asic
lint bpu_qmv_array -GNSlice=3 -GLanes=4 -GRowInterleave=2 -GMaxK=256 -GProdReg=0 -GTreeRegEvery=1      -GI2fReg=0 "-GMulPipe=3'b000" "-GAddPipe=3'b000"                             # tiny

echo "lint clean"
