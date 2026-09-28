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

lint bpu_fvu -GVLanes=16 -GSpmWords=4096 -GSfuLanes=4                                # fpga
lint bpu_fvu -GVLanes=2 -GSpmWords=1024 -GSfuLanes=1 "-GMulPipe=3'b010" "-GAddPipe=3'b010" "-GSfuPipe=5'b01010"  # asic
lint bpu_fvu -GVLanes=4 -GSpmWords=512 -GSfuLanes=2 "-GMulPipe=3'b000" "-GAddPipe=3'b000" "-GSfuPipe=5'b00000"   # tiny
lint bpu_fvu -GVLanes=4 -GSpmWords=512                                               # one SFU per lane

lint bpu_compute_top -GFSfuLanes=1                                                    # asic-like defaults
lint bpu_compute_top -GNSlice=3 -GLanes=4 -GRowInterleave=2 -GMaxK=256 -GVLanes=4 -GSpmWords=16384      -GQProdReg=0 -GQTreeRegEvery=1 -GQI2fReg=0 "-GQMulPipe=3'b000" "-GQAddPipe=3'b000"      "-GFMulPipe=3'b000" "-GFAddPipe=3'b000" "-GFSfuPipe=5'b00000"                     # tiny
lint bpu_compute_top -GNSlice=32 -GLanes=64 -GRowInterleave=4 -GMaxK=6144 -GVLanes=16 -GSpmWords=8192      -GQProdReg=1 -GQTreeRegEvery=2 "-GQMulPipe=3'b111" "-GQAddPipe=3'b111"      "-GFMulPipe=3'b111" "-GFAddPipe=3'b111" "-GFSfuPipe=5'b11111"                     # fpga

echo "lint clean"
