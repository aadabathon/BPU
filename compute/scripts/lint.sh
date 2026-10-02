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

# Every named configuration (generated from compute/model/bpuref/configs.py)
while read -r line; do
  eval "lint $line"
done < <(cd "$here/../model" && python3 -W ignore -m bpuref.configs --lint)

# Edge shapes beyond the named configurations
lint bpu_fvu_reduce -GLanes=1 "-GAddPipe=3'b101"
lint bpu_fvu_reduce -GLanes=16 "-GAddPipe=3'b000"
lint bpu_sram_shared -GNBanks=1 -GNRd=1 -GNWr=1
lint bpu_sram_shared -GNBanks=32 -GBankWords=4096 -GLanes=16 -GNRd=5 -GNWr=3 "-GOutReg=1'b1"
lint bpu_sram_shared "-GHash=1'b0" -GNRd=3
lint bpu_cmd_seq -GQDepth=1
lint bpu_cmd_seq -GNTags=32 -GQDepth=4
lint bpu_qmv_engine -GVLanes=64 -GLanes=16                                         # wide SRAM words
lint tt_um_bpu_fp32 ../tapeout/tt/src/tt_um_bpu_fp32.sv                            # Tiny Tapeout run
lint bpu_fvu_sys ../tb/hdl/bpu_fvu_sys.sv                                           # test harness

echo "lint clean"
