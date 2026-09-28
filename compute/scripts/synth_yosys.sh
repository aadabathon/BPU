#!/usr/bin/env bash
# Generic (technology-independent) Yosys synthesis of the QMV slice at the named
# configurations. Checks that the RTL stays inside what the open ASIC flow accepts,
# and gives a first cell-count/area comparison between configurations.
# Usage: compute/scripts/synth_yosys.sh [asic|fpga|tiny ...]   (default: asic tiny)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl="$here/../rtl"
cd "$rtl"
mapfile -t files < <(grep -v '^//' compute.f | grep -v '^[[:space:]]*$')
out="${BPU_SYNTH_OUT:-$here/../synth_out}"
mkdir -p "$out"

# Keep in sync with compute/model/bpuref/configs.py
declare -A cfg
cfg[fpga]="-GLanes=64 -GRowInterleave=4 -GMaxK=6144"
cfg[asic]="-GLanes=16 -GRowInterleave=1 -GMaxK=2048 -GTreeRegEvery=0 -GMulPipe=3'b010 -GAddPipe=3'b010"
cfg[tiny]="-GLanes=4 -GRowInterleave=2 -GMaxK=256 -GProdReg=0 -GTreeRegEvery=1 -GI2fReg=0 -GMulPipe=3'b000 -GAddPipe=3'b000"

names=("$@")
[ ${#names[@]} -eq 0 ] && names=(asic tiny)

for name in "${names[@]}"; do
  echo "=== synth bpu_qmv_slice [$name]"
  # Memories are kept as $mem cells: on silicon they become SRAM macros via
  # bpu_sram_1r1w, so only the logic around them is counted here.
  yosys -q -m slang -l "$out/qmv_slice_$name.log" -p "
    read_slang -DSYNTHESIS --top bpu_qmv_slice ${cfg[$name]} ${files[*]}
    synth -top bpu_qmv_slice -flatten -run :fine
    memory -nomap
    opt -full
    techmap; opt -fast
    abc -g AND,NAND,OR,NOR,XOR,XNOR,MUX
    opt_clean
    stat
    tee -o $out/qmv_slice_$name.stat stat
  "
  grep -E "[0-9]+ cells$|DFF|\\\$mem" "$out/qmv_slice_$name.stat" | head -16 || true
done
