#!/usr/bin/env bash
# Generic (technology-independent) Yosys synthesis of the compute blocks.
# Proves the RTL stays inside what the open ASIC flow accepts and gives
# first-order cell counts. SRAMs stay as $mem cells (on silicon they become
# macros behind the bpu_sram_* wrappers), so only logic is counted.
# Usage: compute/scripts/synth_yosys.sh [target ...]   (default: every target)
# Targets: slice-{asic,tiny,fpga} array-{asic,tiny} sfu fvu-{asic,tiny} top-{asic,tiny}
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl="$here/../rtl"
cd "$rtl"
mapfile -t files < <(grep -v '^//' compute.f | grep -v '^[[:space:]]*$')
out="${BPU_SYNTH_OUT:-$here/../synth_out}"
mkdir -p "$out"

# Keep in sync with compute/model/bpuref/configs.py
declare -A top params
top[slice-fpga]=bpu_qmv_slice;  params[slice-fpga]="-GLanes=64 -GRowInterleave=4 -GMaxK=6144"
top[slice-asic]=bpu_qmv_slice;  params[slice-asic]="-GLanes=16 -GRowInterleave=1 -GMaxK=2048 -GTreeRegEvery=0 -GMulPipe=3'b010 -GAddPipe=3'b010"
top[slice-tiny]=bpu_qmv_slice;  params[slice-tiny]="-GLanes=4 -GRowInterleave=2 -GMaxK=256 -GProdReg=0 -GTreeRegEvery=1 -GI2fReg=0 -GMulPipe=3'b000 -GAddPipe=3'b000"
top[array-asic]=bpu_qmv_array;  params[array-asic]="-GNSlice=1 ${params[slice-asic]}"
top[array-tiny]=bpu_qmv_array;  params[array-tiny]="-GNSlice=3 ${params[slice-tiny]}"
top[sfu]=bpu_sfu;               params[sfu]="-GPipeMask=5'b01010"
top[fvu-asic]=bpu_fvu;          params[fvu-asic]="-GVLanes=2 -GSpmWords=1024 -GMulPipe=3'b010 -GAddPipe=3'b010 -GSfuPipe=5'b01010"
top[fvu-tiny]=bpu_fvu;          params[fvu-tiny]="-GVLanes=4 -GSpmWords=512 -GMulPipe=3'b000 -GAddPipe=3'b000 -GSfuPipe=5'b00000"
top[top-asic]=bpu_compute_top;  params[top-asic]="-GMaxK=256 -GSpmWords=1024"
top[top-tiny]=bpu_compute_top;  params[top-tiny]="-GNSlice=3 -GLanes=4 -GRowInterleave=2 -GMaxK=256 -GVLanes=4 -GSpmWords=512 -GQProdReg=0 -GQTreeRegEvery=1 -GQI2fReg=0 -GQMulPipe=3'b000 -GQAddPipe=3'b000 -GFMulPipe=3'b000 -GFAddPipe=3'b000 -GFSfuPipe=5'b00000"

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(slice-asic array-asic sfu fvu-asic top-asic slice-tiny fvu-tiny)

printf "%-12s %-16s %10s %8s %6s\n" target module cells flops mems
for t in "${targets[@]}"; do
  yosys -q -m slang -l "$out/$t.log" -p "
    read_slang -DSYNTHESIS --top ${top[$t]} ${params[$t]} ${files[*]}
    synth -top ${top[$t]} -flatten -run :fine
    memory -nomap
    opt -full
    techmap; opt -fast
    abc -g AND,NAND,OR,NOR,XOR,XNOR,MUX
    opt_clean
    tee -q -o $out/$t.stat stat
  " >/dev/null
  cells=$(grep -E '^ +[0-9]+ cells$' "$out/$t.stat" | awk '{print $1}')
  flops=$(grep -E 'DFF' "$out/$t.stat" | awk '{s+=$1} END {print s+0}')
  mems=$(grep -E '\$mem' "$out/$t.stat" | awk '{s+=$1} END {print s+0}')
  printf "%-12s %-16s %10s %8s %6s\n" "$t" "${top[$t]}" "$cells" "$flops" "$mems"
done
