#!/usr/bin/env bash
# Generic (technology-independent) Yosys synthesis of the compute blocks.
# Proves the RTL stays inside what the open ASIC flow accepts and gives
# first-order cell counts. SRAMs stay as $mem cells (on silicon they become
# macros behind the bpu_sram_* wrappers), so only logic is counted.
# Usage: compute/scripts/synth_yosys.sh [target ...]   (default: every target)
# Targets: slice-{asic,tiny,fpga} array-{asic,tiny} sfu fvu-{asic,tiny,fpga} sram-asic seq core-{asic,tiny}
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl="$here/../rtl"
cd "$rtl"
mapfile -t files < <(grep -v '^//' compute.f | grep -v '^[[:space:]]*$')
out="${BPU_SYNTH_OUT:-$here/../synth_out}"
mkdir -p "$out"

# Named configurations come from compute/model/bpuref/configs.py.
cfg() { (cd "$here/../model" && python3 -W ignore -m bpuref.configs --params "$1" "$2"); }
declare -A top params
for c in fpga asic tiny; do top[slice-$c]=bpu_qmv_slice; params[slice-$c]="$(cfg slice $c)"; done
for c in asic tiny; do top[array-$c]=bpu_qmv_array; params[array-$c]="$(cfg array $c)"; done
for c in asic tiny fpga; do top[fvu-$c]=bpu_fvu; params[fvu-$c]="$(cfg fvu $c)"; done
for c in asic tiny; do top[core-$c]=bpu_core; params[core-$c]="$(cfg core $c)"; done
top[sfu]=bpu_sfu;              params[sfu]="-GPipeMask=5'b01010"
top[sram-asic]=bpu_sram_shared; params[sram-asic]="-GNBanks=4 -GBankWords=16384 -GLanes=2 -GNRd=3 -GNWr=3 -GAW=31"
top[seq]=bpu_cmd_seq;          params[seq]="-GNTags=16 -GQDepth=2"

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(slice-asic array-asic sfu fvu-asic sram-asic seq core-asic slice-tiny fvu-tiny)

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
  # (grep finds nothing for blocks without flops or memories: not an error)
  cells=$(grep -E '^ +[0-9]+ cells$' "$out/$t.stat" | awk '{print $1}' || true)
  flops=$( (grep -E 'DFF' "$out/$t.stat" || true) | awk '{s+=$1} END {print s+0}')
  mems=$( (grep -E '\$mem' "$out/$t.stat" || true) | awk '{s+=$1} END {print s+0}')
  printf "%-12s %-16s %10s %8s %6s\n" "$t" "${top[$t]}" "$cells" "$flops" "$mems"
done
