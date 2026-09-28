#!/usr/bin/env bash
# Pre-layout sky130 estimate: map the asic configuration onto the sky130_fd_sc_hd
# standard cells (typical corner) and report cell area and ABC's critical-path
# delay of the mapped logic. No placement, wires or clock tree: a lower bound on
# area and an optimistic delay, useful for sizing a tapeout, not for sign-off.
#
# Needs the liberty file (Apache-2.0 SkyWater PDK data), e.g. from
#   https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/platforms/sky130hd/lib
# Usage: SKY130_LIB=path/to/sky130_fd_sc_hd__tt_025C_1v80.lib compute/scripts/synth_sky130.sh [target ...]
set -euo pipefail

lib="${SKY130_LIB:?set SKY130_LIB to sky130_fd_sc_hd__tt_025C_1v80.lib}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl="$here/../rtl"
cd "$rtl"
mapfile -t files < <(grep -v '^//' compute.f | grep -v '^[[:space:]]*$')
out="${BPU_SYNTH_OUT:-$here/../synth_out}/sky130"
mkdir -p "$out"

declare -A top params extra
top[slice]=bpu_qmv_slice;  params[slice]="-GLanes=16 -GRowInterleave=1 -GMaxK=2048 -GTreeRegEvery=0 -GMulPipe=3'b010 -GAddPipe=3'b010"
top[sfu]=bpu_sfu;          params[sfu]="-GPipeMask=5'b01010"
top[fvu]=bpu_fvu;          params[fvu]="-GVLanes=2 -GSfuLanes=1 -GSpmWords=1024 -GMulPipe=3'b010 -GAddPipe=3'b010 -GSfuPipe=5'b01010"
top[top]=bpu_compute_top;  params[top]="-GMaxK=256 -GSpmWords=1024 -GFSfuLanes=1"
top[tt]=tt_um_bpu_fp32;    params[tt]=""; extra[tt]="../tapeout/tt/src/tt_um_bpu_fp32.sv"   # Tiny Tapeout run

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(slice sfu fvu top)

printf "%-6s %-16s %14s %10s %12s\n" target module "area (um^2)" cells "delay (ps)"
for t in "${targets[@]}"; do
  yosys -l "$out/$t.log" -p "
    read_slang -DSYNTHESIS --top ${top[$t]} ${params[$t]} ${files[*]} ${extra[$t]:-}
    synth -top ${top[$t]} -flatten -run :fine
    memory -nomap
    opt -full
    techmap; opt -fast
    dfflibmap -liberty $lib
    abc -liberty $lib -script +strash;dch;map;topo;stime,-p
    opt_clean
    tee -q -o $out/$t.stat stat -liberty $lib
  " >/dev/null
  area=$(awk '/Chip area/ {print $NF}' "$out/$t.stat")
  cells=$(awk '/ cells$/ {print $1; exit}' "$out/$t.stat")
  delay=$(grep -oE 'Delay = +[0-9.]+ ps' "$out/$t.log" | awk '{print $3}' | sort -n | tail -1 || true)
  printf "%-6s %-16s %14s %10s %12s\n" "$t" "${top[$t]}" "$area" "$cells" "$delay"
done
echo "(SRAMs excluded: they remain \$mem cells here and become macros on silicon.)"
