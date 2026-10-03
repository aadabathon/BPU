#!/usr/bin/env bash
# Assemble a Tiny Tapeout submission tree for tt_um_bpu_fp32: info.yaml, docs/,
# and src/ with the BPU RTL it needs (copied, so the tile always matches the
# verified sources). Usage: compute/tapeout/tt/assemble.sh <output-dir>
set -euo pipefail

out="${1:?usage: assemble.sh <output-dir>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl="$here/../../rtl"

mkdir -p "$out/src" "$out/docs"
cp "$here/info.yaml" "$out/"
cp "$here/docs/info.md" "$out/docs/"
cp "$rtl/common/bpu_compute_pkg.sv" "$rtl/common/bpu_delay.sv" \
   "$rtl/common/bpu_lzc_pow2.sv" "$rtl/common/bpu_lzc.sv" \
   "$rtl/fp/bpu_fp32_mul.sv" "$rtl/fp/bpu_fp32_add.sv" \
   "$here/src/tt_um_bpu_fp32.sv" "$out/src/"
echo "Tiny Tapeout tree in $out (rename the top module with your GitHub username before submitting)"
