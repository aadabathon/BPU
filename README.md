# BPU: Badger Processing Unit

SiliconBadgers' inference accelerator for **Qwen3.5-2B**: a hybrid of Gated
DeltaNet and full-attention layers, prototyped on AWS F2 (VU47P + 16 GiB HBM),
with a smaller silicon tapeout planned.

| Path | Contents |
|---|---|
| [`meeting_docs/`](meeting_docs/) | the team's technical report (rev 2), architecture notes, research links |
| [`proposed_hardware_module_organization.txt`](proposed_hardware_module_organization.txt) | proposed compute-subsystem hierarchy |
| [`compute/`](compute/README.md) | **compute engines**: QMV, FVU + SFU, compute top. Bit-exact RTL, reference model, tests, proofs, docs |
| [`.github/workflows/compute.yml`](.github/workflows/compute.yml) | CI for `compute/` |

## Where the compute work stands

A full decode of a tiny model with Qwen3.5's exact structure runs on the RTL
compute top **bit-exact** against the Python reference, at the F2 configuration
(32 × 64-lane QMV slices, 16-lane FVU) and the tapeout configuration
(1 × 16-lane slice, 2-lane FVU). The reference matches Hugging Face's Qwen3.5 to
~2e-7. Details: [compute/docs/verification.md](compute/docs/verification.md).

### How `compute/` maps onto the proposed module organization

| Proposed | Implemented as |
|---|---|
| `matrix_engine` (tile controller, weight unpack + scale, MAC array, partial sums, output writer) | `bpu_qmv_array` / `bpu_qmv_slice` (W4/W8 × A8 dot, bf16 group scales, fp32 accumulate, argmax) |
| `vector_engine` (elementwise, reductions + RMSNorm, SiLU, RoPE, gather) | `bpu_fvu` + `bpu_sfu` |
| `attention_engine`, `deltanet_engine` | sequences of FVU operations (`bpuref.qwen.Compiler`); dedicated hardware only where [performance.md](compute/docs/performance.md) shows it pays |
| `compute_dispatch` | `bpu_compute_top` (operation stream, QMV ↔ SPM transfers, weight requests) |
| local tensor interface | the SPM host port and the `wreq_*` + per-slice weight streams |
