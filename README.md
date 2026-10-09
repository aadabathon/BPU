# BPU: Badger Processing Unit

SiliconBadgers' inference accelerator for **Qwen3.5-2B**: a hybrid of Gated
DeltaNet and full-attention layers, prototyped on AWS F2 (VU47P + 16 GiB HBM),
with a smaller silicon tapeout planned.

| Path | Contents |
|---|---|
| [`meeting_docs/`](meeting_docs/) | the team's technical report (rev 2), architecture notes, research links |
| [`proposed_hardware_module_organization.txt`](proposed_hardware_module_organization.txt) | proposed compute-subsystem hierarchy |
| [`compute/`](compute/README.md) | **compute + shared memory**: matrix unit (QMV), vector unit (FVU + SFU), banked shared SRAM, command sequencer + scoreboard (`bpu_core`). Bit-exact RTL, reference model, tests, proofs, docs |
| [`.github/workflows/compute.yml`](.github/workflows/compute.yml) | CI for `compute/` |

## Where the compute work stands

A full decode of a tiny model with Qwen3.5's exact structure runs on `bpu_core`
as 443 tagged descriptors, **bit-exact** against the Python reference, at the F2
configuration (16 × 64-lane QMV slices, one per HBM interface in the block
diagram; 16-lane FVU; 32-bank 8 MiB shared SRAM), the tapeout configuration
(1 × 16-lane slice, 2-lane FVU) and a tiny regression configuration. The
reference matches Hugging Face's Qwen3.5 to ~2e-7. Details: [compute/docs/verification.md](compute/docs/verification.md).

### How `compute/` maps onto the proposed module organization

| Proposed | Implemented as |
|---|---|
| `matrix_engine` (tile controller, weight unpack + scale, MAC array, partial sums, output writer) | `bpu_qmv_engine` → `bpu_qmv_array` → `bpu_qmv_slice` (W4/W8 × A8 dot, bf16 group scales, fp32 accumulate, argmax) |
| `vector_engine` (elementwise, reductions + RMSNorm, SiLU, RoPE, gather) | `bpu_fvu` + `bpu_sfu` |
| `attention_engine`, `deltanet_engine` | sequences of FVU operations (`bpuref.qwen.Compiler`); dedicated hardware only where [performance.md](compute/docs/performance.md) shows it pays |
| `compute_dispatch` | `bpu_cmd_seq` + `bpu_scoreboard` (tagged descriptors, per-unit queues, wait-mask dependencies, completion per tag) |
| local tensor interface | `bpu_sram_shared` (banked, request/response, round-robin per bank), the memory manager's SRAM ports and commands, and the matrix unit's `wreq_*` + per-slice weight streams |

The serial `bpu_compute_top` (private scratchpad, one operation at a time) was
replaced by `bpu_core` in the shared-SRAM rework; it remains in git history.
[`NOTES_FOR_AGENTS_README/BPU_RTL_Reuse_Review.md`](NOTES_FOR_AGENTS_README/BPU_RTL_Reuse_Review.md)
reviews that older top; [compute/docs/review-response.md](compute/docs/review-response.md)
lists what changed in response.
