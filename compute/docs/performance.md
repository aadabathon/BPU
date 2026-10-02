# Performance: measured, modelled, projected

## Measured (RTL simulation)

`tb/cocotb_core.py` runs a full decode step of the tiny same-structure Qwen3.5
(hidden 128, 3 DeltaNet + 1 attention layer, vocab 512) on `bpu_core`, bit-exact.
The step is 443 tagged descriptors, 32 of them matrix commands and 112 with
cross-unit waits. Cycles from the first descriptor to the last completion (position 1):

| Configuration | Matrix unit | Vector unit | Shared SRAM | Cycles / token |
|---|---|---|---|---|
| fpga | 16 slices × 64 lanes | 16 lanes, 4 shared SFUs, 3 read ports, 8 slots | 32 banks, 8 MiB, RdLat 2 | 25,191 |
| asic | 1 × 16 | 2 lanes, 1 shared SFU, 1 read port | 4 banks, RdLat 1 | 171,694 |
| tiny | 3 × 4 | 4 lanes, 2 read ports, 2 slots, 2-entry write buffer | 2 banks | 152,832 |

The serial `bpu_compute_top`, with a private scratchpad and twice the fpga slices (32),
took 26,077 / 179,839 / 117,315 cycles. The fpga and asic builds are now faster
through unit overlap and the operand collector. The tiny build is the
regression corner: two banks and minimum buffering, so every credit and conflict
path stalls.

## Cycle model

`bpuref/perf.py` estimates each command from the RTL's rules, then replays the
sequencer. Per command, it takes the largest of these limits:
- operand reads per read port, with expected bank conflicts;
- collector-slot turnaround (`RdLat + 3` cycles per slot);
- the write-buffer and reduction credit loops;
- VVECMAT row recurrence (forwarded rows: latency + 2; read-back rows wait for the
  previous row's write);
- the reduction merge (1.08 cycles per word node);
- the QMV load gearbox and beats per slice.

The replay accepts descriptors in order (tag reuse, queue depth), runs each unit
in order, and honours the cross-unit waits.

| Configuration | Measured | Modelled | Error |
|---|---|---|---|
| fpga | 25,191 | 25,213 | +0.1% |
| asic | 171,694 | 171,670 | −0.0% |
| tiny | 152,832 | 151,134 | −1.1% |

The decode test fails if the model drifts more than 5% on any step at any configuration.

## Projection: Qwen3.5-2B at the fpga configuration, 250 MHz

`python -m bpuref.perf` compiles the real 2B decode step (shapes only), schedules it
into descriptors and applies the calibrated model. It assumes the memory manager
delivers one weight beat per cycle per slice and that operands fit the SRAM. The
real 2B state does not fit (roadmap, capacity), so these are compute-side bounds.
Tokens/s:

| | 128 ctx | 2K ctx | 8K ctx |
|---|---|---|---|
| **fpga: 16 slices (the diagram's 16 HBM interfaces), 16-lane FVU** | **67.9** | **36.3** | **14.6** |
| 32 slices (32 HBM pseudo-channels) | 89.3 | 41.6 | 15.3 |
| 16 slices, 32-lane FVU | 80.9 | 52.7 | 24.9 |
| 16 slices, attention K·q and pᵀV on the matrix unit (INT8 KV) | 88.3 | 70.0 | 64.3 |

Busy cycles per unit for one 2B token at the fpga configuration (units overlap, so the
total is less than the sum):

| Work | 128 ctx | 2K ctx |
|---|---|---|
| Matrix unit (all projections + LM head) | 1.88M | 1.88M |
| Vector element-wise (DeltaNet state decay/update, norms, SiLU…) | 1.08M | 1.12M |
| VVECMAT (DeltaNet state reads, attention pᵀV) | 0.78M | 2.34M |
| Reductions (attention scores) | 0.15M | 1.76M |
| **Token** | **3.68M** | **6.89M** |

**Weight bandwidth now matters.** 16 slices × 64 int4 MACs read 512 bytes of
weights per cycle, all the diagram's 16 × 256-bit HBM interfaces deliver at 250 MHz
(128 GB/s). The ~1.06 GB of W4 weights per token then cost 1.88M cycles, which caps
decode at ~130 tok/s before any vector work. The QMV engine runs at that rate. If the
weights are staged through the shared SRAM instead of streamed past it, they also
take the SRAM's write and read bandwidth.

**Shared-SRAM bandwidth.** The diagram budgets 512 B read + 512 B write per cycle. The
fpga core's clients use at most 320 B read and 192 B write: three 64-byte vector
read ports, matrix activations, the memory manager, and one write port each.
A 32-lane vector unit would double its share, so the SRAM width caps the vector
unit around 32 lanes.

**Overlap.** The sequencer runs the matrix and vector units concurrently, but the
compiled program reuses a few scratch buffers. Most commands therefore wait on
their predecessor, and overlap saves only ~5% today. Double-buffering the scratch
regions in the compiler is the cheapest next gain (roadmap).

## Area and timing

**sky130** (`scripts/synth_sky130.sh`): the asic configuration mapped onto
`sky130_fd_sc_hd` cells, typical corner, SRAMs excluded. These are pre-layout numbers:
no wires, placement or clock tree, and ABC's delay is area-oriented. ABC's results
move by a few percent between runs; read them as ±5%.

AREA_TABLE

**Generic cells** (`scripts/synth_yosys.sh`, technology-independent), asic configuration:

GENERIC_TABLE

**SFU sharing** (`SfuLanes`): the SFU is the largest per-lane cost. Sharing one SFU
across the asic configuration's two FVU lanes saves about 20% of the logic for ~1.5%
more cycles; the fpga configuration shares 4 SFUs among 16 lanes.
