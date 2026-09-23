# 01 -- GDN short convolution, decode path

`_causal_conv1d_update_kernel`, the width-4 causal convolution in front of the
30 linear-attention layers. Qwen3.5-35B-A3B, B200, TP2, ISL 60000 / OSL 400,
bs 1, **DFLASH draft block 7** (`--speculative-num-draft-tokens 7`).

---

## 0. This is a re-measurement at block 7. What it supersedes, and why

Every number in the previous revision of this file was taken at draft block 14.
The customer runs block 7. The gain of this change is the removal of a
loop-carried dependence along the token axis, so it scales with the draft-token
count: at 7 it is between a quarter and a third of what it is at 14, measured
both ways in section 7. The block-14 figures are marked **SUPERSEDED** in
section 7 rather than deleted, so the two can be compared; none of them is
carried into a headline here.

| | block 14 (superseded) | block 7 (this revision) |
|---|---:|---:|
| kernel, in-engine, `BLOCK_N=64`, TP0 | 6.688 -> 4.000 us/call | **3.872 -> 3.136 us/call** |
| kernel, in-engine, per step | -80.8 us/step | **-21.8 us/step** |
| end to end, paired arms | -81.2 us/step from one pair | **-19.7 us/step from four pairs; one pair cannot see it** |

At block 7 the kernel-level result is solid and reproduces on both ranks and in
the standalone bench. The end-to-end result is at the edge of what this harness
can resolve: the effect, 19.7 us of a 4437 us step, is the same size as the
spread between control arms run on the same node in the same allocation, so a
single control/variant pair is a coin flip. It takes four replicates per side --
where all four variant arms land below all four control arms, p = 0.014 -- to
see it at all. Section 4 says so plainly rather than arguing around it.

---

## 1. The change

Unchanged from the previous revision; restated so this file stands on its own.

`_causal_conv1d_update_kernel` has two bodies, selected by
`HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`. The engine passes `retrieve_next_token=None`
on this path -- confirmed again at block 7, in-engine, by the launcher probe in
section 6 (`retrieve_next_token_is_None=True`), by the 28 registers per thread
in the control trace, and by the device-side branch counter in section 5b
(`tree_branch=0`) -- so the **non-tree** body is the one that runs.

That body walked the draft tokens one at a time, rotating `KERNEL_WIDTH - 1`
`[BLOCK_N]` register vectors per iteration (`col0 <- col1 <- col2 <- x[t]`), so
token `t` could not start until `t-1` had finished rotating, and each iteration
issued its own output store and window stores.

A width-`KW` causal convolution is a fixed `KW`-tap FIR,

    out[t] = sum_j src(t - (KW-1) + j) * w_col[j]

with `src(i) = x[i]` for `i >= 0` and conv-state column `i + KW - 1` otherwise.
Every output is independent; the rotation is a software pipeline, not a
recurrence. The body now computes the chunk as one `[NP2_SEQLEN, BLOCK_N]` tile:
`KW` masked gathers, `KW` FMAs, one tile store per destination. The conv-state
columns come from the registers STEP 1 already holds, so `conv_state` is not
read twice. The tree body is untouched and only moves one level deeper under an
explicit `if`.

Accumulation stays in `j` order against `w_col0..w_col{KW-1}`, the order the
previous body used, which is what makes the result bit-identical rather than
merely close.

---

## 2. us/call as a function of T -- the durable result

A single number at one block length stops being true the moment anyone moves
`--speculative-num-draft-tokens`. So the standalone measurement is a sweep and
the reported result is a decomposition:

    us/call(T) = fixed + slope * T

`fixed` is launch, prologue and the conv-state roll -- work that does not depend
on the draft-token count. `slope` is what this change attacks.

`T = 16` is `dflash_config.block_size`, the draft model's trained upper bound.
Nothing above it was measured because nothing above it is reachable.

### Setup

B200, 148 SMs, one rank's shard: batch 1, `dim` 4096, width 4, `state_len` 3,
non-tree path, `SAVE_INTERMEDIATE=True`, `BLOCK_N = 64` (the deployed value; see
section 6). CUDA-graph replay, 50 launches per graph, 30 replays, median of the
30. Contention probe before and after: 221.9 and 221.8 GFLOP/s/SM against a
clean-node reference of ~222 (a contended read is ~108).

Every argument is one the engine actually passed, not a reconstruction:
`r2/gdn_conv7/env/patch_conv1d_dump.py` wrote `x`, `conv_state`, `weight`, the
two slot-index vectors and the strides of all of them, plus the intermediate
window's shape and strides, from inside a live block-7 decode call. Two things
the block-14 round had wrong are fixed by that: the window's geometry at this
block, and the fact that **this convolution has no bias** (`bias=None` in the
dump) -- the block-14 bench compiled `HAS_BIAS=True` against a synthetic one.

For `T != 7` the token axis of the captured activation is cycled (`T > 7`) or
sliced (`T < 7`). The non-tree body has no data-dependent control flow, so the
values do not change the timing; at `T = 7` the activation is used unmodified,
and that is where the bit-equality gate of section 3 runs.

### Measured, `BLOCK_N = 64`

| T | serial (replaced) | tiled (shipped) | change | |
|---:|---:|---:|---:|---:|
| 4 | 2.001 | **1.884** | -0.116 | -5.8% |
| **7 (customer)** | **2.937** | **2.368** | **-0.569** | **-19.4%** |
| 8 | 3.144 | **2.327** | -0.817 | -26.0% |
| 12 | 4.125 | **2.701** | -1.423 | -34.5% |
| 16 (`block_size`, trained bound) | 5.070 | **2.654** | -2.416 | -47.7% |

### Fitted

| body | fixed us | slope us/token | R^2 | max abs residual |
|---|---:|---:|---:|---:|
| serial (replaced) | **1.091** | **0.2515** | 0.9963 | 0.096 |
| tiled (shipped) | **1.802** | **0.0622** | 0.7907 | 0.167 |

**The slope falls 4.04x, 0.2515 -> 0.0622 us per draft token. The intercept
rises 0.711 us, 1.091 -> 1.802.** That is the whole of this optimization in two
numbers: it buys per-token cost and pays a fixed cost for it. Registers per
thread go 28 -> 48, which is where the fixed cost went.

Two things follow from the two lines, and they are worth more than any single
measurement:

- **Break-even at T = 3.76.** Below about four draft tokens the tiled body is
  *slower* than the one it replaces. At the customer's 7 it is 0.569 us/call
  ahead; at the trained bound 16, 2.416 us/call ahead.
- **The serial fit is a straight line (R^2 0.9963); the tiled fit is not
  (R^2 0.7907).** That is structure, not noise: the tile is
  `NP2_SEQLEN = next_power_of_2(T)` rows deep, so the tiled cost is a staircase
  in T. `T = 7` and `T = 8` both compile an 8-row tile and measure 2.368 and
  2.327 -- the 8-token point is the *faster* of the two, 0.041 us apart, which is
  this measurement's own resolution. `T = 12` and `T = 16` both compile a 16-row
  tile and measure 2.701 and 2.654. Refitting the tiled body against
  `NP2_SEQLEN` instead of T gives `1.785 + 0.0578 * NP2`, R^2 0.90.
  **Read practically: block 7 already pays for a tile of 8, so a deployment at
  block 8 would get the eighth draft token for nothing.**

### Reproducibility, and the same harness at block 14

The sweep was run twice, in two independent sessions, the second with `T = 14`
added so the superseded operating point sits on the same footing
(`standalone_block7.json` and `standalone_block7_session2.json`):

| T | serial s1 | serial s2 | tiled s1 | tiled s2 |
|---:|---:|---:|---:|---:|
| 4 | 2.001 | 2.004 | 1.884 | 1.891 |
| 7 | 2.937 | 2.938 | 2.368 | 2.369 |
| 8 | 3.144 | 3.143 | 2.327 | 2.330 |
| 12 | 4.125 | 4.131 | 2.701 | 2.694 |
| 14 | -- | 4.636 | -- | 2.816 |
| 16 | 5.070 | 5.076 | 2.654 | 2.657 |

Every point reproduces to within 0.007 us, so the 0.041 us gap between the T=7
and T=8 tiled points is a real ordering and not drift. Session 2's fits are
`1.087 + 0.2526 * T` (serial, R^2 0.9971) and `1.779 + 0.0670 * T` (tiled,
R^2 0.8259); the tiled slope moves because adding T=14 adds a second point on
the 16-row step of the staircase.

At `T = 14` this harness measures **4.636 -> 2.816, -1.820 us/call**, against the
superseded block-14 round's 4.927 -> 2.747, -2.180. The two bracket each other
to within 6% on the serial body and 2.5% on the tiled one -- the residual is the
block-14 bench's synthetic bias, which the engine does not pass. That is close
enough to say the two rounds measured the same kernel, and it is why the block-14
numbers are marked superseded for being at the wrong T rather than for being
wrong.

### At the in-tree default `BLOCK_N = 256`, which this change does not touch

| T | serial | tiled | change |
|---:|---:|---:|---:|
| 4 | 2.862 | 2.615 | -0.247 |
| **7** | **4.831** | **3.061** | **-1.770** |
| 8 | 5.476 | 2.909 | -2.567 |
| 12 | 9.092 | 3.238 | -5.854 |
| 16 | 12.902 | 3.587 | -9.315 |

Fits: serial `-0.953 + 0.8495 * T` (R^2 0.9933), tiled `2.375 + 0.0752 * T`
(R^2 0.9330). The serial body's slope is 3.4x larger here than at
`BLOCK_N = 64`, because each loop iteration is four times wider; the negative
fitted intercept is the fit telling us it is in fact slightly superlinear over
this range, with per-token cost rising from 0.65 us between T=4 and T=8 to
0.95 us between T=12 and T=16. The tiled body's slope is essentially unchanged
by `BLOCK_N`, which is what a body with no loop-carried state should do.

**On an unpatched tree the change is worth -1.770 us/call at block 7**, three
times what it is worth at the deployed `BLOCK_N = 64`.

### Where the floor is

Bytes moved per call at block 7, `dim` 4096, bf16:

    read  x            4096 * 7 * 2      =  57344
    read  weight       4096 * 4 * 2      =  32768
    read  conv_state   4096 * 3 * 2      =  24576
    write conv_state   4096 * 3 * 2      =  24576
    write out          4096 * 7 * 2      =  57344
    write window       7 * 4096 * 3 * 2  = 172032
    total                                = 368640 B = 0.369 MB

At 8.0 TB/s that is **0.046 us**. The kernel measures 2.368 us standalone and
3.136 us in-engine. **This kernel is not moving data and it is not computing; it
is launch and prologue**, and the tiled body has already taken the per-token
term down to 0.062 us/token. What is left is the 1.80 us fixed term, and the
only way to remove a fixed per-launch cost is to stop launching -- which is the
three-kernel merge sized in section 8, not this change.

---

## 3. Bit-exactness at block 7

Gate is **bit equality**, zero differing bits, not a tolerance band. On a
speculative-decode path a numerics change moves the accepted tokens and
therefore acceptance, which makes a TPOT comparison non-causal. The tiled body
performs the same four bf16 products and accumulates them into fp32 in the same
`j` order, so anything short of bit equality would be a real semantic
difference.

Run by loading the container's pristine `causal_conv1d_triton` and the same file
with `patch_conv1d_inplace.rewrite` applied side by side in one process, and
launching both at `BLOCK_N = 64` on the tensors the engine handed the launcher
at block 7: real activations, the real convolution weight, the real conv-state
pool, the real slot indices, no bias.

| output | differing elements | of | region | RMS of the reference |
|---|---:|---:|---|---:|
| `out` | **0** | 28672 | the whole output, `1 x 4096 x 7` | 0.333 |
| `conv_state` | **0** | 12288 | the one cache line this call writes (slot 7) | 2.543 |
| intermediate window | **0** | 36864 | the one window slot this call writes (slot 0) | 2.414 |
| `conv_state`, whole pool | **0** | 19451904 | all 1583 cache lines | |
| intermediate window, whole buffer | **0** | 1806336 | all 49 slots | |

Max absolute difference on `out` is 0.0. The written regions are quoted
separately from the whole buffers because the engine hands the kernel a pool and
one call writes one slot of it; a denominator of 19.4 M would be a weaker claim
than it looks. Registers per thread 28 -> 48.

The end-to-end consequence is in section 4: **acceptance reads 4.89 in every
arm**, control and variant alike, to two decimals.

---

## 4. End to end at block 7 -- and what it does not show

Eight arms, control and variant adjacently interleaved, one node, one
allocation, **unprofiled**. `BLOCK_N = 64` in every arm, set by the same
pre-existing deployment patch on both sides, so the only difference within an
arm pair is the kernel body. Each arm's resolved configuration is in section 6,
read out of the server process rather than assumed.

| arm | TPOT ms | TTFT ms | accept | step us = TPOT x accept |
|---|---:|---:|---:|---:|
| `cw_ctl_a` control | 0.9085 | 500.1 | 4.89 | 4442.6 |
| `cw_var_a` variant | **0.9039** | 500.7 | 4.89 | **4420.1** |
| `cw_ctl_b` control | 0.9046 | 503.1 | 4.89 | 4423.5 |
| `cw_var_b` variant | **0.9041** | 500.1 | 4.89 | **4421.1** |
| `cw_ctl_c` control | 0.9090 | 499.1 | 4.89 | 4445.0 |
| `cw_var_c` variant | **0.9026** | 500.2 | 4.89 | **4413.7** |
| `cw_ctl_d` control | 0.9073 | 501.8 | 4.89 | 4436.7 |
| `cw_var_d` variant | **0.9027** | 501.6 | 4.89 | **4414.2** |

Per replicate, with the control-to-control spread next to it:

    pair A            0.9085 -> 0.9039   -0.0046 ms/token   -22.5 us/step
    pair B            0.9046 -> 0.9041   -0.0005 ms/token    -2.4 us/step
    pair C            0.9090 -> 0.9026   -0.0064 ms/token   -31.3 us/step
    pair D            0.9073 -> 0.9027   -0.0046 ms/token   -22.5 us/step

    control mean      0.90735   sd 0.00197   spread 0.0044 ms/token = 21.5 us/step
    variant mean      0.90333   sd 0.00079   spread 0.0015 ms/token =  7.3 us/step
    difference of means         -0.00403 ms/token   4436.9 -> 4417.3 us/step
                                                    **-19.7 us/step  (-0.44%)**

**A single pair cannot resolve this.** The effect, -19.7 us/step, is smaller
than the spread between the extreme control arms, 21.5 us/step, and pair B
measured -2.4 us/step -- essentially nothing. Anyone running one control and one
variant at block 7 would have a coin flip.

**Four pairs can, and only just.** All four variant arms fall below all four
control arms -- the largest variant, 0.9041, is below the smallest control,
0.9046. Under an exact permutation test over the eight arms that is one of 70
assignments, **p = 0.014** one-sided; the paired t over the four deltas gives
t = 3.22 on 3 degrees of freedom, p = 0.05 two-sided. The variant arms are also
2.5x tighter than the control arms (sd 0.00079 against 0.00197), which is what a
faster arm with a shorter critical path should look like and not what a
measurement artefact looks like.

The two profiled arms corroborate from a fifth, independent pair: their
unprofiled phases read 0.9081 (control) and 0.9020 (variant), a delta of
-0.0061 ms/token, and adding them keeps the separation complete across five
pairs (p = 1/252).

The three measurements of the same quantity agree on size:

    standalone bench, -0.569 us/call x 30 calls/step            -17.1 us/step
    profiled trace, kernel exclusive and critical path          -21.6 us/step
    unprofiled arms, difference of four-arm means               -19.7 us/step

**So the honest statement is: at block 7 this change is worth about 20 us of a
4437 us step, 0.44%. That is real -- it reproduces in the kernel trace on both
ranks and survives a four-replicate end-to-end test -- and it is the same size as
this harness's own arm-to-arm variation, so it is at the edge of what the
end-to-end measurement can see. At block 14 the same change moved the step by
-81.2 us against a control spread of 28.2 us and needed no statistics at all.**

TTFT is unchanged across every arm, as expected: prefill calls
`causal_conv1d_fn`, which this change does not touch.

Acceptance is 4.89 in every arm. That matches the block-7 baseline on record for
this harness and is what makes `TPOT x accept` a legitimate conversion here; it
is also the end-to-end consequence of having gated the kernel on bit equality
rather than on a tolerance, because a numerically equivalent but non-identical
kernel would move acceptance and make TPOT non-causal.

---

## 5. The kernel itself -- profiled pair

A separate profiled pair, for the kernel's own us/call, its call count and the
step's launch count. The end-to-end magnitude stays with the unprofiled arms
above; the profiled step period is not a valid A/B basis on this harness and
section 5a says why.

Steady-state decode window, 88 steps. The step count is cross-checked against a
marker this optimization does not touch: `gdn_decode_bf16state`, the GDN
recurrence kernel, also fires 30 times per target forward and also gives 88.0
steps, so the normalisation does not assume the thing being measured.

The shipped form is an in-place edit, so the kernel keeps its name in both arms.
Control and variant are separate runs with separate traces; the rows are told
apart by which trace they came from.

| | TP0 control | TP0 variant | TP1 control | TP1 variant |
|---|---:|---:|---:|---:|
| conv kernel | `_causal_conv1d_update_kernel` | same | same | same |
| **calls/step** | **30.00** | **30.00** | **30.00** | **30.00** |
| med us/call | 3.872 | **3.136** | 3.904 | **3.168** |
| us/step | 116.9 | **95.1** | 116.8 | **95.4** |
| crit us/step | 115.2 | **93.5** | 115.4 | **93.9** |
| `fused_qkvzba_split_reshape_cat_contiguous` us/step | 87.5 | 87.4 | 82.6 | 83.0 |
| `fused_qkv_split_gdn_prefill_kernel` us/step | 39.9 | 40.0 | 40.2 | 40.5 |
| **launches/step** | **1246.1** | **1246.1** | **1247.1** | **1247.1** |

Four things this settles.

1. **The kernel ran, 30.00 times per step, on the device, on both ranks, in both
   arms.** This counts real GPU launches in the trace, not installations, and
   the step it is divided by comes from a different kernel.
2. **Launches per step are identical.** The tap pattern is affine in the token
   index, so there is nothing to precompute and no second kernel to charge.
3. **The change is local.** Both stream neighbours are flat to within 0.5%.
4. **The critical-path contribution falls by the same amount as the exclusive
   time.** With `crit(S) = union(all) - union(all\S)` computed inside each arm,
   -21.7 us/step on TP0 and -21.5 on TP1, against exclusive-time deltas of -21.8
   and -21.4. They agree to 0.5%, which is what one expects from a kernel with
   essentially no co-resident work: its time is the step's time.

### 5a. Why the arm totals are not quoted, and what reconciles instead

The kernel block is supposed to reconcile against the busy or union delta. On
this pair it does not, and the reason is measurable rather than rhetorical:

| per step | TP0 control -> variant | TP1 control -> variant |
|---|---:|---:|
| busy | 6228.0 -> 6361.3 (**+133.3**) | 6345.6 -> 6135.8 (**-209.7**) |
| union | 4539.2 -> 4677.4 (**+138.2**) | 4682.2 -> 4467.4 (**-214.7**) |
| profiled step period | 5546.0 -> 5147.2 (-398.8) | 5545.8 -> 5147.9 (-398.0) |

The busy and union deltas **disagree in sign between the two ranks** and are six
to ten times larger than the 21.6 us the kernel moved. They are measuring the
profiler's own per-arm overhead, not the change. The profiled step period moves
-399 us on both ranks, 18x the kernel's saving and 20x the unprofiled arms'
figure -- the known arm-dependence of the profiled step period on this harness.

What does reconcile at this magnitude is the per-kernel critical-path term,
because it is a difference taken *within* one arm and so cancels that arm's
offset:

    kernel exclusive time, TP0            116.9 -> 95.1    -21.8 us/step
    kernel critical path,  TP0            115.2 -> 93.5    -21.7 us/step
    kernel exclusive time, TP1            116.8 -> 95.4    -21.4 us/step
    kernel critical path,  TP1            115.4 -> 93.9    -21.5 us/step

### 5b. Device-side invocation count

Counted on the device, in a separate untimed arm, because an installation count
is not evidence that a kernel ran, and a host-side wrapper count is worthless
under CUDA-graph replay, where the Python launcher does not execute at all.

`r2/gdn_conv7/env/patch_conv1d_count.py` adds one pointer argument and, from one
CTA per launch, atomically records the launch, the `seqlen` it was compiled for
(a `tl.constexpr`, so each block length is its own specialisation and its own
slot) and which of the two bodies was taken. Last reading of each rank, variant
build, block 7:

    CONV1D_COV rank=0 host_calls=1800 device_launches=1800 seqlen_hist={7: 1800} tree_branch=0 tile_branch=1800
    CONV1D_COV rank=1 host_calls=1800 device_launches=1800 seqlen_hist={7: 1800} tree_branch=0 tile_branch=1800

Read across all six samples (300, 600, ... 1800 on each rank): the device launch
count equals the host call count exactly at every sample, every launch was
compiled for `seqlen = 7`, and **every launch took the rewritten non-tree body;
the eagle-tree body was entered zero times.**

What this arm cannot do on its own is give a per-step figure: dividing 1800 by
30 to get 60 forwards assumes the very thing 30/step asserts. The per-step count
comes from the profiled trace above, where it is normalised by a different
kernel. The counter's job is the part a trace cannot state -- which branch ran,
at which `seqlen`, and whether every host call actually reached the device.

This build is never timed. One `atomic_add` per launch cost ~0.5 us/call on this
kernel in the block-14 round, a fifth of its total. The arm also runs with
`--disable-cuda-graph`, because reading a device counter from inside a
CUDA-graph capture aborts the server with `cudaErrorStreamCaptureInvalidated`;
with graphs off there is no capture to land in.

---

## 6. The configuration each arm resolved

A previous round of this work compared a control and a variant that differed in
a tuning knob as well as in the kernel, and the arm logs did not say so. Every
arm here carries `patch_conv1d_probe.py`, which reports, from inside the server
process at the launcher, what that arm actually resolved. Raw log:
`r2/gdn_conv7/out/arms_b7.log` and `r2/gdn_conv7/out/arms_b7_2.log`.

| arm | block arg | `BLOCK_N` | tile branch | accept | probe before / after |
|---|---:|---|---|---:|---|
| `cw_ctl_a` | 7 | 256 -> 64 | absent | 4.89 | 221.9 / 222.0 |
| `cw_var_a` | 7 | 256 -> 64 | **present** | 4.89 | 221.7 / 222.1 |
| `cw_ctl_b` | 7 | 256 -> 64 | absent | 4.89 | 222.1 / 221.8 |
| `cw_var_b` | 7 | 256 -> 64 | **present** | 4.89 | 222.0 / 221.8 |
| `cw_pctl` (profiled) | 7 | 256 -> 64 | absent | 4.89 | 221.7 / 221.6 |
| `cw_pvar` (profiled) | 7 | 256 -> 64 | **present** | 4.89 | 221.7 / 221.8 |
| `cw_ctl_c` | 7 | 256 -> 64 | absent | 4.89 | 222.1 / 221.8 |
| `cw_var_c` | 7 | 256 -> 64 | **present** | 4.89 | 221.9 / 222.1 |
| `cw_ctl_d` | 7 | 256 -> 64 | absent | 4.89 | 222.0 / 221.7 |
| `cw_var_d` | 7 | 256 -> 64 | **present** | 4.89 | 221.9 / 221.6 |

`block arg` is the `--speculative-num-draft-tokens` the server was launched with,
read back from the arm's own `sgl_args.txt`; `BLOCK_N` is what the deployment
patch reported plus what the probe re-read from the imported file at the first
real call; `tile branch` is the probe's report of whether the rewritten `else`
is present in the file the server imported. Every arm also passed the harness's
own validity gate (`GATE PASS`, acceptance >= 2.0).

Every arm ran on the same node in the same allocation. All 24 contention-probe
readings fall in 221.6-222.1 GFLOP/s/SM against a clean-node reference of ~222; a
co-resident compute kernel reads ~108. The node was shared for the duration, so
this is load-bearing rather than ceremonial.

The launcher arguments, read in-engine at block 7 on an eager arm -- the only
kind in which a Python wrapper body runs on a steady-state decode call:

    seqlen        7
    x             (1, 4096, 7)      stride (28672, 1, 4096)
    conv_state    (1583, 4096, 3)   stride (12288, 3, 1)
    window        (49, 7, 4096, 3)  stride (36864, 1, 9, 1)
    weight        (4096, 4)         stride (4, 1)
    bias          None
    retrieve_next_token   None      <- selects the non-tree body
    num_accept_tokens     None      <- state_len = width - 1 = 3
    activation    silu
    pad_slot_id   -1

The first two calls of every arm report `x` as `(48, 4096, 7)`: that is the
startup warmup forward at a padded batch, not decode. The steady-state decode
shape is the `(1, 4096, 7)` above, and the trace confirms it.

The intermediate window is a **rolling buffer**, not a dense `[step, dim, win]`
array: its step and win axes both have stride 1 and address `step + win` along a
single `seqlen + width - 2 = 9` axis. Entries that alias carry the same value,
so writing the window as a tile rather than token by token is safe -- and the
bit-equality gate in section 3 is what proves it at this block length, rather
than inheriting it from the last one.

The trace confirms the launch geometry the standalone bench was built to match:

    grid [1, 64, 1]   block [128, 1, 1]   registers per thread 28   shared 512 B

The standalone harness compiles the same body to **28** registers at the same
grid and block -- an exact match this time; the block-14 round was one register
off. In-engine the kernel reads 1.32x the standalone time on both arms
(3.872/2.937 and 3.136/2.368), so the bench is usable for the ratio and for the
shape of the T curve, and not for absolute in-engine times.

### What sets `BLOCK_N`, and what this diff does not touch

- Nothing in the tree selects it. There is no autotune decorator and no
  heuristic on this kernel:
  `grep -n "autotune\|heuristics" python/sglang/kernels/ops/mamba/causal_conv1d_triton.py`
  returns nothing. The two literals are `causal_conv1d_triton.py:564` (prefill
  launcher) and `:1278` (decode launcher), both **256**.
- The 64 in every arm comes from a pre-existing round-1 deployment patch that
  rewrites the decode literal only, invoked identically for control and variant.
  Every arm's log records `patched BLOCK_N=256 -> 64`, and the probe re-reads
  the file at the first real call and reports `BLOCK_N_literals=['256', '64']`.
- **The commit assigns no `BLOCK_N` literal.** Verified:
  `git show 09a48ff8aa -- python/sglang/kernels/ops/mamba/causal_conv1d_triton.py | grep -E "^[+-].*BLOCK_N *= *[0-9]"`
  returns nothing.

---

## 7. SUPERSEDED -- the block-14 figures

Kept so the two operating points can be compared. **None of these transfers to
block 7 and none is quoted above.**

    standalone, BLOCK_N 64,  T=14   4.927 -> 2.747 us/call   -2.180  (-44.2%)
    standalone, BLOCK_N 256, T=14  11.068 -> 3.396 us/call   -7.671  (-69.3%)
    in-engine,  BLOCK_N 64,  TP0    6.688 -> 4.000 us/call   201.1 -> 120.3 us/step  -80.8
    in-engine,  BLOCK_N 64,  TP1    6.656 -> 4.000 us/call   199.9 -> 120.3 us/step  -79.6
    end to end, one unprofiled pair  TPOT 0.8026 -> 0.7882 ms   step -81.2 us  (-1.79%)
    control-to-control spread        5.0 us/token = 28.2 us/step
    bit equality                     0/57344, 0/98304, 0/1376256, max abs 0.0
    arm set `cvj_*`

Two further block-14 arm sets, already superseded in the previous revision
because they moved `BLOCK_N` to 32 in the variant and so measured the tile plus
a knob, remain superseded:

    tile + knob, 1 pair                        -17.8 us/token   -100.4 us/step
    tile + knob, instrumented build, 4 arms    -16.3 us/token    -91.9 us/step

Why block 14 overstates it: the serial body costs 0.2515 us per draft token
against the tiled body's 0.0622, so what the change buys grows with T. Measured
in this harness at `BLOCK_N = 64`, the saving is **1.820 us/call at T=14 against
0.569 at T=7, a factor of 3.2**; measured in-engine it is 2.688 us/call at block
14 against 0.736 at block 7, a factor of 3.7. (The two factors are not the same
because of the tile staircase: T=14 sits on the 16-row step, so extrapolating
the tiled line to T=14 -- which would predict a saving of 2.65 us/call --
overstates it by 0.8 us. The staircase is why the fit, not the line, is the
result.)

The block-14 standalone numbers also carried two fidelity defects this revision
does not: `HAS_BIAS=True` against a synthetic bias where the engine passes none,
and cache layouts reconstructed from a printed probe rather than taken from the
engine's own tensors.

---

## 8. Sizing the three-kernel merge -- not implemented, reported only

Measured on this round's control trace, same 88-step window, TP0, per step:

| kernel | calls/step | med us | us/step |
|---|---:|---:|---:|
| `_causal_conv1d_update_kernel` | 30.00 | 3.872 | 116.9 |
| `fused_qkvzba_split_reshape_cat_contiguous` | 30.00 | 2.913 | 87.5 |
| `fused_qkv_split_gdn_prefill_kernel` | 30.00 | 1.375 | 39.9 |

    today, per layer   3.872 + 2.913 + 1.375   =  8.16 us
    per step           116.9 + 87.5 + 39.9     = 244.3 us
    the convolution's own fixed term (fit)     =   1.80 of its 2.37 us standalone

The convolution is 76% fixed cost at block 7 -- 1.80 of 2.37 us standalone -- and
the whole three-kernel group costs 8.16 us per layer for 0.369 MB of traffic in
its largest member. **A merge buys launches, and launches are now what this
group is made of.** The fixed fractions of the other two were not measured here,
so the size of the merge is not quoted; what is quoted is that the part this
change could remove has been removed.

That is a larger prize at block 7 than at block 14, not a smaller one: the
per-token work this change took out was the part that scaled with the block, and
what is left is the part that does not. The merge was the right follow-up at
block 14; at block 7 it is the only remaining lever on this kernel.

---

## 9. Files

In this directory:

    bench_t.py                       us/call vs T, both bodies, and the fits
    bitcheck.py                      bit equality over the written regions
    standalone_block7.json           session 1 of section 2
    standalone_block7_session2.json  session 2, with T = 14
    bitcheck_block7.json             every number in section 3
    arms_b7.log                      every arm's result and resolved configuration
    BEFORE_RUN_TAG / AFTER_RUN_TAG   run identifiers and headline numbers

    bench_inplace.py, standalone_inplace.json, arms_ship.log
                                     the SUPERSEDED block-14 round, kept for
                                     the comparison in section 7

In the experiment tree, not committed:

    r2/gdn_conv7/ana_b7.py                 profiled-pair comparison, incl. crit(S)
    r2/gdn_conv7/env/patch_conv1d_dump.py  in-engine capture of the launcher arguments
    r2/gdn_conv7/env/patch_conv1d_probe.py per-arm resolved-configuration report
    r2/gdn_conv7/env/patch_conv1d_count.py device-side invocation/seqlen/branch counter
    r2/gdn_conv7/env/queue7.sh             the pinned harness, block guard untouched
    r2/env/patch_conv1d_inplace.py         the source patcher that produces this diff
    r2/dense/src/clk.cu                    ALU contention probe

Raw traces are not committed; their run tags are in `BEFORE_RUN_TAG` and
`AFTER_RUN_TAG`.
