# [feat] Compute the GDN decode convolution as a tile instead of a serial token loop

## Conversation

`_causal_conv1d_update_kernel` has two bodies, selected by
`HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`. The speculative-decode path takes the
non-tree one, which walks the draft tokens one at a time. Each iteration rotates
`KERNEL_WIDTH - 1` `[BLOCK_N]` register vectors -- `col0 <- col1 <- col2 <- x[t]`
-- so the loop is carried: token `t` cannot begin until `t-1` has finished
rotating. Each iteration also issues its own output store and, when an
intermediate window is requested, three more.

None of that serialisation is necessary. A width-`KW` causal convolution is a
fixed `KW`-tap FIR,

    out[t] = sum_j src(t - (KW-1) + j) * w_col[j]

with `src(i) = x[i]` for `i >= 0` and conv-state column `i + KW - 1` otherwise.
Every output is independent; the rotation is a software pipeline, not a
recurrence. This PR computes the whole chunk as one `[NP2_SEQLEN, BLOCK_N]`
tile: `KW` masked gathers, `KW` FMAs, one tile store for the output and one per
window slot. The tree body is untouched and only moves one level deeper under an
explicit `if`; the `else` is the entire change.

The conv-state columns the tile needs are the ones STEP 1 already read into
registers before STEP 2 overwrote them. Taking them from there rather than
re-reading `conv_state` removes `KW - 1` vector loads per CTA and lets the tile
inherit the ordering guarantee the serial loop already relied on instead of
needing a new one against the state roll.

Accumulation stays in `j` order against `w_col0..w_col{KW-1}`, the order the
previous body used. That is deliberate: on a speculative-decode path a numerics
change moves routing and therefore acceptance, so the target is bit equality
with the code being replaced, not a tolerance band.

**The size of this depends on the draft-token count, so it is reported as a
function of it rather than as one number.** The change removes a dependence
along the token axis, so what it buys is per-token cost. Fitting
`us/call(T) = fixed + slope * T` on a standalone bench at the deployed
`BLOCK_N = 64`, over `T = 4, 7, 8, 12, 16` (16 is `dflash_config.block_size`,
the draft model's trained upper bound):

```
  body                     fixed us    slope us/token    R^2
- serial token loop           1.091            0.2515    0.9963
+ tiled chunk                 1.802            0.0622    0.7907
```

The slope falls 4.04x; the intercept rises 0.711 us, which is the tile's own
prologue and a register count of 48 against 28. Break-even is at `T = 3.76`:
**below about four draft tokens this change is a regression**, at the deployed
7 it is worth -0.569 us/call, at 16 it is worth -2.416 us/call. The tiled fit's
R^2 is low because its cost is a staircase in `NP2_SEQLEN = next_power_of_2(T)`,
not a line in `T` -- `T = 7` and `T = 8` compile the same 8-row tile and measure
2.368 and 2.327 us. The sweep was run in two independent sessions and every
point reproduces to within 0.007 us.

Everything below is measured at **draft block 7**, the configuration this
deployment runs. An earlier revision of this PR measured block 14 and reported
four times the gain; those figures are superseded and are listed as such in
`optimizations/01-gdn-conv1d-occupancy/COMPARISON.md`.

**`BLOCK_N` is not changed by this PR.** The in-tree literals stay at 256. Every
arm below runs at 64, set by a pre-existing deployment patch applied identically
to control and variant. On an unmodified tree the change is worth more, not
less: -1.770 us/call at `BLOCK_N = 256`, block 7.

## Kernel-level comparison

Decode trace, batch 1, draft block 7, TP2, steady-state window of 88 steps,
exclusive time per step, `BLOCK_N = 64`. Control and variant are separate runs
with separate traces -- the shipped form is an in-place edit, so the kernel keeps
its name and the rows are told apart by which trace they came from. TP0 shown,
TP1 in brackets. Step count cross-checked against `gdn_decode_bf16state`, a
kernel this PR does not touch, which also gives 88.0 steps.

```
- _causal_conv1d_update_kernel  (serial token loop)   30.00 calls   3.872 us   116.9 us/step   [116.8]
+ _causal_conv1d_update_kernel  (tiled chunk)         30.00 calls   3.136 us    95.1 us/step   [ 95.4]
-------------------------------------------------------------------------------------------------------
  net -21.8 us/step   [-21.4]
```

No kernel class appears or disappears and nothing else moves:

```
  fused_qkvzba_split_reshape_cat_contiguous_kernel  30.00 -> 30.00 calls    87.5 ->  87.4 us/step   -0.1
  fused_qkv_split_gdn_prefill_kernel                30.00 -> 30.00 calls    39.9 ->  40.0 us/step   +0.1
  launches/step                                                 1246.1 -> 1246.1                    0.0
```

**Launches per step are identical.** The tap pattern is affine in the token
index, so there is nothing to precompute and no second kernel to pay for.

**The net above does not reconcile against the busy or union delta of this pair,
and it should not be expected to at this size.** Per step:

```
  busy                TP0  6228.0 -> 6361.3  (+133.3)     TP1  6345.6 -> 6135.8  (-209.7)
  union               TP0  4539.2 -> 4677.4  (+138.2)     TP1  4682.2 -> 4467.4  (-214.7)
  profiled step       TP0  5546.0 -> 5147.2  (-398.8)     TP1  5545.8 -> 5147.9  (-398.0)
```

Those totals disagree in sign between the two ranks and are six to ten times
larger than the 21.6 us the kernel moved; they are measuring the profiler's own
per-arm overhead. What does reconcile is the kernel's critical-path
contribution, `crit(S) = union(all) - union(all\S)`, which is a difference taken
inside one arm and so cancels that arm's offset:

```
  kernel exclusive   TP0  116.9 -> 95.1   -21.8      TP1  116.8 -> 95.4   -21.4
  kernel crit path   TP0  115.2 -> 93.5   -21.7      TP1  115.4 -> 93.9   -21.5
```

Exclusive and critical-path agree to 0.5% on both ranks, which is what a kernel
with essentially no co-resident work should do: its time is the step's time.

Against the `t = F + B/R` floor: this kernel moves 0.369 MB per call at block 7,
which is 0.046 us at 8.0 TB/s, and it takes 3.136 us in-engine. Essentially none
of its cost is traffic and none is arithmetic. The tiled body has taken the
per-token term to 0.062 us/token and the standalone fixed term is 1.80 us of a
2.37 us call -- 76% -- so what is left inside this kernel can only be removed by
removing the launch, which is a separate change.

## End-to-end comparison

Batch 1, ISL 60000, OSL 400, draft block 7, TP2, `BLOCK_N = 64` in both arms.
Eight arms, control and variant adjacently interleaved on one node in one
allocation, unprofiled harness. Every arm bracketed by an ALU contention probe:
all 24 readings fall in 221.6-222.1 GFLOP/s/SM against a clean-node reference of
~222, where a co-resident compute kernel reads ~108. Step period is
`TPOT x accept`; the conversion is exact because acceptance is 4.89 in every arm.

| | control | variant | change |
|---|---:|---:|---:|
| TPOT, mean of four | 0.9074 ms | **0.9033 ms** | **-0.45%** |
| step period | 4436.9 us | **4417.3 us** | **-19.7 us** |
| accept | 4.89 | 4.89 | unchanged |
| TTFT | 499.1-503.1 ms | 500.1-501.6 ms | unchanged |

Per replicate, with the control-to-control spread beside it:

```
  pair A   0.9085 -> 0.9039   -0.0046 ms/token   -22.5 us/step
  pair B   0.9046 -> 0.9041   -0.0005 ms/token    -2.4 us/step
  pair C   0.9090 -> 0.9026   -0.0064 ms/token   -31.3 us/step
  pair D   0.9073 -> 0.9027   -0.0046 ms/token   -22.5 us/step

  control  mean 0.90735  sd 0.00197  spread 0.0044 ms/token = 21.5 us/step
  variant  mean 0.90333  sd 0.00079  spread 0.0015 ms/token =  7.3 us/step
```

**At block 7 this is at the edge of what the end-to-end harness can resolve, and
that is stated rather than smoothed over.** The effect is the same size as the
spread between control arms, and pair B on its own measured -2.4 us/step -- a
single control/variant pair at this block is a coin flip. What makes the result
stand is the replication: all four variant arms fall below all four control arms
(the largest variant, 0.9041, is below the smallest control, 0.9046), which is
one of 70 assignments under an exact permutation test, p = 0.014; the paired t
over the four deltas is 3.22 on 3 df. The two profiled arms' unprofiled phases
add a fifth pair in the same direction, 0.9081 -> 0.9020, and keep the
separation complete.

The three independent measurements of the same quantity agree on size:

```
  standalone bench, -0.569 us/call x 30 calls/step      -17.1 us/step
  profiled trace, kernel exclusive and critical path    -21.6 us/step
  unprofiled arms, difference of four-arm means         -19.7 us/step
```

For contrast, the same change at draft block 14 moved the step by -81.2 us
against a control spread of 28.2 us and needed no statistics.

TTFT does not move because prefill calls `causal_conv1d_fn`, which this PR does
not touch. Acceptance is identical in every arm, which is the end-to-end
consequence of the bit-equality gate below.

## Accuracy

Gate is **bit equality**, not a tolerance band, run against the code being
replaced by loading the pristine and rewritten modules side by side in one
process, on the tensors the engine actually handed the launcher at block 7: real
activations, the real convolution weight, the real conv-state pool, the real
slot indices, and no bias -- which is what this layer has.

| output | mismatching bf16 elements | of | region |
|---|---:|---:|---|
| `out` | 0 | 28672 | the whole output |
| `conv_state` | 0 | 12288 | the one cache line this call writes |
| intermediate window | 0 | 36864 | the one window slot this call writes |
| `conv_state`, whole pool | 0 | 19451904 | all 1583 cache lines |
| intermediate window, whole buffer | 0 | 1806336 | all 49 slots |

Max absolute difference on `out` is 0.0; reference RMS 0.333 on `out`, 2.543 on
the conv-state line, 2.414 on the window slot, so this is not a comparison
against zeros. Registers per thread 28 -> 48.

A device-side counter in a separate untimed build confirms that what ran is the
branch this PR rewrites: over 1800 launches on each rank, the device launch
count equals the host call count exactly, every launch was compiled for
`seqlen = 7`, and the eagle-tree body was entered zero times.

## Notes for reviewers

**Where `BLOCK_N` comes from.** Nothing in the tree selects it: there is no
autotune decorator and no heuristic on this kernel, and the two literals are
`causal_conv1d_triton.py:564` for the prefill launcher and `:1278` for the
decode launcher, both `256`. The 64 used in every measurement above is set by a
pre-existing deployment patch that rewrites the decode literal only, applied
identically to both arms; the in-tree defaults are untouched by this PR, and the
diff assigns no `BLOCK_N` literal.

**The intermediate window is a rolling buffer**, not a dense
`[step, dim, win]` array: at block 7 the engine passes shape `(49, 7, 4096, 3)`
with stride `(36864, 1, 9, 1)`, so its step and win axes both have stride 1 and
address `step + win` along a single `seqlen + width - 2 = 9` axis. Entries that
alias carry the same value, so writing the window as a tile rather than token by
token is safe -- but a benchmark that allocates it densely gets both the traffic
and the addressing wrong.

`causal_conv1d_update`'s `validate_data` branch asserts
`conv_state.stride(-2) == 1`, which does not hold on this path; the real stride
is `state_len`. That assert is off by default and is a misleading comment rather
than a contract.

**This PR's value is concentrated at large draft blocks.** At the block this
deployment runs it is a small kernel-level win -- unambiguous in the trace,
0.44% of the step end to end, and only separable from the harness's own noise
with four replicates per side. At the draft model's trained upper bound of 16 it
is worth 2.4 us/call, and on an unmodified tree at `BLOCK_N = 256`, 9.3 us/call.
**The two fitted lines, not any single measurement, are what should be carried
forward**; a reviewer who wants one number should say which `T` it is for.

**Do not merge this for a draft block below four.** Break-even is at T = 3.76 and
the tiled body is slower below it.
