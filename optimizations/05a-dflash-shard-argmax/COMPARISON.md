# 05a -- DFLASH draft greedy select: split the vocab-shard reduction across CTAs

## What the kernel is and where the time goes

The DFLASH draft borrows the target `lm_head` and picks its `block_size - 1`
draft tokens greedily inside the draft CUDA graph
(`_DflashDraftSampler.__call__`, `python/sglang/srt/speculative/dflash_worker_v2.py:150`).
At TP2 the shard-local step is

    logits = hs @ weight[:num_org].T          # [6, 124160] bf16
    torch.max(logits, dim=-1, out=(local_max, local_arg))

Measured in the block-7 control capture `runs/sz7_ctl`
(`optimizations/sizing-block7/call_sites.csv`, graph 9239):

    void at::native::reduce_kernel<512, 1, ReduceOp<c10::BFloat16, MaxOps<...>>>
      grid [6, 1, 1]   block [128, 4, 1]   smem 8208
      1.00 calls/step  40.32 us/call  40.68 us/step busy  40.41 us/step critical
      coverage 0.0067  -> 99.3% exposed, so its busy time is its critical time

### The bound, two ways

`n = bs * (block_size - 1) = 6` rows, `num_org = 124160` per rank. The vocab
shard is confirmed from the `lm_head` weight bytes in the same capture:
`124160 x 2048 x 2 = 508,559,360 B`, and `510,326,272` total with activations
and the logits write.

    read      6 x 124160 x 2            = 1,489,920 B
    write     6 x 2 (bf16) + 6 x 8 (i64) =      60 B
    FLOPs     0 (a max reduction)

    t_SOL = max(FLOPs / P_peak, F + bytes / R)
          = max(0, 2.08 us + 1,489,980 / 7.37e12)
          = 2.08 + 0.20 = 2.28 us

    measured 40.32 us   ->  gap 17.7x   ->  achieved 37.0 GB/s

Neither term explains 40 us. 1.49 MB is 0.20 us of streaming on this device and
there is no arithmetic at all, so the kernel is neither bandwidth-bound nor
FLOP-bound: it is bound by having **6 CTAs on a 148-SM device**. 6 x 512
threads = 3072 threads over 744,960 bf16 elements is 242 elements per thread,
with 4% of the machine resident and nothing to hide memory latency behind.

`torch.max(x, dim=-1)` is a TensorIterator reduction over the contiguous inner
dimension. It assigns one CTA per output row and there are 6 output rows;
`grid = [6, 1, 1]` with `grid.y = 1` says the reduced dimension was **not split
at all**. This is a decomposition problem, not a tuning problem.

### The same decomposition is already in this tree

`python/sglang/kernels/ops/speculative/topk1.py:79`, `draft_topk1_postprocess`:

> PyTorch eager argmax reduces each row with too little parallelism for the
> GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes the
> vocab dimension across CTAs, then finalizes one token per row.

Same diagnosis, same shape, already written and already tested. Its consumers:

    $ grep -rn "draft_topk1_postprocess" --include=*.py . | grep -v ops/speculative/topk1.py
    python/sglang/srt/speculative/eagle_worker_v2.py:9
    python/sglang/srt/speculative/eagle_worker_v2.py:775
    python/sglang/srt/debug_utils/pr_fix_toggle.py:22
    test/registered/kernels/ops/speculative/test_spec_topk1.py  (5 hits)
    test/registered/kernels/benchmark/speculative/bench_spec_topk1.py  (5 hits)
    python/sglang/kernels/ops/speculative/__init__.py:21

Nothing under `srt/speculative/dflash*`. **The EAGLE draft path adopted it; the
DFLASH draft path never did.** It cannot be called unchanged here because the
DFLASH sampler needs the max *value* as well as the index for its cross-rank
select, and `draft_topk1_postprocess` returns only the index, so the two kernels
are written out once more rather than reused.

## The change

Two kernels replacing one. The partial kernel gives each `(row, split)` pair its
own CTA and reduces `BLOCK` vocab entries; the finalize kernel reduces the
`num_splits` partials per row and folds `org_vocab_start` into its store, which
also removes the separate `local_arg.add_()` pass the stock path would need when
that offset is non-zero (it is 0 at this configuration, so no launch is saved
here -- stated so the launch accounting below is not mistaken for a gain).

At `BLOCK = 8192`, the value the in-tree EAGLE path already uses:
`num_splits = 16`, partial grid `[6, 16]` = **96 CTAs**, finalize grid `[6]`.

`BLOCK` is **not** tuned by this entry. The sweep is reported separately below.

## Standalone comparison

B200, 148 SMs, inside the serving container, CUDA-graph captured and replayed
exactly as the production call site is, medians over 200 replays.
`r2/dflash/bench/bench_shard_argmax.py`, result `r2/dflash/out7/bench_synth.json`.

```
- torch.max(logits, dim=-1, out=(max, arg))     6 CTAs    min 29.856  med 30.656  max 35.296
+ split-vocab partial + finalize                96+6 CTAs min  7.296  med  7.824  max 11.200
-----------------------------------------------------------------------------------------
  net -22.832 us/call median  (-74.5%)
```

Eager, same inputs, for reference: 33.504 -> 26.432 us. The graph numbers are
the ones that transfer, because the production call site is inside the draft
graph; the eager gap is smaller because eager pays two Python dispatches for the
two kernels instead of one.

ALU contention probe (`r2/dense/src/clk.cu`) bracketing every measurement:

    CLK_BEFORE 222.1    CLK_MID 222.1    CLK_AFTER 222.1   GFLOP/s/SM

Clean reads are ~222 and contended ~108, so all three readings are uncontended
and the standalone numbers are not a co-residency artefact.

Against the bound the kernel goes from **7.4% of SOL to 29.2%**. It is still
3.4x its floor at 96 CTAs; what remains is the launch constant (2.08 of the
2.28 us bound) plus a 6-row finalize that cannot use the machine either. The
entry does not claim to have reached SOL, only to have removed the
decomposition.

## The knob is not the change

`BLOCK` swept at the decode shape, on real captured logits, same harness, same
CUDA-graph replay, medians over 200 replays (`r2/dflash/out7/sweep_real6.json`):

| `BLOCK` | splits | partial CTAs | before | after | delta | bit-equal |
|---:|---:|---:|---:|---:|---:|:--:|
| 2048 | 61 | 366 | 30.496 | 7.904 | -22.592 | yes |
| 4096 | 31 | 186 | 30.528 | 7.968 | -22.560 | yes |
| **8192 (shipped, = in-tree EAGLE value)** | 16 | **96** | 30.512 | **7.856** | **-22.656** | yes |
| 16384 | 8 | 48 | 30.416 | 9.088 | -21.328 | yes |
| 32768 | 4 | 24 | 30.784 | 9.472 | -21.312 | yes |

**The sweep is flat from 2048 to 8192 and 8192 is already the best of the five.**
There is no tuning gain to claim on top of the decomposition: the knob is worth
at most 0.1 us against the change's 22.66 us, and moving it the other way costs
1.2 us. The entry ships the value that was already in the tree and the sweep is
here only to show the lever is spent.

For contrast, at the *extend* shape the same sweep is not flat -- 288 rows
already supply enough CTAs at any block, so the reduction becomes genuinely
bandwidth-bound and wider blocks help:

| `BLOCK` | before | after (288 rows) |
|---:|---:|---:|
| 2048 | 44.448 | 35.808 |
| 8192 | 44.320 | 26.176 |
| 32768 | 44.448 | 23.536 |

That is the expected signature of the diagnosis: the decode shape is CTA-starved
and the extend shape is not.

## Kernel-level comparison

Profiled pair `dfl_pctl` / `dfl_pvar`, block 7, TP2, 88 decode steps in each
trace, medians over 88 launches, both ranks. Both arms ran the same patched
source and differ only by `DFLASH_SHARD_ARGMAX`, read back from each arm's own
`server.log` (`r2/dflash/out7/resolved.log`), not assumed.

```
  TP-0
- void at::native::reduce_kernel<512,1,ReduceOp<BFloat16,MaxOps<...>>>   1.00/step  40.416 us   40.44 us/step
+ _dflash_shard_max_partial_kernel                                       1.00/step   3.264 us    3.26 us/step
+ _dflash_shard_max_finalize_kernel                                      1.00/step   1.856 us    1.86 us/step
----------------------------------------------------------------------------------------------------------
  net -35.32 us/step

  TP-1
- void at::native::reduce_kernel<512,1,ReduceOp<BFloat16,MaxOps<...>>>   1.00/step  40.912 us   40.95 us/step
+ _dflash_shard_max_partial_kernel                                       1.00/step   3.169 us    3.20 us/step
+ _dflash_shard_max_finalize_kernel                                      1.00/step   1.696 us    1.71 us/step
----------------------------------------------------------------------------------------------------------
  net -36.04 us/step
```

The device-side invocation count confirms the substitution rather than inferring
it (`r2/analysis2/gate_arms.py`, invocation section):

```
  kernels ADDED by dfl_pvar      1.0/step  _dflash_shard_max_partial_kernel
                                 1.0/step  _dflash_shard_max_finalize_kernel
  kernels REMOVED                1.0/step  at::native::reduce_kernel<512,1,ReduceOp<c10::BFloat16,...
  launches/step                  1246.1 -> 1247.1    +1.0
```

**Launches per step go up by exactly one**, which is the whole cost of the
change: one kernel becomes two. That one extra launch is worth 2.08 us of floor
and buys 35 us, and it is stated here rather than buried because the ledger's
other entries have held launches flat.

The immediate neighbours of the replaced kernel do not move:

```
  reduce_kernel<256,2,ArgMaxOps<float>>  (the post-all-gather select)
        TP-0  6.52 -> 6.03 us/step        TP-1  6.54 -> 6.60 us/step
  _scatter_gather_elementwise_kernel     (torch.gather of the winning id)
        TP-0  4.87 -> 4.81 us/step        TP-1  5.29 -> 5.25 us/step
  ncclDevKernel_AllGather_RING_LL        (both all-gathers)
        TP-0 14.80 -> 13.40 us/step       TP-1 13.54 -> 14.82 us/step
```

### Why the busy totals do not reconcile and the step period does

The whole-trace busy delta is **+23.6 us/step on TP-0 and -51.1 us/step on
TP-1** -- opposite signs on the two ranks of the same pair. The rows responsible
are the MNNVL AllReduce families, which move +46.3 us/step on TP-0 while moving
-14.7 us/step on TP-1 at an unchanged call count and an unchanged 5.2 us/call
median. The AllGather pair above does the same thing, 14.80 -> 13.40 on one rank
and 13.54 -> 14.82 on the other.

That is peer-rank wait being redistributed, not work appearing or disappearing:
a collective's duration on one rank is the time it spends waiting for the other,
so removing 35 us of GPU time ahead of it on both ranks re-phases which rank
arrives first. It is the round-2 finding that GPU savings turn into rank skew,
reproduced here on a much smaller change. **Busy time is therefore not a valid
ledger for this entry, and it is not quoted as one.** What is quoted is the
kernel row, which is unambiguous and agrees on both ranks, and the step period,
which is measured on the unprofiled arms below.

The two reconcile:

```
  kernel net, profiled pair       -35.32 us/step (TP-0)   -36.04 us/step (TP-1)
  step period, unprofiled median  -34.8 us/step
```

Agreement to 1.5% is expected here: the replaced kernel has coverage 0.0067, so
it is essentially fully exposed and its time is the step's time.

## End-to-end comparison

Batch 1, ISL 60000, OSL 400, block 7, TP2, unprofiled harness, three pairs with
control and variant adjacent and interleaved in one allocation on one node.
Step period is `TPOT x acceptance`; the conversion is
exact because acceptance is identical in every arm to four decimals.

| arm | TPOT ms | TTFT ms | accept | resolved toggle |
|---|---:|---:|---:|---|
| dfl_ctl_a | 0.9138 | 500.2 | 4.8858 | 0 |
| dfl_var_a | **0.9050** | 501.6 | 4.8858 | 1 |
| dfl_ctl_b | 0.9109 | 500.1 | 4.8858 | 0 |
| dfl_var_b | **0.9048** | 500.7 | 4.8858 | 1 |
| dfl_ctl_c | 0.9141 | 500.7 | 4.8858 | 0 |
| dfl_var_c | **0.9069** | 501.3 | 4.8858 | 1 |

Per-replicate paired deltas, and the spread they have to beat:

```
  a   0.9138 -> 0.9050   -8.7 us/token   -42.6 us/step
  b   0.9109 -> 0.9048   -6.1 us/token   -29.6 us/step
  c   0.9141 -> 0.9069   -7.1 us/token   -34.8 us/step
  median                 -7.1 us/token   -34.8 us/step   (-0.78% TPOT)

  control-to-control spread   3.2 us/token  over 3 controls
  variant-to-variant spread   2.1 us/token  over 3 variants
```

**All three pairs are negative and the smallest of them, -6.1 us/token, is
1.9x the control-to-control spread.** The three variants also sit in a tighter
band than the three controls.

TTFT moves +0.5 ms on the median, inside its own noise, and is expected not to
move: the draft sampler runs only on the decode path.

For completeness, the profiled pair's own step periods, which are not differenced
against the unprofiled arms because the profiler term is arm-dependent:

```
  dfl_pctl 4465.7 us/step   dfl_pvar 4434.4 us/step   -31.4
```

### Validity gates

`r2/analysis2/gate_arms.py` on the pairs:

- **Gate 1 (accept > 2.0, trace_steps << OSL)**: PASS on both profiled arms,
  `accept=4.89 trace_steps=88.0 osl=400`, and the same 88 steps as the reference
  capture `sz7_ctl`.
- **Gate 2 (non-zero on-device delta)**: PASS -- the invocation diff above names
  the kernel removed and the two added. This is the check that caught a previous
  patch which never reached the GPU.
- **Gate 3 (profile the measured request)**: satisfied by the harness.
- **Gate 4 (replicate tolerance)**: range -42.6..-29.6 us/step over the pairs the
  gate was given, median -36.1, spread 13.0 against a tolerance of 100.0. PASS.

The gate additionally reports `missing trace, on-device delta unverifiable` for
the unprofiled pairs. That is by construction -- unprofiled arms have no trace,
which is why the profiled pair exists -- and is not a failure of this change.

## Accuracy

Gate is **bit equality** of both outputs -- the bf16 maximum and the int64 index
-- against the `torch.max` call being replaced, not a tolerance band, because a
numerics change here moves which tokens the draft proposes and therefore
acceptance, which would make every timing below incomparable.

Run on **real captured activations**: the draft's own `lm_head` output, taken
in-engine from the live server by `r2/env/patch_dflash_dump_logits.py` on the
first sampler call that is not inside a graph capture, on both TP ranks
(`r2/dflash/data/logits.rank{0,1}.pt`, `[288, 124160]` bf16).

| input | rows | value mismatches | index mismatches |
|---|---:|---:|---:|
| real logits, rank 0, all rows | 288 | **0** | **0** |
| real logits, rank 1, all rows | 288 | **0** | **0** |
| real logits, rank 0, decode shape | 6 | **0** | **0** |
| synthetic N(0,4) | 6 | **0** | **0** |

Zero at every one of the five `BLOCK` values in the sweep as well, so the
equality is a property of the decomposition and not of one block size.

Why it is exact rather than close: bf16 widens to fp32 exactly, so no candidate
value is perturbed; maximum is associative and commutative, so splitting the
reduction cannot change which value wins; and both stages break ties to the
left, so the winning split is the lowest-numbered one holding the maximum and
the winning lane within it is the lowest offset, which composes to the globally
first maximal index -- the same index `torch.max` returns and the invariant
`_DflashDraftSampler` already documents for its cross-rank select.

**One stated divergence.** `torch.max` propagates NaN; this reduction does not,
because the masked tail is filled with `-inf` and `tl.max` ignores NaN. That is
the same choice the in-tree `draft_topk1_postprocess` makes. It cannot arise
from finite logits, and it did not arise on the captured activations: the gate
returned zero index mismatches over all 576 real rows, which it could not have
done had any row contained a NaN. It is recorded rather than hidden.
