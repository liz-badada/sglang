# [feat] Split the DFLASH draft's vocab-shard argmax across CTAs instead of one CTA per token

## Conversation

The DFLASH draft borrows the target `lm_head` and picks its `block_size - 1`
draft tokens greedily inside the draft CUDA graph. At tensor parallelism the
shard-local step of that select is

    logits = hs @ weight[:num_org].T          # [block_size - 1, num_org]
    torch.max(logits, dim=-1, out=(local_max, local_arg))

`torch.max(x, dim=-1)` is a TensorIterator reduction over the contiguous inner
dimension, and it gives **one CTA per output row**. The draft has
`block_size - 1` rows. At the customer configuration that is 6 rows over a
124160-wide vocab shard, so the reduction runs on `grid = [6, 1, 1]` -- six CTAs
of a 148-SM device, with `grid.y = 1`, meaning the reduced dimension is not split
at all. It measures **40.32 us/call** at **37 GB/s**, against a
`max(FLOPs/P_peak, F + bytes/R)` bound of **2.28 us**: a gap of **17.7x**, and
with coverage 0.0067 essentially all of it is critical path.

Neither term of the bound explains it. The kernel reads 1.49 MB, which is 0.20 us
of streaming here, and a maximum has no arithmetic at all. What it is short of is
places to run: 6 CTAs x 512 threads is 242 elements per thread with 4% of the
machine resident and nothing to hide memory latency behind. **This is a
decomposition problem, not a tuning problem**, and the sweep at the end confirms
that -- the block size is worth at most 0.1 us, the decomposition is worth 22.7.

**The same decomposition is already in this tree and this path never adopted it.**
`sglang/kernels/ops/speculative/topk1.py`, `draft_topk1_postprocess`, records the
identical diagnosis in its docstring -- "PyTorch eager argmax reduces each row
with too little parallelism ... in CUDA graph replay. This split reduction
exposes the vocab dimension across CTAs, then finalizes one token per row" -- and
is wired only into `eagle_worker_v2.py`. Nothing under `srt/speculative/dflash*`
calls it. It cannot be called unchanged here either, because the DFLASH sampler
needs the shard maximum *value* as well as its index to run the cross-rank
select, and `draft_topk1_postprocess` returns only the index. So this PR writes
the two kernels once more rather than reusing it, and keeps them beside it in the
same module.

The vocab offset folds into the finalize store instead of costing the separate
`local_arg.add_()` pass the stock path needs when `org_vocab_start` is non-zero.
It is zero at the measured configuration, so **no launch is saved by that** and
the launch count below is flat; it is mentioned so the accounting is not
misread.

**The block size is not changed by this PR.** It ships at 8192, the value the
in-tree EAGLE top-1 path already uses, and the sweep shows that value is already
the best of the five tried. There is no knob gain folded into any number here.

Accumulation order is not a free choice on a speculative-decode path: a numerics
change moves which tokens the draft proposes and therefore acceptance, which
would make every timing incomparable. The target is bit equality with the
`torch.max` being replaced, and that is what is measured -- on real captured
logits, on both ranks.

## Kernel-level comparison

Profiled pair, block 7, TP2, 88 decode steps per trace, medians over 88
launches, both ranks. Both arms run the same patched source and differ only by a
toggle whose resolved value is read back from each arm's own `server.log`.

```
  TP-0
- void at::native::reduce_kernel<512,1,ReduceOp<BFloat16,MaxOps<...>>>  1.00/step  40.416 us  40.44 us/step
+ _dflash_shard_max_partial_kernel                                      1.00/step   3.264 us   3.26 us/step
+ _dflash_shard_max_finalize_kernel                                     1.00/step   1.856 us   1.86 us/step
---------------------------------------------------------------------------------------------------------
  net -35.32 us/step

  TP-1
- void at::native::reduce_kernel<512,1,ReduceOp<BFloat16,MaxOps<...>>>  1.00/step  40.912 us  40.95 us/step
+ _dflash_shard_max_partial_kernel                                      1.00/step   3.169 us   3.20 us/step
+ _dflash_shard_max_finalize_kernel                                     1.00/step   1.696 us   1.71 us/step
---------------------------------------------------------------------------------------------------------
  net -36.04 us/step
```

Device-side invocation count, so the substitution is observed and not inferred:

```
  kernels ADDED    1.0/step  _dflash_shard_max_partial_kernel
                   1.0/step  _dflash_shard_max_finalize_kernel
  kernels REMOVED  1.0/step  at::native::reduce_kernel<512,1,ReduceOp<c10::BFloat16,...
  launches/step    1246.1 -> 1247.1   +1.0
```

**Launches per step rise by exactly one**: one kernel becomes two. That launch
costs 2.08 us of floor and buys 35 us. Neighbours are flat:

```
  reduce_kernel<256,2,ArgMaxOps>   TP-0 6.52 -> 6.03   TP-1 6.54 -> 6.60  us/step
  _scatter_gather_elementwise      TP-0 4.87 -> 4.81   TP-1 5.29 -> 5.25  us/step
```

**The whole-trace busy delta is not quoted, because it is not a valid ledger for
this change.** It is +23.6 us/step on TP-0 and -51.1 us/step on TP-1 -- opposite
signs on the two ranks of the same pair -- and the rows responsible are the MNNVL
AllReduce families, which move +46.3 us/step on one rank and -14.7 on the other
at an unchanged call count and an unchanged 5.2 us/call median. A collective's
duration on a rank is the time it waits for its peer, so removing 35 us ahead of
it re-phases which rank arrives first. That is the known result that GPU savings
turn into rank skew. What reconciles instead is the kernel row against the step
period:

```
  kernel net, profiled pair       -35.32 us/step (TP-0)   -36.04 us/step (TP-1)
  step period, unprofiled median  -34.8 us/step
```

to 1.5%, which is what should happen for a kernel at coverage 0.0067.

## End-to-end comparison

Batch 1, ISL 60000, OSL 400, block 7, TP2, unprofiled harness, three pairs,
control and variant adjacent and interleaved in one allocation on one node.
Step period is `TPOT x acceptance`; the conversion is exact because acceptance is
identical in every arm to four decimals.

| | before | after | change |
|---|---|---|---|
| TPOT (median of 3 pairs) | 0.9138 ms | **0.9069 ms** | **-0.78%** |
| step period | - | - | **-34.8 us** |
| accept | 4.8858 | 4.8858 | unchanged |
| TTFT | 500.7 ms | 501.3 ms | unchanged |

```
  a   0.9138 -> 0.9050   -8.7 us/token   -42.6 us/step
  b   0.9109 -> 0.9048   -6.1 us/token   -29.6 us/step
  c   0.9141 -> 0.9069   -7.1 us/token   -34.8 us/step

  control-to-control spread   3.2 us/token
  variant-to-variant spread   2.1 us/token
```

All three pairs are negative and the smallest, -6.1 us/token, is 1.9x the
control-to-control spread. TTFT is not expected to move and does not: the draft
sampler runs only on the decode path.

## Accuracy

Gate is **bit equality** of both outputs -- the bf16 maximum and the int64 index
-- against the `torch.max` being replaced, not a tolerance band, because a
numerics change here moves which tokens the draft proposes and therefore
acceptance.

Run on real captured activations: the draft's own `lm_head` output taken
in-engine from the live server on both ranks, `[288, 124160]` bf16.

| input | rows | value mismatches | index mismatches |
|---|---:|---:|---:|
| real logits, rank 0 | 288 | **0** | **0** |
| real logits, rank 1 | 288 | **0** | **0** |
| real logits, decode shape | 6 | **0** | **0** |
| synthetic N(0,4) | 6 | **0** | **0** |

Zero at all five block sizes swept, so the equality is a property of the
decomposition rather than of one configuration.

Why exact and not merely close: bf16 widens to fp32 exactly, so no candidate is
perturbed; maximum is associative and commutative, so splitting cannot change
which value wins; and both stages break ties to the left, which composes to the
globally first maximal index -- what `torch.max` returns, and the invariant the
sampler already documents for its cross-rank select.

**Speculative acceptance is 4.8858 in all six end-to-end arms**, identical to
four decimals, which is the end-to-end consequence of the bit-equality gate and
is what makes the TPOT table above a legitimate comparison.

One divergence, stated rather than hidden: `torch.max` propagates NaN and this
reduction does not, the same choice the in-tree `draft_topk1_postprocess` makes.
It cannot arise from finite logits and did not arise here -- the gate returned
zero index mismatches over all 576 real rows, which it could not have done had
any row contained a NaN.

## Notes for reviewers

**Why the shard maximum is needed at all.** At TP1 the sampler takes a different
branch and calls `torch.argmax` directly; only the TP>1 branch needs the value,
to pick a winner across ranks. This PR changes the TP>1 branch, which is the one
measured. The TP1 branch has the same 6-CTA pathology and is left alone here
because it was not measured at this configuration.

**The vocab offset.** `org_vocab_start` is 0 at this configuration, so folding it
into the finalize store saves no launch today; it is folded in because the
alternative is a second full pass over the index buffer whenever it is non-zero.

**Where the standalone and in-engine numbers differ.** Standalone the pair costs
7.86 us against 5.12 us in the engine, and the replaced kernel costs 30.5 us
standalone against 40.4 in the engine. Both gaps are the same sign and the same
cause -- a cold L2 and per-replay event instrumentation in the standalone
harness -- so the standalone figure is the conservative one and the entry is
quoted from the trace.
