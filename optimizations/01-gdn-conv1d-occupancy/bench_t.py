"""us/call as a function of the draft-token count T, for the shipped tiled body
and the serial body it replaces, plus the block-7 bit-equality gate.

Why as a function of T. The gain of this optimization comes from removing a
loop-carried dependence along the token axis, so it scales with the number of
draft tokens. A single number measured at one block length stops being true the
moment anyone moves ``--speculative-num-draft-tokens``. Fitting

    us/call(T) = fixed + slope * T

separates the part that does not depend on T -- launch, prologue, the conv-state
roll -- from the part this change attacks.

Both bodies are loaded in one process: the container's pristine
``causal_conv1d_triton`` and the same file with
``r2/env/patch_conv1d_inplace.rewrite`` applied. Nothing is renamed and nothing
is instrumented, so what is compared is the diff that would be committed against
the code it replaces.

Every layout comes from ``r2/gdn_conv7/env/patch_conv1d_dump.py``, which wrote
the tensors the engine actually handed the launcher on a block-7 decode call,
together with their strides. Nothing here is inferred.
"""

import importlib.util
import json
import os
import pathlib
import statistics
import sys

import torch
import triton

DUMP = os.environ.get("CONV1D_DUMP_DIR", "")
RANK = os.environ.get("BENCH_RANK", "0")
OUT = os.environ.get("BENCH_OUT", "/tmp/bench_t.json")
BLOCK_NS = [int(x) for x in os.environ.get("BENCH_BLOCK_NS", "64,256").split(",")]
TS = [int(x) for x in os.environ.get("BENCH_TS", "4,7,8,12,16").split(",")]
REPS = int(os.environ.get("BENCH_REPS", "50"))
OUTER = int(os.environ.get("BENCH_OUTER", "30"))

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env")
)
import patch_conv1d_inplace as P  # noqa: E402

WIDTH = 4
PAD_SLOT_ID = -1

try:
    from sglang.srt.utils import is_arch_support_pdl

    PDL = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
except Exception:
    PDL = {}


def load_module(name, text):
    """Import a module from source text under a private name."""
    p = pathlib.Path("/tmp") / f"{name}.py"
    p.write_text(text)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def restrided(vals, shape, stride, dev):
    """Materialise ``vals`` in a buffer laid out with the engine's strides.

    :param vals: contiguous tensor holding the values
    :param shape: engine shape
    :param stride: engine stride
    :return: (view with the engine's strides, backing buffer)
    """
    n = sum((s - 1) * st for s, st in zip(shape, stride)) + 1
    buf = torch.zeros(n, dtype=vals.dtype, device=dev)
    view = buf.as_strided(tuple(shape), tuple(stride))
    view.copy_(vals.to(dev))
    assert tuple(view.stride()) == tuple(stride)
    return view, buf


def read_dump(dev):
    """Load the tensors the engine handed the launcher at block 7."""
    d = os.path.join(DUMP, f"r{RANK}")
    meta = json.load(open(os.path.join(d, "conv_meta.json")))
    out = {"meta": meta}

    def rd(key):
        e = meta.get(key)
        if e is None:
            return None
        dt = getattr(torch, e["dtype"].split(".")[-1])
        raw = torch.from_file(
            os.path.join(d, e["file"]),
            dtype=torch.uint8,
            size=os.path.getsize(os.path.join(d, e["file"])),
        )
        return raw.view(dt).view(*e["shape"])

    for k in (
        "x",
        "conv_state",
        "weight",
        "bias",
        "conv_state_indices",
        "intermediate_state_indices",
    ):
        out[k] = rd(k)
    return out


def build(dev, dmp, T):
    """Build every kernel argument at draft-token count ``T``.

    The token axis of ``x`` is cycled from the captured block-7 activation when
    ``T > 7`` and sliced when ``T < 7``. Neither body has data-dependent control
    flow on the non-tree path, so the values do not affect the timing; at ``T=7``
    the activation is used unmodified and that is the configuration the
    bit-equality gate runs at.
    """
    m = dmp["meta"]
    dim = m["x"]["shape"][1]
    T0 = m["x"]["shape"][2]
    xs = dmp["x"][0]  # (dim, T0) real
    idx = [i % T0 for i in range(T)]
    xv = xs[:, idx].unsqueeze(0).contiguous()  # (1, dim, T)
    # engine layout for x: (batch, dim, seqlen), stride (dim*seqlen, 1, dim)
    xstride = (dim * T, 1, dim)
    assert tuple(m["x"]["stride"]) == (dim * T0, 1, dim), m["x"]["stride"]
    x, _xb = restrided(xv, (1, dim, T), xstride, dev)

    cs_shape, cs_stride = m["conv_state"]["shape"], m["conv_state"]["stride"]
    conv_state, _cb = restrided(dmp["conv_state"], cs_shape, cs_stride, dev)

    weight = dmp["weight"].to(dev)
    bias = None if dmp["bias"] is None else dmp["bias"].to(dev)
    csi = dmp["conv_state_indices"].to(dev)
    isi = dmp["intermediate_state_indices"].to(dev)

    iw = m["intermediate_conv_window"]
    ncl, _T0, idim, iwin = iw["shape"]
    WIN = T + WIDTH - 2
    assert tuple(iw["stride"]) == (idim * (T0 + WIDTH - 2), 1, T0 + WIDTH - 2, 1), iw[
        "stride"
    ]
    ishape = (ncl, T, idim, iwin)
    istride = (idim * WIN, 1, WIN, 1)
    n = sum((s - 1) * st for s, st in zip(ishape, istride)) + 1
    ibuf = torch.zeros(n, dtype=torch.bfloat16, device=dev)
    inter = ibuf.as_strided(ishape, istride)

    out = torch.empty_strided((1, dim, T), xstride, dtype=x.dtype, device=dev)
    return dict(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        conv_state_indices=csi,
        intermediate_state_indices=isi,
        inter=inter,
        ibuf=ibuf,
        out=out,
        dim=dim,
        T=T,
        num_cache_lines=cs_shape[0],
        state_len=cs_shape[2],
    )


def launch(mod, t, block_n):
    """Launch ``mod``'s kernel on the non-tree decode path."""
    x, cs, w, bias, inter, out = (
        t["x"],
        t["conv_state"],
        t["weight"],
        t["bias"],
        t["inter"],
        t["out"],
    )
    grid = (1, triton.cdiv(t["dim"], block_n))
    return mod._causal_conv1d_update_kernel[grid](
        x,
        w,
        bias,
        cs,
        None,
        t["conv_state_indices"],
        None,
        inter,
        t["intermediate_state_indices"],
        None,
        None,
        None,
        out,
        1,
        t["dim"],
        t["T"],
        t["state_len"],
        t["num_cache_lines"],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        cs.stride(0),
        cs.stride(1),
        cs.stride(2),
        t["conv_state_indices"].stride(0),
        inter.stride(0),
        inter.stride(1),
        inter.stride(2),
        inter.stride(3),
        t["intermediate_state_indices"].stride(0),
        0,
        0,
        0,
        0,
        0,
        0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=WIDTH,
        SILU_ACTIVATION=True,
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=triton.next_power_of_2(t["state_len"]),
        NP2_SEQLEN=triton.next_power_of_2(t["T"]),
        USE_PAD_SLOT=True,
        BLOCK_N=block_n,
        SAVE_INTERMEDIATE=True,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False,
        **PDL,
    )


def time_graph(fn, reps=REPS, outer=OUTER):
    """Time ``fn`` under a captured CUDA graph; returns (min, median, max) us."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(outer):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / reps)
    ts.sort()
    return min(ts), statistics.median(ts), max(ts)


def fit(xs, ys):
    """Ordinary least squares of ``y = a + b x``; returns (a, b, r2, max_resid)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in res)
    return a, b, 1.0 - ss_res / ss_tot, max(abs(r) for r in res)


def main():
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)

    spec = importlib.util.find_spec("sglang.kernels.ops.mamba.causal_conv1d_triton")
    src = pathlib.Path(spec.origin).read_text()
    if P.MARK in src:
        raise SystemExit(
            "the installed file is already patched; run this before patching"
        )
    orig = load_module("cc_pristine", src)
    tiled = load_module("cc_tiled", P.rewrite(src))

    dmp = read_dump(dev)
    res = {
        "gpu": torch.cuda.get_device_name(0),
        "sm": torch.cuda.get_device_properties(0).multi_processor_count,
        "dump_meta": dmp["meta"],
        "block_ns": BLOCK_NS,
        "ts": TS,
        "reps": REPS,
        "outer": OUTER,
    }

    # ---- bit equality at the block-7 decode shape --------------------------
    T7 = dmp["meta"]["x"]["shape"][2]
    res["bitexact_T"] = T7
    t = build(dev, dmp, T7)
    cs0 = t["conv_state"].clone()
    t["ibuf"].zero_()
    k1 = launch(orig, t, 64)
    torch.cuda.synchronize()
    ref_o = t["out"].clone()
    ref_c = t["conv_state"].clone()
    ref_i = t["ibuf"].clone()
    t["conv_state"].copy_(cs0)
    t["ibuf"].zero_()
    t["out"].zero_()
    k2 = launch(tiled, t, 64)
    torch.cuda.synchronize()
    bd = lambda a, b: [
        int(
            (a.reshape(-1).view(torch.int16) != b.reshape(-1).view(torch.int16))
            .sum()
            .item()
        ),
        a.numel(),
    ]
    res["n_regs"] = {
        "serial": getattr(k1, "n_regs", None),
        "tiled": getattr(k2, "n_regs", None),
    }
    res["equivalence"] = {
        "out": bd(t["out"], ref_o),
        "conv_state": bd(t["conv_state"], ref_c),
        "intermediate_window": bd(t["ibuf"], ref_i),
    }
    res["max_abs_out"] = float((t["out"].float() - ref_o.float()).abs().max().item())
    res["out_rms"] = float(ref_o.float().pow(2).mean().sqrt().item())
    t["conv_state"].copy_(cs0)

    # ---- us/call as a function of T ----------------------------------------
    tm = {}
    for bn in BLOCK_NS:
        for T in TS:
            tt = build(dev, dmp, T)
            for tag, mod in (("serial", orig), ("tiled", tiled)):
                tt["conv_state"].copy_(tt["conv_state"])  # no-op, keeps the buffer live
                tm[f"{tag}_bn{bn}_T{T}"] = time_graph(
                    lambda mod=mod, tt=tt, bn=bn: launch(mod, tt, bn)
                )
            del tt
    res["timings_us"] = {
        k: {"min": a, "med": b, "max": c} for k, (a, b, c) in tm.items()
    }

    res["fit"] = {}
    for bn in BLOCK_NS:
        for tag in ("serial", "tiled"):
            ys = [tm[f"{tag}_bn{bn}_T{T}"][1] for T in TS]
            a, b, r2, mr = fit([float(T) for T in TS], ys)
            res["fit"][f"{tag}_bn{bn}"] = {
                "fixed_us": a,
                "slope_us_per_token": b,
                "r2": r2,
                "max_abs_residual_us": mr,
                "points": {str(T): y for T, y in zip(TS, ys)},
            }

    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
