"""Bit equality at block 7, counted over the regions the kernel actually writes.

``bench_t.py`` compares whole destination buffers. That is a valid statement but
a weak one for ``conv_state`` and the intermediate window, because the engine
hands the kernel a pool of 1583 cache lines and 49 window slots and one decode
call writes exactly one of each. This repeats the comparison restricted to the
written sub-regions, so the denominators are elements the kernel touched.

Inputs are the tensors the engine handed the launcher on a block-7 decode call
(``patch_conv1d_dump.py``), unmodified: real activations, the real convolution
weight, the real conv-state pool, the real slot indices, no bias -- which is what
this layer has.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_t as B  # noqa: E402


def main():
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec("sglang.kernels.ops.mamba.causal_conv1d_triton")
    src = pathlib.Path(spec.origin).read_text()
    if B.P.MARK in src:
        raise SystemExit("installed file already patched; run before patching")
    orig = B.load_module("bc_pristine", src)
    tiled = B.load_module("bc_tiled", B.P.rewrite(src))

    dmp = B.read_dump(dev)
    T = dmp["meta"]["x"]["shape"][2]
    t = B.build(dev, dmp, T)
    csi = int(dmp["conv_state_indices"][0].item())
    isi = int(dmp["intermediate_state_indices"][0].item())
    dim, sl = t["dim"], t["state_len"]
    win = T + 4 - 2

    cs0 = t["conv_state"].clone()
    t["ibuf"].zero_()
    B.launch(orig, t, 64)
    torch.cuda.synchronize()
    ref_o, ref_c, ref_i = t["out"].clone(), t["conv_state"].clone(), t["ibuf"].clone()

    t["conv_state"].copy_(cs0)
    t["ibuf"].zero_()
    t["out"].zero_()
    B.launch(tiled, t, 64)
    torch.cuda.synchronize()

    def bd(a, b):
        return [
            int(
                (a.reshape(-1).view(torch.int16) != b.reshape(-1).view(torch.int16))
                .sum()
                .item()
            ),
            a.numel(),
        ]

    iv = t["ibuf"].as_strided((49, dim, win), (dim * win, win, 1))
    rv = ref_i.as_strided((49, dim, win), (dim * win, win, 1))
    res = {
        "T": T,
        "block_n": 64,
        "conv_state_slot": csi,
        "window_slot": isi,
        "written_region": {
            "out": bd(t["out"], ref_o),
            "conv_state_slot": bd(t["conv_state"][csi], ref_c[csi]),
            "intermediate_window_slot": bd(iv[isi], rv[isi]),
        },
        "whole_buffer": {
            "out": bd(t["out"], ref_o),
            "conv_state_pool": bd(t["conv_state"], ref_c),
            "intermediate_window_buffer": bd(t["ibuf"], ref_i),
        },
        "max_abs_out": float((t["out"].float() - ref_o.float()).abs().max().item()),
        "out_rms": float(ref_o.float().pow(2).mean().sqrt().item()),
        "conv_state_slot_rms": float(ref_c[csi].float().pow(2).mean().sqrt().item()),
        "window_slot_rms": float(rv[isi].float().pow(2).mean().sqrt().item()),
        "has_bias": dmp["meta"]["bias"] is not None,
        "provenance": "r2/gdn_conv7/data/r0/conv_meta.json, decode call 300, block 7",
    }
    print(json.dumps(res, indent=1))
    json.dump(
        res, open(os.environ.get("BITCHECK_OUT", "/tmp/bitcheck.json"), "w"), indent=1
    )


if __name__ == "__main__":
    main()
