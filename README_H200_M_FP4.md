# H200 MXFP4 DFlash benchmark

This branch integrates the standalone Humming MoE backend for mixed-FP4 checkpoints and the model hooks needed
to execute DFlash. Its Python dependency pins Humming to the following source revision:

```text
https://github.com/vllm-project/humming/commit/a74973b5079e42ef861720b62f847ce9d33447f5
```

The commands below reproduce the `batch={4,8}` and `draft_tokens={6,8}` benchmark on one 8xH200 node with
tensor parallelism 8 and expert parallelism 1.

## Select W4A8 instead of W4A16

The MXFP4 expert weights are W4 in both modes. Humming uses BF16 activations (W4A16) by default. To run W4A8,
set the following environment variable in the server process:

```bash
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype":"float8e4m3","input_scale_group_size":128}'
```

This enables dynamic group-128 FP8 E4M3 activation quantization for the Humming FC1 and FC2 paths. Merely
selecting `--moe-runner-backend humming` does not enable W4A8; if the variable is absent, the run remains
W4A16. Record `activation_dtype=float8e4m3`, `input_quantization=dynamic_group`, and
`input_scale_group_size=128` in the benchmark manifest to make the selected path explicit.

## Correctness status

The branch can start the server and execute DFlash. It contains speculative layer-capture configuration,
multi-layer auxiliary hidden-state capture, and the capture-layer offset fix.

It does not yet contain all forward-correctness changes from the unmerged enablement work. In particular, the
FlashInfer routing/top-k/SwiGLU-limit changes still need to be ported. Until then, real-weight runs may report an
average accepted length close to 1.05. Step latency is benchmarkable, but effective throughput at that acceptance
is not the intended final DFlash performance.

Use real target and draft weights when measuring acceptance. `--load-format dummy` is suitable only for startup
and latency smoke tests because dummy router and model weights do not preserve routing or acceptance behavior.

## Install from the branch

The normal editable install resolves the pinned Humming revision from `python/pyproject.toml`:

```bash
git clone --branch h200_m_fp4 https://github.com/AichenF/sglang.git
cd sglang
python -m pip install -e ./python
python -c 'import importlib.metadata as m; print(m.version("humming-kernels"))'
```

For the container benchmark below, derive a small image that replaces the image-bundled Humming wheel with the
pinned source revision. The SGLang Python package itself is bind-mounted from this checkout.

```bash
docker build --network host -t sglang:h200-m-fp4-humming-a74973b -f - . <<'DOCKERFILE'
FROM lmsysorg/sglang:nightly-dev-cu13-20260910-00840301
RUN python -m pip install --no-cache-dir --force-reinstall --no-deps \
    "humming-kernels @ git+https://github.com/vllm-project/humming.git@a74973b5079e42ef861720b62f847ce9d33447f5"
DOCKERFILE
```

## Start the DFlash server

Run all commands from the root of this SGLang checkout. `/raid` is node-local; change `NODE_SCRATCH` to an
equivalent local NVMe path such as `/tmp/sglang_benchmark` on nodes without `/raid`.

```bash
export NODE_SCRATCH=/raid/data/sglang_benchmark
export MODEL_PATH=/path/to/target/model
export DRAFT_MODEL_PATH="$MODEL_PATH/dflash"
export DRAFT_TOKENS=6
export SERVER_NAME=h200_m_fp4_dflash_d${DRAFT_TOKENS}

mkdir -p "$NODE_SCRATCH/rootcache" "$NODE_SCRATCH/dgcache"

docker run --rm --init -d \
  --name "$SERVER_NAME" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 64g \
  -e 'SGLANG_HUMMING_INPUT_QUANT_CONFIG={"a_dtype":"float8e4m3","input_scale_group_size":128}' \
  -v "$NODE_SCRATCH:$NODE_SCRATCH" \
  -v "$PWD/python/sglang:/sgl-workspace/sglang/python/sglang:ro" \
  -v "$NODE_SCRATCH/rootcache:/root/.cache" \
  -v "$NODE_SCRATCH/dgcache:/root/.deep_gemm" \
  --entrypoint python3 \
  sglang:h200-m-fp4-humming-a74973b \
  -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --context-length 65536 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
  --speculative-num-draft-tokens "$DRAFT_TOKENS" \
  --tp-size 8 \
  --ep-size 1 \
  --trust-remote-code \
  --moe-runner-backend humming \
  --mem-fraction-static 0.75 \
  --host 0.0.0.0 \
  --port 29999
```

Do not add `--disable-cuda-graph`: this DFlash decode path requires CUDA-graph replay. Keep `--init`, and manage
the long-running server separately from the benchmark client so a client startup timeout cannot SIGKILL the
server process tree.

Wait until the server is ready:

```bash
until curl -fsS http://127.0.0.1:29999/health >/dev/null; do sleep 5; done
docker logs "$SERVER_NAME" | tail -n 50
```

## Run the batch-4 and batch-8 benchmark

```bash
docker run --rm \
  --network host \
  -v "$PWD/python/sglang:/sgl-workspace/sglang/python/sglang:ro" \
  --entrypoint python3 \
  sglang:h200-m-fp4-humming-a74973b \
  -m sglang.benchmark.one_batch_server \
  --model None \
  --base-url http://127.0.0.1:29999 \
  --batch-size 4 8 \
  --input-len 1024 \
  --output-len 64 \
  --show-report
```

Record at least the exact SGLang and Humming revisions, GPU model/count, draft-token count, batch size, input and
output lengths, clean step latency, output throughput, throughput per user, and average accepted length.

Stop the server atomically after the benchmark:

```bash
docker kill "$SERVER_NAME"
```

Set `DRAFT_TOKENS=8`, choose a fresh `SERVER_NAME`, restart the server, and repeat the same client command to
complete the four points: `bs4d6`, `bs8d6`, `bs4d8`, and `bs8d8`. Draft length is a server-side setting, so the
server must be restarted between d6 and d8.
