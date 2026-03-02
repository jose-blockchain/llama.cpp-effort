# BucketMul / Effort feature

## What it is

**BucketMul** is an optional CPU path for matrix-vector multiplication used during inference. It builds “bucket” views from the model weights at load time and uses an **effort** value (0.0–1.0) to trade off speed vs accuracy: higher effort gives more accurate results and uses more CPU; lower effort is faster and uses less CPU.

Use it to compare CPU inference speed with the baseline, or to reduce CPU usage when many processes run on the same machine (dynamic effort).

- **GGUF-only**, no new file format. Optionally caches built buckets in a `.bucket_mul` file next to the model for faster restarts.
- **Server only**: dynamic effort (based on CPU usage) is implemented in `llama-server`; the CLI uses a fixed effort.

---

## How to enable

- **Enable bucket-mul:** `--bucket-mul`  
Build bucket views when loading the model and use effort-based matmul for the relevant ops.
- **Disable (default):** `--no-bucket-mul` or omit the flag.

---

## Parameters


| Parameter                    | Type  | Default | Description                                                                                                                       |
| ---------------------------- | ----- | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--bucket-mul`               | flag  | off     | Enable bucket-mul (build buckets at load, use effort-based matmul).                                                               |
| `--bucket-mul-effort`        | FLOAT | 0.5     | Nominal effort 0.0–1.0 when CPU is not busy (or fixed effort if dynamic is disabled).                                             |
| `--bucket-mul-effort-min`    | FLOAT | 0.25    | Minimum effort when CPU usage is high (dynamic mode only).                                                                        |
| `--bucket-mul-cpu-threshold` | PCT   | 999     | Average CPU usage % (0–100). When usage ≥ this, effort is reduced toward `effort-min`. **999 = disabled** (no dynamic reduction). |


**Environment:** `LLAMA_ARG_BUCKET_MUL=1` to enable bucket-mul from the environment.

---

## Fixed vs dynamic effort

- **Fixed effort (default):**  
Set `--bucket-mul-cpu-threshold 999` (default). Effort stays at `--bucket-mul-effort` (e.g. 0.5).
- **Dynamic effort (server):**  
Set `--bucket-mul-cpu-threshold` to a value 0–100 (e.g. 80). The server samples average CPU usage about once per second:
  - If CPU usage **< threshold**: use **nominal** effort (`--bucket-mul-effort`).
  - If CPU usage **≥ threshold**: use **minimum** effort (`--bucket-mul-effort-min`).

So when the machine is busy, each process can lower its effort to reduce contention. Typical threshold: slightly above idle CPU %, e.g. **80**.

---

## Cache file

If the model path is `model.gguf`, the cache path is `model.gguf.bucket_mul`. On first run with `--bucket-mul`, buckets are built and optionally saved there; on later runs they are loaded from the cache so startup is faster. You can delete the `.bucket_mul` file to force a rebuild.

---

## Examples

**Server – fixed effort (0.5):**

```bash
./build/bin/llama-server -m model.gguf --port 8080 --bucket-mul
```

**Server – dynamic effort (reduce when CPU ≥ 80%):**

```bash
./build/bin/llama-server -m model.gguf --port 8080 --bucket-mul --bucket-mul-cpu-threshold 80
```

**Server – custom nominal and min effort:**

```bash
./build/bin/llama-server -m model.gguf --port 8080 --bucket-mul \
  --bucket-mul-cpu-threshold 80 --bucket-mul-effort 0.5 --bucket-mul-effort-min 0.25
```

**CLI – fixed effort:**

```bash
./build/bin/llama-cli -m model.gguf --bucket-mul --bucket-mul-effort 0.5 -p "Hello" -n 32
```

**Disable bucket-mul:**

```bash
./build/bin/llama-server -m model.gguf --port 8080 --no-bucket-mul
```

---

## Logs (server)

When dynamic effort is active you may see:

- `bucket_mul dynamic effort active (cpu_usage% threshold=80), sampling every 1s`
- `bucket_mul cpu_usage%=X.X effort=Y.YY (nominal|min)`

If CPU usage cannot be sampled (e.g. platform unsupported): `bucket_mul: cpu usage unavailable`; effort stays at nominal.