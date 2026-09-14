# Edge0 35B A3B — Tensor Atlas v4

Offline, single-file WebGL2 architecture atlas for **Edge0/Edge0-35B-A3B-preview**, built by
retargeting the nisten *Tensor Atlas v4* engine (`LLMViz-DeepSeek-V4.1-Flash`, MIT © 2026 netsin)
with the `tensor-atlas-v4-retarget` skill. Open `index.html` in a browser — no server, no CDN.

## What is in the page

| | |
|---|---|
| Layers | 40: **30 gated linear attention** + **10 full attention** (every 4th layer: 3, 7, 11 … 39) |
| Linear attention | fused q/k/v (16 key heads, 32 value heads at 128 dims), gate z, decay a/b, kernel-4 conv, constant-size state |
| Full attention | 16 query heads over **2 KV heads** at head_dim 256, q/k norms, quarter-rotary rope at theta 1e7 |
| MoE | 256 experts per layer, top-8, plus an always-on shared expert (512 wide) |
| Edge extras | Recover-LoRA adapters (620 tensors, 42.3 MB) and a prerouter head (99 tensors, 138.4 MB over 33 layers) |
| Vision | 27-block tower in the base models — **absent from the edge build** (measured: zero visual tensors) |
| Context | 262,144 positions; only the ten full-attention layers keep a growing KV cache |

## Measured bytes

Per-tensor sizes read from the safetensors headers over HTTP Range (no weights downloaded), with
per-expert tensors folded into the fused banks the BF16 base stores and quantization
scales/biases folded into their owner.

| Mode | Repository | Shards | Tensors | Payload |
|---|---|---|---|---|
| BF16 base | `Qwen/Qwen3.6-35B-A3B` | 26 | 1,045 | **71.9036 GB** |
| FP8 build | `Qwen/Qwen3.6-35B-A3B-FP8` | 42 | 64,196 | **37.4548 GB** |
| Edge0 int4 | `Edge0/Edge0-35B-A3B-preview` | 6 files | 2,476 | **19.6895 GB** |

`build.py` refuses to write the page unless *page total == measured payload* for all three modes
(exact integers, tolerance 0). The FP8 repository additionally ships one MTP layer
(`mtp.layers.0`, 0.8537 GB) that neither the BF16 base nor the edge build has — it is shown as its
own module (D0) rather than folded away.

## Files

| file | role |
|---|---|
| `index.html` | the deliverable — self-contained, offline, ~296 KB |
| `build.py` | builder: fold → archetypes → data half → splice → panel rewrites → code patches → gates |
| `objectcode.py` | anchored rewrite engine with a structural balance check (tag nesting, JSX pairs, backticks) |
| `panels.py` | the model-specific panel copy and engine-function rewrites, anchored on the engine's own text |
| `src.html` | upstream engine, unmodified (MIT, © 2026 netsin) |
| `measure.py` | safetensors header measurement over HTTP Range (`HF_MEASURE_DIR=/tmp/e0-measure python3 measure.py <repos…>`) |
| `hf-bf16.json` / `hf-fp8.json` / `hf-edge0.json` | per-tensor measurement output |
| `hf-config-*.json` | the three published configs, embedded in the config modal |
| `README-model-card.md` | the Edge0 model card (provenance) |

Rebuild: `python3 build.py` (add `--data-only` to emit just the generated data half).

## Method notes

* **Folding.** MLX names (`language_model.model.*`, `mlp.switch_mlp.*`, `.scales`/`.biases`),
  FP8 names (`mlp.experts.N.*`, `weight_scale_inv`) and BF16 fused banks all map onto one logical
  tensor list; a tensor that maps to nothing aborts the build.
* **Structure-checked rewrites.** Every whole-line panel rewrite must keep the line's nesting
  depth, JSX open/close pairs and backtick parity — that check is what stops htm from silently
  dropping a wrapper element (which reads as "the 3D scene does not render").
* **Acceptance run.** `#root`'s child must be `.app-shell`, the canvas a sane landscape box, every
  animation phase and view must run without an exception, and each mode's rendered solids must sum
  to `sumB` for every module.

Apache-2.0 weights (base model), MIT engine. No weight data is bundled.
