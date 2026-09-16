#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build edge0-35b-atlas/index.html — skill: tensor-atlas-v4-retarget.

src.html is the nisten "Tensor Atlas v4" engine (LLMViz-DeepSeek-V4.1-Flash, MIT (c) 2026
netsin) kept whole for the shell. Only the model half is replaced.

Subject: Edge0/Edge0-35B-A3B-preview — a 4-bit MLX build of Qwen/Qwen3.6-35B-A3B made for the
edge0 streaming runtime: hybrid attention (30 gated-linear-attention layers + 10 full-attention
layers), 256-expert MoE with top-8 routing plus a shared expert, a 27-block vision tower in the
base model (absent from the edge build), a Recover-LoRA adapter set and a trained prerouter head.

Three measured modes, per-tensor bytes read from the safetensors headers over HTTP Range:
  bf16   Qwen/Qwen3.6-35B-A3B            26 shards  1,045 tensors  71.9036 GB
  fp8    Qwen/Qwen3.6-35B-A3B-FP8        42 shards 64,196 tensors  37.4548 GB
  edge0  Edge0/Edge0-35B-A3B-preview      6 shards  2,476 tensors  19.6895 GB
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
SRC, OUT = DIR / "src.html", DIR / "index.html"
BF_JSON, FP8_JSON, E0_JSON = DIR / "hf-bf16.json", DIR / "hf-fp8.json", DIR / "hf-edge0.json"
CFG_BASE, CFG_FP8, CFG_E0 = DIR / "hf-config-base.json", DIR / "hf-config-fp8.json", DIR / "hf-config-edge0.json"

# ---- name folding -----------------------------------------------------------------------------
SCALE_SUFFIX = re.compile(r"\.(weight_scale_inv|weight_scale_2|weight_scale|scales|biases|input_scale|output_scale)$")
PREFIX = [(r"^language_model\.model\.", "model.language_model."),
          (r"^language_model\.lm_head\.", "lm_head.")]
# FP8 splits experts; MLX stacks them in switch_mlp
EXPERT = re.compile(r"^(?:model\.language_model\.layers\.(\d+)|mtp\.layers\.(\d+))\.mlp\.(?:experts\.\d+|switch_mlp)\.(gate_proj|up_proj|down_proj|gate_up_proj)(\.weight)?$")
LORA = re.compile(r"^model\.language_model\.layers\.\d+\.(.+)\.lora_([AB])$")
PREROUTER = re.compile(r"^layers\.(\d+)\.(fc1|fc2|linear_init)\.weight$")


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def make_folder(bfset: set[str]):
    def fold_expert(n: str) -> str | None:
        m = EXPERT.match(n)
        if not m:
            return None
        tail = "gate_up_proj" if m.group(3) in ("gate_proj", "up_proj") else "down_proj"
        base = f"model.language_model.layers.{m.group(1)}" if m.group(1) is not None else "mtp.layers.0"
        return f"{base}.mlp.experts.{tail}"

    def synthetic(n: str) -> str | None:
        """LoRA and prerouter tensors exist only in the edge build: give them their own names."""
        m = LORA.match(n)
        if m:
            return f"lora::{m.group(1)}.lora_{m.group(2)}"
        m = PREROUTER.match(n)
        if m:
            return f"prerouter::{m.group(2)}.weight"
        return None

    def folder(name: str) -> str | None:
        n = name
        for pat, rep in PREFIX:
            n = re.sub(pat, rep, n)
        syn = synthetic(n)
        if syn:
            return syn
        stripped = SCALE_SUFFIX.sub("", n)
        for c in (n, stripped, stripped + ".weight"):
            if c in bfset:
                return c
            f = fold_expert(c)
            if f and f in bfset:
                return f
        return None
    return folder


def collect(path: Path, folder) -> tuple[dict, list[str]]:
    out: dict[str, int] = {}
    unmapped: list[str] = []
    for n, t in load(path)["tensors"].items():
        lg = folder(n)
        if lg is None:
            unmapped.append(n)
            continue
        out[lg] = out.get(lg, 0) + t["bytes"]
    return out, unmapped


def templates(allow_unmapped: bool = False) -> dict:
    bfset = set(load(BF_JSON)["tensors"])
    folder = make_folder(bfset)
    bf, un_bf = collect(BF_JSON, folder)
    fp8, un_fp8 = collect(FP8_JSON, folder)
    e0, un_e0 = collect(E0_JSON, folder)
    for label, un in (("bf16", un_bf), ("fp8", un_fp8), ("edge0", un_e0)):
        if un and not allow_unmapped:
            raise SystemExit(f"{label}: {len(un)} tensors map to nothing, e.g. {un[:4]}")
        if un:
            print(f"  unplaced {label} tensors: {len(un)} (e.g. {un[0]})")
    names = sorted(set(bf) | set(fp8) | set(e0))
    dims = {n: t["shape"] for src in (BF_JSON, FP8_JSON, E0_JSON)
            for n, t in load(src)["tensors"].items()}  # raw; logical dims come from the first shard seen

    def logical_dims(name: str) -> list[int]:
        """Shape of the logical tensor, taken from a mode that stores it directly."""
        for src in (BF_JSON, FP8_JSON, E0_JSON):
            for raw, t in load(src)["tensors"].items():
                if folder(raw) == name:
                    sh = t["shape"]
                    if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
                        if len(sh) == 3:      # fused bank [E, out, in]
                            return sh
                    elif name.endswith("switch_mlp"):
                        return sh
                    elif len(sh) <= 2:
                        return sh
        return [1]

    both = {n: {"dims": logical_dims(n), "b16": bf.get(n, 0), "b8": fp8.get(n, 0), "b4": e0.get(n, 0)}
            for n in names}
    del dims

    def layer_of(i: int) -> dict:
        pre = f"model.language_model.layers.{i}."
        return {n[len(pre):]: v for n, v in both.items() if n.startswith(pre)}

    sigs: dict[str, dict] = {}
    for i in range(40):
        got = layer_of(i)
        if not got:
            raise SystemExit(f"layer {i} has no tensors")
        key = json.dumps({k: v["dims"] for k, v in sorted(got.items())})
        if key in sigs:
            sigs[key]["indices"].append(i)
        else:
            sigs[key] = {"indices": [i], "items": got}
    arch = {}
    for s in sigs.values():
        dom = "linear" if "linear_attn.in_proj_qkv.weight" in s["items"] else "full"
        arch[f"lm_{dom}"] = s
    if len(arch) != 2:
        raise SystemExit(f"expected two layer archetypes, got {list(arch)}")

    vis_layers: dict[str, list[int]] = {}
    for n in both:
        m = re.match(r"^model\.visual\.blocks\.(\d+)\.(.+)$", n)
        if m:
            vis_layers.setdefault(m.group(2), []).append(int(m.group(1)))
    counts = {k: len(v) for k, v in vis_layers.items()}
    if counts and set(counts.values()) != {27}:
        raise SystemExit(f"visual blocks are not uniform: {counts}")
    vis = {}
    for suffix in vis_layers:
        v = both[f"model.visual.blocks.*.{suffix}"] if f"model.visual.blocks.*.{suffix}" in both else None
        full = f"model.visual.blocks.{vis_layers[suffix][0]}.{suffix}"
        v = both[full]
        vis[f"model.visual.blocks.*.{suffix}"] = {"dims": v["dims"], "count": 27,
                                                 "b16": v["b16"] * 27, "b8": v["b8"] * 27, "b4": v["b4"] * 27}
    io = {n: dict(v, count=1) for n, v in both.items()
          if not n.startswith("model.language_model.layers.")
          and not re.match(r"^model\.visual\.blocks\.\d+\.", n)}
    return {"arch": arch, "vis": vis, "io": io}


CAT = [(r"experts|switch_mlp", "expert"), (r"\.mlp\.gate", "router"), (r"self_attn|linear_attn", "attn"),
       (r"shared_expert", "shared"), (r"norm|_log$|dt_bias|\.bias", "norm"), (r"lora|prerouter", "adapter")]


def cat_of(name: str) -> str:
    if "lora_" in name:
        return "lora"
    if name.startswith("prerouter"):
        return "prerouter"
    if "visual" in name:
        return "vision"
    if "embed_tokens" in name:
        return "vocab"
    if name.startswith("lm_head"):
        return "head"
    for pat, c in CAT:
        if re.search(pat, name):
            return c
    return "norm"


NOTE = {
    "mlp.experts.gate_up_proj": "256 experts × (gate ⊕ up), fused",
    "mlp.experts.down_proj": "256 experts × (512 → 2048), fused",
    "mlp.gate.weight": "router: 256 logits per token",
    "mlp.shared_expert_gate.weight": "shared-expert gate",
    "mlp.shared_expert.gate_proj.weight": "shared expert, gate",
    "mlp.shared_expert.up_proj.weight": "shared expert, up",
    "mlp.shared_expert.down_proj.weight": "shared expert, down",
    "linear_attn.in_proj_qkv.weight": "gated linear attention, fused q/k/v",
    "linear_attn.in_proj_z.weight": "gate z",
    "linear_attn.in_proj_a.weight": "decay a",
    "linear_attn.in_proj_b.weight": "decay b",
    "linear_attn.out_proj.weight": "linear-attention output",
    "linear_attn.conv1d.weight": "short conv, kernel 4",
    "linear_attn.A_log": "state decay (log)",
    "linear_attn.dt_bias": "time-step bias",
    "self_attn.q_proj.weight": "Q projection",
    "self_attn.k_proj.weight": "K projection (GQA)",
    "self_attn.v_proj.weight": "V projection (GQA)",
    "self_attn.o_proj.weight": "attention output",
    "self_attn.q_norm.weight": "per-head Q norm",
    "self_attn.k_norm.weight": "per-head K norm",
}


def js_array(var: str, items, comment: str, vision: bool = False) -> str:
    lines = [f"const {var}=[", f"  /* {comment} */"]
    for name, v in items:
        display = name
        tail = name.split(".", 3)[-1] if name.startswith("model.language_model.layers.") else (
            name.split(".", 2)[-1] if name.startswith("model.visual.blocks.*.") else name)
        note = NOTE.get(tail, "")
        count = v.get("count", 1)
        extra = f",{count}" if count > 1 else ""
        label = display + (f"  ({note})" if note else "")
        lines.append(f'  W({json.dumps(label)},{json.dumps(v["dims"])},{json.dumps(cat_of(name))},'
                     f'"",{{bf16:{v["b16"]},fp8:{v["b8"]},edge0:{v["b4"]}}}{extra}),')
    lines.append("];")
    return "\n".join(lines)


DATA_HEAD = r"""// ---------- data.js ----------
/* Edge0 35B A3B — Tensor Atlas. Architecture from the published config of
   Edge0/Edge0-35B-A3B-preview (base Qwen/Qwen3.6-35B-A3B). Every byte is measured from the
   safetensors headers of the three checkpoints named in Sources, summed per tensor, with
   quantization scales/biases folded into their owner and per-expert tensors folded into the
   fused banks the BF16 checkpoint stores. */
const CFG = {
 text_config:{
  model_type:'qwen3_5_moe',num_hidden_layers:40,hidden_size:2048,
  intermediate_size:6144,shared_expert_intermediate_size:512,moe_intermediate_size:512,
  num_experts:256,num_experts_per_tok:8,
  num_attention_heads:16,num_key_value_heads:2,head_dim:256,
  linear_num_key_heads:16,linear_num_value_heads:32,linear_key_head_dim:128,linear_value_head_dim:128,
  linear_conv_kernel_dim:4,attn_output_gate:true,
  rms_norm_eps:1e-6,vocab_size:248320,max_position_embeddings:262144,
  tie_word_embeddings:false,full_attention_interval:4,
  layer_types:["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],
  kv_source_layer_ids:[3,7,11,15,19,23,27,31,35,39],index_source_layer_ids:[],
  rope_parameters:{rope_theta:10000000,rope_type:'default',partial_rotary_factor:0.25,
                   mrope_interleaved:true,mrope_section:[11,11,10]},
  bos_token_id:248045,eos_token_id:[248046,248044]
 },
 quantization:{group_size:64,bits:4,mode:'affine',eight_bit_overrides:80},
 vision_config:{model_type:'qwen3_5_moe',depth:27,hidden_size:1152,num_heads:16,
   intermediate_size:4304,patch_size:16,spatial_merge_size:2,temporal_patch_size:2,
   out_hidden_size:2048,num_position_embeddings:2304},
 edge0:{active_memory:'under 3 GiB',decode:'15 tok/s',prefill:'140 tok/s',
        lora_delta_vs_fp16:'3.9 points',prerouter_gain:'up to +59%'},
 prerouter_layers:[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
};
const CT=CFG.text_config, VC=CFG.vision_config;
"""

DATA_TAIL = r"""const COL={blue:'#638bff',enc:'#5ca7ff',dec:'#55d7c1',engram:'#b99bff',expert:'#76b9ff',shared:'#d6e99c',attn:'#55ddd0',router:'#f4b765',mhc:'#e3a8dc',vision:'#80ceea',head:'#c4d0ff',full:'#efbc71',reindex:'#b798ff',reuse:'#709bbd',swa:'#758090',norm:'#a7b5cc',muted:'#8390a6',lora:'#e9a6dd',prerouter:'#9be6b4'};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=x=>x*x*(3-2*x);
const num=x=>Math.round(x).toLocaleString('en-US');
function fmtP(p){return p>=1e12?(p/1e12).toFixed(3)+'T':p>=1e9?(p/1e9).toFixed(2)+'B':p>=1e6?(p/1e6).toFixed(2)+'M':p>=1e3?(p/1e3).toFixed(1)+'K':num(p);}
function bytes(n,binary=false){let b=binary?1024:1000,u=binary?['B','KiB','MiB','GiB','TiB']:['B','KB','MB','GB','TB'],k=0;while(n>=b&&k<4){n/=b;k++;}return (k===0?num(n):n.toFixed(n>=100?1:2))+' '+u[k];}
const SOURCE_BASE='https://huggingface.co/Edge0/Edge0-35B-A3B-preview';
const BASE_MODEL='https://huggingface.co/Qwen/Qwen3.6-35B-A3B';
const SOURCES=[
 ['Edge0 model card',SOURCE_BASE,'The subject of this atlas: the edge0 preview of Qwen3.6-35B-A3B as int4 plus LoRA and prerouter adapters, streamed from storage by the edge0 runtime. Runs in under 3 GiB of active memory at 15 tok/s decode, 140 tok/s prefill.'],
 ['Edge0 config',SOURCE_BASE+'/blob/main/config.json','Embedded field for field: MLX affine 4-bit with group_size 64, plus 80 per-module 8-bit overrides (every mlp.gate and shared_expert_gate). Layer types, the linear-attention shapes and the vision config come from the same file.'],
 ['BF16 base',BASE_MODEL,'26 shards, 1,045 tensors, 71.9036 GB. The teacher checkpoint this edge build was distilled from; every byte in the bf16 mode was read from its shard headers.'],
 ['FP8 base',BASE_MODEL+'-FP8','42 layer-wise shards, 64,196 tensors, 37.4548 GB: e4m3 weights with per-block weight_scale_inv, experts written per expert rather than as fused banks.'],
 ['Edge0 checkpoint',SOURCE_BASE+'/tree/main','4 model shards plus lora_edge0_35b.safetensors (42.3 MB) and prerouter_edge0_35b.safetensors (138.4 MB). Text only: the edge build contains no vision tensors at all.'],
 ['Method',SOURCE_BASE,'Every figure on this page is the sum of tensor byte ranges read from those shard headers over HTTP Range. No weights were downloaded, and no number here is a shape × bytes estimate.']
];
const MODE_INFO={
 bf16:{label:'BF16 base',short:'BF16',color:COL.enc,note:'Qwen/Qwen3.6-35B-A3B · 26 shards · 2 bytes per parameter'},
 fp8:{label:'FP8 base',short:'FP8',color:COL.dec,note:'Qwen/Qwen3.6-35B-A3B-FP8 · 42 shards · e4m3 with per-block scales'},
 edge0:{label:'Edge0 int4',short:'EDGE0',color:COL.expert,note:'Edge0 preview · MLX affine int4 · group 64 · + LoRA and prerouter'}
};
const MODE_KEYS=['bf16','fp8','edge0'];
/* This checkpoint has no KV compression and no borrowed cache: thirty gated-linear layers keep
   a constant-size recurrent state, ten full-attention layers keep a growing KV cache. */
function modeFor(i){return CT.layer_types[i]==='full_attention'?'full':'swa';}
function ownerFor(i){return modeFor(i)==='full'?i:null;}
function indexOwnerFor(i){return null;}
function W(name,shape,cat,note='',ex=null,count=1){
 return {name,shape,cat,note,count,
   p:shape.reduce((a,b)=>a*b,1)*count,
   format:ex&&ex.edge0&&ex.edge0<ex.bf16*0.4?'edge0':(ex&&ex.fp8&&ex.fp8<ex.bf16*0.9?'fp8':'bf16'),
   ex};
}
function wBytes(w,mode='bf16'){return w.ex?w.ex[mode]:2*w.p;}
function wFormat(w,mode){return {bf16:'BF16',fp8:'FP8 / block',edge0:'int4 / group 64'}[mode];}
const sumP=ws=>ws.reduce((a,w)=>a+w.p,0);
const sumB=(ws,m)=>ws.reduce((a,w)=>a+wBytes(w,m),0);
@@TABLES@@
const FP8_CONFIG=@@FP8CFG@@;
const E0_CONFIG=@@E0CFG@@;
function layerWeights(i){return modeFor(i)==='full'?FULL_W:LINEAR_W;}
const LAYERS=Array.from({length:40},(_,i)=>({id:'L'+i,index:i,label:'Layer '+String(i).padStart(2,'0'),
 part:modeFor(i)==='full'?'full':'linear',mode:modeFor(i),owner:ownerFor(i),indexOwner:indexOwnerFor(i),
 ratio:1,ws:layerWeights(i)}));
const AUX_TABLES=[];
const DRAFT=[{id:'D0',index:40,label:'MTP layer · FP8 build only',ws:DRAFT_W}];
const EMBED={id:'embed',label:'Token embedding',ws:EMBED_W};
const HEAD={id:'head',label:'Output norm + untied head',ws:HEAD_W};
const SELF={id:'adapters',label:'Recover-LoRA adapters',ws:LORA_W};
const VISION={id:'vision',label:'Vision tower · 27 blocks',ws:VISION_W};
const ALIGNER={id:'aligner',label:'Prerouter head',ws:PRE_W};
const MODULES=[EMBED,...LAYERS,HEAD,...DRAFT,SELF,VISION,ALIGNER];
const ALL_W=MODULES.flatMap(m=>m.ws);
const TOTALS=Object.fromEntries(MODE_KEYS.map(m=>[m,sumB(ALL_W,m)]));
const TOTAL_P=sumP(ALL_W);
const MOE_EXPERT_P=(2*2048*512+512*2048);
const MOE_TOTAL_P=MOE_EXPERT_P*256*40;
const FP8_DELTA=TOTALS.bf16-TOTALS.fp8;
const E0_DELTA=TOTALS.bf16-TOTALS.edge0;
const KV_FULL_PER_TOKEN=2*2*256*2*10;
const KV_LINEAR_STATE=32*128*128*2*30;
const CATEGORIES=[
 ['expert','Routed experts · 256 per layer',COL.expert,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='expert'))],
 ['attn','Attention (linear + full)',COL.attn,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='attn'))],
 ['shared','Shared expert',COL.shared,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='shared'))],
 ['router','Routers and gates',COL.router,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='router'))],
 ['vision','Vision tower + merger',COL.vision,VISION.ws],
 ['adapters','LoRA adapters',COL.lora,SELF.ws],
 ['prerouter','Prerouter heads',COL.prerouter,ALIGNER.ws],
 ['vocab','Embedding + head',COL.head,[...EMBED.ws,...HEAD.ws]],
 ['norm','Norms and scalars',COL.norm,ALL_W.filter(w=>w.cat==='norm')]
];
const EXP={
 overview:{title:'A 35B MoE in phone memory.',body:'The edge0 preview is the same 40-layer Qwen3.6-35B-A3B MoE — 256 experts per layer, 8 routed per token, 3B active — re-quantized to MLX affine int4 and shipped with two small trained adapters. The trick is not the file size: it is that the 19.7 GB checkpoint stays on storage and experts are streamed in as they are routed, so only the active set is ever in RAM.'},
 expert:{title:'256 experts, 8 per token.',body:'Each layer stores its experts fused in the BF16 base: [256, 1024, 2048] for gate_up and [256, 2048, 512] for down — 8.4 GB of the 71.9 GB base is expert mass at BF16. Only 8 of the 256 fire per token, and the shared expert (512-wide) always runs beside them. In the int4 build the whole bank is one SwitchLinear per projection, packed as U32 weights with group-64 scales and biases.'},
 full:{title:'Every fourth layer sees everything.',body:'Layers 3, 7, 11 … 39 are full attention: 16 query heads, 2 KV heads, head_dim 256, with per-head q and k norms and a quarter-rotary rope at theta 1e7. Only these ten layers hold a KV cache that grows with context; the other thirty hold a fixed-size recurrent state instead.'},
 swa:{title:'Thirty gated-linear layers.',body:'The default layer is a gated linear attention block: a fused q/k/v projection (16 key heads and 32 value heads at 128 dims), gate z, decay a/b, a kernel-4 short convolution, a per-head norm and a 2048→2048 output projection. Its state is constant-size — which is what makes a long prompt affordable on a phone.'},
 selfcond:{title:'42.3 MB of LoRA that buy back the quantization.',body:'Recover-LoRA keeps the int4 base frozen and trains adapters by distillation from the FP teacher. Measured here: 620 tensors, 42.3 MB over all 40 layers — linear-attention projections, the shared expert, and q/k/v/o of the full-attention layers. The card reports the int4 model landing within 3.9 points of its fp16 base.'},
 vision:{title:'The base model has eyes; this build does not.',body:'The BF16 base carries a 27-block vision tower (hidden 1152, 16 heads, patch 16, spatial merge 2) plus a merger, ~1.0 GB of the checkpoint. The edge preview ships text only: measured across all six of its files, there is not one visual tensor. That is a real difference between the modes on this page, not a rounding artifact.'},
 bf16:{title:'The teacher checkpoint.',body:'71.9036 GB across 26 shards and 1,045 tensors: 40 layers, 256 experts per layer written as fused banks, a 248,320-row untied embedding and output head, and the vision tower.'},
 fp8:{title:'FP8 is the middle rung, not a small file.',body:'37.4548 GB in 42 layer-wise shards. Qwen ships this build with e4m3 weights and per-block weight_scale_inv, and it writes experts per expert instead of fused — 64,196 tensors against the base 1,045. Same architecture, different bookkeeping: 52% of the BF16 size.'},
 edge0:{title:'int4 group 64, and two adapters that matter.',body:'19.6895 GB in four model shards — 3.65x smaller than the base — plus lora_edge0_35b.safetensors (42.3 MB, 620 tensors) and prerouter_edge0_35b.safetensors (138.4 MB, 99 tensors over 33 layers). Every linear is packed as U32 with group-64 scales and biases; 80 router modules are kept at 8-bit on purpose, because routing errors are not recoverable by distillation.'},
 storage:{title:'Where 71.9 GB goes, and what survives int4.',body:'Virtually all of it is expert mass. Flip the mode and watch the same architecture re-price itself: FP8 halves it, the edge build takes it to a quarter — and the two adapters together are 180 MB, 0.9% of the edge checkpoint.'},
 cache:{title:'Ten caches that grow, thirty that do not.',body:'Full-attention layers keep a KV cache at 2 heads × 256 dims; the thirty gated-linear layers keep a fixed recurrent state whose size does not depend on the context. That is the whole reason a 262,144-token model can run from storage on a phone.'}
};
const BENCH=[['GPQA Diamond','Reasoning',[79.8,93.4,94.1,92.9,88.1,92.4,89.9,90.9]],['Terminal-Bench 2.1','Agentic',[null,89.1,88.8,88.3,88.2,87.9,82.7,90.6]],['Terminal-Bench 3.0','Agentic',[null,43.3,34.4,17.7,28.3,11.8,7.6,30]],['Terminal-Bench 4.0','Agentic',[null,51.8,39.9,12.6,37.9,12.4,7,31.2]],['DeepSWE v1.1','Agentic',[null,74,73,67.5,66.9,62.7,54.4,74.2]],['ProgramBench','Agentic',[null,37,23,17.5,19,15.5,null,20.3]],['NL2Repo-Bench','Agentic',[null,75.3,56.8,58,58,61.5,54.2,64]],['CyberGym','Agentic',[null,null,84.5,80,84.5,83.3,76.7,88.1]],['SEC-Bench Pro','Agentic',[null,null,74.3,null,null,56.4,30.9,62.8]],['ExploitGym','Agentic',[null,22.1,33.7,null,15,5.4,1.8,15.3]],['HLE with tools','Agentic',[null,63.6,null,59.8,62.5,60,51.5,63.9]],['AutomationBench','Agentic',[null,50.3,45.8,46.7,48.8,43.2,37.7,54.8]],['Agent\'s Last Exam','Agentic',[null,28.6,26.7,27.6,28.5,25.7,25.2,31.8]],['Chartography with tools','Visual',[null,84,79.9,68.1,null,null,null,78.9]],['BabyVision with tools','Visual',[null,94.1,88.9,85.7,null,null,null,89.6]],['ZeroBench-main (Pass@5)','Visual',[null,52,53,41,null,null,null,49]]];
const BENCH_MODELS=['Edge0 35B A3B (this atlas)','Opus-5.0','GPT-5.6 Sol','K3','GLM-5.3','DS V4 Pro','DS V4 Flash','DS V4.1 Flash']; const CARD_BENCH={"src": "Edge0/Edge0-35B-A3B-preview", "headers": ["Benchmark", "edge0-35b (int4)", "Qwen3.6-35B-A3B (fp16)"], "rows": [["AIME 2026", "86.6", "92.7"], ["HumanEval", "90.9", "95.1"], ["GPQA-Diamond", "79.8", "81.8"], ["MMLU-Pro", "81.0", "84.6"], ["IFBench", "57.9", "61.7"]]};
/* Publisher-reported leaderboard as published on the original atlas page (each model's
   own card). Edge0-35B-A3B is not in that publisher set; its own card figure (int4 build) is shown first for the benchmarks its card publishes, and a blank cell means the card does not publish that benchmark. */

/* Publisher-reported leaderboard as published on the original atlas page (each model's
   own card). Edge0-35B-A3B is not in that publisher set; its own card figure (int4 build) is shown first for the benchmarks its card publishes, and a blank cell means the card does not publish that benchmark. */

function pickExperts(seed,n=256,k=8){let x=(seed+1)*2654435761>>>0;const s=new Set;while(s.size<k){x=(Math.imul(x,1664525)+1013904223)>>>0;s.add(x%n);}return [...s].sort((a,b)=>a-b);}
const PHASES=[
 {name:'Route',label:'Route',from:0,to:6,active:'8 of 256',color:COL.router,caption:'The router picks, the prerouter guesses ahead.',desc:'Each token hits the router (256 logits) and the top 8 experts are chosen. The trained prerouter runs one step ahead of the forward pass, so the expert fetch for layer N+1 starts while layer N is still computing.'},
 {name:'Fetch',label:'Fetch',from:6,to:11,active:'from SSD',color:COL.expert,caption:'Only the routed experts move.',desc:'Expert banks live on storage, not in RAM. The runtime fetches the active 8 experts as they are routed; peak memory is bounded by the active set, which is how a 19.7 GB checkpoint runs in under 3 GiB.'},
 {name:'Compute',label:'Compute',from:11,to:17,active:'3B active',color:COL.enc,caption:'Attention plus the expert FFN.',desc:'Thirty gated-linear layers keep a constant-size state; ten full-attention layers keep a KV cache. The MoE output is mixed with the always-on shared expert, then the LoRA delta is added on top of the int4 weights.'},
 {name:'Generate',label:'Generate',from:17,to:20,active:'15 tok/s',color:COL.dec,caption:'One token, then the next.',desc:'The card reports 15 tok/s decode and 140 tok/s prefill in the edge0 runtime. This atlas measures bytes; those numbers are the publisher\u2019s throughput claim.'},
 {name:'Adapt',label:'Adapt',from:20,to:22,active:'+LoRA',color:COL.lora,caption:'Adapters correct the 4-bit error.',desc:'Recover-LoRA was trained by distillation from the FP teacher with the int4 base frozen. 42.3 MB of adapters across 40 layers carry the correction.'}
];
const TRAIN_PHASES=[
 {name:'Teacher',from:0,to:8,active:'FP forward',color:COL.enc,caption:'Run the FP teacher.',desc:'Conceptual schematic of the Recover-LoRA recipe: the frozen 4-bit student and the FP teacher are compared, not trained against text labels alone.'},
 {name:'Distil',from:8,to:11,active:'Compare',color:COL.engram,caption:'Match the teacher\u2019s distribution.',desc:'Only the LoRA matrices receive gradients; the packed int4 weights stay frozen. This is what the 42.3 MB of adapters buys.'},
 {name:'Backward',from:11,to:18,active:'Adapters only',color:COL.mhc,caption:'Update A and B only.',desc:'Gradients reach the adapters, never the quantized base — which is also why the checkpoint ships base and adapters as separate files.'}
];
function phasesFor(mode){return mode==='training'?TRAIN_PHASES:PHASES;}
function phaseAt(t,mode='inference'){const ps=phasesFor(mode);return ps.find(p=>t>=p.from&&t<p.to)||ps[ps.length-1];}
const TC={q:'#8caaff',local:'#ffc477',kv:'#61dccb',index:'#f5a078',out:'#b2bfff',expert:'#66deb0',shared:'#eee081',router:'#f494be',mhc:'#c9a3fa',norm:'#90a9b5',engram:'#d7b57b',vision:'#62d5d0',head:'#a5b9eb',attn:'#55ddd0',lora:'#e9a6dd',prerouter:'#9be6b4',vocab:'#a5b9eb'};
function tensorKind(w){return TC[w.cat]?w.cat:'head';}
function tensorColor(w){return TC[tensorKind(w)]||COL.attn;}
function tensorShort(w){return w.name.replace(/^model\.language_model\.layers\.\d+\./,'').replace(/^model\.visual\.blocks\.\*\./,'ViT.').replace(/^model\.visual\./,'visual.').replace(/^model\.language_model\./,'').replace(/^lora::/,'LoRA ').replace(/^prerouter::/,'Prerouter ').replace(/\.weight$/,'').replace(/\.linear$/,'');}
function weightParts(w,mode){const total=wBytes(w,mode);return {data:total,scales:0,aux:0,total};}
function displayWeights(m){return m.ws;}
function findWeight(m,name){return displayWeights(m).find(w=>w.name===name)||m.ws.find(w=>w.name===name);}
function orderWeights(m){const ws=displayWeights(m);const rank=w=>{let n=w.name;
 if(n.includes('experts.gate_up'))return 0; if(n.includes('experts.down'))return 1;
 if(n.endsWith('mlp.gate.weight'))return 2; if(n.includes('shared_expert_gate'))return 3;
 if(n.includes('shared_expert.gate'))return 4; if(n.includes('shared_expert.up'))return 5; if(n.includes('shared_expert.down'))return 6;
 if(n.includes('in_proj_qkv'))return 7; if(n.includes('in_proj_z'))return 8; if(n.includes('in_proj_a'))return 9; if(n.includes('in_proj_b'))return 10;
 if(n.includes('conv1d'))return 11; if(n.includes('dt_bias'))return 12; if(n.includes('A_log'))return 13;
 if(n.includes('linear_attn.norm'))return 14; if(n.includes('linear_attn.out_proj'))return 15;
 if(n.includes('q_proj'))return 16; if(n.includes('k_proj'))return 17; if(n.includes('v_proj'))return 18; if(n.includes('o_proj'))return 19;
 if(n.includes('q_norm')||n.includes('k_norm'))return 20;
 if(n.includes('lora'))return 21; if(n.includes('prerouter'))return 0;
 if(n.includes('input_layernorm'))return 22; if(n.includes('post_attention'))return 23;
 if(n.includes('embed_tokens'))return 0; if(n.includes('lm_head'))return 1;
 if(n.includes('visual'))return 2; return 24;};
 return [...ws].sort((a,b)=>rank(a)-rank(b));}
function isAttentionTensor(w){return w.cat==='attn';}
function tensorInfo(w,m){
 const p=fmtP(w.p),size=bytes(wBytes(w,'bf16'));
 let t='Stored tensor.',b='Two bytes per parameter in the BF16 base. Flip the precision to see what FP8 and the int4 edge build do with it.';
 if(w.name.includes('experts.gate_up')){t='Fused expert gate and up banks.';b='[256, 1024, 2048] in the BF16 base: every expert gate (512×2048) and up (512×2048) side by side. In the FP8 base the same mass is written per expert, and the edge build stores it as one MLX SwitchLinear packed in U32 with group-64 scales and biases. Only 8 of the 256 banks are ever active for a token.';}
 else if(w.name.includes('experts.down')){t='Fused expert down banks.';b='[256, 2048, 512]: each of the 256 experts projects its 512-wide activation back to the 2048-dim residual stream. Fused in BF16, per-expert in FP8, one packed SwitchLinear in the int4 build.';}
 else if(w.name.endsWith('mlp.gate.weight')){t='Router.';b='2048 → 256 logits, one per expert. The top 8 fire and are mixed by routing weight. This module is deliberately kept at 8-bit in the edge build (one of 80 such overrides): routing is discrete, so quantization damage there is not recoverable by adapter training.';}
 else if(w.name.includes('shared_expert_gate')){t='Shared-expert gate.';b='512-wide gate controlling how much of the always-on shared expert path enters the residual stream. Also kept at 8-bit in the int4 build.';}
 else if(w.name.includes('shared_expert')){t='Shared expert.';b='The always-on path beside the routed experts: 2048 → 512 → 2048. Every token passes through it, so it is dense by construction; its three matrices are also LoRA targets in the edge build.';}
 else if(w.name.includes('in_proj_qkv')){t='Gated linear attention, fused q/k/v.';b='16 key heads and 32 value heads at 128 dims come out of one projection. Together with in_proj_z (the gate), in_proj_a/b (the per-channel decay) and the kernel-4 convolution, this is the state-space block that replaces attention in 30 of the 40 layers.';}
 else if(w.name.includes('in_proj_z')){t='Gate z.';b='Input gate of the gated delta-rule recurrence: 2048 → 4096 (32 value heads × 128), multiplied elementwise into the block output.';}
 else if(w.name.includes('in_proj_a')||w.name.includes('in_proj_b')){t='Decay a / b.';b='Two small projections (2048 → 32) that produce the per-head decay used to forget the recurrent state. 65,536 parameters each — the cheapest part of the block and the reason the state can be constant-size.';}
 else if(w.name.includes('conv1d')){t='Short convolution.';b='Kernel 4, 8,192 channels, depthwise: a four-token causal convolution over the fused q/k/v stream before the recurrence.';}
 else if(w.name.includes('A_log')){t='State decay, stored in log space.';b='32 values per layer. Stored as a log so the runtime can exponentiate it into a positive decay.';}
 else if(w.name.includes('dt_bias')){t='Time-step bias.';b='32 values per layer added to the learned step size of the recurrence.';}
 else if(w.name.includes('A_log')||w.name.includes('dt_bias')){t='Linear-attention scalar.';b='Part of the gated delta-rule recurrence state.';}
 else if(w.name.includes('q_proj')){t='Query projection.';b='Full-attention layers only: 2048 → 4096 (16 heads × 256). A per-head q norm follows.';}
 else if(w.name.includes('k_proj')){t='Key projection.';b='Full-attention layers only: 2048 → 512 (2 KV heads × 256) — grouped-query attention with a very small KV footprint, which is what keeps the growing cache cheap.';}
 else if(w.name.includes('v_proj')){t='Value projection.';b='Full-attention layers only: 2048 → 512 (2 KV heads × 256).' ;}
 else if(w.name.includes('o_proj')){t='Attention output projection.';b='4096 → 2048 in the ten full-attention layers, 2048 → 2048 in the thirty linear ones.';}
 else if(w.name.includes('q_norm')||w.name.includes('k_norm')){t='Per-head norm.';b='256 scales applied per head to queries and keys before the dot product in the full-attention layers.';}
 else if(w.name.includes('lora')){t='LoRA adapter matrix.';b='620 adapter tensors over all 40 layers, 42.3 MB, trained by distillation from the FP teacher while the int4 base stayed frozen. For attention and the shared expert, never for the routed experts.';}
 else if(w.name.includes('prerouter')){t='Prerouter head.';b='Three matrices per layer (fc1, fc2 and a linear init) across 33 layers: 138.4 MB that predict the next layer\u2019s routing one step early, so the expert fetch overlaps computation instead of stalling it.';}
 else if(w.name.includes('embed_tokens')){t='Token embedding.';b='[248,320, 2048] — the vocabulary is untied here, so the input table and the output head are separate tensors. The int4 build packs both with group-64 scales and biases.';}
 else if(w.name.includes('lm_head')){t='Output head.';b='2048 → 248,320 logits, untied from the embedding: its own 1.0 GB at BF16, and its own packed copy in the int4 build.';}
 else if(w.name.includes('visual')){t='Vision tower tensor.';b='Part of the 27-block tower (hidden 1152, 16 heads, patch 16, spatial merge 2) plus the merger. Present in both base modes and completely absent from the edge build.';}
 else if(w.name.includes('norm')){t='RMSNorm.';b='Layer normalisation (eps 1e-6). Norms stay high precision in every mode here — they are 0.00 GB of the ledger, and quantizing them would save nothing.';}
 return {title:t,body:b,size,p};}
function layerStory(m){
 if(!m.mode)return null;let t,body,detail;
 if(m.mode==='swa'){t='Gated-linear layer';body='A fused q/k/v projection, per-channel decay, a kernel-4 convolution and a 2048 → 2048 output, then 8 of 256 experts plus the shared expert.';detail='This is 30 of the 40 layers. Its recurrent state has a fixed size: a long prompt does not grow its memory. Expert mass dominates the block — at BF16 that is 8.4 GB across the thirty of them.';}
 else {t='Full-attention layer';body='16 query heads over 2 KV heads at head_dim 256, with per-head q and k norms, then the same 8-of-256 expert bank.';detail='Ten layers — 3, 7, 11 … 39 — keep a KV cache that grows with the context at 2 heads × 256 dims. They are the only blocks whose attention shapes differ from the rest.';}
 return {title:t,body,detail};}
"""


def build_data_js(t: dict) -> str:
    arch, io = t["arch"], t["io"]
    linear = sorted(arch["lm_linear"]["items"].items())
    full = sorted(arch["lm_full"]["items"].items())
    vis = sorted(t["vis"].items())

    def is_layer(n):
        return n.startswith("model.language_model.layers.")

    embed = sorted((n, v) for n, v in io.items() if "embed_tokens" in n)
    head = sorted((n, v) for n, v in io.items() if n.startswith("lm_head") or n.endswith('language_model.norm.weight'))
    lora = sorted((n, v) for n, v in io.items() if "lora::" in n or re.search(r"lora_[AB]", n))
    pre = sorted((n, v) for n, v in io.items() if n.startswith("prerouter::") or "prerouter" in n)
    visio = sorted((n, v) for n, v in io.items() if n.startswith("model.visual.") and n not in t["vis"])
    mtp = sorted((n, v) for n, v in io.items() if n.startswith('mtp.'))
    placed = {n for g in (embed, head, lora, pre, visio, mtp) for n, _ in g} | set(t["vis"])
    missing = sorted(set(io) - placed)
    if missing:
        raise SystemExit(f"top-level tensors placed in no module: {missing[:6]}")
    tables = "\n".join([
        js_array("LINEAR_W", linear, "one gated-linear layer: fused q/k/v + gate + decay + conv + 8-of-256 experts"),
        js_array("FULL_W", full, "one full-attention layer: q/k/v/o + the same expert bank"),
        js_array("EMBED_W", embed, "the untied input embedding"),
        js_array("HEAD_W", head, "the untied output head"),
        js_array("LORA_W", lora, "Recover-LoRA adapters, aggregated across the layers that carry them", True),
        js_array("PRE_W", pre, "prerouter heads, aggregated across 33 layers", True),
        js_array("VISION_W", vis, "one vision block ×27 (bytes already summed)", True),
        js_array("VISIO_W", visio, "merger tensors (vision present in both base modes, absent in the edge build)", True),
        js_array("DRAFT_W", mtp, "the MTP layer the FP8 repository ships (absent from the base and the int4 build)", True),
    ])
    js = DATA_HEAD + DATA_TAIL
    js = js.replace("const SELF={id:'adapters',label:'Recover-LoRA adapters',ws:LORA_W};",
                    "const SELF={id:'adapters',label:'Recover-LoRA adapters',ws:LORA_W};")
    js = js.replace("const VISION={id:'vision',label:'Vision tower · 27 blocks',ws:VISION_W};",
                    "const VISION={id:'vision',label:'Vision tower · 27 blocks',ws:[...VISION_W,...VISIO_W]};")
    js = js.replace("@@TABLES@@", tables)
    js = js.replace("@@FP8CFG@@", json.dumps(load(CFG_FP8), ensure_ascii=False))
    js = js.replace("@@E0CFG@@", json.dumps(load(CFG_E0), ensure_ascii=False))
    return js


REGEX_SUBS = [
 [
  "'native'",
  "'bf16'"
 ],
 [
  "'nvfp4'",
  "'edge0'"
 ],
 [
  "'ENCODER / L00-L19'",
  "'LAYERS 00-19'"
 ],
 [
  "'First 20 layers / runs in decode'",
  "'first reading group'"
 ],
 [
  "'DECODER / L20-L39'",
  "'LAYERS 20-39'"
 ],
 [
  "'Next 20 layers / runs in decode'",
  "'second reading group'"
 ],
 [
  "'ONE CHECKPOINT / L00-L39'",
  "'ONE STACK / 30 GATED-LINEAR + 10 FULL'"
 ],
 [
  "'Ordered layers, not physical shard offsets\\.'",
  "'One stack, two block types, four layers apart.'"
 ],
 [
  "'L19 -> L20'",
  "'L19 → L20'"
 ],
 [
  "'Optimized prefill: prepare decoder KV'",
  "'The stack continues: no group switch'"
 ],
 [
  "'Forward pass continues\\. No model swap\\.'",
  "'Forward pass continues over the same stack.'"
 ],
 [
  "'Prefill'",
  "'Route'"
 ],
 [
  "'SWA replay'",
  "'Fetch'"
 ],
 [
  "'Decode'",
  "'Compute'"
 ],
 [
  "'DSpark'",
  "'Adapt'"
 ],
 [
  "'Decode / all 40 layers'",
  "'Compute / 3B active'"
 ],
 [
  "'Optimized prompt prefill'",
  "'Route + fetch'"
 ],
 [
  "const n=m\\.id\\.startsWith\\('D'\\)\\?128:384",
  "const n=CT.num_experts"
 ],
 [
  "cols=n===128\\?16:24",
  "cols=16"
 ],
 [
  "n\\+' EXPERTS / TOP-'\\+\\(n===128\\?3:6\\)",
  "n+' EXPERTS / TOP-'+CT.num_experts_per_tok"
 ],
 [
  "\\$\\{selected\\.id\\.startsWith\\('D'\\)\\?'128':'384'\\} experts",
  "256 experts"
 ],
 [
  "pickExperts\\(m\\.index\\*1000\\+Math\\.floor\\(this\\.clock\\*\\.65\\),g\\.n,g\\.n===128\\?3:6\\)",
  "pickExperts(m.index*1000+Math.floor(this.clock*.65),g.n,CT.num_experts_per_tok)"
 ],
 [
  "w\\.name\\.includes\\('\\.w1\\.'\\)\\?'W1 / gate':w\\.name\\.includes\\('\\.w3\\.'\\)\\?'W3 / up':'W2 / down'",
  "w.name.includes('gate_up')?'gate + up bank':'down bank'"
 ],
 [
  "'Expert '\\+String\\(i\\)\\.padStart\\(3,'0'\\)",
  "'Expert '+String(i).padStart(3,'0')+' of 256'"
 ],
 [
  "const pairs=\\[\\[.*?\\]\\];",
  "const pairs=[['linear_attn.in_proj_qkv.weight','linear_attn.conv1d.weight'],['linear_attn.conv1d.weight','linear_attn.out_proj.weight'],['mlp.gate.weight','mlp.experts.gate_up_proj'],['mlp.experts.gate_up_proj','mlp.experts.down_proj'],['self_attn.q_proj.weight','self_attn.o_proj.weight']];"
 ],
 [
  "m\\.id\\[0\\]==='E'\\?TC\\.engram:m\\.id\\[0\\]==='D'\\?TC\\.mhc",
  "m.id===('adapters')?TC.lora:m.id==='D0'?TC.prerouter"
 ],
 [
  "m\\.id\\[0\\]==='E'\\?\\{glow:\\.02,style:2,dim\\}",
  "m.id==='adapters'?{glow:.02,style:2,dim}"
 ],
 [
  "m\\.id\\[0\\]==='E'\\?TC\\.engram:m\\.id\\[0\\]==='D'\\?TC\\.mhc:TC\\.vision",
  "m.id==='adapters'?TC.lora:m.id==='aligner'?TC.prerouter:TC.vision"
 ],
 [
  "if\\(isLayer\\)\\{const c=COL\\[m\\.mode\\]\\|\\|TC\\.mhc;",
  "if(isLayer){const c=COL[m.mode]||TC.expert;"
 ],
 [
  "glow:m\\.index<40\\?this\\.layerHeat\\(m\\.index\\):ph==='Adapt'\\?\\.42:0",
  "glow:this.layerHeat(m.index)"
 ],
 [
  "'PREDICTION vs TARGET','Loss \\(schematic\\)',COL\\.engram",
  "'DISTILLATION LOSS','LoRA receives the gradient',COL.auxa"
 ],
 [
  "'GRADIENTS','Backward path, not reverse inference\\.',COL\\.mhc",
  "'GRADIENTS','Adapters only, the int4 base stays frozen.',COL.auxb"
 ],
 [
  "COL\\.engram",
  "COL.auxa"
 ],
 [
  "TC\\.engram",
  "TC.auxa"
 ],
 [
  "COL\\.mhc",
  "COL.auxb"
 ],
 [
  "TC\\.mhc",
  "TC.auxb"
 ],
 [
  "engram:'#",
  "auxa:'#"
 ],
 [
  "mhc:'#",
  "auxb:'#"
 ],
 [
  "\\bENGRAM\\b",
  "AUX_TABLES"
 ],
 [
  "cache\\(\\)\\{const s=this\\.state,ids=\\[2,8,14,20\\];",
  "cache(){const s=this.state,ids=CT.kv_source_layer_ids;"
 ],
 [
  "let n=id<20\\?Math\\.floor\\(s\\.context/2\\):s\\.context,b=n\\*356",
  "let n=s.context,b=n*2*2*256*2"
 ],
 [
  "i===3\\?COL\\.dec:COL\\.full",
  "COL.full"
 ],
 [
  "\\('L'\\+String\\(id\\)\\.padStart\\(2,'0'\\),\\('2:1 / ':'1:1 / '\\)\\+bytes\\(b,s\\.binary\\)",
  "this.label([p[0],2.0+side/2,0],'L'+String(id).padStart(2,'0'),'KV / '+bytes(b,s.binary)"
 ],
 [
  "const selected=pickExperts\\(Math\\.floor\\(this\\.clock\\*\\.4\\),2048,512\\),set=new Set\\(selected\\);",
  "const selected=pickExperts(Math.floor(this.clock*.4),CT.num_experts,CT.num_experts_per_tok),set=new Set(selected);"
 ],
 [
  "for\\(let i=0;i<2048;i\\+\\+\\)\\{let x=\\(i%64-31\\.5\\)\\*\\.077,y=-\\.5-Math\\.floor\\(i/64\\)\\*\\.052;this\\.point\\(\\[x,y,0\\],set\\.has\\(i\\)\\?COL\\.dec:'#4a6280',set\\.has\\(i\\)\\?3\\.3:2,set\\.has\\(i\\)\\?\\.72:\\.22\\);\\}",
  "for(let i=0;i<256;i++){let x=(i%16-7.5)*.26,y=-.5-Math.floor(i/16)*.26;this.point([x,y,0],set.has(i)?COL.expert:'#4a6280',set.has(i)?3.6:2,set.has(i)?.8:.22);}"
 ],
 [
  "'2,048 CANDIDATE BLOCKS x 8'",
  "'256 EXPERTS PER LAYER'"
 ],
 [
  "'A schematic pool; the true cap is 16,384 candidate positions\\.'",
  "'Eight of them are routed for a sample token; all 256 stay in the file.'"
 ],
 [
  "'TOP-512 POSITIONS PER QUERY'",
  "'8 ROUTED OF 256'"
 ],
 [
  "'512 illuminated sample marks illustrate sparse selection\\.'",
  "'Illuminated marks show one token\\u2019s routing, not extra bytes.'"
 ],
 [
  "'FOUR SHARED GLOBAL BANKS'",
  "'TEN FULL-ATTENTION LAYERS'"
 ],
 [
  "'890 B / original token across the model, not per layer\\.'",
  "'20,480 B per token in BF16, and they grow with the context.'"
 ],
 [
  "'THE CHECKPOINT, BY VOLUME','One cubic world unit = one decimal GB\\. No category is enlarged\\.'",
  "'THE CHECKPOINT, BY VOLUME','One cubic world unit = one decimal GB. Measured payloads, not estimates.'"
 ],
 [
  "or silently replace the estimated 3D scene",
  "or silently replace the measured 3D scene"
 ],
 [
  "SCHEMA-DERIVED PAYLOAD",
  "MEASURED FROM SHARD HEADERS"
 ],
 [
  "if\\(!train&&ph\\.name==='Adapt'\\)for\\(let i=0;i<3;i\\+\\+\\)this\\.curve\\(pos\\('L'\\+\\(37\\+i\\)\\),pos\\('D'\\+i\\),TC\\.auxb,\\.55,i\\*\\.3,3,2\\);",
  "if(!train&&ph.name==='Fetch'){this.curve(pos('L38'),pos('D0'),TC.prerouter,.5,.3,3,2);this.label(V3.add(pos('D0'),[0,1.2,0]),'MTP / FP8 BUILD',null,TC.prerouter,8);}"
 ]
]
LITERAL_SUBS = [
    ("<p class=\"guide-copy\">The ledger remains schema-derived. A complete set of checkpoint headers is required before calling its totals exact on-disk measurements.</p>",
     "<p class=\"guide-copy\">Every total on this page is measured: the three checkpoints were read header by header over HTTP Range and summed tensor by tensor. The audit panel in Sources runs the same arithmetic on shards you already have.</p>"),
    ("<${Note} color=${s.precision==='edge0'?COL.dec:COL.enc}><strong>${s.precision==='bf16'?'A reference, not a checkpoint.':EXP[s.precision].title}</strong> ${s.precision==='bf16'?'Every logical value is charged two bytes. This baseline is hypothetical, not an available BF16 repository or a device-memory prediction.':EXP[s.precision].body}<//>",
     "<${Note} color=${MODE_INFO[s.precision].color}><strong>${EXP[s.precision].title}</strong> ${EXP[s.precision].body}<//>"),
    ("<span>Schema-derived tensor payload estimate</span>",
     "<span>Measured: every shard header read, tensor by tensor</span>"),
    ("aria-label=\"DeepSeek V4.1 Flash architecture explorer home\"",
     "aria-label=\"Edge0 35B A3B architecture explorer home\""),
    ("<h1>DeepSeek <em>V4.1 Flash</em>",
     "<h1>Edge0 <em>35B A3B</em>"),
    ("Model data is attributed to DeepSeek-AI's card, configuration and reference implementation. The model card identifies the repository and weights as MIT-licensed.",
     "Model data is attributed to the Edge0 model card and to Qwen's published configuration for the base checkpoint; the three byte inventories come from the safetensors headers of Edge0/Edge0-35B-A3B-preview, Qwen/Qwen3.6-35B-A3B and Qwen/Qwen3.6-35B-A3B-FP8. Apache-2.0 weights; the page engine is MIT. No weight data is bundled."),
    ("'deepseek-exact-selected-shard-audit.json'",
     "'edge0-35b-shard-audit.json'"),
    ("{style:id==='engram'?2:0,glow:id==='routed'?.06:0}",
     "{style:(id==='adapters'||id==='prerouter')?2:0,glow:id==='expert'?.06:0}"),
    ("CFG,NV_CONFIG,TOTALS,TOTAL_P,MODULES",
     "CFG,FP8_CONFIG,E0_CONFIG,TOTALS,TOTAL_P,MODULES"),
    ("'One forward pass, shown in two groups'",
     "'One stack, two block types'"),
    ("'Decode: L00-L19 then L20-L39 then head. No layers skipped.'",
     "'Thirty gated-linear layers and ten full-attention layers, interleaved every fourth block.'"),
    ("'The checkpoint in layer order'",
     "'The checkpoint in block order'"),
    ("'MTP / FP8 BUILD',null",
     "'MTP / FP8 BUILD','FP8 build only'"),
    ("const f=Math.sqrt(b/sumB(m.ws,'bf16'))",
     "const f=Math.max(.06,Math.sqrt(b/sumB(m.ws,'bf16'))||0)"),
    ("f=Math.sqrt(total/sumB(m.ws,'bf16'))",
     "f=Math.max(.06,Math.sqrt(total/sumB(m.ws,'bf16'))||0)"),
    ("maxError=Math.max(maxError,Math.abs(v-b.bytes)/Math.max(1,b.bytes))",
     "if(Number.isFinite(v))maxError=Math.max(maxError,Math.abs(v-b.bytes)/Math.max(1,b.bytes))"),
    ("coords=[[-2.6-rSide/2,0,0],[-1.4+eSide/2,0,0],[-1.4+eSide+2.4,0,-2.1],[-1.4+eSide+5.4,0,-2.1],[-1.4+eSide+5.4,0,2],[-1.4+eSide+2.4,0,2],[-4,0,Math.max(rSide,eSide)/2+2.4],[.6,0,Math.max(rSide,eSide)/2+2.4]];",
     "sizes=CATEGORIES.map(c=>Math.cbrt(sumB(c[3],s.precision)/VOLUME_UNIT)),cols=3,gap=Math.max(...sizes,1)+2.6,rows=Math.ceil(CATEGORIES.length/cols),coords=CATEGORIES.map((c,i)=>[(i%cols-(cols-1)/2)*gap,0,(Math.floor(i/cols)-(rows-1)/2)*gap]);"),
    ("if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();else this.architecture();",
     "try{if(this.state.view==='storage')this.storage();else if(this.state.view==='cache')this.cache();else this.architecture();}catch(err){if(!this.viewError){this.viewError=1;console.error('atlas: '+this.state.view+' view failed:',err);}}"),
    ("pos('D2')", "pos('D0')"),
    ("<div class=\"format-notice\"><b>+${bytes(NV_DELTA,s.binary)}</b> modeled vs native / smaller scale groups</div>",
     "<div class=\"format-notice\"><b>${bytes(TOTALS.bf16-TOTALS[s.precision],s.binary)} smaller</b> than the BF16 base / measured from shard headers</div>"),
]

def main() -> int:
    import objectcode
    import panels
    t = templates()
    totals = {"bf16": load(BF_JSON)["payload_bytes"], "fp8": load(FP8_JSON)["payload_bytes"],
              "edge0": load(E0_JSON)["payload_bytes"]}
    arch, io = t["arch"], t["io"]
    key = {"bf16": "b16", "fp8": "b8", "edge0": "b4"}

    def page_total(mode: str) -> int:
        k = key[mode]
        lin = sum(v[k] for v in arch["lm_linear"]["items"].values()) * len(arch["lm_linear"]["indices"])
        ful = sum(v[k] for v in arch["lm_full"]["items"].values()) * len(arch["lm_full"]["indices"])
        return lin + ful + sum(v[k] for v in io.values()) + sum(v[k] for v in t["vis"].values())

    print(f"data · {len(arch['lm_linear']['indices'])} linear / {len(arch['lm_full']['indices'])} full / "
          f"{len(t['vis'])} vision x27 / {len(io)} top-level")
    for m in totals:
        if page_total(m) != totals[m]:
            print(f"GATE FAIL page {m} {page_total(m):,} vs measured {totals[m]:,}")
            return 3
    print("GATE payload == measured ✔ " + " · ".join(f"{m} {page_total(m)/1e9:.4f} GB" for m in totals))

    if "--data-only" in sys.argv:
        (DIR / "data.js").write_text(build_data_js(t), encoding="utf-8")
        print("wrote data.js")
        return 0

    data_js = build_data_js(t)
    lines = SRC.read_text(encoding="utf-8").split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.startswith("const CFG = {"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("class CanvasRenderer{"))
    text = "\n".join(lines[:i0] + data_js.split("\n") + [''] + lines[i1:])
    print(f"splice · lines {i0+1}..{i1} -> data.js ({len(data_js.splitlines())} lines)")

    text, rep = objectcode.apply_rewrites(text, panels.REWRITES)
    print(f"panels · {len(rep)} whole-line rewrites applied (structure checked)")

    n_re = 0
    for pat, sub in REGEX_SUBS:
        text, k = re.subn(pat, lambda _m, sub=sub: sub, text)
        n_re += k
    print(f"code   · {n_re} regex substitutions, {len(LITERAL_SUBS)} literal subs")
    text, rep2 = objectcode.apply_literals(text, LITERAL_SUBS)
    for r in rep2:
        print(f"         · {r[:78]}")

    must_have = ["Edge0", "Edge0-35B-A3B-preview", "256 experts", "gated linear", "LoRA", "prerouter",
                 "int4", "Recurr", "Qwen3.6-35B-A3B"]
    must_not = ["DeepSeek", "DSpark", "Engram", "CSA2", "129,280", "5,120", "384 routed", "mHC",
                "890", "Top-512", "40 backbone", "schema-derived", "hypothetical", "wo_a", "indexer"]
    body = text.split("</head>", 1)[-1]
    miss = [tok for tok in must_have if tok.lower() not in body.lower()]
    bad = [tok for tok in must_not if tok.lower() in body.lower()]
    if miss:
        print(f"RESIDUE FAIL missing: {miss}")
        return 5
    if bad:
        for tok in bad:
            k = body.lower().find(tok.lower())
            print(f"RESIDUE {tok!r} at {k}: ...{body[max(0,k-120):k+100]!r}...")
        return 6
    # mandatory preflight (skill: tensor-atlas-v4-retarget) — runs on the candidate page BEFORE it
    # lands on disk, and proves the checker itself can still fail (--selftest)
    import subprocess
    import tempfile
    pre = DIR / "preflight.py"
    if pre.exists():
        st = subprocess.run([sys.executable, str(pre), "--selftest"], capture_output=True, text=True)
        print((st.stdout or st.stderr).strip().splitlines()[-1] if (st.stdout or st.stderr).strip() else "self-test: no output")
        if st.returncode != 0:
            print("PREFLIGHT SELF-TEST FAILED — the guard is broken, refusing to ship", file=sys.stderr)
            return 8
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "candidate"
            cand.mkdir()
            (cand / "index.html").write_text(text, encoding="utf-8")
            r = subprocess.run([sys.executable, str(pre), str(cand)], capture_output=True, text=True)
            print("\n".join(r.stdout.strip().splitlines()[-12:]))
            if r.returncode != 0:
                print("PREFLIGHT FAILED — index.html NOT written", file=sys.stderr)
                return 7
    # WebMCP tools: webmcp-tools.js 를 자리표시자에 인라인 주입(외부 스크립트 미사용 → CSP 원문 유지)
    _wm = DIR / "webmcp-tools.js"
    if "@@WEBMCP@@" in text:
        if not _wm.exists():
            print("RESIDUE FAIL: webmcp-tools.js missing", file=sys.stderr)
            return 9
        _code = _wm.read_text(encoding="utf-8").rstrip() + "\n"
        if "</script" in _code.lower():
            print("RESIDUE FAIL: webmcp-tools.js contains </script", file=sys.stderr)
            return 9
        text = text.replace("@@WEBMCP@@", _code)
        if "@@WEBMCP@@" in text:
            print("RESIDUE FAIL: WebMCP placeholder not fully replaced", file=sys.stderr)
            return 9
        print(f"webmcp · inlined {len(_code):,} B of tools")
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name} · {len(text):,} B")

    # the NTT internal badge + copyright footer are part of the published page
    import badge
    print("badge ·", badge.inject(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
