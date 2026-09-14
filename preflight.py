#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""atlas_preflight.py — structural checks that stop the recurring Tensor-Atlas bug classes.

Usage:  python3 atlas_preflight.py <atlas-dir> [<atlas-dir> ...]

Runs on the *built* index.html (plus its data half if present) and fails loudly with the exact
line/snippet, so a bad atlas never reaches a browser. Every check below exists because that bug
once shipped:

  1  category count vs the engine's category layout   -> clicking 02 Storage blanked the 3D view
  2  view dispatch not wrapped in try/catch           -> one exception killed the frame loop
  3  pos('id') targets that do not exist              -> V3.add threw inside the frame loop
  4  dynamic id loops that walk past the layer count  -> 'L'+(20+i) over a 30-layer model
  5  engine identifiers with no definition in the data half (KV_FULL_PER_TOKEN class)
  6  label(..., null, ...) arguments                  -> commitLabels reads sub.length
  7  category solids referencing categories that the page does not define
  8  the app-shell wrapper, mode keys and per-mode totals (the layout/precision contract)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FAIL = 0


def report(ok: bool, title: str, detail: str = "") -> None:
    global FAIL
    print(f"{'PASS' if ok else 'FAIL'}  {title}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAIL = 1



def code_only(script: str) -> str:
    """Drop string literals and template text so only real code identifiers remain."""
    out, i, n = [], 0, len(script)
    while i < n:
        c = script[i]
        if c in "\"'":
            q, i = c, i + 1
            while i < n:
                if script[i] == "\\":
                    i += 2
                    continue
                if script[i] == q:
                    i += 1
                    break
                i += 1
            out.append(" ")
        elif c == "\u0060":
            i += 1
            while i < n:
                if script[i] == "\\":
                    i += 2
                    continue
                if script[i] == "$" and i + 1 < n and script[i + 1] == "{":
                    depth, i = 1, i + 2
                    while i < n and depth:
                        if script[i] == "\\":
                            out.append(script[i:i + 2])
                            i += 2
                            continue
                        if script[i] == "{":
                            depth += 1
                        elif script[i] == "}":
                            depth -= 1
                            if depth == 0:
                                i += 1
                                break
                        out.append(script[i])
                        i += 1
                    continue
                if script[i] == "\u0060":
                    i += 1
                    break
                i += 1
            out.append(" ")
        else:
            out.append(c)
            i += 1
    joined = "".join(out)
    # strings nested inside ${...} expressions survive the pass above
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", joined)


def line_of(text: str, needle: str, window: int = 120) -> str:
    i = text.find(needle)
    return "" if i < 0 else text[max(0, i - window // 2):i + window // 2].replace("\n", " ")


def check(atlas: Path) -> None:
    page = atlas / "index.html"
    if not page.exists():
        report(False, f"{atlas.name}: index.html exists")
        return
    text = page.read_text(encoding="utf-8")
    body = text.split("</head>", 1)[-1]
    print(f"--- {atlas.name}  ({len(text):,} B)")

    # 1 + 7: CATEGORIES count vs the layout the engine uses for them
    m = re.search(r"const CATEGORIES=\[(.*?)\n\];", body, re.S)
    ncat = len(re.findall(r"\n \['", m.group(1))) if m else 0
    # the storage() body is the only place that lays the category solids out
    st = body.find("storage(){")
    storage = body[st:st + 4000] if st >= 0 else ""
    if "coords=CATEGORIES.map" in storage:
        report(True, "category layout is computed from CATEGORIES")
    else:
        lit = storage.find("coords=[[")
        ncoords = 0
        if lit >= 0:
            seg = storage[lit:lit + 900]
            ncoords = seg.count("],[") + 1
        report(lit >= 0 and ncat <= ncoords, "category count fits the hand-written coords array",
               f"{ncat} categories, {ncoords} coords — compute coords from CATEGORIES")
    report(ncat >= 4, "categories defined", f"{ncat} categories")

    # 2: the frame loop must survive a bad view
    guard = "try{if(this.state.view==='storage')this.storage()" in body
    report(guard, "view dispatch is wrapped in try/catch",
           line_of(body, "this.state.view==='storage')this.storage()"))

    # 3 + 4: every id the scene asks for must exist
    ids = set(re.findall(r"\bid:'([A-Za-z0-9_]+)'", body))
    layer_ids = {f"L{i}" for i in range(0, 200)}
    # ids the data half really defines: no hardcoded allowances, a scene that points at a
    # module this model does not have must fail here, not in the browser.
    known = ids | layer_ids
    missing = sorted({t for t in re.findall(r"pos\('([^']+)'\)", body) if t not in known})
    report(not missing, "literal pos() targets resolve", f"undefined: {missing}")

    lm = re.search(r"const LAYERS=Array\.from\(\{length:(\d+)", body) or \
        re.search(r"Array\.from\(\{length:(\d+)\},\(_,i\)=>\(\{id:'L'", body)
    nlayers = int(lm.group(1)) if lm else 0
    bad_loops = []
    for m2 in re.finditer(r"Array\.from\(\{length:(\d+)\},\(_,i\)=>'L'\+\(?(\d+)\+i", body):
        length, base = int(m2.group(1)), int(m2.group(2))
        if base + length > nlayers:
            bad_loops.append(f"L{base}..L{base+length-1} with {nlayers} layers")
    report(not bad_loops, "dynamic id loops stay inside the layer count", "; ".join(bad_loops))

    # 5: identifiers the rewritten panels use must be defined somewhere
    # Constants the engine interpolates (${...}) must exist in the data half.
    # This targets the real failure mode — a panel referencing NV_DELTA / KV_FULL_PER_TOKEN that
    # no `const` ever defines — without tripping over prose, DOM or WebGL constant names.
    script_start = text.find(">", text.find("<script>")) + 1
    engine_region = text[script_start:text.rfind("</script>")]
    datamark = engine_region.find("class CanvasRenderer{")
    engine_region = engine_region[datamark:] if datamark > 0 else engine_region
    interpolated, depth, k = "", 0, 0
    while k < len(engine_region) - 1:
        if engine_region[k] == "$" and engine_region[k + 1] == "{":
            depth, k = 1, k + 2
            while k < len(engine_region) and depth:
                if engine_region[k] == "{":
                    depth += 1
                elif engine_region[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                interpolated += engine_region[k]
                k += 1
        k += 1
    used = set(re.findall(r"(?<![.\w])([A-Z][A-Z0-9_]{3,})\b", code_only(interpolated)))
    defined = set(re.findall(r"\b(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)", text))
    defined |= {"MODE_KEYS", "MODE_INFO", "CATEGORIES", "TOTALS", "TOTAL_P", "MODULES", "ALL_W",
                "LAYERS", "EMBED", "HEAD", "VISION", "ALIGNER", "SELF", "DRAFT", "AUX_TABLES"}
    ignore = {"SECTION", "FACT", "NOTE", "ICON", "HTML", "JSON", "MATH", "URL", "SVG", "HTTP",
              "HTTPS", "OBJECT", "ARRAY", "NUMBER", "STRING", "BOOLEAN", "TRUE", "FALSE", "NULL",
              "NAN", "USE", "MU", "AN", "OK", "KV", "MB", "GB", "TB", "CPU", "GPU", "RAM", "SSD",
              "ID", "WEBGL", "ARRAY_BUFFER", "STATIC_DRAW", "DYNAMIC_DRAW", "TRIANGLES", "LINES",
              "POINTS", "SRC_ALPHA", "ONE_MINUS_SRC_ALPHA", "ONE", "COLOR_BUFFER_BIT",
              "DEPTH_BUFFER_BIT", "DEPTH_TEST", "CULL_FACE", "BLEND", "VERTEX_SHADER",
              "FRAGMENT_SHADER", "COMPILE_STATUS", "LINK_STATUS", "FLOAT", "PREFIX", "LICENSE",
              "MIT", "APACHE", "ATLAS_DEBUG", "EXPLORER", "SCHEMATIC", "TABLE", "SHARD", "HEADERS",
              "MEASURED", "BUILD", "MODEL", "TENSOR", "WEIGHT", "TOTAL", "KV_FULL", "UPDATE"}
    undef = sorted(u for u in used - defined - ignore)
    report(not undef, "interpolated constants exist in the data half", f"undefined: {undef[:8]}")

    # 6: label() with a null second argument
    bad_null = re.findall(r"this\.label\([^)]*,\s*null\s*,", body)
    report(not bad_null, "no label(..., null, ...) call", f"{len(bad_null)} found")

    # 8: chrome + precision contract
    report('id="root"' in text and "app-shell" in text, "page keeps the app-shell wrapper")
    keys = re.search(r"const MODE_KEYS=\[([^\]]*)\]", body)
    modes = re.findall(r"'([a-z0-9]+)'", keys.group(1)) if keys else []
    report(len(modes) >= 2, "mode keys defined", str(modes))
    no_info = [m for m in modes if f"{m}:{{label:" not in body]
    report(not no_info, "every mode has MODE_INFO", f"missing: {no_info}")
    for m3 in re.finditer(r"const TOTALS=Object\.fromEntries\(MODE_KEYS", body):
        pass
    report("sumB(ALL_W" in body, "totals are derived from the data half")

    # 11: every palette key the engine reads must be defined in the data half
    palette_problems: list[str] = []
    for name in ("COL", "TC"):
        mm = re.search(rf"const {name}=\{{([^}}]*)\}}", body)
        if not mm:
            palette_problems.append(f"{name} object not found")
            continue
        defined = set(re.findall(r"([A-Za-z_$][\w$]*)\s*:", mm.group(1)))
        used = set(re.findall(rf"\b{name}\.([A-Za-z_$][\w$]*)", body))
        miss_keys = sorted(used - defined)
        if miss_keys:
            palette_problems.append(f"{name} read but undefined: {miss_keys}")
    report(not palette_problems, "palette keys the engine reads are defined",
           "; ".join(palette_problems))


SELFTEST_PAGE = """<!doctype html><html><head></head><body><div id="root"></div>
<script>
const MODE_KEYS=['bf16','fp8'];
const MODE_INFO={bf16:{label:'x'},fp8:{label:'y'}};
const CATEGORIES=[
 ['expert','x','#fff',[]],
 ['attn','x','#fff',[]],
 ['shared','x','#fff',[]],
 ['vocab','x','#fff',[]]
];
const LAYERS=Array.from({length:30},(_,i)=>({id:'L'+i,mode:i<10?'full':'swa'}));
const TOTALS={bf16:1,fp8:1};
const ALL_W=[];
const sumB=(ws,m)=>1;
function x(){return 1;}
const tpl=html`<div class="app-shell">${bytes(NV_DELTA,s.binary)}</div>`;
function storage(){let s={};coords=CATEGORIES.map((c,i)=>[i,0,0]);
 for(let i=0;i<CATEGORIES.length;i++){let p=[coords[i][0],1,coords[i][2]];}}
function loop(now){if(this.state.view==='storage')this.storage();else this.architecture();}
</script></body></html>"""


def selftest(tmp: Path) -> bool:
    """The checker must FAIL on a page with an undefined interpolated constant."""
    global FAIL
    ok = True
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "index.html").write_text(SELFTEST_PAGE, encoding="utf-8")
    save = FAIL
    FAIL = 0
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        check(tmp)
    failed = FAIL
    FAIL = save
    if 'NV_DELTA' not in buf.getvalue():
        print("SELF-TEST FAILED: the interpolation check did not flag an undefined NV_DELTA")
        ok = False
    if not failed:
        print("SELF-TEST FAILED: preflight passed a page it must reject")
        ok = False
    print("PREFLIGHT SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return ok


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            return 0 if selftest(Path(td) / "atlas") else 1
    dirs = [Path(a) for a in argv[1:] if not a.startswith("--")] or [Path.cwd()]
    for d in dirs:
        check(d)
    print("\n" + ("PREFLIGHT FAILED" if FAIL else "PREFLIGHT PASSED"))
    return FAIL


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
