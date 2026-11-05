#!/usr/bin/env python3
"""gen_stubs.py — the Vulkan trap-roster generator (Task 1.2).

Reads the engine's own kernel contract headers (include/strata/kernels/*.hpp) through clang's AST dump —
the metal/tools/gen_stubs.py mechanism — and emits vulkan/src/engine/stubs_gen.cpp: one definition per
host-callable strata::kernels entry point that has no Vulkan implementation yet, each calling
strata::vk_engine::missing("<qualified name>") then returning a contract-legal neutral value ({}; a
reference return gets a static zero-initialised object of its own type).

THE MACHINE LIST OF NOT-PORTED ENTRY POINTS is two sources that must agree (the script errors on drift):
  * the trap set: every bodyless strata::kernels free declaration in the headers MINUS PORTED,
    MINUS the functions vulkan/src/engine/missing.cpp defines itself (HOST_DEFINED, the metal pattern:
    cvec() answers the empty vector because the engine's layer loop asks every request),
  * the ledger: vulkan/PORT-STATUS.md — every row whose state starts with `todo` must have its header
    name covered by a trap (or be PORTED), and every PORTED name must appear on a 'ported+parity-green'
    row. PORT-STATUS.md is the source of truth for WHY; the headers are the source for signatures.

Ported kernels live in vulkan/src/kernels/<name>_vk.cpp and are NEVER redefined here.
Regenerate after porting one: add its name to PORTED, re-run this, rebuild:
    python3 vulkan/tools/gen_stubs.py
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HDR_DIR = os.path.join(ROOT, "include", "strata", "kernels")
OUT = os.path.join(ROOT, "vulkan", "src", "engine", "stubs_gen.cpp")
LEDGER = os.path.join(ROOT, "vulkan", "PORT-STATUS.md")

# ---- PORTED: entry points with a real Vulkan launcher — the PORT-STATUS
# 'ported+parity-green' rows (s2_gemv, dequant_s2). These must NOT be trapped.
PORTED = {"s2_gemv", "dequant_s2", "native_quantize_q8_1", "native_mmvq", "native_mmvq_slots"}

# ---- HOST_DEFINED: defined without a trap in vulkan/src/engine/missing.cpp
# (metal/src/engine/missing.cpp's pattern): the engine's host loop calls cvec()
# unconditionally on every layer; the honest answer is the empty, unloaded vector.
# The native_mmvq capability/byte-count queries (Task 1.4): pure type tables and byte arithmetic in
# the CUDA build too — the engine's load path asks them before any kernel is launched, and trapping
# them would route the whole load through a different format than the CUDA engine picks.
HOST_DEFINED = {"cvec", "native_mmvq_supported", "native_mmvq_ids_supported", "native_mmvq_il_supported",
                "native_q8_1_bytes", "native_q8_1_il_bytes",
                # Phase 7.1: the native-expert tier's capability/byte queries — pure host arithmetic
                # transcribed verbatim into missing.cpp (same rule as native_mmvq_supported: trapping a
                # load-path QUESTION re-routes the whole model through another tier than CUDA's build).
                "iq_row_bytes", "native_expert_supported", "native_expert_layout",
                "native_expert_scratch_bytes",
                # the rope table builders + the rope config: host float64 loops (rope.cu:45-86),
                # transcribed verbatim above; bitwise what CUDA computes
                "rope_scaling_set", "rope_scaling", "build_rope_table",
                "iq_supported", "embed_type_supported", "native_mmvq_weight_bytes",
                "native_q8_0_packed_enabled", "native_q8_0_packed_eligible",
                # gr/fused_gr host surface (setters, geometry query, workspace layout): transcribed
                # above; the read/write kernels and the multi/persist launches stay named traps
                "gr_set_fp32_activations", "gr_set_native_mmvf", "gr_native_mmvf_enabled",
                "gr_workspace_init", "fused_gr_set_fast", "fused_gr_supported",
                "fused_gr_variant", "fused_gr_check", "fused_gr_persist_init",
                # ple host surface (mode setters, availability, scratch arithmetic)
                "ple_set_native_bf16", "ple_native_bf16_enabled", "ple_set_native_postops",
                "ple_native_postops_enabled", "ple_block_available", "ple_block_scratch_bytes",
                # the config-flag family (setters/getters/geometry predicates/byte arithmetic only):
                # they pick WHICH trap a run reaches, exactly as on CUDA; no kernel path invented here
                "native_gdn_set_enabled", "native_gdn_enabled",
                "native_rope_set_enabled", "native_rope_enabled",
                "native_router_set_enabled", "native_router_enabled", "native_router_topk_supported",
                "native_qsa_set_enabled", "native_qsa_enabled",
                "native_qsa_indexer_set_enabled", "native_qsa_indexer_enabled",
                "native_qsa_score_set_enabled", "native_qsa_score_enabled",
                "native_moe_combine_set_enabled", "native_moe_combine_enabled",
                "native_moe_combine_hits_supported",
                "shared_expert_set_native_bf16", "shared_expert_native_bf16_enabled",
                "iq_set_old_kernels", "native_grouped_set_v1",
                "native_mmvq_set_b6_rows", "native_mmvq_set_multi_exact",
                "qsa_prompt_attn_set_imma",
                "native_expert_set_expdec", "native_expert_expdec",
                "native_expert_set_expdec_params", "native_expert_expdec_params",
                "native_expert_expdec_takes",
                "native_q6_k_packed_enabled", "native_expert_set_mode",
                # load-path scratch arithmetic + type tables (transcribed verbatim above)
                "moe_hit_grouped_scratch_bytes", "shared_expert_scratch_bytes",
                "argmax_rows_scratch_bytes", "row_top_prob_scratch_bytes", "coupled_draft_scratch_bytes",
                "kv_block_bytes", "dequant_bf16_supported", "dequant_set_wide",
                "cvec_enabled", "cvec_set_enabled", "qsa_decode_attn_scratch_floats",
                # the rope/mrope table registries (pointer stores + the opt-in query): host state only,
                # the native_rope launches that read them stay traps
                "rope_table_set", "rope_table_release", "rope_table_for", "mrope_table_set", "mrope_table"}

# ---- N/A: ledger rows whose state is `n/a` — these must NOT trap (the engine's
# verify.cpp evaluates pdl_supported() on EVERY run, graphs or not; a trap there
# would abort before any kernel is demanded). They get neutral definitions in
# vulkan/src/engine/missing.cpp instead: PDL is an NVIDIA launch feature, the
# honest answer is "not supported, scope off" — the CUDA build's own fallback.
NA = {"pdl_scope", "pdl_supported"}

# ---- HOST_TU: bodyless in the headers only because their .cu sibling shares the library; a PURE-HOST
# translation unit defines them, and the Phase-7.1 pack target compiles that TU unchanged
# (src/kernels/ngram.cpp — its own CMake note: host-only, in strata_kernels only for the geometry
# constants). Trapping them would collide with the real definitions; poly never references them, so
# dropping the traps costs the poly link nothing.
HOST_TU = {"bf16_dequant_row", "fp8_e4m3_dequant_row", "iq4nl_code", "iq4nl_dequant_row",
           "ngram_mixed", "ngram_rows", "ple_artifact_consts", "ple_format_count",
           "ple_format_for_type", "ple_format_info", "ple_format_list", "ple_formats",
           "ple_prefetch_enable", "ple_prefetch_enabled"}

CLANG = os.environ.get("CLANGXX", "clang++")
INCLUDES = ["-I", os.path.join(ROOT, "vulkan", "include", "strata", "vulkan_compat"),
            "-I", os.path.join(ROOT, "vulkan", "include"),
            "-I", os.path.join(ROOT, "include")]

def header_files():
    return sorted(f for f in os.listdir(HDR_DIR) if f.endswith(".hpp"))

def header_compilable(h):
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write('#include "strata/kernels/%s"\n' % h)
        name = f.name
    r = subprocess.run([CLANG, "-fsyntax-only", "-std=c++20", '-DSTRATA_VERSION="0"'] + INCLUDES + [name],
                       capture_output=True, text=True)
    os.unlink(name)
    return r.returncode == 0, r.stderr

# ---- declarations from the AST dump (metal/tools/gen_stubs.py's walker, free functions)
def parse_headers(hdrs):
    tu = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False)
    for h in hdrs:
        tu.write('#include "strata/kernels/%s"\n' % h)
    tu.close()
    cmd = ([CLANG, "-fsyntax-only", "-std=c++20", '-DSTRATA_VERSION="0"',
            "-Xclang", "-ast-dump", "-Xclang", "-ast-dump-filter=strata", "-fno-color-diagnostics"]
           + INCLUDES + [tu.name])
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tu.name)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        sys.exit("gen_stubs: AST dump of the kernel headers failed")
    decls = []
    stack = []   # (depth, name, is_record)
    cur = None
    def flush():
        nonlocal cur
        if cur and not cur["body"]:
            decls.append(cur)
        cur = None
    for line in r.stdout.splitlines():
        if line.startswith("Dumping "):
            flush(); stack = []; continue
        m = re.match(r"^([| `-]*)(\w+) ", line)
        if not m: continue
        depth = len(m.group(1)) // 2
        kind = m.group(2)
        if cur and depth <= cur["depth"]:
            flush()
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if cur:
            if kind == "CompoundStmt" and depth == cur["depth"] + 1:
                cur["body"] = True
            if kind == "ParmVarDecl" and depth == cur["depth"] + 1:
                pm = re.search(r"(?:col|line):[\d:]+(?: used| referenced)?(?: (\w+))? '([^']*)'(?::'[^']*')?", line)
                if pm: cur["params"].append((pm.group(1) or "", pm.group(2)))
            continue
        if kind == "NamespaceDecl":
            nm = re.search(r"> (?:prev 0x\w+ )?(?:line|col):[\d:]+ (\w+)", line)
            stack.append((depth, nm.group(1) if nm else "(anonymous)", False))
        elif kind in ("CXXRecordDecl", "ClassTemplateDecl", "FunctionTemplateDecl"):
            nm = re.search(r"(?:class|struct|union) (\w+)", line)
            if kind == "FunctionTemplateDecl" or " implicit " in line or not nm:
                stack.append((depth, "(template)" if kind == "FunctionTemplateDecl" else "(implicit)", True))
            else:
                stack.append((depth, nm.group(1), True))
        elif kind == "FunctionDecl" and " implicit " not in line:
            fm = re.findall(r"(~?\w+) '([^']*)'(?::'[^']*')?", line)
            if not fm: continue
            name, ty = fm[-1]
            names = [s[1] for s in stack]
            if any(s[2] for s in stack): continue   # inside a record or template: not a contract free fn
            cur = {"qual": "::".join(names + [name]), "type": ty, "params": [], "depth": depth, "body": False}
    flush()
    return decls

# ---- the ledger: PORT-STATUS rows. Names are the first cell with parenthesised
# variant/setter notes stripped; state is the last cell.
def ledger():
    todo, ported, na = set(), set(), set()
    for line in open(LEDGER):
        if not line.startswith("| "): continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7: continue
        if cells[0].startswith("poly:"): continue   # the strata::poly surface: gen_anystrata_stubs.py's ledger
        state = cells[6].strip("* ")
        cell0 = re.sub(r"\([^)]*\)", " ", cells[0])
        names = [t.strip() for t in re.split(r"[/,]| and ", cell0) if re.fullmatch(r"[a-z_0-9]+", t.strip())]
        # also plain-space splits for single-name rows: "s2_gemv variants ..." after paren-strip has extra words
        if len(names) == 0:
            names = [t for t in re.split(r"\s+", cell0) if re.fullmatch(r"[a-z_0-9]+", t)]
            names = names[:1]   # only the leading entry-point name without a separator
        names = [n.strip() for n in names]
        if state.startswith("todo"): todo.update(names)
        elif state.startswith("ported"): ported.update(n for n in names if n in PORTED or True)
        elif state.startswith("n/a"): na.update(names)
    return todo, ported, na

def ret_of(ty):
    s = re.sub(r"\)\s*(const\s*)?(noexcept\s*)?$", ")", ty)
    i = len(s) - 1; d = 0
    while i >= 0:
        if s[i] == ")": d += 1
        elif s[i] == "(":
            d -= 1
            if d == 0: break
        i -= 1
    return s[:i].strip()

def main():
    hdrs, skipped = [], []
    for h in header_files():
        ok, err = header_compilable(h)
        if ok: hdrs.append(h)
        else: skipped.append(h)
    decls = parse_headers(hdrs)

    # ledger cross-check 1: PORTED must be exactly the 'ported+parity-green' row names
    todo, lported, lna = ledger()
    if lported != PORTED:
        sys.exit("gen_stubs: PORTED=%s disagrees with PORT-STATUS 'ported+parity-green' rows %s"
                 % (sorted(PORTED), sorted(lported)))

    kernels_decls = [d for d in decls if d["qual"].startswith("strata::kernels::")]
    def simple(d): return d["qual"].rsplit("::", 1)[1]
    to_trap = [d for d in kernels_decls
               if simple(d) not in PORTED and simple(d) not in HOST_DEFINED and simple(d) not in NA
               and simple(d) not in HOST_TU]
    seen = {simple(d) for d in kernels_decls}

    # ledger cross-check 2: every todo-row name that IS a header contract function got a trap
    trapped = {simple(d) for d in to_trap}
    drift = sorted(n for n in todo if n in seen and n not in trapped)
    if drift:
        sys.exit("gen_stubs: PORT-STATUS says todo but no trap generated (PORTED/HOST_DEFINED/NA drift?): %s" % drift)

    print("headers: %d scanned, %d skipped (device-only inline): %s" % (len(hdrs), len(skipped), skipped))
    print("strata::kernels bodyless declarations: %d -> traps: %d (ported %d, host-defined %d, n/a %d)"
          % (len(kernels_decls), len(to_trap), len(PORTED), len(HOST_DEFINED & seen), len(NA & seen)))
    print("todo-ledger names with no header declaration (variants, device-only, or prose): %d"
          % len([n for n in todo if n not in seen]))

    lines = ["""// vulkan/src/engine/stubs_gen.cpp - GENERATED by vulkan/tools/gen_stubs.py: do not edit.
// The strata::kernels entry points with no Vulkan implementation yet: each names itself and stops the
// run (vulkan/src/engine/missing.cpp; STRATA_VK_STUBS=log prints one line per entry and returns).
// Regenerate after porting one: add its name to PORTED in the script, re-run, rebuild.""", ""]
    lines += ['#include "strata/kernels/%s"' % h for h in hdrs]
    lines += ["#include <type_traits>", "",
              "namespace strata::vk_engine { void missing(const char* name); }", ""]
    def ret_stmt(r):
        if r == "void": return ""
        if r.endswith("&"):
            return " static std::remove_cv_t<std::remove_reference_t<%s>> s{}; return s;" % r
        return " return {};"
    for d in sorted(to_trap, key=lambda x: (x["qual"], len(x["params"]))):
        ns, fn = d["qual"].rsplit("::", 1)
        ty, params = d["type"], d["params"]
        r = ret_of(ty)
        ps = []
        for k, (pn, pt) in enumerate(params):
            if "(" in pt and "*" in pt:            # function-pointer parameter: name inside the declarator
                ps.append(re.sub(r"\(\*", "(*p%d)" % k, pt, count=1))
            else:
                ps.append("%s p%d" % (pt, k))
        body = 'strata::vk_engine::missing("%s");' % d["qual"] + ret_stmt(r)
        nx = " noexcept" if re.search(r"\)\s*(const\s*)?noexcept\s*$", ty) else ""
        lines.append("namespace %s { %s %s(%s)%s { %s } }" % (ns, r, fn, ", ".join(ps), nx, body))
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("wrote %s: %d trap definitions" % (OUT, len(to_trap)))

if __name__ == "__main__":
    main()
