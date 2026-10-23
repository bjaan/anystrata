#!/usr/bin/env python3
"""gen_poly_stubs.py — the Task-1.4 trap roster for the POLY kernel surface.

include/strata/poly/kernels.hpp is the strata::poly engine's own float-kernel contract: its definitions
live in src/poly/kernels.cu and src/poly/kernels_dsv4.cu, which hold BOTH host wrappers and __global__
kernels and are not compilable without nvcc. Per the plan (the sycl port's X.cu -> migrated-copy rule,
applied here as "traps replace the lib"), the Vulkan build does NOT compile those .cu files: this script
emits vulkan/src/poly/stubs_gen.cpp — one trap definition per bodyless strata::poly free declaration in
the header, same mechanism as gen_stubs.py (clang AST dump, exact signatures), naming each entry point
through strata::vk_engine::missing(). Porting one = delete/branch its trap and write the real launcher.

The header is the source of truth for signatures; PORT-STATUS rows "poly: ..." describe the .cu kernels.
Regenerate with:  python3 vulkan/tools/gen_poly_stubs.py
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HDR = os.path.join(ROOT, "include", "strata", "poly", "kernels.hpp")
OUT = os.path.join(ROOT, "vulkan", "src", "poly", "stubs_gen.cpp")

CLANG = os.environ.get("CLANGXX", "clang++")
INCLUDES = ["-I", os.path.join(ROOT, "vulkan", "include", "strata", "vulkan_compat"),
            "-I", os.path.join(ROOT, "vulkan", "include"),
            "-I", os.path.join(ROOT, "include")]

def main():
    tu = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False)
    tu.write('#include "strata/poly/kernels.hpp"\n')
    tu.close()
    cmd = ([CLANG, "-fsyntax-only", "-std=c++20", '-DSTRATA_VERSION="0"',
            "-Xclang", "-ast-dump", "-Xclang", "-ast-dump-filter=strata", "-fno-color-diagnostics"]
           + INCLUDES + [tu.name])
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tu.name)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        sys.exit("gen_poly_stubs: AST dump of strata/poly/kernels.hpp failed")

    decls = []
    stack = []
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

    poly = [d for d in decls if d["qual"].startswith("strata::poly::")]
    # HOST: pure host-side helpers of the contract, no kernel involved - defined with the same
    # arithmetic as src/poly/kernels.cu's body (not trapped: trapping a launcher-size question
    # would abort a run before any device work is demanded). blas_create/blas_destroy are the
    # cuBLAS handle's host lifecycle, and the compat cublas IS a real host library here
    # (strata_vk_blas's CPU fallback), so taking the handle is honest without any kernel.
    HOST = {"sum_threads": """namespace strata::poly {
int sum_threads(int n) {
    int threads = 128;
    while (threads < 1024 && threads * 4 < n) threads *= 2;
    return threads;
}
}""",
        "attend_split_floats": """namespace strata::poly {
size_t attend_split_floats(int chunk, int n_head, int max_rows, int dim) {
    return size_t(chunk) * n_head * ((size_t) max_rows + 63) / 64 * (dim + 2);
}
}""",
        "blas_create": """namespace strata::poly {
void* blas_create(void* stream) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("blas_create: cublasCreate failed");
    cublasSetStream(h, (cudaStream_t) stream);
    return h;
}
}""",
        "blas_destroy": """namespace strata::poly {
void blas_destroy(void* blas) { if (blas) cublasDestroy((cublasHandle_t) blas); }
}"""}
    poly = [d for d in poly if d["qual"].rsplit("::", 1)[1] not in HOST]
    # PORTED: entry points with a real Vulkan launcher (mirrors gen_stubs.py's rule; the launcher
    # file named, so the exclusion is auditable).  Regenerate after porting one: add the name, re-run.
    PORTED = {"copy2": "vulkan/src/poly/kernels/copy2_vk.cpp",
              "rms_norm": "vulkan/src/poly/kernels/rms_norm_vk.cpp",
              "gemv_f32": "vulkan/src/poly/kernels/gemv_vk.cpp",
              "conv4_silu": "vulkan/src/poly/kernels/conv4_silu_vk.cpp",
              "gdn_step": "vulkan/src/poly/kernels/gdn_vk.cpp",
              "axpy": "vulkan/src/poly/kernels/elementwise_vk.cpp",
              "swiglu_clamp": "vulkan/src/poly/kernels/elementwise_vk.cpp",
              "mul_sigmoid": "vulkan/src/poly/kernels/elementwise_vk.cpp",
              "add_scaled": "vulkan/src/poly/kernels/elementwise_vk.cpp",
              "swiglu_clamp_rows": "vulkan/src/poly/kernels/elementwise_vk.cpp",
              "head_norm_rope": "vulkan/src/poly/kernels/head_norm_rope_vk.cpp",
              "store_f16": "vulkan/src/poly/kernels/store_f16_vk.cpp",
              "attend": "vulkan/src/poly/kernels/attend_vk.cpp"}
    poly = [d for d in poly if d["qual"].rsplit("::", 1)[1] not in PORTED]
    # every header contract function must appear; the ones WITH bodies (inline helpers) do not need traps
    with open(HDR) as f:
        hdr_text = f.read()
    hdr_names = set(re.findall(r"^\s*(?:[\w:*&<> ]+?)\b(\w+)\s*\(", hdr_text, re.M))
    trapped = {d["qual"].rsplit("::", 1)[1] for d in poly}
    missing_decls = sorted(n for n in hdr_names
                           if n not in trapped and re.search(r"\b%s\s*\(" % n, hdr_text)
                           and "struct" not in n)
    print("strata::poly bodyless declarations: %d" % len(poly))
    print("trap set: %s" % sorted(trapped))

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

    lines = ["""// vulkan/src/poly/stubs_gen.cpp - GENERATED by vulkan/tools/gen_poly_stubs.py: do not edit.
// The strata::poly float-kernel contract (include/strata/poly/kernels.hpp) with no Vulkan implementation
// yet: each names itself and stops the run (vulkan/src/engine/missing.cpp). src/poly/kernels.cu and
// kernels_dsv4.cu hold the CUDA definitions together with __global__ kernels and are NOT compiled by
// this backend (Task 1.4: the trap roster replaces the lib; Phase 2 swaps traps for real launchers).""", ""]
    lines += ['#include "strata/poly/kernels.hpp"', '#include <cublas_v2.h>', '#include <cuda_runtime.h>',
              "#include <stdexcept>", "#include <type_traits>", "",
              "namespace strata::vk_engine { void missing(const char* name); }", ""]
    def ret_stmt(r):
        if r == "void": return ""
        if r.endswith("&"):
            return " static std::remove_cv_t<std::remove_reference_t<%s>> s{}; return s;" % r
        return " return {};"
    for d in sorted(poly, key=lambda x: (x["qual"], len(x["params"]))):
        ns, fn = d["qual"].rsplit("::", 1)
        ty, params = d["type"], d["params"]
        r = ret_of(ty)
        ps = []
        for k, (pn, pt) in enumerate(params):
            if "(" in pt and "*" in pt:
                ps.append(re.sub(r"\(\*", "(*p%d)" % k, pt, count=1))
            else:
                ps.append("%s p%d" % (pt, k))
        body = 'strata::vk_engine::missing("%s");' % d["qual"] + ret_stmt(r)
        lines.append("namespace %s { %s %s(%s) { %s } }" % (ns, r, fn, ", ".join(ps), body))
    lines.append("// HOST-defined contract helpers (no device work): the same arithmetic as kernels.cu's.")
    for h in sorted(HOST): lines.append(HOST[h])
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("wrote %s: %d trap definitions (+1 host-defined sum_threads)" % (OUT, len(poly)))

if __name__ == "__main__":
    main()
