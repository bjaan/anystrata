#!/usr/bin/env python3
"""gen_pack_stubs.py — the Phase-7.1 trap roster for the PACK engine's host-kernel surface.

`anystrata-pack` (src/program/generate.cpp + strata_engine/strata_prefill/strata_spec/q36) reaches its
device work through THREE contracts the Vulkan tree has no implementation of yet:

  include/strata/prefill/kernels.hpp   (strata::prefill free launchers)
  include/strata/prefill/gemm.hpp      (strata::prefill::gemm)
  include/strata/prefill/moe_mmq.hpp   (strata::prefill::mmq)

...while generate.cpp's other device surface (strata::kernels) is ALREADY trapped by the Task-1.2
roster (vulkan/src/engine/stubs_gen.cpp), and q36_engine.cu's kernels are TU-internal (its public
contract is one function: strata::qwen35::run(argc, argv) — trapped at whole-run level, by name).

Per the port's law (the sycl "migrated copy" rule as applied in Task 1.4): the .cu files are NOT
compiled without nvcc; every free declaration WITHOUT a body gets a trap that names itself through
strata::vk_engine::missing() (exit 86; STRATA_VK_STUBS=log returns instead, for roster inspection).
A member function of a class (Prefill, ...) is NOT generated here: those TUs (prefill.cpp) are pure
host and compile UNCHANGED through the compat layer; their kernel calls land on these traps.

Porting one = write the real launcher in vulkan/src/pack/, delete nothing here (the trap object sits
at LINK TAIL: real definitions win the link), rebuild. Regenerate: python3 vulkan/tools/gen_pack_stubs.py

Signatures come from the headers' clang AST (same mechanism as gen_stubs.py), never by hand.
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# kernels.hpp + gemm.hpp: the free launchers with .cu-only bodies. moe_mmq.hpp is NOT trapped:
# its mmq:: query/gather API is host-defined in prefill.cpp itself (measured by link: the first
# definition is there) — trapping it collided, and the CUDA-only edges inside resolve against
# strata::kernels traps (Task 1.2 roster) at link. Filter in main(): keep exactly the names the
# host link leaves UNDEFINED (undefined-pack-symbols.txt from ld), never more.
# moe_mmq.hpp is NOT in the roster: its whole mmq:: API has host bodies in prefill.cpp, which this
# target compiles unchanged (link proved it: every mmq collision resolves there). The trapped surface
# is exactly what .cu files alone define: prefill/kernels.hpp + prefill/gemm.hpp.
HDRS = [os.path.join("include", "strata", "prefill", h)
        for h in ("kernels.hpp", "gemm.hpp")]
UNDEF_FILE = os.path.join(ROOT, "vulkan", "tools", "undefined-pack-symbols.txt")
OUT = os.path.join(ROOT, "vulkan", "src", "pack", "stubs_gen.cpp")

CLANG = os.environ.get("CLANGXX", "clang++")
INCLUDES = ["-I", os.path.join(ROOT, "vulkan", "include", "strata", "vulkan_compat"),
            "-I", os.path.join(ROOT, "vulkan", "include"),
            "-I", os.path.join(ROOT, "include")]

def ast_of(header):
    tu = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False)
    tu.write(f'#include "{os.path.relpath(header, os.path.join(ROOT, "include"))}"\n')
    tu.close()
    cmd = ([CLANG, "-fsyntax-only", "-std=c++20", '-DSTRATA_VERSION="0"',
            "-Xclang", "-ast-dump", "-Xclang", "-ast-dump-filter=strata", "-fno-color-diagnostics"]
           + INCLUDES + [tu.name])
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tu.name)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        sys.exit(f"gen_pack_stubs: AST dump of {header} failed")
    return r.stdout

def decls_of(dump):
    decls = []
    stack = []
    cur = None
    with_body = []   # quals DEFINED in the headers (inline host helpers): never trap them twice
    def flush():
        nonlocal cur
        if cur and cur["body"]:
            with_body.append(cur["qual"])
        elif cur:
            decls.append(cur)
        cur = None
    for line in dump.splitlines():
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
            name = nm.group(1) if nm else "(anonymous)"
            if name in ("referenced", "used", "inline", "nested", "abstract", "final"):
                # clang puts linkage/ownership markers before the name: `line:13:11 referenced strata`
                rest = line.split(f"{name} ", 1)
                nm2 = re.match(r"(\w+)", rest[1]) if len(rest) > 1 else None
                name = nm2.group(1) if nm2 else "(anonymous)"
            stack.append((depth, name, False))
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
            if any(s[2] for s in stack): continue   # member/template: not a free contract fn
            cur = {"qual": "::".join(names + [name]), "type": ty, "params": [], "depth": depth, "body": False}
    flush()
    return decls, with_body

def main():
    all_decls, defined_list = [], []
    for h in HDRS:
        d, w = decls_of(ast_of(os.path.join(ROOT, h)))
        all_decls += d; defined_list += w
    defined = set(defined_list)
    # keep only symbols the real link reports undefined (mangled names resolved via c++filt match)
    import subprocess as sp
    want = set()
    if os.path.exists(UNDEF_FILE):
        for line in open(UNDEF_FILE):
            line = line.strip()
            if line.startswith("_Z"):
                dem = sp.run(["c++filt", line], capture_output=True, text=True).stdout.strip()
            else:
                dem = line
            want.add(dem.split("(")[0].strip())
    # HEADER_HOST: declarations whose CUDA body is entirely #if defined(__HIPCC__) (non-HIP builds get
    # a no-op / fixed false). The AST sees only the declaration, so the roster would trap host code that
    # CUDA compiles to nothing; these are defined by hand at the top of the emitted file instead.
    HEADER_HOST = {"prompt_f16", "gemm_set_dense_f16", "gemm_set_dense_f32", "gemm_f16_range_read",
                   "gemm_set_f16_range",
                   # kernels.cu too: same shape (CUDA builds compile them empty; the state lives in
                   # HIP __constant__ device globals these builds never declare)
                   "set_act_f16", "set_f16_range", "f16_range_read"}
    pack = [d for d in all_decls
            if d["qual"].startswith("strata::prefill::") and "type" in d
            and d["qual"] not in defined
            and d["qual"].split("::")[-1] not in HEADER_HOST
            and (not want or d["qual"] in want or ("::".join(d["qual"].split("::")[:3]) in want))]
    # unique on qualified name
    seen, uniq = set(), []
    for d in pack:
        if d["qual"] in seen: continue
        seen.add(d["qual"]); uniq.append(d)
    uniq.sort(key=lambda d: d["qual"])
    if not uniq:
        sys.exit("gen_pack_stubs: no bodyless strata::prefill free declarations found — headers changed?")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("""// vulkan/src/pack/stubs_gen.cpp - GENERATED by vulkan/tools/gen_pack_stubs.py: do not edit.
// The strata::prefill entry points with no Vulkan implementation yet (the pack engine's kernel
// surface): each names itself and stops the run (vulkan/src/engine/missing.cpp; STRATA_VK_STUBS=log
// prints one line per entry and returns). Porting one = write the real launcher in vulkan/src/pack/
// (it wins the link: this object sits at link tail); re-run the generator only after header moves.

#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <stdexcept>
#include <type_traits>

namespace strata::vk_engine { void missing(const char* name); }

// HEADER_HOST (see the generator): declarations whose CUDA bodies are wholly #if defined(__HIPCC__) —
// non-HIP builds compile them to no-ops, so trapping them would abort runs CUDA completes silently.
namespace strata::prefill {
bool prompt_f16() { return false; }
void gemm_set_dense_f32(int) {}
bool gemm_f16_range_read(float*, unsigned long long*, bool) { return false; }
void gemm_set_f16_range(bool) {}                    // gemm.cu
void set_act_f16(bool) {}                           // kernels.cu: CUDA's activation image is always BF16
void set_f16_range(bool) {}
bool f16_range_read(float&, unsigned long long&, bool) { return false; }
}  // namespace strata::prefill

""")
        emitted = 0
        for d in uniq:
            ty = d["type"]
            m = re.match(r"(?s)^(.*?)\((.*)\)$", ty)
            if not m:
                print(f"skip (no AST function type): {d['qual']}", file=sys.stderr); continue
            ret = m.group(1).strip()
            # same rendering as the poly roster: positional names p0..pN, types from ParmVarDecls
            plist = ", ".join(f"{t} p{i}" for i, (_nm, t) in enumerate(d["params"]))
            qual = d["qual"]
            # qual is like strata::prefill::mmq::built -> wrap in the innermost namespace block style:
            # emit as a qualified definition; strip the leading "strata::" for the namespace block
            mm = re.match(r"(strata::[a-z0-9_]+(?:::[a-z0-9_]+)*)::([~\w]+)$", qual)
            ns, fn = mm.group(1), mm.group(2)
            if ret and ret != "void":
                body = (ret + ' ' + fn + '(' + plist + ') { strata::vk_engine::missing("'
                        + qual + '"); return {}; }')
            else:
                body = 'void ' + fn + '(' + plist + ') { strata::vk_engine::missing("' + qual + '"); }'
            f.write(f"namespace {ns} {{ {body} }}\n")
            emitted += 1
    print(f"gen_pack_stubs: {emitted} traps into vulkan/src/pack/stubs_gen.cpp")

if __name__ == "__main__":
    main()
