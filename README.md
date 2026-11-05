## Credits and license

The engines are PolyStrata's, by VecSzn and contributors, built on [Strata](https://github.com/Niko1221/Strata) by
Niko1221. The front end is built from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)'s server, common and
mtmd code. Qwen3.8-Flash-Next and Qwen3.6-35B-A3B are by the Qwen team; the IQ3_XXS GSQ-RCO file is
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)'s, the Qwen3.6 files are
[Unsloth](https://huggingface.co/unsloth)'s and [chtisgit](https://huggingface.co/chtisgit)'s GSQ conversions.
Upstream's full credits: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
MIT License ([LICENSE](LICENSE)); some parts and every model have their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

# anystrata — Strata for every GPU

<table>
  <tr>
    <td width="280" valign="top">
      <img src="vulkan/docs/any-cyan.png" alt="Any Cyan, the anystrata mascot" width="260" />
    </td>
    <td valign="top">
      <p>
        One <code>make</code>, no vendor SDK: an inference engine and OpenAI-compatible server for
        Mixture-of-Experts LLMs, built around one idea — <strong>the experts live in VRAM</strong>:
        each token reads only the few experts it routes to, so a 35B MoE runs at chat speed on
        the card you already own.
      </p>
      <p>
        The default build speaks <strong>Vulkan</strong>: any AMD under Mesa, any APU,
        any GPU ROCm abandoned — just the driver already installed. No installer, no downloads
        at build or start, no Python between a request and a token. The CUDA, HIP, SYCL and
        Metal backends stay for the cards where they win; CUDA remains the reference everything
        is measured against.
      </p>
    </td>
  </tr>
</table>

## Why this exists

MoE models are cheap to run if you exploit their structure: a token activates a few percent of the weights.
Strata's engines have scheduled exactly that — expert caches, routed GEMVs, speculative decoding — for NVIDIA
first. This repo carries the port of that engine, kernel by kernel and parity test by parity test, to Vulkan
(`vulkan/`): the cards nobody's CUDA serves — AMD APUs Mesa keeps as RADV homes, integrated graphics, and
oddities like the BC-250, a repurposed PS5 board with 16 GB of unified memory that ROCm refuses to see. On one
such board, Qwen3.6-35B-A3B at 2 bits keeps all 10,240 routed experts resident in VRAM, decodes ~19 tok/s, and
answers `pi` and other coding agents over the OpenAI API. It is not one card's port: the same Vulkan engine
served Qwen3.6-35B-A3B successfully on an **RTX 4070 Laptop with only 8 GB of VRAM** — experts streaming
around a card too small to hold them (Windows section below; the measured table there is its record).

The port is held to the fork's correctness culture: integer arithmetic is **bitwise** against the CUDA tree (a
parity suite of ~50,000 outputs, zero differences, across every quantized weight format the models use — Q2_0,
Q8_0, the K-quants, the IQ-quants), float paths at the tree's own tolerances with the measured driver laws
recorded per GPU family, and the teacher-forced harness (`bench/correctness/`) as the acceptance gate. Any format
or path not yet ported refuses loudly, by name — it never answers with a wrong kernel. The CUDA/HIP sources under
`src/` and `include/strata/` are untouched by the port: `git diff` against the import commit is still the whole
story, the rule [UPSTREAM.md](UPSTREAM.md) exists to keep.

## Origins

The lineage, spelled out because the name changed:

1. **[Strata](https://github.com/Niko1221/Strata)** (Niko1221) — the original expert-streaming engine, NVIDIA first.
2. **[PolyStrata](https://github.com/VecSzn/PolyStrata)** (VecSzn and contributors) — many model families
   (`strata-poly` reading GGUFs in place: Qwen3.6, GLM, Gemma 4, DeepSeek, LFM, Ornith, Tiel…) and many GPUs:
   AMD/HIP, Intel/SYCL, Apple/Metal. Its v0.1.45-poly.1 release is imported into this repo **byte for byte**
   (verified tree hash, commit `0197ae86`).
3. **UnbiasedStrata** (robertkarl) — the suckless/dwm treatment: the installer, the build- and run-time downloads,
   and the Python middle layer are gone; in their place a C++ front end built from llama.cpp's own server code.
   The human note that opened this README — *"Strata minus the bullshit. clone, rsync to compile and go"* — is
   still the rule.
4. **anystrata** — renamed 2026-10-29 because the Vulkan port made GPU coverage the headline, and the
   name says it in plain English: Strata for *any* GPU — one engine family, every GPU that speaks a
   backend, and a search finds only this repo. The binaries carry it: `anystrata` the front end,
   `anystrata-gguf` the engine, `anystrata-gguf-vk` the Vulkan build, `anystrata-pack` the original
   pack engine. "Poly" and "unbiased" live on in the history and the credits, not in the current names.

```
Issues with Strata / PolyStrata, from the human who forked it:
* piping instructions from markdown into agent == bad idea
* replacing components at build-time or runtime isn't great
* desired: clone; submodule init; make; situation

It's been vetted in a few ways:
* agents went through and verified kernels to the difference between CPU/GPU within llama.cpp
* full a run of Qwen 3.8 Flash Next over Bob-bench wide test set with a high score.
* and, in the Vulkan port, ~50,000 kernel outputs held bitwise against the CUDA tree (vulkan/PORT-STATUS.md).

This wouldn't exist without llama.cpp.

without Niko's Strata.

Without PolyStrata either.

Use at your own risk. Works on my machine — and now, on yours.
```

## Box setup — CliffsNotes

**AMD GPU or APU under Mesa — the Vulkan path, no CUDA, no ROCm, no vendor SDK.** Debian, Ubuntu and
Linux Mint need only CMake ≥ 3.24, a C++20 compiler, the Vulkan loader/headers, and `glslc` to
compile the compute kernels. (`glslc` ships from Debian bookworm and Ubuntu noble — i.e. Mint 22;
on Mint 21 / Ubuntu 22.04 there is no glslc package, and on anything older the CMake ≥ 3.24 is the
problem instead: use a newer release, the backports cmake, or the kit's own CMake ≥ 3.30 build.)
On this path the two probes worth running before the build are the glslc check and the capability
check below; the build then takes one command:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git cmake build-essential python3 libvulkan-dev glslc
# glslc is in universe on Ubuntu/Mint: add it with `add-apt-repository universe` if prompted.

git clone https://github.com/bjaan/anystrata && cd anystrata
git submodule update --init                    # third_party/llama.cpp (pinned), shallow, no downloads at build
make -j$(nproc)                                # the default target: the Vulkan engine
ctest --test-dir build-vk --output-on-failure  # the whole parity suite on your card (tests skip cleanly without one)

# check the card speaks the two load-bearing capabilities the runtime demands:
apt install -y vulkan-tools && vulkaninfo --summary
#   needs: Vulkan >= 1.2, VK_KHR_shader_integer_dot_product, VK_EXT_subgroup_size_control
#   (RDNA1/2/3, GCN5+ under RADV all qualify; the engine refuses a card without them, by name)
```

Run it (models are downloaded on another machine and copied over; the engine fetches nothing):

```bash
build-vk/bin/anystrata --engine build-vk/bin/anystrata-gguf-vk \
  -m /path/to/Qwen3.6-35B-A3B.gguf --host 127.0.0.1 --port 18090 \
  -- --model /path/to/Qwen3.6-35B-A3B.gguf --max-context 8192
```

Context sizes on a unified-memory APU (shared RAM+VRAM): 16K/32K are safe with ≥ 1.5x the model
size in total RAM; 64K needs `--kv q8_0` (docs/VULKAN.md, "Context sizes on a unified-memory
APU"). On a discrete card there is more slack. `make build` (the CUDA path) needs the NVIDIA
toolkit instead and is the section below.

Guest, both cards:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends git cmake build-essential gcc-13 nvidia-cuda-toolkit linux-headers-amd64
python3 -m venv ~/venv-strata && ~/venv-strata/bin/pip install -r tools/requirements.txt   # offline pack tools only
nvcc --version | tail -1                                        # CUDA 12.4.131
g++-13 --version | head -1                                      # template default is 14.2; nvcc 12.4 accepts <= 13 (#1645)
cmake --version | head -1                                       # 3.31.6, needs >= 3.24
python3 --version; curl --version | head -1                     # make test needs both
nvidia-smi --query-gpu=driver_version --format=csv,noheader     # 550.163.01
ls /lib/modules/$(uname -r)/updates/dkms/ | head -1             # DKMS module must exist for the running kernel
sudo apt-get install -y --no-install-recommends linux-headers-amd64   # before the first reboot after an apt kernel upgrade
df -h /                                                         # toolkit costs ~8 GB
```

Pascal — GTX 1080 Ti, compute 6.1:

```bash
nvcc --list-gpu-arch | grep -q compute_61                       # CUDA 13 has no sm_61; stay on 12.4
nvidia-smi --query-gpu=driver_version --format=csv,noheader     # 550 is below CUDA 13's requirement (>= 580)
git clone https://github.com/bjaan/anystrata && cd anystrata
JOBS=4 make test                                                # no GPU, no model
JOBS=4 make build CUDA_ARCHS=61                                 # -DSTRATA_EXPERIMENTAL_SM60=ON, -ccbin /usr/bin/g++-13; 15 min, 398 MB, 53 warnings
```

Ada — RTX 4090, compute 8.9:

```bash
git clone https://github.com/bjaan/anystrata && cd anystrata
make test
make build CUDA_ARCHS=89                                        # 6m40 on 16 cores, 557 MB, 0 warnings
```

## Run

```
anystrata --engine <anystrata-pack|anystrata-gguf|anystrata-gguf-dsv41|path> -m <model.gguf> [llama-server options] -- <engine options>
```

`-m` names the GGUF the vocabulary and chat template are read from; no tensors are loaded from it by the front end.
`--engine` takes a path, or a bare name looked up beside the program. Everything before `--` is a `llama-server`
option (`--host`, `--port`, `--api-key`, `--alias`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`,
`--reasoning-format`, `--chat-template-kwargs`, `--sse-ping-interval`, ...); everything after `--` goes to the
engine, started as a child with `--serve` added and spoken to over its line protocol
([docs/BACKENDS.md](docs/BACKENDS.md)). Where the model ends up (GPU cache, pinned RAM, SSD) and the three expert
modes: [docs/MEMORY_LAYOUT.md](docs/MEMORY_LAYOUT.md). The server answers `/v1/chat/completions` (streaming and
not), `/v1/models`, `/health`, `/props`, `/tokenize`, `/detokenize`, one request at a time.

Vulkan, the default — Qwen3.6-35B-A3B 2-bit GSQ ([chtisgit's hybrid conversion](https://huggingface.co/chtisgit/Qwen3.6-35B-A3B-2Bit-GSQ-hybrid-GGUF)
of [ISTA-DASLab's checkpoint](https://huggingface.co/ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ); 12.2 GB, sha256
`8e50912f5d0703401ca21b09deb2ee8a84d03536ade6fb44815fcb40a0b03596` — the file this port gates on; the engine
fetches nothing, download it and point `--model` at it; all 10,240 routed experts fit in VRAM on this class of
card) on a Mesa-AMD card, with the Qwen3.6 sampling defaults:

```
build-vk/bin/anystrata --engine build-vk/bin/anystrata-gguf-vk \
  -m /models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf --host 127.0.0.1 --port 18090 \
  --alias qwen3.6-35b-a3b-gsq-hybrid --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 \
  -- --model /models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf --max-context 16384
```

(Captured graphs are the default and bitwise-verified inert on this backend, in **both** regimes:
full residency and the expert-SWAP regime (`--vram-reserve-mib` big enough that experts leave the
card). The swap regime's collapse bug — six weeks open, graphs replaying a stale windowed head
capture after the halves shifted rows — is fixed by recording head captures fresh while the cache
moves (`vulkan/PORT-STATUS.md` Task 6.0): byte-for-byte canary at 16 and 96 tokens, and 7.4 tok/s
against the old workaround's 5.2-6.2, so `--no-graphs` stays only as an A/B toggle for debugging.
Every full-residency serve config on this box — including 64K with `--kv q8_0` — runs on graphs
and is verified bitwise.)

`--max-context` on this APU: **16K / 32K with the fp16 cache, and 64K with `--kv q8_0`** — the
q8_0 KV cache (int8 codes, one fp16 scale per 32 values: 544 bytes a row where fp16 costs 1,024)
halves the cache and is what makes the 64K window fit this box's pinned arena; its store,
prefill-read and fused decode-read kernels are all ported and parity-tested. 128K does not fit
this card with this model; it is a machine for the larger pods, same as upstream's long-context
records. The law and the measurements are in `vulkan/PORT-NOTES.md` (rules 10-11).

CUDA, the reference path — Qwen3.6-35B-A3B from an Unsloth GGUF:

```
build/bin/anystrata --engine build/bin/anystrata-gguf \
  -m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 127.0.0.1 --port 18091 --alias qwen3.6-35b-a3b-ud-q3_k_m \
  -- --model /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --max-context 32768 \
     --expert-profile /models/expert-profile-q3km.txt --tail-skip 6
```

Hybrid models (some experts cached, some computed on the CPU) take the streaming knobs after the `--`:

```
    --stage-rows N         from N tokens a call copies the experts outside the cache to the card
                           (--stage-mib) and computes them there while the CPU takes its share
                           meanwhile; under N the CPU computes all of them (default 32;
                           1 = also while an answer is written — keeps card and bus busy instead
                           of the CPU; --stage-overlap 0|1 copies a staging round under the
                           cached experts' kernels)
    --mmq-tab 0|1          CUDA only: a layer's cached experts in pointer-table launches of 16
                           wherever follow() has traded their slots, instead of one launch per
                           run (STRATA_POLY_MMQ_TAB=ON build; on the Vulkan path this is a no-op
                           trap, named, per backend law)
```

With pictures, add `--mmproj /models/mmproj-F16.gguf` (the projector from the same repository) and `--threads N`
before the `--`; the front end encodes each picture once (picture cache, keyed by its SHA-256) and hands the rows
to the engine over the `GENI` protocol. A picture is at most 300 rows by default (`--image-max-tokens` changes it).
The pack engine (`anystrata-pack`, Qwen3.8-Flash-Next) takes its own BF16 projector and also needs `--vision`
after the `--`. Pack preparation (`tools/iq_pack.py`, `tools/mtp_fetch.py`, `tools/mtp_pack.py`, `tools/mtp_rt.py`)
is unchanged; published packs stay at https://huggingface.co/robertkarl/UnbiasedStrataPacks (name kept for
continuity). Model files are downloaded on a network box and copied in; the served files with sizes and SHA-256
are in [docs/MODEL-FILES.md](docs/MODEL-FILES.md).

## Windows

Both GPU backends build natively with MSVC: the CUDA engine and the Vulkan engine, plus the front end that serves them.
Verified 2026-10-30 on Windows 11 with an RTX 4070 Laptop GPU (8 GB, sm_89), 32 GB RAM, CUDA 12.6, Visual Studio 2022
17.14 and Vulkan SDK 1.4.363, serving LM Studio's `lmstudio-community/Qwen3.6-35B-A3B-GGUF` Q4_K_M file (21 GB).

What you need:

| For | Install |
| --- | --- |
| everything | Visual Studio 2022 (or Build Tools) with *Desktop development with C++*, which brings CMake and Ninja; Git |
| the CUDA engine | a CUDA 12.x or 13 toolkit (its installer puts `cudart64_*.dll` on `PATH`) |
| the Vulkan engine | the LunarG Vulkan SDK (`winget install KhronosGroup.VulkanSDK`: glslc, spirv-as, headers) and Python 3 on `PATH` (a real interpreter, not the Microsoft Store alias) |

Run everything below in an **x64 Native Tools Command Prompt for VS 2022** (`cmd`) at the repository root.
`%CD:\=/%` is cmd's way of writing the current directory with forward slashes. The engines and the front end must end
up in the same `bin` folder, because `--engine` takes a bare name and finds it beside `anystrata.exe`.

```bat
git submodule update --init
```

CUDA (set `CMAKE_CUDA_ARCHITECTURES` to your card: 75 RTX 20, 86 RTX 30, 89 RTX 40, 120 RTX 50):

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON ^
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_BUILD_TESTS=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=%CD:\=/%/build/bin -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%CD:\=/%/build/bin
cmake --build build --target anystrata anystrata-gguf strata_kq anystrata-pack anystrata-gguf-dsv41
```

Vulkan (its tree has no front end; take `anystrata.exe` from the CUDA build, or build the CUDA-free front end below):

```bat
cmake -S . -B build-vk -G Ninja -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=%CD:\=/%/build-vk/bin
cmake --build build-vk
ctest --test-dir build-vk --output-on-failure
```

The front end alone, without a CUDA toolkit:

```bat
cmake -S . -B build-frontend -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=%CD:\=/%/build-frontend/bin
cmake --build build-frontend --target anystrata
```

Serve with CUDA:

```bat
set M=%USERPROFILE%\.lmstudio\models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf
build\bin\anystrata.exe --engine anystrata-gguf -m "%M%" --host 127.0.0.1 --port 18091 ^
  --alias qwen3.6-35b-a3b-q4_k_m --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 ^
  -- --model "%M%" --max-context 8192
```

Serve with Vulkan (`--engine` takes the engine's path; `--no-graphs` is needed when the model does not fit in VRAM,
see below):

```bat
build-frontend\bin\anystrata.exe --engine %CD%\build-vk\bin\anystrata-gguf-vk.exe -m "%M%" --host 127.0.0.1 --port 18092 ^
  --alias qwen3.6-35b-a3b-q4_k_m-vk --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 ^
  -- --model "%M%" --max-context 8192 --no-graphs
```

Ask it something (Windows ships `curl.exe`):

```bat
curl http://127.0.0.1:18091/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23?\"}]}"
```

Ctrl+C stops the server, and the engine goes with it. Measured on the laptop above, at 8,192 context:

| Engine | Experts in VRAM | Decode | Notes |
| --- | --- | --- | --- |
| `anystrata-gguf` (CUDA 12.6) | 2,159 of 10,240 (3.8 GiB) | 28.4 tok/s | the rest of the experts run on the CPU (24 threads) |
| `anystrata-gguf-vk` (NVIDIA Vulkan driver) | 2,537 of 10,240 (4.5 GiB) | 10.0 tok/s | `--no-graphs`; 14.2 tok/s with graphs and `--lookup 0` |

Both answered the test prompt correctly, and their first 32 generated token ids matched exactly.

Known limits on Windows:

* **Vulkan with a model larger than VRAM:** the graphs + expert-SWAP + lookup-guesser bug that made that
  configuration collapse to id-0s (BC-250) or crash mid-generation (NVIDIA/Windows) is FIXED on branch
  `task-6.0-head-capture` (swap-regime head captures record fresh; byte-for-byte canary on the BC-250, and
  7.4 vs 5.2-6.2 tok/s over the old `--no-graphs` workaround). Until it merges into your build, run the
  swap regime with `--no-graphs` or `--lookup 0`. The Windows card should re-run its arm once merged.
* **`ctest` in `build-vk` passes 13 of 17 on a card that is not a BC-250.** The other four are pinned to that board:
  `vk_core_smoke` asserts its 24 CUs, `vk_doorbell_probe` spins a kernel longer than Windows' 2 s GPU timeout allows,
  and `native_mmvq_vk_parity` / `poly_kernels_vk_parity` expect RADV's bitwise floats, where NVIDIA's shader compiler
  differs in the last bits (Q5_K mmvq, rope).
* With a CUDA 12.x toolkit the build links the shared CUDA runtime (#585), so `cudart64_12.dll` must be on `PATH` when
  the engine starts.

## Supported models

Three engines, and what each one runs. The `general.architecture` string in a GGUF decides the engine family;
the files actually served, with sizes and SHA-256, are listed in [docs/MODEL-FILES.md](docs/MODEL-FILES.md).

| Engine | Architecture (`general.architecture`) | Models | CUDA | HIP | SYCL | Metal | **Vulkan** |
| --- | --- | --- | :-: | :-: | :-: | :-: | :-: |
| `anystrata-pack` | `qwen4exp` | Qwen3.8-Flash-Next and its fine-tunes (Coder, Swift 1.5) — pack + MTP directories, GSQ-RCO quants | ✓ | – | ✓ | ✓ | not ported |
| `anystrata-gguf` | `qwen35moe` | Qwen3.6-35B-A3B, Ornith 1.5, Tiel Coder, MiniCPM-V 4.7 (Unsloth / chtisgit-GSQ / community GGUFs) | ✓ | ✓ | ✓ | ✓ | **✓ end-to-end** (the port's gate model) |
| | `glm5-next` | GLM 5.3 Flash (GSQ-RCO, REAP, UD quants) | ✓ | ✓ | – | – | untried |
| | `gemma4` | Gemma 4 26B-A4B | ✓ | ✓ | – | – | untried |
| | `deepseek4` | DeepSeek V4 Flash | ✓ | ✓ | – | – | untried |
| | `lfm2moe` | LFM2.5-24B-A2B | ✓ | – | – | – | untried |
| `anystrata-gguf-dsv41` | `deepseek41` | DeepSeek-V4.1-Flash-1017 (Strata v0.2.4's engine: the four shards, Engram table, DSpark drafter; IQ1_M / IQ2_XXS / IQ2_XS GSQ-RCO) | ✓ | – | – | – | not ported |

The Vulkan column reports what has run end-to-end on a Vulkan device (the BC-250 gate: `qwen35moe` files, all
expert and dense quant formats held bitwise against the CUDA tree, the q8_0 KV cache — store, prefill
read and fused decode read all ported, making a 64K serve fit this box — see `vulkan/PORT-STATUS.md`); the other
families build against the same shim but wait for their own bring-up gate. Anything the Vulkan port has not
carried yet refuses loudly, by name, and never answers with a wrong kernel.

Dense (non-MoE) models are a different story: Strata's design — streaming routed experts to VRAM — only applies
to MoE. A dense GGUF like `qwen35` Qwen3.8-27B is served best by llama.cpp's own Vulkan backend
([tools/vulkan/serve-swift27.sh](tools/vulkan/serve-swift27.sh)), and no Strata engine takes that architecture
on any backend.

## Backends

| Backend | Command | Needs | Status |
| --- | --- | --- | --- |
| **Vulkan** (default) | `make` | a Vulkan loader, glslc; any Vulkan 1.2+ driver | the port: [docs/VULKAN.md](docs/VULKAN.md) |
| CUDA | `make build` | CUDA 12.x | upstream's, unchanged; the reference |
| AMD HIP | `-DSTRATA_ENABLE_HIP=ON` | ROCm 7 (gfx1100/gfx1201 validated) | upstream's |
| Intel SYCL | `-DSTRATA_ENABLE_SYCL=ON` | oneAPI/icpx | upstream's (community) |
| Metal | `metal/` | macOS | upstream's (community) |

### Example lines (the pack and DeepSeek-V4.1 engines, CUDA path)

```
build/bin/anystrata --engine build/bin/anystrata-pack \
  -m /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --host 127.0.0.1 --port 18090 --alias qwen3.8-flash-next-iq3_xxs --sse-ping-interval 10 \
  -- --pack /models/pack-iq3-xxs \
     --native /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
     --ple-gguf /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
     --expert-profile /models/pack-iq3-xxs/expert-profile.bin \
     --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp /models/mtp \
     --max-context 65536 --kv int8 --ple-io ram --pool-workers 15 --kv-resident 32768
```


**DeepSeek-V4.1-Flash-1017, ISTA-DASLab IQ2_XXS** (`anystrata-gguf-dsv41`, Strata v0.2.4's third engine; the four shards,
the shared Engram file, Unsloth's DSpark drafter and a pack directory made from them): [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

```
build/bin/anystrata --engine build/bin/anystrata-gguf-dsv41 \
  -m /models/dsv41/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf \
  --host 127.0.0.1 --port 11434 --alias deepseek-v4.1-flash-1017-ista-iq2_xxs --sse-ping-interval 10 \
  --chat-template-file data/dsv41/chat_template-ista-1017.jinja --temp 1.0 --top-p 1.0 --min-p 0.0 --top-k 64 \
  -- --pack /models/dsv41/IQ2_XXS/maya-ds4 --max-context 131072 --dspark auto --gpus 0
```

### Preparing a Flash-Next pack

`anystrata-gguf` reads a GGUF as it is. `anystrata-pack` needs a pack directory and an MTP draft directory made once, offline
except for the draft download, with upstream's tools (`tools/requirements.txt`: numpy, regex; `gguf-py` comes from the
submodule). This is what upstream's installer ran. Re-run in this fork on 2026-10-20 on three hosts; the MTP directory
came out byte-identical each time, so it is published, with the commands and hashes, at
https://huggingface.co/robertkarl/UnbiasedStrataPacks: copy it instead of fetching and quantizing.

```
python tools/iq_pack.py --gguf <shard1> --out <pack>             # --compat-bf16 for Unsloth files
STRATA_SOURCE=huggingface python tools/mtp_fetch.py fetch --out <mtp>
python tools/mtp_pack.py --src <mtp> --experts q2_0 --out <mtp>/mtp-q2_0.gguf
python tools/mtp_rt.py --gguf <mtp>/mtp-q2_0.gguf --out <mtp>/rt
cp data/draft_vocab.bin <mtp>/rt/draft_vocab.bin
```

Model files themselves are downloaded on a machine with network access and copied to the inference box; the fork does
not fetch them. The files that have been served, with sizes and SHA-256, are in
[docs/MODEL-FILES.md](docs/MODEL-FILES.md).


## Measured

The CUDA and upstream-vs-fork tables (RTX 3090 / A6000 / B200 numbers, PLAN.md steps 6-13, the teacher-forced
correctness record with its USTRAT tickets, the pi acceptance runs) are unchanged and live in `docs/`,
`bench/correctness/` and git history at 7551f138. New, from the port itself — **BC-250 (gfx1013 APU, Mesa/RADV:
no CUDA, no ROCm)** with Qwen3.6-35B-A3B GSQ-hybrid 2-bit, 10,240/10,240 experts resident:

These numbers qualify **main** (engine version `0.1.45`, `CMakeLists.txt`; the `windows-build` merge `7ece380` —
native MSVC + the discrete-card planner fixes — and upstream's hybrid-performance merge `1d474dd`, canary-verified
inert at full residency, are both in; rows re-run on 2026-11-04 carry that date, older rows say when).

| Gate | Result |
| --- | --- |
| decode | 25.4 tok/s (96-token leg, full residency, 2026-11-04 re-run); 16-token legs are load-boundary-bound (10.6-11.5); up to 24.3 across the older 128-token gates |
| prefill (736 / 4,954 token prompts) | **36.2** / 50 tok/s (736-token leg re-run on current main 2026-11-04, canary ids correct after; 4,954 from the 2026-10-27 tiled-GEMM era) |
| `pi` coding agent (1.6K-token system prompt) | served repeatedly: 52 s round trip, KV reuse, grammar parse |
| parity vs the CUDA tree | ~50,000 outputs bitwise; runtime driver laws per GPU, measured |
| the canary | `17*23?` -> 391, 16 ids bitwise-fixed per configuration; run after every change ([docs/CANARY.md](docs/CANARY.md)) |
| upstream merge 1d474dd (`--stage-rows` streaming + `--mmq-tab`, shared engine core) | canary bitwise-identical on the BC-250; decode at full residency 25.4 tok/s (96-token leg) — the best measured here; upstream's own cards gained 20.6 -> 26.5-29.3 tok/s decode streaming and 94.6 vs 68.1 t/s prefill with `--mmq-tab` (5090/1060-class, CUDA; their numbers, not this box's) |
| context on the APU | 16K / 32K stable (14.1 / 13.9 GB resident); 64K live with `--kv q8_0` — the whole q8 chain (store, prefill workspace read, fused decode read) ported, parity-green and correctness-scored — PORT-NOTES laws 10-13 |
| swap regime (experts streaming, 4096-ctx legs, current main) | eager: 5.2-6.2 tok/s at 50-55.5% residency, canary **byte-for-byte vs the recorded line** (re-verified 2026-11-04); graphs+`--lookup 0`: no collapse, answers 391 (its guess-off continuation drifts one echo token — a second-order path difference, not the bug); graphs+`--lookup` (defaults) collapsed to id-0s — the Task 6.0 bug: the windowed HEAD capture replays a stale snapshot/row state once the halves shift; fixed by recording heads fresh while the cache moves (byte-exact at 16/96 tokens, 7.4 vs 5.2-6.2 tok/s over the workaround; merge `1d474dd` proven innocent, pre-merge binary collapsed identically) |
| memory on the 15.7 GiB unified board | boot: 15,714 MiB total / 14,780 MiB free. Full serve: 8.4 GiB expert arena, ~14.1 GB resident at 64K q8_0 — ~1.5-2 GB free for the desktop afterward; `leg.sh` blocks a second engine load under 10,500 MB available (law 10, the crash-week lesson): one load at a time, always |
| correctness gate (teacher-forced vs llama.cpp at the pin) | base arm 1,678 steps: step-weighted KL 1.39x llama.cpp's own CPU-vs-GPU floor (bar: 1.5x); argmax agrees at every gap>=0.5-nat position but one near tie; draft arm bitwise-inert (T0); q8_0-KV arm 1,096 steps at 1.82x (the storage trade, zero argmax flips); graphs bitwise-inert — table in `vulkan/PORT-STATUS.md` |

(llama.cpp's own Vulkan backend on the same card and model: ~76 tok/s decode — the ceiling this backend climbs
toward while keeping Strata's expert-cache design, which is what lets a 12 GB model sit entirely inside
15.7 GB of unified memory.)

## The canary

Before any build is trusted, it answers one question: `17*23?` — and the answer must be **391**,
as the same 16 token ids, byte-identical to the recorded list, on every run. One multiplication
crosses the whole engine — load, expert cache, kernels, attention, sampler — and it is the
fastest way to learn that a single float moved somewhere it should not have. Silly the seabird
carries the 18-byte Q2_0 crates this runs on; the number on Any Cyan's hairpin is the answer.
The full story, and how to run it: [docs/CANARY.md](docs/CANARY.md).

## Known limits

Read [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and the backend pages before filing anything. The Vulkan
port's open items are enumerated in `vulkan/PORT-STATUS.md`: captured graphs are the verified default (Phase 4.2
green and bitwise-inert, in the expert-SWAP regime too since the Task 6.0 capture-staleness fix; `--no-graphs`
remains as the A/B toggle), prefill tiles are under tuning, and every unported
format or path traps by name — loudly, never silently.
pi's compaction defaults assume >64K windows; serve more context or lower them (USTRAT-8).
On this APU, llama.cpp reference runs must obey `vulkan/PORT-NOTES.md` rule 10 (their loader copies the
whole GGUF into pinned anon RAM on an iGPU and the OOM killer then takes the desktop): one load at a
time, mmap forced, and an oom-score above the session's.

## Mascot

<table>
  <tr>
    <td width="280" valign="top">
      <img src="vulkan/docs/any-cyan-approves.png"
           alt="Any Cyan, the anystrata mascot"
           width="260" />
    </td>
    <td valign="top">
      <strong>Any Cyan</strong> — anystrata's mascot.
      See her <a href="vulkan/docs/mascot.md">character sheet</a>.
      <p>
        An early-twenties engineer in a PCB-teal cardigan;
        a three-disc hair-tie (the expert cache: cached / warm / cold),
        a strata-banded skirt (the bitwise quant formats),
        a hairpin engraved <strong>391</strong> —
        the answer she first computed on the BC-250 —
        and Silly the crate-bird hauling an 18-byte Q2_0 block.
      </p>
      <p>
        She belongs to every card that runs the engine.
        Cute, but she documents it.
      </p>
    </td>
  </tr>
  <tr>
    <td width="280" valign="top">
      <img src="vulkan/docs/silly.png"
           alt="Silly the crate-bird"
           width="260" />
    </td>
    <td valign="top">
      <strong>Silly</strong> — the crate-bird, and the load path.
      A round teal seabird (skillfish species: the BC-250's die codename is <em>Cyan
      Skillfish</em>) hauling an <strong>18-byte crate</strong>: exactly one Q2_0 block —
      32 weights at 2 bits plus its fp16 scale — the unit most routed experts of an MoE
      arrive at the card in. Every expert the engine streams is one crate in her beak;
      she is how 12.2 GB gets where 8.4 of it fits.
    </td>
  </tr>
</table>
