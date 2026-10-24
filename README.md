## Credits and license

The engines are PolyStrata's, by VecSzn and contributors, built on [Strata](https://github.com/Niko1221/Strata) by
Niko1221. The front end is built from [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)'s server, common and
mtmd code. Qwen3.8-Flash-Next and Qwen3.6-35B-A3B are by the Qwen team; the IQ3_XXS GSQ-RCO file is
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)'s, the Qwen3.6 files are
[Unsloth](https://huggingface.co/unsloth)'s and [chtisgit](https://huggingface.co/chtisgit)'s GSQ conversions.
Upstream's full credits: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
MIT License ([LICENSE](LICENSE)); some parts and every model have their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

# omnistrata — Strata for every GPU

One `make`, no vendor SDK: an inference engine and OpenAI-compatible server for Mixture-of-Experts LLMs, built
around one idea — **the experts live in VRAM**: each token reads only the few experts it routes to, so a 35B MoE
runs at chat speed on the card you already own. The default build speaks **Vulkan**: any AMD under Mesa, any APU,
any GPU ROCm abandoned — just the driver already installed. No installer, no downloads at build or start, no
Python between a request and a token. The CUDA, HIP, SYCL and Metal backends stay for the cards where they win;
CUDA remains the reference everything is measured against.

```
git clone <this repo> && cd omnistrata
git submodule update --init      # shallow, third_party/llama.cpp @ robertkarl/llama.cpp (a6de9af3)
make                             # the Vulkan engine: build-vk/bin/{omnistrata-gguf-vk, omnistrata}
make test                        # the front end against a fake engine; no GPU, no model
make build                       # the CUDA engine, opt-in: build/bin/{omnistrata-pack, omnistrata-gguf, libstrata_kq.so, omnistrata}
```

## Why this exists

MoE models are cheap to run if you exploit their structure: a token activates a few percent of the weights.
Strata's engines have scheduled exactly that — expert caches, routed GEMVs, speculative decoding — for NVIDIA
first. This repo carries the port of that engine, kernel by kernel and parity test by parity test, to Vulkan
(`vulkan/`): the cards nobody's CUDA serves — AMD APUs Mesa keeps as RADV homes, integrated graphics, and
oddities like the BC-250, a repurposed PS5 board with 16 GB of unified memory that ROCm refuses to see. On one
such board, Qwen3.6-35B-A3B at 2 bits keeps all 10,240 routed experts resident in VRAM, decodes ~19 tok/s, and
answers `pi` and other coding agents over the OpenAI API.

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
4. **omnistrata** — the same project, renamed 2026-10-24 because the Vulkan port made GPU coverage the headline:
   one engine family, every GPU that speaks a backend. Hence *omni*strata, and the binaries carry it
   (`omnistrata` the front end, `omnistrata-gguf` the engine, `omnistrata-gguf-vk` the Vulkan build,
   `omnistrata-pack` the original pack engine). "Poly" and "unbiased" live on in the history and in the credits,
   not in the names. The GitHub repo keeps its old name until it is pushed elsewhere; the tree is what changed.

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
git clone https://github.com/robertkarl/UnbiasedStrata && cd UnbiasedStrata   # the repo directory keeps the old name until it is renamed upstream
JOBS=4 make test                                                # no GPU, no model
JOBS=4 make build CUDA_ARCHS=61                                 # -DSTRATA_EXPERIMENTAL_SM60=ON, -ccbin /usr/bin/g++-13; 15 min, 398 MB, 53 warnings
```

Ada — RTX 4090, compute 8.9:

```bash
git clone https://github.com/robertkarl/UnbiasedStrata && cd UnbiasedStrata   # the repo directory keeps the old name until it is renamed upstream
make test
make build CUDA_ARCHS=89                                        # 6m40 on 16 cores, 557 MB, 0 warnings
```

## Run

```
omnistrata --engine <omnistrata-pack|omnistrata-gguf|omnistrata-gguf-dsv41|path> -m <model.gguf> [llama-server options] -- <engine options>
```

`-m` names the GGUF the vocabulary and chat template are read from; no tensors are loaded from it by the front end.
`--engine` takes a path, or a bare name looked up beside the program. Everything before `--` is a `llama-server`
option (`--host`, `--port`, `--api-key`, `--alias`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`,
`--reasoning-format`, `--chat-template-kwargs`, `--sse-ping-interval`, ...); everything after `--` goes to the
engine, started as a child with `--serve` added and spoken to over its line protocol
([docs/BACKENDS.md](docs/BACKENDS.md)). Where the model ends up (GPU cache, pinned RAM, SSD) and the three expert
modes: [docs/MEMORY_LAYOUT.md](docs/MEMORY_LAYOUT.md). The server answers `/v1/chat/completions` (streaming and
not), `/v1/models`, `/health`, `/props`, `/tokenize`, `/detokenize`, one request at a time.

Vulkan, the default — Qwen3.6-35B-A3B 2-bit GSQ (chtisgit's hybrid conversion of ISTA-DASLab's checkpoint; 12.2 GB;
all 10,240 routed experts in VRAM) on a Mesa-AMD card, with the Qwen3.6 sampling defaults:

```
build-vk/bin/omnistrata --engine build-vk/bin/omnistrata-gguf-vk \
  -m /models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf --host 127.0.0.1 --port 18090 \
  --alias qwen3.6-35b-a3b-gsq-hybrid --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 \
  -- --model /models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf --no-graphs --max-context 16384
```

CUDA, the reference path — Qwen3.6-35B-A3B from an Unsloth GGUF:

```
build/bin/omnistrata --engine build/bin/omnistrata-gguf \
  -m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 127.0.0.1 --port 18091 --alias qwen3.6-35b-a3b-ud-q3_k_m \
  -- --model /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --max-context 32768 \
     --expert-profile /models/expert-profile-q3km.txt --tail-skip 6
```

With pictures, add `--mmproj /models/mmproj-F16.gguf` (the projector from the same repository) and `--threads N`
before the `--`; the front end encodes each picture once (picture cache, keyed by its SHA-256) and hands the rows
to the engine over the `GENI` protocol. A picture is at most 300 rows by default (`--image-max-tokens` changes it).
The pack engine (`omnistrata-pack`, Qwen3.8-Flash-Next) takes its own BF16 projector and also needs `--vision`
after the `--`. Pack preparation (`tools/iq_pack.py`, `tools/mtp_fetch.py`, `tools/mtp_pack.py`, `tools/mtp_rt.py`)
is unchanged; published packs stay at https://huggingface.co/robertkarl/UnbiasedStrataPacks (name kept for
continuity). Model files are downloaded on a network box and copied in; the served files with sizes and SHA-256
are in [docs/MODEL-FILES.md](docs/MODEL-FILES.md).

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
build/bin/omnistrata --engine build/bin/omnistrata-pack \
  -m /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --host 127.0.0.1 --port 18090 --alias qwen3.8-flash-next-iq3_xxs --sse-ping-interval 10 \
  -- --pack /models/pack-iq3-xxs \
     --native /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
     --ple-gguf /models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
     --expert-profile /models/pack-iq3-xxs/expert-profile.bin \
     --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp /models/mtp \
     --max-context 65536 --kv int8 --ple-io ram --pool-workers 15 --kv-resident 32768
```


**DeepSeek-V4.1-Flash-1017, ISTA-DASLab IQ2_XXS** (`omnistrata-gguf-dsv41`, Strata v0.2.4's third engine; the four shards,
the shared Engram file, Unsloth's DSpark drafter and a pack directory made from them): [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

```
build/bin/omnistrata --engine build/bin/omnistrata-gguf-dsv41 \
  -m /models/dsv41/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf \
  --host 127.0.0.1 --port 11434 --alias deepseek-v4.1-flash-1017-ista-iq2_xxs --sse-ping-interval 10 \
  --chat-template-file data/dsv41/chat_template-ista-1017.jinja --temp 1.0 --top-p 1.0 --min-p 0.0 --top-k 64 \
  -- --pack /models/dsv41/IQ2_XXS/maya-ds4 --max-context 131072 --dspark auto --gpus 0
```

### Preparing a Flash-Next pack

`omnistrata-gguf` reads a GGUF as it is. `omnistrata-pack` needs a pack directory and an MTP draft directory made once, offline
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

| Gate | Result |
| --- | --- |
| decode (16 / 128 token runs) | 18.3 / up to 24.3 tok/s |
| prefill (735 / 4,954 token prompts) | 32 / 50 tok/s (5.5x after the GPU tiled GEMM; tuning continues) |
| `pi` coding agent (1.6K-token system prompt) | served repeatedly: 52 s round trip, KV reuse, grammar parse |
| parity vs the CUDA tree | ~50,000 outputs bitwise; runtime driver laws per GPU, measured |

(llama.cpp's own Vulkan backend on the same card and model: ~76 tok/s decode — the ceiling this backend climbs
toward while keeping Strata's expert-cache design, which is what lets a 12 GB model sit entirely inside
15.7 GB of unified memory.)

## Known limits

Read [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and the backend pages before filing anything. The Vulkan
port's open items are enumerated in `vulkan/PORT-STATUS.md`: captured graphs are pending Phase 4.2 (run
`--no-graphs`), prefill tiles are under tuning, and every unported format or path traps by name — loudly, never
silently. pi's compaction defaults assume >64K windows; serve more context or lower them (USTRAT-8).

## Mascot

**Omni Cyan** — the engine's character sheet: [vulkan/docs/mascot.md](vulkan/docs/mascot.md). An early-twenties
engineer in a PCB-teal cardigan; a three-disc hair-tie (the expert cache: cached / warm / cold), a strata-banded
skirt (the bitwise quant formats), a hairpin engraved **391** — the answer she first computed on the BC-250 — and
Silly the crate-bird hauling an 18-byte Q2_0 block. She belongs to every card that runs the engine.
Cute, but she documents it.
