# DeepSeek-V4.1-Flash-1017 (`strata-dsv41`)

DeepSeek-V4.1-Flash-1017 is llama.cpp's architecture `deepseek41`: 40 layers, 384 routed experts in each (6 a token),
an n-gram table (Engram) read for every token, and a small draft model (DSpark). Neither of this fork's two engines reads
it (`strata-poly` knows `deepseek4`, the V4 Flash of [DEEPSEEK.md](DEEPSEEK.md)), so it runs on a third, upstream
Strata's `strata-dsv41` (`src/dsv41`, a port of Project Maya's `maya-ds4`, MIT; `third_party/project-maya/NOTICE.md`),
imported unmodified from Strata v0.2.4 ([UPSTREAM.md](../UPSTREAM.md#deepseek-v41-engine-srcdsv41-from-strata-v024)).
The server, the API and the fork's rules are the README's: no installer, no download at start, the front end in front.

It is early. The fork has run it on one machine (below) with the ISTA-DASLab 2.00 bpw set.

## The files

ISTA-DASLab's GSQ-RCO sets, [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
revision `cb02422c900df489cea5833e8a92171224059b3c` (the one upstream v0.2.4 pins). Three sets share one Engram file; the
drafter is Unsloth's.

| File | Size | What it is |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-0000{1..4}-of-00004.gguf` | 120.7 GB | the model at 1.75 bpw |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1..4}-of-00004.gguf` | 138.2 GB | the model at 2.00 bpw, the set this fork runs |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-0000{1..4}-of-00004.gguf` | 155.2 GB | the model at 2.25 bpw |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110.6 GB | the 4-bit Engram table, one file for all three sets; a few rows are read from it for every token, so it lives on NVMe |
| `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` `DSpark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | 8.0 GB | the DSpark drafter |
| `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` `UD-Q8_K_XL/...-00001-of-00009.gguf` | 5.8 MB | metadata only: `pack_gguf.py --engram-meta` reads the Engram tensor shapes from it (its first shard is the header and nothing else) |

2.00 bpw: 138 + 111 + 8 = 257 GB on the disk the engine reads from. Upstream's quality numbers for the sets, from
its own measurement (4,995 positions): 2.00 bpw KL 0.170, perplexity +15.5%; 2.25 bpw KL 0.107, +7.3%; 1.75 bpw KL 0.217,
+21%, weaker on prose. ISTA-DASLab's card: 98.7% of the base model's task average kept at 2.00.

SHA-256 of every file is in the repository's `SHA256SUMS`-style listing in the Hugging Face tree; this fork checks each
file with `sha256sum -c` before packing (the homelab's copies: `/mnt/ds4/dsv41/{IQ2_XXS,engram}/SHA256SUMS`).

## Preparing the pack

`strata-dsv41` reads a pack directory, not the shards: metadata files that point into the shards, the Engram and the
drafter. `tools/dsv41/pack_gguf.py` writes it from the files in place (stdlib Python, no copy of the tensors), the plain
recipe upstream's installer runs for these sets. Do not add `--skip-tensors` or `--dense-override`: upstream says that
packs without an error and answers garbage.

```
python3 tools/dsv41/pack_gguf.py --model /mnt/ds4/dsv41/IQ2_XXS \
  --shards DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1,2,3,4}-of-00004.gguf \
  --engram /mnt/ds4/dsv41/engram \
  --engram-meta /mnt/ds4/dsv41/meta/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
  --dspark-gguf /mnt/ds4/dsv41/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --dspark-noise-token 128799 --dspark-markov-rank 256 \
  --out /mnt/ds4/dsv41/IQ2_XXS/maya-ds4
cp data/dsv41/profile-2.00.bin /mnt/ds4/dsv41/IQ2_XXS/maya-ds4/expert-profile.bin   # profile-2.25.bin for IQ2_XS
build/bin/strata-dsv41 --pack /mnt/ds4/dsv41/IQ2_XXS/maya-ds4 --pack-info             # every file present
python3 tools/dsv41/smoke.py --exe build/bin/strata-dsv41 --pack /mnt/ds4/dsv41/IQ2_XXS/maya-ds4   # 16 tokens; must name Paris
```

The hot-expert profile upstream ships (`data/dsv41/profile-2.00.bin`, sha256 beside it) decides which experts are loaded
into VRAM first; the engine updates it as it runs.

## Run

```
build/bin/unbiased-strata --engine build/bin/strata-dsv41 \
  -m /mnt/ds4/dsv41/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf \
  --host 0.0.0.0 --port 11434 --alias deepseek-v4.1-flash-1017-ista-iq2_xxs --sse-ping-interval 10 \
  -- --pack /mnt/ds4/dsv41/IQ2_XXS/maya-ds4 --max-context 131072 --dspark auto --gpus 0
```

- `-m` is shard 1: the front end reads the vocabulary and the chat template from it and no tensors. The pinned
  llama.cpp does not know `deepseek41`, so the vocabulary load takes the front end's `general.architecture` override path
  (USTRAT-6), as for any architecture it has not met.
- Engine options are upstream's (`strata-dsv41 --help`): `--max-context`, `--dspark auto|on|off|none` (`auto`, never
  `off`: `off` still loads the drafter and its experts take VRAM), `--gpus`, `--vram-experts N` (how many experts to keep
  on the card; the engine sizes it from free VRAM by default), `--ssd-cache` (the RAM store for the experts beyond VRAM,
  sized by itself), `--expert-profile`, `--threads`. The latent KV cache is FP4 in the engine; there is no `--kv` type flag
  as `strata` has.
- Thinking is on by default, from the template. `reasoning_effort` takes `low`, `medium`, `high` (the default), `max` or a
  number 1 to 100; `chat_template_kwargs.enable_thinking: false` turns it off. Tool calls are DSML
  (`<｜DSML｜ calls>` ... `<｜DSML｜ invoke name="...">`); the front end parses them with llama.cpp's
  `common/parsers/deepseek.cpp`, which carries PR 30817's V4.1 hunks ([UPSTREAM.md](../UPSTREAM.md)), and emits OpenAI
  `tool_calls`. The front end caps `top_k` at 64 (USTRAT-5); DeepSeek's recommended `top_k 0` is not reachable.
- Not in this engine version: several requests at once and pictures (both ignored with a warning). The engine's `STAT` and
  `NOTE` lines are ignored by the front end.

## Where it goes

| | size | where |
| --- | ---: | --- |
| routed experts (15,360) | ~110 GB at 2.00 bpw | as many as fit in VRAM after the dense weights and the context; the rest on the CPU lane (`STRATA_DSV41_TQ_CPU=1`, ggml-cpu's i-quant dots on AVX2 or AVX-512), read from the RAM store or the model files |
| attention and other dense weights | 16.8 GB | VRAM |
| Engram | 110.6 GB | the SSD; a few rows a token |

The experts the card does not hold are read from the files while the model answers, so the OS file cache wants as much of
them as the RAM allows. Upstream's measurements: RTX 5070 (12 GB) with 64 GB RAM 4.7 tokens/s; RTX 3060 (12 GB) with 46 GB
RAM 7.2 tokens/s; four R9700 (32 GB each) 31 tokens/s; "under about 40 GB of RAM it is untested and slow".

## What it was measured on

| | |
| --- | --- |
| Graphics card | RTX A6000, 48 GB (Proxmox VM 220, passthrough) |
| CPU | 12 vCPU of a Ryzen 7 1700 (AVX2, no AVX-512) |
| RAM | 52 GiB configured, 50 GiB in the guest |
| Disk | 420 GB NVMe (thin volume) for the shards, Engram, drafter and pack |
| Built with | CUDA 12.x, `make build` (`STRATA_BUILD_DSV41=ON`, `CUDA_ARCHS` for sm_86), 0 warnings |
| Model | IQ2_XXS set above, 2.00 bpw |
| Context | 128K (`--max-context 131072`) |

Results: to be filled in at first light (decode and prefill tokens/s, VRAM split, load time).

## Tests

- `make test` (the front end against the fake engine) passes unchanged with the engine built in.
- `iq_avx2_parity`: 0 failures after the swap to upstream's `iq_avx2.cpp` / `iq_avx512.cpp`, so `strata`'s and
  `strata-poly`'s CPU i-quant rows are what they were.
- Upstream's engine tests (`ds4_tq_test`, `ds4_iq_test`, `ds4_slot_test`, `ds4_gg_bench`) build with
  `STRATA_BUILD_DSV41=ON`; they are not part of `make test` (they need a GPU, and `ds4_slot_test` a pack).
