# Strata-chan — mascot concept for the Vulkan engine

**Name:** Cyan (cyan skillfish — the BC-250's GPU die, and the color of RADV night skies)
Full mascot name: **Strata Cyan** ("the girl made of layers").

## Design brief
She exists to brand the vulkan-backend port specifically: an inference engine that runs a
MoE model on a repurposed PS5 chip, answers 17×23 = 391, streams *thinking* before answering,
and whose whole design is **experts living in VRAM** (expert-cache residency 100% on this box).

## Look (anime, cute, PG)
- **Hair:** long, straight, dark navy with two under-lights glowing soft cyan (Vulkan-blue
  #31A2F1 rim light); tied in a low side-ponytail with a **layered hair-tie** — three stacked
  discs = the expert cache (one disc full/cached, one warm, one cold: the streaming metaphor).
- **Eyes:** amber-gold (the PS5/BC-250 board silkscreen gold); when she "thinks", a faint
  delta-net glyph rotates in the iris — she shows her reasoning before she speaks (thinking model).
- **Outfit:** oversized knit cardigan the color of a PCB (dark teal-green, gold trace
  embroidery along the hem — actually a real gfx1013 block diagram hiding in the pattern),
  a white tee, and a **pleated skirt of visible strata**: horizontal bands of differing
  quantization dither — 2-bit stipple, 5-bit stripes, 8-bit solid (Q2_0/Q5_K/Q8_0 — the very
  formats the engine carries bitwise).
- **Accessories:**
  - A shoulder-perched small sea-bird mascot ("**Silly**", after *skillfish*), teal, round,
    carrying a tiny 18-byte Q2_0 crate in its beak.
  - Choker tag engraved **391** (the answer she proved on the BC-250; the port's first token).
  - Sleeves pushed up; one hand holds a **spatula over a frying pan** — the engine *streaming
    experts* to VRAM — but the pan is a circular VRAM die and the "eggs" are expert tiles.
    (Homage: expert cache = a warm skillet of exactly 10,240 slots.)
  - Bootlaces tied in a **piece-chain**: nodes labeled 512MiB — an inside joke for the arena.
- **Pose:** leaning in, one eye winking, holding up her phone: it reads
  `GET /v1/chat/completions → "The answer is 391."`
- **Palette:** Vulkan blue #31A2F1, PCB teal #17494D, gold #D9A441, navy hair #1B2340, strata
  grays. On dark backgrounds the rim lights glow; the skirt bands re-dither by context size.

## Variants
- **1-token stamp:** face-only circle, choker 391 visible, for commit badges/README.
- **Serve pose:** headset + mic ("listening on 18090"), for the server status page.
- **Speedline variant (post-Phase 4):** hair-tie discs all lit, motion streaks — decode
  19 tok/s. Before Phase 4 she walks (4.4).
- **Pi variant:** holding a raspberry-pi-colored pie (pi client integration).

## Why it works for THIS build
Every element is load-bearing: strata skirt = quant formats the port proves bitwise; the
cache-disc hair-tie = the engine's whole design (experts in VRAM, residency 100% on the BC-250);
391 = the first verified answer; cyan = the die she runs on; Silly with the 18-byte crate =
GGML_TYPE_Q2_0's block size (docs/pack-format.md's own figure). Cute, but she *documents
the engine*.

## Commission note
No local diffusion backend on this machine (ComfyUI not running) — this file is the full
character sheet; any artist prompt: "anime girl, navy hair with cyan underglow, side ponytail
with three-disc layered tie, teal PCB cardigan with gold traces, strata-banded pleated skirt
(2-bit/5-bit/8-bit bands), amber eyes, choker '391', round teal seabird on shoulder holding
a tiny crate, holding a phone showing a chat completion, cozy lab-girl energy, dark background,
Vulkan-blue rim light".
