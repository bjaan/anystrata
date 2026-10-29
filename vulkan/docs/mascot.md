# Any Cyan — mascot character sheet (anystrata)

## Identity (fixed canon — never varies)
- **Name:** **Any Cyan** ("the girl made of layers, for every GPU"). *Any* from the project
  name — one engine, any GPU; *Cyan* from the Vulkan spec's cyan light, "the color of any GPU
  at night". She was drawn as Strata Cyan, then Omni Cyan as the project renamed twice; the
  character never changed, only her prefix — now the honest one.
- **What she brands:** anystrata as a whole — by default its Vulkan backend: any Vulkan 1.2+
  GPU (Radeon, Intel Arc, APUs, MoltenVK bridges), with CUDA/HIP/SYCL/Metal as opt-in backends
  where they win. Cyan = the Vulkan light, not a vendor color.
- **Core metaphor:** experts living in VRAM, streamed to the card. Everything below encodes it.
- **Origin (a biographical fact, not a setting):** the BC-250 board (die: *Cyan Skillfish*),
  where the port was written and first answered **17 × 23 = 391**. Reference her birthplace
  only in easter eggs — she belongs to every card that runs the engine.
- **Age canon: early-to-mid twenties — an adult engineer.** She is the person who ported the
  engine, not a prodigy kid: think grad-school age, comfortable in the lab, coffee on the desk.
- **Tone:** cozy lab-woman — competent, a little playful, shows her thinking before she speaks
  (the model is a reasoning model). Cute, PG, never sexualized.
- **Age-reading rules for every render:** adult proportions — head ~1/6.5 of standing height
  (never chibi/baby-headed), defined chin and brow (not a round toddler face), eye height ~1/14
  of face height (big doll-eyes read young), light or no blush. Close-up busts crop at the
  collarbones so the shoulders read grown-up. If a render looks like a schoolgirl, it is wrong —
  re-render with a taller face and calmer eyes.

## Fixed visual constants
| Element | Constant |
|---|---|
| Hair | long straight dark navy **#1B2340**, low side-ponytail, two cyan **#31A2F1** under-light strands |
| Hair-tie | three stacked discs = expert cache: full cyan / half-warm **#1F6F9E** / cold dark **#173A52** |
| Eyes | amber-gold **#D9A441**; a faint rotating delta-net glyph (triangle wave) in the iris when thinking |
| Outfit | oversized PCB-teal **#17494D** knit cardigan, gold-trace hem (GPU block diagram: SMs, L2, VRAM ctl), white tee |
| Skirt | pleated with three horizontal strata bands: 2-bit stipple, 5-bit stripes, 8-bit solid (Q2_0 / Q5_K / Q8_0) |
| Hairpin | small brass token-ribbon pin engraved **391** |
| Companion | **Silly** — round teal seabird, carries an 18-byte Q2_0 crate in its beak (skillfish species = the BC-250 easter egg only) |
| Palette | #31A2F1 Vulkan blue · #17494D PCB teal · #D9A441 gold · #1B2340 navy · strata grays #243248/#2E3D55/#F2F6FB |
| Lighting | dark backgrounds: cyan rim-light glows; her under-light strands read as emissive |

## Poses (use the matching one per context)
- **Base pose** — leaning in, one eye winking, holding up her phone reading
  `POST /v1/chat/completions → "391."`. Default single character art.
- **Cook pose** — sleeves up, spatula over a skillet whose pan is a circular VRAM die and the
  eggs are expert tiles ("streaming experts to cache"). Use for anything about the expert cache.
- **Serve pose** — headset + mic, hand raised gesturing at a small "listening :18090" badge.
  Use for server/status UI.
- **Speed pose** — all three hair-tie discs lit, motion streaks. Use for benchmark/perf news.
  (Her unbatched walk — discs dim — is the before-state; don't mix.)
- **Workshop pose** — sitting cross-legged with kernels on a pegboard (a .comp on one hand, a
  parity report on the other), Silly holding the crate. Use for changelog / dev-log headers.
- **Multi-GPU banner** — cache discs split into rings, one per card. Use for layer-split docs.

## Expressions (turnaround sheet)
neutral · thinking (iris glyph spins, eyes up) · happy (wink, base pose) · proud (arms crossed,
choker-free: point at hairpin) · worried/debug (sweat drop, discs half-dark, holding a broken
crate Silly fled from) · celebrating confetti of 0/1 bytes.

## Size/scale rules for reuse
- **Stamp/icon:** face-only circle, hairpin readable at ≥16 px, or Silly alone as a 16 px favicon.
- **Avatar:** bust crop to the ponytail tie (tie must stay in frame — it's her signature).
- **Banner (wide):** base pose right-third, engine name + status left; keep her off-center left
  only in mirror variants. Never stretch the skirt bands below legible dither: at small sizes
  render bands as solid tritone stripes.
- **Print/sticker:** flat version — no gradients, exact 5-color palette above.

## Master prompt (any image model or artist)
> anime character sheet, one girl: long straight dark-navy hair with two glowing soft-cyan
> under-light strands, low side ponytail tied with three stacked discs (bright cyan, half blue,
> dark blue), amber-gold eyes with a tiny triangle-wave glyph in the iris, oversized dark
> teal-green knit cardigan with thin gold circuit traces along the hem, white tee, pleated
> skirt with three horizontal bands — fine stipple, thin stripes, plain white —, a small brass
> hairpin engraved "391", a round teal seabird perched on her shoulder carrying a tiny wooden
> crate marked "18B", cozy lab-girl energy, dark navy background with cyan rim light, clean
> cel shading

Per-pose one-liners:
- base: "…, leaning toward the viewer, one eye winking, holding up a phone showing a chat reply '391.'"
- cook: "…, cooking at a round black skillet drawn like a GPU die, sunny-side eggs are tiny square tiles"
- serve: "…, wearing a support headset, gesturing at a small glowing badge ':18090'"
- speed: "…, running, all three hair discs glowing, blue motion streaks"
- workshop: "…, seated with a pegboard of recipe cards behind her, the bird stealing one"

## Do / don't
- DO keep the three-disc tie, the strata skirt bands, and one cyan under-light strand in every
  render — those three read her identity at any size.
- DO vary context (kitchen, lab, stage, server rack) — she is portable by design.
- DON'T add chokers/necklaces as identity marks (rejected early), vendor logos, or
  number-badges other than 391 (numbers change; 391 is canon).
- DON'T make the eyes cyan — gold eyes + cyan hair-light is the signature contrast.

## Why the parts mean what they mean (for anyone extending her)
Skirt bands = the quant formats the engine proves bitwise; hair-tie discs = the expert cache
(cached/warm/cold); 391 hairpin = first verified answer; Silly's 18-byte crate = Q2_0's block
size (docs/pack-format.md); bootlaces as a node chain (optional detail) = the arena's piece
chain. She documents the engine — when adding a variant, encode a real fact.

MEDIA: no artwork is shipped with the sheet (the early SVG concept sketches were removed
2026-10-29 — they predated the name settling on Any Cyan and rendered the wrong age reading).
The sheet IS the canon: render from its constants, or not at all.
