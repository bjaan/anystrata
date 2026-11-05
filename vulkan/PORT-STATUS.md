# Vulkan port status ledger

Work queue for porting the CUDA engine (`src/kernels/cuda/*.cu`, 49 files incl.
`.cuh`) to the Vulkan backend (`vulkan/`). Target: BC-250 / gfx1013 (RADV) —
the GPU ROCm does not serve. A future session resumes by picking the first
`todo` row the model run actually reaches (see *Model-demanded order* below).

Legend — **decision**: `translate` (mechanical translation expected),
`fp32-fallback` (a HIP-portable ordered-FP32 variant exists or is the plan —
docs/AMD_HIP.md: "CUDA-only QSA matrix instructions have an ordered FP32
fallback"; CMakeLists.txt: the tf32 `mma` in the QSA scorer has a portable
fp32-FMA fallback, STRATA_SELECT_SIMT / STRATA_ATTN_PRE75),
`host-free` (runs host-side, no device kernel needed),
`n/a` (NVIDIA-only formats / arch-specific paths gfx1013 cannot run —
nvfp4, int8-IMMA, gfx11/gfx12 WMMA, PDL).
**state**: `ported+parity-green` | `todo` | `compiled-unchanged` (pure-CPU
header helpers, one row per header) | `n/a` (decision `n/a`).

| entry point | header | .cu file(s) | in-tree parity test | CUDA-only constructs | decision | state |
|---|---|---|---|---|---|---|
| s2_gemv (variants s2_gemv_fast, s2_gemv_quads) | s2_gemv.hpp / s_gemv.hpp | src/kernels/cuda/s2_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | s2_gemv_parity.cpp | `__shared__` (variants), `__constant__` (fast), `__ldg` | translate | **ported+parity-green** (vulkan/src/kernels/s2_gemv.comp, s2_gemv_vk.cpp, s2_gemv_vk_parity.cpp) |
| dequant_s2 | dequant_s2.hpp | src/kernels/cuda/dequant_s2.cu | dequant_s2_parity.cpp | none | translate | **ported+parity-green** (vulkan/src/kernels/dequant_s2.comp, dequant_s2_vk.cpp, dequant_s2_vk_parity.cpp) |
| s2_gemv_q8 | s2_gemv_q8.hpp | src/kernels/cuda/s2_gemv_q8.cu | s2_gemv_q8_parity.cpp | `__shared__` | translate | todo |
| s_gemv | s_gemv.hpp | src/kernels/cuda/s_gemv.cu | s_gemv_parity.cpp | `__shfl`, `__shared__`, `__ldg`, `__constant__` | translate | todo |
| s_gemv_split / s_gemv_split_async | s_gemv.hpp | src/kernels/cuda/s_gemv.cu | s_gemv_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| s_gemv_q8k / s_gemv_q8k_split / s_gemv_q8_0_split | s_gemv.hpp | src/kernels/cuda/s_gemv.cu | s_gemv_q8k_parity.cpp | `__shfl`, `__shared__`, `__constant__` | translate | todo |
| bf16_gemv / bf16_gemv_split | bf16_gemv.hpp | src/kernels/cuda/bf16_gemv.cu | bf16_gemv_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| bf16_gemv_fp32_mmvf / _cols / _multi / _multi_aux | bf16_gemv.hpp | src/kernels/cuda/bf16_gemv.cu | bf16_gemv_parity.cpp, mmvf_rows_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| cvec_upload / cvec_apply (rest host-side) | cvec.hpp | src/kernels/cuda/cvec.cu | cvec_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| dequant_bf16 / dequant_f16 / dequant_f16_ld / dequant_f32 | dequant_bf16.hpp | src/kernels/cuda/dequant_bf16.cu | dequant_wide_parity.cpp | `__constant__` | translate | todo |
| strata_dp4a (device dot helper) | dp4a.hpp | (header; used by iq/qsa .cu) | none | PTX dp4a via asm | translate (GLSL integer-dot emulation; gfx1013 has no dp4a) | todo |
| elementwise: embedding_gather, gdn_gate, scale_inplace, add_inplace, f32_to_f16_bulk, f32_to_bf16_bulk, silu_inplace, rms_norm_weighted, scatter_rows_f32, copy_rows_from_mapped(_after), copy_i32_from_mapped, copy_from_mapped_multi | elementwise.hpp | src/kernels/cuda/elementwise.cu | elementwise_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| elementwise: doorbell_* (ring/wait/publish/payload — host-side pinned-memory plumbing) | elementwise.hpp | (host side of elementwise.cu) | elementwise_parity.cpp | none (host) | host-free (compiled-unchanged) | todo — **handshake measured (Task 1.3): host stores are NOT visible to a running kernel on RADV** (see "Device / pinned / vmm twins": doorbell FAILS live, OK across a submit boundary / staged copy); the ported waiter must ring-then-launch, not spin live |
| fused_gdn_conv_l2 / fused_gdn_ab / fused_gdn_step_norm / fused_gdn_step_norm_silu | fused_gdn.hpp | src/kernels/cuda/fused_gdn.cu | none (covered via spec_verify_parity) | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| gdn_step / gdn_conv_step / gdn_l2_norm / gdn_beta_gate / gdn_out_norm | gdn.hpp | src/kernels/cuda/gdn.cu | gdn_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| fused_gr_read / _multi / _persist (config setters host-side) | fused_gr.hpp | src/kernels/cuda/fused_gr.cu | gr_multi_parity.cpp, hc_persist_parity.cpp | `__shfl`, `__shared__`, `cp.async`, `mma`, `asm volatile`, `__ldg` | fp32-fallback (ordered-FP32 variant, docs/AMD_HIP.md QSA-fallback lineage; STRATA_ATTN_PRE75-style portable path) | todo |
| fused_gr_dequant_q8_0 / fused_gr_quantize_q8_0 / hc_q8_to_bf16 | fused_gr.hpp | src/kernels/cuda/fused_gr.cu | hc_persist_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| gr_read / gr_read_multi / gr_write / gr_write_multi (workspace_init host-side) | gr.hpp | src/kernels/cuda/gr.cu | gr_parity.cpp, gr_multi_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_gr_rms_norm_weighted / _multi | native_gr_norm.hpp | src/kernels/cuda/native_gr_norm.cu | gr_multi_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_gr_down_silu / pre_gated / _multi / post / post_multi | native_gr_postops.hpp | src/kernels/cuda/native_gr_postops.cu | gr_multi_parity.cpp | none | translate | todo |
| quantize_q8_1_rows / iq_mmvq / iq_mmvq_slots / iq_mmvq_ids | iq_kernels.hpp | src/kernels/cuda/iq_kernels.cu | iq_parity.cpp, iq_multi_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| iq_dequant_f16 / iq_dequant_f32 / iq_embed_rows / iq_dequant_gu_f16 | iq_kernels.hpp | src/kernels/cuda/iq_kernels.cu | iq_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| iq_xmx_grouped | iq_kernels.hpp | src/kernels/cuda/iq_kernels.cu | native_grouped_parity.cpp | `mma` (matrix-instruction path) | fp32-fallback (docs/AMD_HIP.md: matrix instructions have an ordered-FP32 fallback; CMakeLists.txt tf32-mma fallback at sm_75) | todo |
| native_expert_grouped (the grouped launch) | iq_kernels.hpp | src/kernels/cuda/iq_kernels.cu | native_expert_parity.cpp, native_grouped_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_expert_supported / _layout / _scratch_bytes, iq_row_bytes | iq_kernels.hpp | src/kernels/cuda/iq_kernels.cu:3570ff | (pure arithmetic; transcribed body vs the CUDA table) | none (type tables) | host (free) | host-defined (missing.cpp, Phase 7.1: the CPU expert tier answers, the kernels stay the launch traps) |
| nvfp4 expert path | iq_kernels.hpp (cpu/nvfp4_*.hpp) | (nvfp4 — NVIDIA-only format) | nvfp4_expert_gpu_parity.cpp | NVIDIA nvfp4 format | n/a (nvfp4 is an NVIDIA-only format; gfx1013 has no path) | n/a |
| native_quantize_q8_1 | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu | (unit: poly_kernels_vk_parity.cpp; in-tree family tests cover the mmvq consumers) | `__shfl` (warp chains) | translate | **ported+parity-green** (vulkan/src/kernels/quantize_q8_1.comp, native_quantize_q8_1_vk.cpp) |
| q8_1 (add_rms_norm_q8_1, swiglu_quantize_q8_1) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| q4_0 / q5_0 / q8_0 (mmvq + f32 pair) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | q8_0 mmvq **ported+parity-green** (Task 2.13, mmvq_q8_0.comp, bitwise); q4_0/q5_0 mmvq + the f32 pair todo |
| q4_k / q5_k / q6_k / q2_0 / q2_k / q3_k (mmvq + f32 pair) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| iq4_xs / iq4_nl (mmvq + f32 pair) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | iq4_xs mmvq **ported+parity-green** (Task 2.12, mmvq_iq4_xs.comp, bitwise); iq4_nl mmvq + the f32 pair todo |
| native_mmvq_slots (types 16/13/14; others: false with nothing launched, exactly CUDA slots_kind==0) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu:3178 + :1710 slots kernel, iq_kernels.cu:3491 mmvq_slots_kernel<16> | native_mmvq_vk_parity.cpp (slots: bitwise y_slots == y_single AND host refs; unported type returns false, y poison untouched) | 24-25 storage bindings (w/x/y per slot + iq2xxs grid), flattened (gx,count) grid, macro per-slot dot | translate | **ported+parity-green** (mmvq_slots_iq2_xxs/q5_k/q6_k.comp + native_mmvq_vk.cpp; pair + ids still todo) |
| native_mmvq_pair / native_mmvq_ids | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_mmvq (iq2_xxs + q5_k + q6_k + iq3_xxs + iq4_xs; other types trap-by-name `native_mmvq type <n>`) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu:3110 + iq_kernels.cu launch_mmvq<16> + native_q5_k_mmvq + native_q6_k_mmvq | native_mmvq_vk_parity.cpp (bitwise) | `__shfl` (warp_sum), `__vcmpne4/__vsubss4/dp4a` (scalar-emulated), `iq2xxs_grid` (arena-uploaded table), fp contraction (the CUDA leg builds with --use_fast_math; the vk dot mirrors the fused order) | translate | **ported+parity-green** (vulkan/src/kernels/mmvq_iq2_xxs.comp, mmvq_q5_k.comp, mmvq_q6_k.comp, mmvq_iq3_xxs.comp, mmvq_iq4_xs.comp, native_mmvq_vk.cpp; iq-family row: format 16 iq2_xxs, format 13 q5_k, format 14 q6_k, format 18 iq3_xxs, format 23 iq4_xs. Other types trap by number) |
| q8_1 interleave: native_quantize_q8_1_il / native_q8_1_interleave / native_mmvq_il / native_q8_1_il_cp | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_moe_combine / _multi / _hits / _multi_hits / _multi_hits_gated | native_moe.hpp | src/kernels/cuda/native_moe.cu | native_expert_parity.cpp, native_multi_parity.cpp | `__shared__`, `__ldg` | translate | todo |
| moe_hit_grouped_s2 / _dev / _multi / _cpu_order, moe_hit_select / _multi, moe_hit_add, moe_grouped_s2, moe_group_resident (setters host-side) | s2_expert_grouped.hpp | src/kernels/cuda/s2_expert_grouped.cu | s2_expert_grouped_parity.cpp, native_grouped_parity.cpp, decode_cluster_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| shared_expert / shared_expert_multi / moe_combine (bf16-mode setters host-side) | shared_expert.hpp | src/kernels/cuda/shared_expert.cu | shared_expert_parity.cpp, native_multi_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| fwht256_cuda / fwht256_inplace_cuda | kv_q4.hpp | src/kernels/cuda/kv_q4.cu | kv_q4_parity.cpp | `__shfl`, `__ldg` | translate | todo |
| kv_append_q4 / _step / _steps / kv_gather_q4_step | kv_q4.hpp | src/kernels/cuda/kv_q4.cu | kv_q4_parity.cpp, kv_hybrid_parity.cpp, kv_steps_parity.cpp | `__shfl`, `__ldg` | translate | todo |
| kv_append_q8_step / _steps / kv_gather_q8_step | kv_q8.hpp | src/kernels/cuda/kv_q8.cu | kv_q8_parity.cpp, kv_steps_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| kv_stream_resolve / kv_ring_restore / kv_stage_from_host / kv_unstage_to_host (counters/map_bytes host-side) | kv_stream.hpp | src/kernels/cuda/kv_stream.cu | kv_stream_parity.cpp, kv_window_parity.cpp, route_window_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| qsa: kv_append / kv_gather / _step / kv_append_step / kv_gather_step | qsa.hpp | src/kernels/cuda/qsa.cu | qsa_parity.cpp, qsa_slots_parity.cpp, kv_steps_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| qsa: indexer_key_append / qsa_index / qsa_index_step | qsa.hpp | src/kernels/cuda/qsa.cu | qsa_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| qsa: topk_512 / topk_512_step | qsa.hpp | src/kernels/cuda/qsa.cu | qsa_topk_parity.cpp, qsa_topk_active_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| qsa: qsa_attend / qsa_attend_step / qsa_gate_apply / _f32 | qsa.hpp | src/kernels/cuda/qsa.cu | qsa_parity.cpp, qsa_merge_daily_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| qsa_block_scores / _rows / _slots | qsa_select.hpp | src/kernels/cuda/qsa_select.cu | qsa_scores_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| qsa_block_scores_tc (tensor-core scorer; gfx1100 variant qsa_scores_gfx1100_test.cu) | qsa_select.hpp | src/kernels/cuda/qsa_select.cu | qsa_scores_parity.cpp, qsa_scores_gfx1100_parity.cpp | `mma`, `cp.async`, `asm volatile`, `__shfl`, `__shared__` | fp32-fallback (docs/AMD_HIP.md: the QSA scorer takes the same ordered-FP32 fallback without matrix cores; STRATA_SELECT_SIMT) | todo |
| qsa_block_topk / _cluster / _ref | qsa_select.hpp | src/kernels/cuda/qsa_select.cu | qsa_topk_parity.cpp, qsa_topk_active_parity.cpp, decode_cluster_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| qsa_decode_attn_step / _batch / _slots / _batch_rows | qsa_decode_attn.hpp | src/kernels/cuda/qsa_decode_attn.cu | qsa_parity.cpp, decode_cluster_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| qsa_decode_attn_chunks_only / _merge_only (chunked pre-sm75 path) | qsa_decode_attn.hpp | src/kernels/cuda/qsa_decode_attn.cu | qsa_merge_daily_parity.cpp | `__shfl`, `__shared__`, `__ldg` | fp32-fallback (STRATA_ATTN_PRE75 portable ordered variant — the HIP path for sub-sm80 cards, qsa_decode_attn.cu:466) | todo |
| qsa_prompt_attn_batch / qsa_decode_attn_split (set_imma host-side) | qsa_prompt_attn.hpp | src/kernels/cuda/qsa_prompt_attn.cu | qsa_prompt_attn_parity.cpp | `mma` (int8 IMMA), `cp.async`, `asm volatile`, `__shfl`, `__shared__`, `__ldg` | fp32-fallback (docs/AMD_HIP.md: prompt QSA attention runs the FP32 kernel without tensor cores; STRATA_HIP_WMMA gfx11/gfx12 paths n/a on gfx1013) | todo |
| native_qsa_rms_norm_weighted / _grouped / gate_apply | native_qsa.hpp | src/kernels/cuda/native_qsa.cu | qsa_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_qsa_score | native_qsa_score.hpp | src/kernels/cuda/native_qsa_score.cu | qsa_scores_parity.cpp | `mma`, `asm volatile`, `__shared__` | fp32-fallback (ordered-FP32 scorer, as on HIP without matrix cores) | todo |
| native_qsa_indexer_append / _batch / _steps / _steps_layers | native_qsa_indexer.hpp | src/kernels/cuda/native_qsa_indexer.cu | qsa_slots_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_flash_attn_short_step | native_flash_attn.hpp | src/kernels/cuda/native_flash_attn.cu | none (covered via batch_rows_parity) | `__shfl`, `__shared__` | translate | todo |
| rope_neox_pair / rope_neox_apply | rope.hpp | src/kernels/cuda/rope.cu | rope_parity.cpp | none | translate | todo |
| build_rope_table (both overloads) | rope_scaling.hpp | src/kernels/cuda/rope.cu:48-86 | (host float64 loops, transcribed verbatim) | none | host (free) | host-defined (missing.cpp, Phase 7.1) |
| rope_tab_cs / mrope_pos (rope/mrope table setters host-side) | mrope.hpp | (inline / rope.cu) | rope_parity.cpp | none | translate | todo |
| rope_scaling / rope_scaling_set (the process config pair; yarn math header-inline) | rope_scaling.hpp | src/kernels/cuda/rope.cu:42-46 | rope_parity.cpp | none | host (free) | host-defined (missing.cpp, Phase 7.1) |
| native_rope_apply / native_qsa_rms_norm_rope | native_rope.hpp | src/kernels/cuda/native_rope.cu | rope_parity.cpp | `__shfl`, `__shared__`, `asm volatile` | translate | todo |
| router_top10 / router_top10_variant | router_top10.hpp | src/kernels/cuda/router_top10.cu | router_top10_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_router_top10 / _multi / native_router_topk / native_route_resident | native_router.hpp | src/kernels/cuda/native_router.cu | router_top10_parity.cpp, route_resident_parity.cu | `__shfl` | translate | todo |
| route_prior_top10 | route_prior.hpp | src/kernels/cuda/route_prior.cu | route_window_parity.cpp | `__shfl` | translate | todo |
| quantize_q8_0 / _scaled / dequant_q8_0 / quantize_q8_K / dequant_q8_K | quantize_act.hpp | src/kernels/cuda/quantize_act.cu | quantize_act_parity.cpp | `__shfl` | translate | todo |
| ple_block / ple_block_projected / ple_history_advance (native-mode setters host-side) | ple.hpp | src/kernels/cuda/ple.cu | ple_parity.cpp, ple_fp8_parity.cpp, ple_q5_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_ple_postops / _batch / _batch_snap | native_ple_postops.hpp | src/kernels/cuda/native_ple_postops.cu | ple_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| ngram: PleTable open/gather/prefetch + format table (host-side file IO, no device kernel) | ngram.hpp | (host; no cuda .cu) | ple_parity.cpp (indirect) | none | host-free (compiled-unchanged) | n/a |
| q8_1_finite / q8_1_quant / q8_1_ds (inline device helpers) | q8_1_finite.hpp | (header-inline; s2*/iq .cu) | (exercised via gemv parity) | none | translate (compiled with consumers) | todo |
| native_bf16 variants (bf16 kernels used by the native paths) | — (no header contract) | src/kernels/cuda/native_bf16.cu | native_multi_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| sample_tokens / sample_tokens_spec / sample_greedy_cluster | sampler.hpp | src/kernels/cuda/sampler.cu | sampler_parity.cpp | `__shfl`, `__shared__`, `asm volatile` | translate | todo |
| coupled_draft_stage / coupled_draft_sample / spec_draft_sample / penalty_rows | sampler.hpp | src/kernels/cuda/sampler.cu | spec_verify_parity.cpp | `__shfl`, `__shared__`, `asm volatile` | translate | todo |
| gdn_conv_l2_multi / gdn_conv_commit / gdn_conv_ab_multi / gdn_ab_multi / gdn_step_norm_groups / gdn_step_norm_multi / gdn_commit_layers | verify_kernels.hpp | src/kernels/cuda/verify_kernels.cu | spec_verify_parity.cpp, verify_batch_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| resident_plan / ident_hits / mtp_select | verify_kernels.hpp | src/kernels/cuda/verify_kernels.cu | resident_plan_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| argmax_rows (+ scratch) / row_top_prob / _split / logits_nonfinite_rows / dense_steps / window_ids | verify_kernels.hpp | src/kernels/cuda/verify_kernels.cu | verify_parity.cpp, batch_rows_parity.cpp | `__shfl`, `__shared__`, `asm volatile` | translate | todo |
| wait_flag_ge* / handoff_publish / wait_handoff / gpu_stamp / gpu_noop (device-side sync flags) | verify_kernels.hpp | src/kernels/cuda/verify_kernels.cu | verify_batch_parity.cpp | `asm volatile` (atomics/spin) | translate (GLSL atomics; ordered spin) | todo |
| copy_rows_strided / copy_indexed / fetch_blobs / res_patch / rebase_ptrs / broadcast_streams / add_streams_broadcast / copy_i32_from_mapped_unless / copy_or_zero_from_mapped / copy_row_to_first / gather_rows / map_ids / embedding_gather_dev | verify_kernels.hpp | src/kernels/cuda/verify_kernels.cu | verify_parity.cpp, verify_batch_parity.cpp | `__shfl`, `__shared__`, `__ldg` | translate | todo |
| launch_pdl / pdl_* (programmatic dependent launch) | pdl.hpp | (host launch glue; pdl helpers) | pdl_parity.cpp | cudaGraph, PDL (programmatic dependent launch) | n/a (NVIDIA PDL/graph feature — run with --no-graphs; the Vulkan port uses its own submission ordering) | n/a |
| poly: gemv_kernel, dequant_q8_kernel | src/poly/kernels.cu (no include/strata header) | src/poly/kernels.cu | none (poly engine tests) | `__shared__`, `__constant__` | translate | **gemv_kernel ported via the gemv_f32 launcher (Task 2.4): vulkan/src/poly/kernels/gemv.comp (5 workgroup-size variants), gemv_vk.cpp, unit poly_kernels_vk_parity BITWISE on 5 shapes; dequant_q8_kernel todo** |
| poly: rms_norm, copy2 (launchers) | include/strata/poly/kernels.hpp | src/poly/kernels.cu | unit poly_kernels_vk_parity + model-run advance | `__shfl`, `__shared__` (rms_norm block_sum) | translate | **ported+parity-green** (vulkan/src/poly/kernels/{rms_norm,copy2}.comp, *_vk.cpp) |
| poly: layer_norm_kernel, bf16_to_f32_kernel, store_f16_kernel/**PORTED 2.10**/_at, store_q8_kernel/_at, add_bias_kernel, axpy_kernel, add_scaled_kernel, mul_sigmoid_kernel, softcap_kernel, likeliest_kernel | src/poly/kernels.cu | src/poly/kernels.cu | unit poly_kernels_vk_parity | none | translate | **axpy/add_scaled/mul_sigmoid PORTED (Task 2.7): vulkan/src/poly/kernels/elementwise.comp (EW_VARIANT arms) + elementwise_vk.cpp; parity: axpy/add_scaled bitwise, mul_sigmoid rel 5.0e-7 (exp vs expf, tree tol 1e-5); likeliest PORTED (Task 2.19): likeliest.comp + likeliest_vk.cpp, parity token-bitwise + prob rel 6.96e-8; store_q8/_at, add_bias, softcap, store_f16_at, route_rows/permute_groups/moe_sum (moe router families) still todo** |
| poly: hc_mix/read/post/head/mean kernels | src/poly/kernels.cu | src/poly/kernels.cu | hc_persist_parity.cpp (host-side contract) | `__shfl`, `__shared__` | translate | todo |
| poly: kda_conv/_rows/_state, kda_step/gates/state/out kernels | src/poly/kernels.cu | src/poly/kernels.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | todo |
| poly: gdn_norm/gates/state/state_reg/out kernels | src/poly/kernels.cu | src/poly/kernels.cu | gdn_parity.cpp (host-side contract) | `__shfl`, `__shared__` | translate | **gdn_step PORTED (Task 2.5): gdn_norm.comp/gdn_gates.comp/gdn_state.comp (plain+reg<1,2,4,8>)/gdn_out.comp + gdn_vk.cpp; unit poly_gdn_vk_parity rel_l1 <= 1.6e-7 vs both contraction arms (tree tol 1e-5)** |
| poly: conv4_silu/_rows/_state, shortconv/_rows/_state kernels | src/poly/kernels.cu | src/poly/kernels.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | **conv4_silu (all three kernels: walk/rows/state) ported (Task 2.4): vulkan/src/poly/kernels/conv4_silu.comp, conv4_state.comp, conv4_silu_vk.cpp, unit poly_kernels_vk_parity rel 5.9e-7 (tree tol 1e-5); shortconv todo** |
| poly: swiglu_clamp/_rows, geglu/_rows kernels | src/poly/kernels.cu | src/poly/kernels.cu | unit poly_kernels_vk_parity | none | translate | **swiglu_clamp PORTED (Task 2.7, elementwise.comp): parity rel 7.0e-7 (exp vs expf); swiglu_clamp_rows PORTED (Task 2.8, elementwise.comp EW_VARIANT 3: rows of gate|up pairs, flattened grid + 65535-row chunking like mul_sigmoid; parity worst rel reported in-test); geglu todo** |
| poly: dsa_keys/softmax, idx_store/pool/weigh, route_kernel, moe_sum_kernel | src/poly/kernels.cu | src/poly/kernels.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | todo |
| poly: attn_score/softmax/value/group/ungroup/softmax_block/scale, head_norm_rope kernels | src/poly/kernels.cu | src/poly/kernels.cu | unit poly_kernels_vk_parity (head_norm_rope), poly_attend_vk_parity (attend) | `__shfl`, `__shared__`, `cp.async` (some paths), `asm volatile` | fp32-fallback for the cp.async/mma-attn paths; ordered-FP32 score/softmax otherwise | **head_norm_rope PORTED (Task 2.10): vulkan/src/poly/kernels/head_norm_rope.comp (-DHNR_THREADS=128/256) + head_norm_rope_vk.cpp; parity rel 7e-7..1.2e-6 all arms (host pow table + Cody-Waite + in-kernel Taylor sin/cos — the CUDA libdevice drifts vs RADV v_sin/v_cos, see the halt para); attend PORTED (Task 2.11): the general chain — attend_score.comp (-DATT_N/-DATT_STEPS, the four reg<> variants), attend_softmax.comp, attend_value.comp (-DATT_VARIANT t1/t2/plain) + attend_vk.cpp; parity rel_l1 4e-8..1.4e-7 vs the exact-order AND naive references (exp-vs-expf family bound); split/join (dim 64 per 4) and kq8/vq8 arms trap-by-name** |
| poly dsv4: rope/state/squeeze/idx_score/select kernels | src/poly/kernels_dsv4.cu | src/poly/kernels_dsv4.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | todo |
| poly dsv4: attend/attend_rows/permute kernels | src/poly/kernels_dsv4.cu | src/poly/kernels_dsv4.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | todo |
| bf16_bits.hpp (bf16_from_f32 / f32_from_bf16 inline) | bf16_bits.hpp | (header-inline) | none | none | host-free | compiled-unchanged (n/a for device) |
| f16_bits.hpp (f16_from_f32 / f32_from_f16 inline) | f16_bits.hpp | (header-inline) | none | none | host-free | compiled-unchanged (n/a for device) |
| gfx_arch.hpp (arch predicates) | gfx_arch.hpp | (header-inline) | none | none | host-free | compiled-unchanged (host-side only) |


## How to resume

1. **Find the first non-green entry point the model run reaches.** Build and
   run the Vulkan engine with graphs off (`build-vk/bin/strata-poly-vk` IS
   built since Task 1.4; the engine aborts at the first missing launcher and
   names the kernel it demands — first demand on the qwen35moe path is
   `strata::poly::copy2`, see *Model-demanded order*):
   `build-vk/bin/strata-poly-vk <model> ... --no-graphs`
2. **Per-kernel recipe** (proven on s2_gemv and dequant_s2):
   1. translate the CUDA body to GLSL compute: `vulkan/src/kernels/<name>.comp`
      — subgroup shuffles become `subgroup*` builtins, `__shared__` becomes
      `shared`, `cp.async`/tensor cores disappear (ordered-FP32 fallback path);
   2. write the launcher `vulkan/src/kernels/<name>_vk.cpp` matching the
      header's contract signature (`namespace strata::kernels`);
   3. port the in-tree parity test
      (`src/kernels/<name>_parity.cpp`) to
      `vulkan/src/tests/<name>_vk_parity.cpp` — keep the same host-side
      reference, run it against the .comp;
   4. `ctest -R <name>` in `build-vk/` until green;
   5. flip the row's state to `ported+parity-green` in this file and commit.
3. **Tolerance law:** each parity test keeps the in-tree bound exactly —
   dequant paths **zero tolerance** (bit-exact), gemv paths **rel 1e-5**,
   fp16 paths **1e-3**. Never loosen a bound to make a port pass; if it
   cannot pass at the bound, the translation is wrong.

## Trap roster (Task 1.2)

The 358 host-callable `strata::kernels` entry points on `todo` rows above are trapped
(`vulkan/src/engine/stubs_gen.cpp`, generated by `vulkan/tools/gen_stubs.py` from the headers' exact
signatures; `vulkan/src/engine/missing.cpp` prints `strata-vk: <name> has no Vulkan implementation yet`
and `_Exit(86)`; `STRATA_VK_STUBS=log` prints one `STUB` line per entry and returns). The two
`ported+parity-green` rows (`s2_gemv`, `dequant_s2`), the host-side `cvec()`/PDL predicates, and the
`native_mmvq` capability/byte-count queries the load path asks before any launch (Task 1.4, see
"strata-poly-vk link shape") carry no trap. The roster regenerates mechanically after each port; the
ctest cases `vk_stubs_trap` / `vk_stubs_trap_log` prove both arms on this box. Link rule: kernel libs
come BEFORE `strata_vk_stubs` on the link line (`vulkan/CMakeLists.txt`). The poly contract's 48 traps
(gen_poly_stubs.py) are the same pattern, one lib earlier on the link line.

## Device / pinned / vmm twins (Task 1.3) — measured on this box (2026-10-22)

**doorbell: FAILS on RADV (host stores NOT visible to a RUNNING kernel).**
`vk_doorbell_probe` (ctest, probe case E: `src/probe/doorbell_probe.cpp` + `probe_doorbell.comp`):
a 40M-`atomicAdd`-read spin (~0.74 s) on a HOST_VISIBLE|HOST_COHERENT buffer, flag flipped from the
host 200 ms into the spin — `observed=0, reads=40,000,000, timeout=1`: the store never reaches the
running kernel (reproduced 4/4, and identically with an uncached/WC host allocation + explicit
`vkFlushMappedMemoryRanges`: arm 3, also NEVER SEEN). Same verdict as the sycl port's Arc A750
(i915). **What DOES work:** the staged handshake — arm 2, a flag copied into device memory by
`vkCmdCopyBuffer` before the spin launches: `observed=7, reads=0`, sentinel set; and arm 4, a host
store that predates the LAUNCH is visible to the next kernel on the same coherent buffer
(`VISIBLE at next launch`). Consequence for the engine's `doorbell_ring`/`doorbell_wait` ports:
ring = host store + submit boundary; a waiter kernel must be launched AFTER the ring (or the flag
flushed by a copy node before it), never spun live inside an already-running graph. The pinned twins
keep the compat `cudaHostAlloc` buffers as the host side of the doorbell — the store travels through
the queue-submit ordering, which is proven by arm 4.

| contract | CUDA twin | Vulkan twin | state |
|---|---|---|---|
| `include/strata/core/device.hpp` (device_count/device_info/device_summary/gpu_arch_problem/primary/CudaError/DeviceArena) | src/core/device.cu | `vulkan/src/core/vk_device_impl.cpp` | implemented, green in `vk_core_smoke` (cc 10.1, arch gfx1013, 24 SMs via VK_AMD_shader_core_properties [2 engines x 2 arrays x 6 CUs, queried WITHOUT enabling the extension; a BIOS-unlocked 40-CU count is NOT visible to it — STRATA_VK_CU_OVERRIDE=N rewrites for occupancy experiments], 15715 MiB total / 14042 MiB free, DeviceArena bump + compat pointer contract round-trip) |
| `include/strata/core/pinned.hpp` + `pinned_ring.hpp` | src/core/pinned.cu | `vulkan/src/core/vk_pinned.cpp` (pinned.cu's body against the compat layer — the file has no device code; PinnedRing is header-inline over compat) | implemented, green in `vk_core_smoke` (PinnedArena 8 MiB: `cudaHostRegister PORTABLE ok` on NormalPages — hugetlb absent (vm.nr_hugepages=0), THP reported as found; ring h2d bit-exact). PCIe probe: `stream_bandwidth` routes to a NAMED SKIP by default (uma=1, no bus tier; `seconds=-1` "not measured", never a fake number; `STRATA_VK_SKIP_PCIE_PROBE=0` measures anyway) |
| `include/strata/core/vmm.hpp` | src/core/vmm.cpp (driver entry points resolve null through compat — the engine's documented off-switch) | `vulkan/src/core/vk_vmm.cpp` | **grow-never arena fallback** (`host-free fallback`): reserve() maps the whole span up front, mapped(i) true from the start, unmap/freed nothing (bump arena). The elastic K/V reports full at all times: `--kv-grow` costs its max context at startup instead of at token 4000; take()-donated spare chunks waste their bytes (bounded, recorded here). Green in `vk_core_smoke` |
| `multiProcessorCount` / `l2CacheSize` for cudaDeviceProp | (CUDA query) | vk_device.cpp: CU count via VK_AMD_shader_core_properties (=24 measured); **L2: no Vulkan query exists on radv — stays 0, documented**, readers only tune prefetch/bench cycling with it (`s2_expert_grouped_parity.cpp:532`) | gap closed for CUs, honestly open for L2 |

`src/core/device.cu`, `src/core/pinned.cu` and `src/core/vmm.cpp` are NOT compiled by this backend; the
twins live in `vulkan/src/core/` (static lib `strata_vk_core`). `vk_core_smoke` is the ctest proof
(28 checks, all green on this box).


## cublas CPU fallback (Task 1.1)

| entry point | header | .cu file(s) | in-tree parity test | CUDA-only constructs | decision | state |
|---|---|---|---|---|---|---|
| dense GEMM family (cublasSgemm/GemmEx/GemmStridedBatchedEx/SgemmStridedBatched/Hgemm) | cublas_v2.h (vulkan_compat) | src/prefill/gemm.cu, src/poly/kernels.cu, src/qwen35/q36_engine.cu | gemm_f32_parity.cpp, gemm_bf16_parity.cpp | cublas handle + device operands | host-free **speed-todo**: vk_blas.cpp runs a plain blocked CPU GEMM (correct, fp32 accumulation, staged arena round-trips per operand); Phase 3 (Task 3.1) replaces it with the dequant+GEMM `.comp` family | compiled-unchanged (CPU fallback green in vk_compat_smoke: Sgemm + GemmEx fp16-in/fp32-out bit-match the host reference) |

## Model-demanded order

Task 1.4 fills this table with the trap/kernel names from the first
`strata-poly-vk` model run (`--no-graphs`), in encounter order. Until then the
row order above (gemv/dequant first — already green) is the fallback.

Measured 2026-10-22 on this box: `build-vk/bin/strata-poly-vk --model
Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --no-graphs --max-context 4096 --tokens 1 --max-new 1`
device line -> full mmap load ("loaded in 82.1 s; 10496 of 10496 routed experts in VRAM (9.4 GiB)")
-> `strata-vk: strata::poly::copy2 has no Vulkan implementation yet`, exit 86.
Re-run with `STRATA_VK_STUBS=log` enumerates the rest of the qwen35moe decode path's demand:

| # | kernel demanded by the run | row above |
|---|---|---|
| 1 | `strata::poly::copy2` | poly: copy2_kernel — **PORTED (Task 2.1)** |
| 2 | `strata::poly::rms_norm` | poly: rms_norm_kernel — **PORTED (Task 2.1)** |
| 3 | `strata::kernels::native_quantize_q8_1` | q8_1 (native_quantize_q8_1, …) — **PORTED (Task 2.1)** |
| 4 | `strata::kernels::native_mmvq` | native_mmvq / _pair / _slots / _ids — **iq2_xxs (type 16) PORTED (Task 2.2)**; **iq3_xxs (type 18) PORTED (Task 2.9)**; other types trap-by-name |
| 5 | `strata::poly::gemv_f32` | poly: gemv_kernel — **PORTED (Task 2.4)** |
| 6 | `strata::poly::conv4_silu` | poly: conv4_silu/_rows/_state kernels — **PORTED (Task 2.4)** |
| 7 | `strata::poly::gdn_step` | poly: gdn_norm/gates/state/state_reg/out kernels — **PORTED (Task 2.5)** |
| 7b | `strata::kernels::native_mmvq` type 14 (Q6_K) | native_mmvq family — **PORTED (Task 2.6)** |
| 8 | `strata::poly::axpy` | poly: axpy_kernel — **PORTED (Task 2.7)** |
| 9 | `strata::poly::swiglu_clamp` | poly: swiglu_clamp/_rows — **swiglu_clamp PORTED (Task 2.7)**; _rows todo |
| 10 | `strata::poly::mul_sigmoid` | poly: mul_sigmoid_kernel — **PORTED (Task 2.7)** |
| 11 | `strata::kernels::native_mmvq_slots` | native_mmvq / _pair / _slots / _ids — **PORTED (Task 2.8, types 16/13/14)** |
| 12 | `strata::poly::swiglu_clamp_rows` | poly: swiglu_clamp/_rows — **PORTED (Task 2.8)** |
| 13 | `strata::poly::add_scaled` | poly: add_scaled_kernel — **PORTED (Task 2.7)** |
| 14 | `strata::poly::head_norm_rope` | poly: head_norm_rope kernels — **PORTED (Task 2.10)** |
| 15 | `strata::poly::store_f16` | poly: store_f16_kernel/_at — **store_f16 PORTED (Task 2.10)**; _at todo |
| 16 | `strata::poly::attend` | poly: attn_score/softmax/value/… kernels — **PORTED (Task 2.11)** |
| 17 | `strata::kernels::native_mmvq` type 23 (IQ4_XS) | native_mmvq family — **PORTED (Task 2.12)** |
| 18 | `strata::kernels::native_mmvq` type 8 (Q8_0) | native_mmvq family — **PORTED (Task 2.13)** |
| 19 | `strata::kernels::native_mmvq` type 22 (IQ2_S) | native_mmvq family — **PORTED (Task 2.14)** |
| 20 | `strata::kernels::native_mmvq` type 12 (Q4_K) | native_mmvq family — **PORTED (Task 2.15)** |
| 21 | `strata::poly::gemv_bf16` | poly: gemv_kernel<uint16_t> — **PORTED (Task 2.16)** |
| 22 | `strata::kernels::native_mmvq` type 10 (Q2_K) | native_mmvq family — **PORTED (Task 2.17)** |
| 23 | `strata::kernels::native_mmvq` type 11 (Q3_K) | native_mmvq family — **PORTED (Task 2.18)** |

(The log arm returns untouched outputs, so entries after #1 are the path's *shape*, not a
guaranteed per-token sequence; the first trap, `copy2`, is exact — the default arm stops there.)

**Halt point after Task 2.1 (2026-10-22):** `strata::kernels::native_mmvq` (demand #4) — the trio
above cleared, 9/9 ctest, model load unaffected. NOTE the load time swung 82 s -> 332 s on this run
(same command, same file; swap/THP noise on a 14 GiB box under desktop load) — the harness timing
story, not a port regression: gate the port by the HALT POINT moving, not by load seconds.

**Halt point after Task 2.2 (2026-10-22):** `strata::kernels::native_mmvq type 13` — demand #4's
IQ2_XXS leg is PORTED (mmvq_iq2_xxs.comp + native_mmvq_vk.cpp, bitwise parity
native_mmvq_vk_parity, 387/387 exact, 10/10 ctest): the 10496 routed experts (type 16 gate/up and
down) now COMPUTE on the GPU and the decode run advances to the Q5_K (type 13) matrices — a
different format's mmvq entry, trap-by-name until its decoder is ported. Demand #4 row below reads
"iq2_xxs ported; other types trap-by-name".

**Halt point after Task 2.3 (2026-10-22):** `strata::poly::gemv_f32` — demand #4's Q5_K leg (type 13,
the dense projections) is PORTED: STRATA_VK_TRACE_MMVQ13 shows exactly two calls per token (n_in 2048
n_out 8192 / n_out 4096, ncols 1 -> SmallK=false ROWS=1). mmvq_q5_k.comp transcribes
native_q5_k_mmvq_kernel's partition (both ROWS variants; ncols via the pinned multi_exact per-column
equality) and q5_q8_dot/q5_q8_dot_impl word-exactly. native_mmvq_vk_parity is now 908/908 outputs
BITWISE (5 q5_k shapes incl. both partitions + the iq2_xxs set), ctest 10/10. Q6_K (14) is not
demanded by this model's decode and stays trap-by-number. Next stop: the poly-contract gemv_f32.
CONTRACTION LAW (this box): the CUDA native_mmvq.cu builds with --use_fast_math and no
-ffp-contract=off, so its exact dot semantics are fused; the device answers `sumf = fma(d8, dot*sc,
sumf)` per i and the final pair as `fma(-dm_y, sumf_m, dm_x*sumf_d)` — the host reference mirrors
exactly that (all other fusion orders differ on 108-321 outputs).

**Halt point after Task 2.4 (2026-10-22):** `strata::poly::gdn_step` — demands #5 and #6 cleared in
one gate run: gemv_f32 (gemv.comp, 5 workgroup-size variants 32..1024 matching the CUDA wrapper's
`n_in >= 128 ? sum_threads(n_in) : 32` choice, grid dim3(n_out,rows) flattened) is **BITWISE**
against the transcribed strided+shuffle-tree reference on all 5 tested shapes (worst rel 0.000e+00
— and here, unlike the mmvq dots, the device does NOT contract `acc += w*x`: the plain fp32
reference matches to the bit); conv4_silu — all three kernels (walk with snapshots, rows, state;
conv4_silu.comp + conv4_state.comp + conv4_silu_vk.cpp with CUDA's CONV_ROWS=16 arm switch
transcribed) — the window chain `s0*w0+s1*w1+s2*w2+v*w3` measures UNFUSED on this box (worst rel
5.9e-7 vs the unfused host reference; an explicit OpExtInst Fma in the .spv lowers to identical
values — RADV/GFX1013 turns FMA back into mul+add here), 2.9e-4 vs the fma arm, inside the tree's
1e-5 gemv-family tol against the closer arm (never looser). ctest 10/10. Model load 126 s, halt
moved #5 -> gdn_step (demand #7): the gate passed gemv_f32 and conv4_silu — gdn_step is what
qwen35moe's recurrent layers ask for next.

**Halt point after Task 2.5 (2026-10-23):** `strata::kernels::native_mmvq type 14` (Q6_K) — the
pre-digested guess (#8 axpy) did NOT hold: gdn_step cleared, and the model's next demand is the
mmvq family's third type (the table's #4 row: iq2_xxs=16 and q5_k=13 ported; 14 now demanded
before any poly elementwise). gdn_step ported as the five-kernel chain of src/poly/kernels.cu
(:1988 norm, :2002 gates, :2017 state / :2058 state_reg<PER>, :2118 out) — gdn_norm.comp +
gdn_out.comp compile per dim (32..256 in 32s: block_sum's part[] warp count IS the fp contract),
gdn_state.comp as plain + reg<1,2,4,8> variants (the CUDA wrapper's `switch (dim / WARP)`,
threads = columns*32 = 1024, one warp a column of the dim x dim state, the warp walking the
tokens; snapshots per snap_of(), stride = rec_n_ = n_head*dim*dim floats), gdn_gates.comp
elementwise. Shuffle trees force_subgroup32=false (offsets <= 16 stay warp-aligned, the rms_norm
reasoning). Unit poly_gdn_vk_parity (new ctest): model shape dim=128 nh=32 kh=16 (the GGUF's
ssm.state_size/group_count/time_step_rank) rows=1/5/8 + snapshots 0x41, plus dim=32/64 (reg arms)
and 96 (plain arm) — the device matches the closer host arm at tree rel_l1 <= 1.6e-7 (worst
ABSOLUTE deviation 7e-7 = ~4 ULP; exp/softplus/sqrt transcendental deltas dominate, contraction is
secondary here), the tree's OWN metric (gdn_parity.cpp:88 rel_l1 = sum|d|/sum|ref|, tol 1e-5 at
:202-203 — per-element rel with the 1e-3 floor is NOT the GDN row's bound: the RMS norm's
1/sqrt(sum) divides small cells, so cancellation cells read 1e-4 per-element while the tensor's
rel_l1 sits at 1e-7; never looser, it is the tree's number). NOT bitwise: the reg kernel's
explicit __fma spellings vs RADV's lowering land between the arms (Task 2.4's measurement
pattern; both arms tested, closer one governs). ctest 11/11. Model load 87 s; the gate cleared
#7 and named `native_mmvq type 14` — demand 7b, port iq2_s before the poly elementwise trio.

**Halt point after Task 2.6 (2026-10-23):** `strata::poly::axpy` — demand #7b cleared: Q6_K (type
14) ported as mmvq_q6_k.comp (native_q6_k_mmvq_kernel :806-842 + q6_q8_dot/_impl :773-803; VDR=1:
kbx = tid/32 + 4t, kqs = tid%32, SmallK threshold nb < 4, Q6KBlock 210 B read byte-wise — the
iq2_xxs precedent, not q5_k's direct word indexing). STRATA_VK_TRACE_MMVQ14: the model asks one
Q6_K mmvq per token, n_in 4096 n_out 2048 ncols 1 -> nb 16 >= 4, ROWS=1 arm (grid n_out).
native_mmvq_vk_parity is now 1443/1443 outputs **BITWISE** (6 q6_k shapes incl. both ROWS arms
+ q5_k + iq2_xxs sets). CONTRACTION LAW (this box, q6 leg): the intra-dot `sumf += d8*(dot*sc)`
fuses to fma exactly as q5_k's, AND the kbx accumulation `tmp += d*sumf` fuses to
fma(d, sumf, tmp) — visible only when a thread takes >1 kbx iteration (the rows=1 arm): the
un-fused host arm differs by 1-2 ULP on 38/64 of the first shape, the fused arm matches to the
bit; rows=4 (nb<4, one iteration) matched either way. ctest 11/11. Model load 94 s, gate reached
decode and stopped at poly demand #8 (axpy) — the pre-digested demand order resumes: axpy,
swiglu_clamp, mul_sigmoid (actual halt decides).

**Halt point after Task 2.7 (2026-10-23):** `strata::kernels::native_mmvq_slots` — the poly
elementwise batch (demands #8 axpy, #9 swiglu_clamp, #10 mul_sigmoid, + #13 add_scaled) ported as
vulkan/src/poly/kernels/elementwise.comp (one .comp, -DEW_VARIANT 0/1/2: axpy reused as
add_scaled's per-vector pass — the CUDA kernel's k-loop == count sequential axpy launches, no
cross-pass ordering; mul_sigmoid flattens dim3(grid,rows) and replays the wrapper's 65535-row
chunking) + elementwise_vk.cpp. PURE ELEMENTWISE, no contraction bites: axpy and add_scaled are
BITWISE against the device's chosen arm (un-fused arm matches at 0.0 rel on this box; scale arm
trivially); swiglu_clamp/mul_sigmoid are NOT bitwise-only because CUDA expf vs GLSL exp is not a
documented twin — MEASURED worst rel: swiglu_clamp 7.0e-7, mul_sigmoid 5.0e-7 / 3.6e-7 (one_gate
arms), inside the tree's silu/gemv-family 1e-5 bound. ctest 11/11. Model gate: load 32.9 s, decode
ran past demands 8-10 and 13, stopped at native_mmvq_slots (#11) — the expert-path's slot gather:
port native_mmvq_slots before the next poly demand (swiglu_clamp_rows, #12).

**Halt point after Task 2.8 (2026-10-23):** `strata::kernels::native_mmvq type 18` (IQ3_XXS — the
single-matrix entry trap-by-number, NOT a demand-list name: the model's mixed-format expert stack
demands an unported iq family format directly). native_mmvq_slots (#11) ported for 16/13/14 as
mmvq_slots_iq2_xxs/q5_k/q6_k.comp (one W/X/Y triple per slot = 24 storage buffers, +25th iq2xxs
grid for type 16; the CUDA dim3(gx,count) grid FLATTENED — slot = gid/gx — since the runtime
dispatches groups_x only; per-slot dot via token-paste macro instances, one source of truth;
unused slots alias slot 0's buffers at 4-byte spans, arms never run past count) + the launcher
entry in native_mmvq_vk.cpp; unported formats return FALSE with nothing launched, exactly CUDA's
slots_kind==0 wrapper leg (native_mmvq.cu:3179) — mirrored and poison-verified in the test. iq2's
slots kernel transcribes iq_kernels.cu's mmvq_slots_kernel<16> (one warp per row, no c-loop);
q5_k/q6_k transcribe native_mmvq.cu:1710 slots_kernel with the ROWS=1/2 SmallK pick — and per-row
it is the SAME kbx chain + subrow-ascending + xor tree as the single kernel, so slots == single is
BITWISE across ROWS arms (512-wide ROWS=2 cases included): 9073 outputs compared bitwise, 0 differ
(count 2 and 3, shared x column across slots). swiglu_clamp_rows (#12) as elementwise.comp
EW_VARIANT 3 (flattened grid + 65535-row chunking, mul_sigmoid's pattern): worst rel 7.0e-7
(exp vs expf, inside the 1e-5 family bound). native_mmvq_ids (:3194, the sibling contract) NOT
ported — still trap-by-number; record: the next halt may demand it. ctest 11/11. Model gate:
load 32.6 s, decode ran past #11 and #12, stopped at native_mmvq type 18 = IQ3_XXS (ggml.h:408).

**Halt point after Task 2.10 (2026-10-23):** `strata::poly::attend` (demand #16 — the run walks
straight past #14/#15: head_norm_rope + store_f16 both ported this task, one demand each, and the
900 s gate ends at `strata-vk: strata::poly::attend has no Vulkan implementation yet`).
head_norm_rope as vulkan/src/poly/kernels/head_norm_rope.comp (HNR_THREADS=128/256 variants =
the CUDA launcher's `dim >= 256 ? 256 : 128`, kernels.cu:2445; one workgroup per token-and-head,
the block_sum xor chain + part-walk line-for-line, NeoX pairing, mrope axes / freq / on-card at
arms all wired — the engine passes axes+sections+at together (qwen35.cpp:351/386) and the kernel
takes them as independent bindings) + head_norm_rope_vk.cpp (7 storage buffers; the theta_scale^j
mantissa table is host-pow'd ONCE per (theta,half) and cached — RADV's hardware pow drifts ~4e-6
rel where CUDA's powf is ~1 ulp, and `pow` in .comp has no honest path; the rope angle then goes
through a three-word Cody-Waite 2*pi reduction + pi/2 quadrant fold + Taylor sin/cos in-kernel,
because glslc's sin/cos land on v_sin_f32/v_cos_f32 whose own reduction drifts ~2e-6 abs and
WRAPS at large angles (probe: 0.43 rel at 1e6 rad) — and the freq divide goes through double per
PORT-NOTES law 2 (the f32 `/` and one Newton step both left 1.2e-5; the double path lands at
1.1e-6). Parity (poly_kernels_vk_parity, +7 checks): all six head_norm_rope arms rel <= 1.3e-6
vs a host transcription (double ss, launcher-matched pow table, f32-rounded libm trig) — the
tree tol is 1e-5; the pure-norm tail and on-card `at` arms included. store_f16 as store_f16.comp
+ store_f16_vk.cpp: f16 pairs built with f16_rn_bits (NOT packHalf2x16, law 1), u32 words —
**BITWISE** against the host f32_to_f16 over 1600 values incl. 0/-0/±1 and prior-row-untouched.
store_f16_at (kernels.cu:522, the at_-reading replay arm) stays a trap — the recording_ path is
--no-graphs territory; if attend's launcher wants it, port it then. ctest 11/11. Model gate: load
32.1 s, 10496/10496 experts in VRAM, decode ran past #14 and #15 and stopped at attend (#16),
the last big demand of the decode path.

**Halt point after Task 2.9 (2026-10-23):** `strata::poly::head_norm_rope` (demand #14 — the
pre-digested order holds; store_f16 #15 and any type-8 Q8_0 / slots-18 demand did NOT show up).
IQ3_XXS (type 18) ported as mmvq_iq3_xxs.comp — a straight swap of the proven iq2_xxs template:
Fmt<18> (iq_kernels.cu:613) is qk 256/ipb 8/step 2, identical launch geometry (ROWS=4, grid
ceil(n_out/4)); vec_dot_iq3_xxs_q8_1 (:191-214) differs from iq2_xxs' only in (a) block_iq3_xxs =
98 B (half d + 64 grid-index bytes + 32 aux bytes: aux32 at byte 66+2*iqs, read word-wise through
the wb/w32 byte assembler — 98 is not 4-aligned), (b) two uint32 grid words per l0 pair
(make_int2 of CONSECUTIVE iq3xxs_grid entries, not one uint64's halves — the table is 256 x uint32,
extracted verbatim to iq3xxs_grid_vk.h and uploaded once to the arena like iq2xxs'), (c) the scale
step `(ls*sumi + sumi/2)/2` with ls = aux32>>28 (trunc_div2 by hand, both divisions truncate toward
zero). native_mmvq_slots(18) still returns false (no slots kernel for 18 — the engine falls back to
per-matrix native_mmvq, which is what the model exercised: single-launch arms ran, halt advanced).
native_mmvq_vk_parity: iq3_xxs 5 shapes (2048x64 ncols 1/2, 2048x768, 768x64 ncols 3, 256x3 —
expert dims included) = 1155 outputs **BITWISE**, 0 differ; suite now 10228 compared, 0 differ.
ctest 11/11. Model gate: load 30.7 s, 10496/10496 experts in VRAM, decode ran past type-18 mmvq
and stopped at head_norm_rope — demand #14 next.

Phase-2 first picks by demand order: `copy2`, `rms_norm`, then the mmvq family
(`native_quantize_q8_1`, `native_mmvq`) — matching the plan's expectation (quantize_act /
expert GEMV) with `copy2` + `rms_norm` arriving one step earlier.

## strata-poly-vk link shape (Task 1.4)

`strata_vk_poly_kernels` (`vulkan/src/poly/stubs_gen.cpp` — 48 traps for the whole
`include/strata/poly/kernels.hpp` contract, generated by `vulkan/tools/gen_poly_stubs.py` — plus
`vulkan/src/poly/mmq_vk.cpp`): **mmq prompt kernels: trapped (Phase 3 covers prefill)**; `Mmq::fits()`
answers false honestly so the engine's batched prompt path stays off and the run reaches the decode
traps. Host-only contract parts (`sum_threads`, `attend_split_floats`, `blas_create`/`blas_destroy`) are
defined without traps. `src/poly/kernels.cu` and `kernels_dsv4.cu` are NOT compiled (host+`__global__`
in one TU, no nvcc); `src/kernels/cuda/*.cu` likewise (the strata_vk_stubs roster + the two ported
launchers replace `strata_kernels`); `device.cu`/`pinned.cu` by the Task-1.3 twins; `src/core/vmm.cpp`
not needed by this target. ggml-base/ggml-cpu come from the pinned llama.cpp submodule (the CPU
experts), `add_subdirectory` in `vulkan/CMakeLists.txt`. Compat gaps closed on the way are listed in
PORT-NOTES.md "Task 1.4 compat gaps".

**Halt point after Task 2.11 (2026-10-23):** `strata::kernels::native_mmvq` **type 23 (IQ4_XS)** —
demand #16 cleared and the run walks ATTEND for the whole first decode step: the halt moves to the
mmvq family's fourth type. attend as the general chain, exactly the CUDA wrapper's dispatch
(kernels.cu:2449-2549): attend_vk.cpp replays the chunk loop (first/keys/wt predicate/value-arm
pick mirrored), attend_score.comp compiles the four `attn_score_reg_kernel<N, STEPS, false>`
variants (-DATT_N/-DATT_STEPS: 4/1, 8/2, 4/2, 4/4) — THE MODEL RUNS reg<8,2>: the trace shows
`pos=0 rows=1 chunk=256 dim=256 n_head=16 kv_heads=2 scale=0.0625 window=0 ring=0 whole=0 at=0`
(key_length 256, NOT the pre-dugested 128), the scores_sum<N> xor tree unrolled lane-exactly
(off 16/8 keep-and-hand, off 4/2/1 finish; which = 2·bit16 + bit8 (+ bit4 at N=8));
attend_softmax.comp = attn_softmax_kernel (strided mx/exp/sum passes, 8-warp part[] walk, the wt
transpose arm); attend_value.comp = the three value arms (-DATT_VARIANT: attn_value_t<1>, <2>, the
plain kernel — its __shfl weight broadcast becomes per-lane loads, the same float in the same
order: on wave64 an absolute subgroupShuffle is not SPIR-V-uniform). fp16 cache decode is a
BIT-EXACT half→float bit build (subnormals normalized, the CUDA u64 read as two words + the +8
tail slack). Parity (poly_attend_vk_parity, NEW ctest): 6 cases (dim128 plain/wt/chunked + the
MODEL dim256 rows=1/chunked arms) — rel_l1 vs the exact-ORDER reference 4.3e-8..1.4e-7, vs the
naive DIRECT reference 9.3e-8..3.1e-7 (tree tol 1e-5; exp-vs-expf is the only freedom, never
bitwise claimed; worst abs dev 6e-8 on outputs of max 0.28). NOT ported, trap-by-name inside
attend(): attn_split/join (dim 64 per 4 — the CUDA 2460 gate is evaluated so the message names the
shape), kq8/vq8 quantized-KV, plain attn_score_kernel (non-reg dims/per). ALSO fixed the arena
layout class attend exposed: the "512 MiB RADV max allocation" was FICTIONAL (measured: 8 GiB
buffers allocate fine) — max_allocation_size is now 6 GiB by default, arena_add piece-aligns every
region that fits one piece (single-piece by construction; this killed the score-scratch straddle),
probe D pins 512 MiB via setenv to keep the chained-region arms deterministic, and ctest
vk_piece_align guards the new law (5 checks: near-boundary realign, consecutive small allocs,
chained full-span rejection + piece-local bind). ctest 13/13. Model gate: load 33.9 s,
10496/10496 experts in VRAM, decode ran through all ported poly demands incl. attend, stopped at
`strata-vk: strata::kernels::native_mmvq type 23 has no Vulkan implementation yet` (demand #17 —
IQ4_XS; port it next, the iq3_xxs template is the proven shape to extend).

**Halt point after Task 2.12 (2026-10-23):** `strata::kernels::native_mmvq` **type 8 (Q8_0)** —
demand #17 cleared: IQ4_XS (type 23) is PORTED bitwise. mmvq_iq4_xs.comp clones the iq3_xxs
template with Fmt<23> = {qk 256, ipb 8, step 4} (iqs = 4*(k%8), one 32-value sub-block per dot
call): 146-byte unaligned blocks assembled byte-wise (wb/w32), the two-level 6-bit scale decode
(scales_l low 4 + scales_h high 2), and the nibble->kvalues_iq4nl path — the CUDA
get_int_from_table_16 byte_perm tree emulated and VERIFIED against a host prmt model (even
nibbles -> v.x, odd -> v.y). No grid table (the table rides inlined in the .comp). Parity: 5
iq4_xs shapes (2048x64 c1/c2, 2048x768 c1, 768x64 c3, 256x3 c1) BITWISE — full suite
11383/11383 outputs, 0 differ; ctest 13/13. Model gate: load 36.5 s, 10496/10496 experts in VRAM,
the decode walked past all IQ-type mmvq legs and stopped at
`strata-vk: strata::kernels::native_mmvq type 8 has no Vulkan implementation yet` (demand #18 —
Q8_0 mmvq, the down-expert format of this UD-IQ2_XXS quant; vec_dot_q8_0_q8_1 in
native_mmvq.cu is the shape to port — integer dot, 32-elem blocks, no SIMD emu needed).

**Halt point after Task 2.13 (2026-10-23):** `strata::kernels::native_mmvq` **type 22 (IQ2_S)** —
demand #18 cleared: Q8_0 (type 8) is PORTED bitwise. mmvq_q8_0.comp transcribes
small_mmvq<Q80Block, 8> (native_mmvq.cu:2118-2145): 32-elem blocks, small-K pick nb=n_in/32 < 32
(ROWS=4/grid ceil(n_out/4) vs ROWS=1/grid n_out, the rows push constant), Qi=8 partition
(kbx = tid/4 + 32t, kqs = 2*(tid%4), KBY=1), the two-dp4a small_q8_dot (:888-900) with the
unaligned 34 B block assembled byte-wise (wb/w32, the iq4_xs pattern). Parity: 5 q8_0 shapes
(2048x64 c1, 2048x768 c2, 2048x2048 c3, 768x64 c2, 256x7 c1) BITWISE — full suite now
19262/19262 outputs, 0 differ; ctest 13/13. CONTRACTION PROBE: unlike the q5_k/q6_k dots, this
dot's `acc += dm*sumi` stays UNFUSED on gfx1013 (the fma(dm,sumi,acc) host arm differed on
36/64, 941/1536, 3649/6144 rows=1 outputs; the unfused arm matches all 7879 q8_0 outputs —
dm and (float)sumi are both exact, so only the adds' rounding order separates the arms). Model
gate: load 31.3 s, 10496/10496 experts in VRAM, the decode walked past every type-8 leg (trace
mmvq8 hits incl. n_in 512 -> rows=4 arm) and stopped at
`strata-vk: strata::kernels::native_mmvq type 22 has no Vulkan implementation yet` (demand #19 —
IQ2_S mmvq, iq_kernels.cu's mmvq_kernel<22> path; the iq3_xxs/iq4_xs templates are the proven
shapes to extend).

**Halt point after Task 2.14 (2026-10-23):** `strata::kernels::native_mmvq` **type 12 (Q4_K)** —
demand #19 cleared: IQ2_S (type 22) is PORTED bitwise. mmvq_iq2_s.comp clones the iq2_xxs template
with Fmt<22> = {qk 256, ipb 8, step 2}: 82-byte unaligned blocks (half d @0, qs[64] @2 — grid bytes
at 2+4*(iqs/2)+au, 2-bit sign bytes at 34+2*iqs+au = get_int_b2(qs, 8+iqs/2), qh[8] @66, scales[8]
@74 nibble pair ls0/ls1), the 10-bit grid composite qs | ((qh << (8-l0)) & 0x300) into
iq2s_grid (1024 x uint64, ggml-common.h:758 — extracted value-by-value diff-clean to
iq2s_grid_vk.h, uploaded once as uvec2 pairs like iq2xxs), the split sumi0/sumi1 accumulators and
(sumi0*ls0 + sumi1*ls1 + trunc_div2(sumi0+sumi1))/4. Parity: 6 iq2_s shapes (2048x64 c1, 2048x768
c1, 2048x64 c2, 768x64 c3, 256x64 c1, 256x3 c1) BITWISE — full suite now 20481/20481 outputs,
0 differ; ctest 13/13. Contraction: the row accumulation stays UNFUSED (the dw*d8*sumi -> s +=
term arm matches; same evidence as iq2_xxs/iq3_xxs — the fused arm was never needed). Model gate:
load 35.0 s, 10496/10496 experts in VRAM, the decode walked 8 type-22 legs (trace mmvq22) and
stopped at `strata-vk: strata::kernels::native_mmvq type 12 has no Vulkan implementation yet`
(demand #20 — Q4_K mmvq, native_q4_k_mmvq in native_mmvq.cu: the Fmt<12> geometry is
{qk 256, ipb QI4_K/VDR, step VDR} — super-block scale decode, no grid table).

**Halt point after Task 2.15 (2026-10-23): FIRST TOKEN — no halt.** Demand #20 cleared: Q4_K
(type 12) is PORTED bitwise. mmvq_q4_k.comp clones the q5_k template — Q4KBlock is the 144 B =
36-word sibling (static_asserts native_mmvq.cu:139-140: dm half2 @0, scales[12] @4, qs[128] @16,
4-aligned, direct word indexing), q4_q8_dot/q4_q8_dot_impl (675-727) verbatim: the SAME 6-bit
scale + 4-bit min aux decode (0x3f3f/0xc0c0>>2, s4 >>0 / >>4 & 0x0f0f, hi-masking, sc/m = aux
byte split) and the same DP4A pair structure, minus the qh plane (v masked straight through
0x0f0f0f0f). Wrapper pick (:2933) n_in/256 < WARPS*WARP/16 = 8 -> ROWS=4 grid ceil(n_out/4),
else ROWS=1 grid n_out — identical to q5_k (VDR=2, QI=16, BPI=8, kqs = 2*(tid%16)); rows is the
4th push constant. Contraction mirrors the q5_k lineage (fused fma sumf arms + final
fma(-dm_y, sumf_m, dm_x*sumf_d)) — matched BITWISE first try, no probe needed. Parity: 7 q4_k
shapes (2048x64 c1, 2048x768 c1/c3, 768x64 c2, 768x3 c1, 256x3 c2, 256x768 c1) — full suite now
24522/24522 outputs, 0 differ; ctest 13/13. slots stays types 16/13/14 (the unported-type-12
slots arm in the parity test unchanged). Model gate (fresh binary): load 31.5 s, 10496/10496
experts in VRAM, prompt 2.4 tok/s, **generated 1 token (first token emitted — the whole mmvq
family the model routes is now ported)**; the run exits clean with no `has no Vulkan
implementation yet` line. Candidates for the NEXT halt (deeper/multi-token runs): mmvq types 11
(Q3_K)/10 (Q2_K)/7 (IQ1_S) if a tensor uses them, native_mmvq_slots beyond 16/13/14, or the
sampler batch route (row_norm/softcap/moe_sum/permute_groups).

**Halt point after Task 2.16 (2026-10-23):** `strata::kernels::native_mmvq` **type 10 (Q2_K)** —
demand #21 cleared: the 18-token prompt run's `gemv_bf16 has no Vulkan implementation yet` is gone
and the prefill advances to the mmvq family's Q2_K leg (the candidate list's type 10, now
measured). gemv_bf16 as gemv_bf16.comp — the f32 gemv.comp's line-for-line twin with the CUDA
ldw(uint16_t) (kernels.cu:72, `__uint_as_float(w[i] << 16)`) swap: the w buffer binds as uint[]
(two bf16 LE per word; even j takes word<<16, odd j word & 0xFFFF0000 — exactly the shift either
way), and bf16->f32 is an EXACT widening, so the partition/warp-sum contract is the f32 one's,
unchanged. Same five -DGEMV_THREADS variants (32..1024: the thread pick lives in the shared CUDA
gemv<> wrapper); gemv_bf16_vk.cpp mirrors gemv_vk.cpp (w span = n_out*n_in*2). Parity
(poly_kernels_vk_parity, +6 cases incl. odd-n_in word-straddle shapes 100 and 257): weights encoded
with the tree's bf16_from_f32 (bf16_bits.hpp:37 RNE, transcribed host-side) and the reference
decodes with the exact <<16 shift — **BITWISE 0.000e+00 on all six shapes** (the device does not
contract `acc += w*x` here, same as f32 gemv). gen_poly_stubs.py PORTED += gemv_bf16, regen: 34
traps. ctest 13/13. Model gate: load 33.8 s, 10496/10496 experts, halt moved from poly gemv_bf16
to `native_mmvq type 10` (Q2_K; native_mmvq.cu's Q2_K mmvq is the next shape to port — the
K-quants' super-block scale decode, the q4_k template's closest sibling).

**Halt point after Task 2.17 (2026-10-23):** `strata::kernels::native_mmvq` **type 11 (Q3_K)** —
demand #22 cleared: Q2_K mmvq as mmvq_q2_k.comp, the q4_k clone with the Q3_K partition (VDR=1:
kqs = tid%16, BLOCKS_PER_ITER = WARPS*WARP/16 = 8, SmallK n_in/256 < 8 -> ROWS=4) and the 84 B
Q2KBlock = 21 words (scales[16]@0, qs[64]@16, dm half2@80 — 4-aligned at 84 B stride, so qs[iqs]
and dm are direct word reads; only the scales bytes go through wb). The dot is q2k_q8_dot
(:501-523) line-for-line: per-i `vi = (v >> 2i) & 0x03030303` DP4A vs the q8 word at (iqs%8),
sc = scales[2i] split 0xF/>>4 replicated, and the fused return fma(-dm.y, sumf_m, dm.x*sumf_d)
(the q4_k contraction arm went bitwise first try — both arms verified in one run). Single-matrix
dispatch only: the CUDA wrapper launches per-column (:2862 j loop), the comp's c loop re-bases
x/y exactly like that; slots stays false (mirrors slots_kind == 0). Parity (+6 shapes
2048x64/768/768x3 + 768x64x2/768x3/256x3x2, rows 1 and 4): **BITWISE 0 differ, 3273 outputs
added -> 27795 total compared, 0 differ**. ctest 13/13. Model gate: load 32.8 s, 10496/10496
experts, 8 mmvq10 launches traced, and the halt moves from type 10 to `native_mmvq type 11`
(Q3_K — the q5_k/q4_k decode tree with the 5-bit qh plane's low bit folded, the next sibling).

**Halt point after Task 2.18 (2026-10-23):** `strata::poly::likeliest` — demand #23 cleared:
Q3_K mmvq as mmvq_q3_k.comp, the q2_k clone with the Q3_K partition it documents (VDR=1:
kqs = tid%16, kbx = tid/16 + 8t, SmallK n_in/256 < 8 -> ROWS=4) and the 110 B Q3KBlock
(hmask[32]@0, qs[64]@32, scales[12]@96, half d@108 — alignof 2, 110 % 4 = 2, so like Q6_K every
W byte goes through wb/w16/rd32; load_int_b2 = plain 4-byte LE). The dot is q3_q8_dot +
q3_q8_dot_impl (:422-460) line-for-line: vl = qs pair, vh = ~hmask pair >> bq8_offset (the
>> i << 2 fold never crosses bit 31), vsubss4 exact without clamping (bytes in [-4,3]), the
6-bit scale decode indexing scales[isc%8] nibbles OR scales[8+isc%4] 2-bit high field minus 32.
Contraction law 6 (the q6_k lineage): fused fma per i, and tmp += d*sumf -> fma(d, sumf, acc)
(all shapes 1 kbx iteration; first draft failed 100% only because the host ref read scales at
byte isc instead of 96+isc — kernel was right first try). Single-matrix dispatch; slots stays
false (CUDA slots_kind(11)==1 but no slots comp ported, the q2_k precedent). Parity (+6 shapes
2048x64/768/768x3 + 768x64x2/768x3/256x3x2, both rows arms): **BITWISE 0 differ, 3273 outputs
added -> 31068 total compared, 0 differ**. ctest 13/13. Model gate: load 34.2 s, 10496/10496
experts, and the halt moves from `native_mmvq type 11` to `strata::poly::likeliest` — the mmvq
family is fully walked through the first decode step; the next halt is a poly-path kernel
(likeliest_kernel, src/poly/kernels.cu, still todo in the poly table).

**COMPLETION after Task 2.19 (2026-10-23): THE 16-TOKEN GATE COMPLETES — no halt.** Demand #24
cleared: likeliest as likeliest.comp (one workgroup of 1024, the CUDA <<<1,1024>>> mirrored;
the strided loops re-guard their bound INSIDE the body -- glslc -O hoists a bare `for (i=tid;
i<n; i+=1024)` into a do-while whose first read past the buffer folds garbage into m/sum on
RADV) + likeliest_vk.cpp (writes an 8-byte arena scratch, copies back with copy_d2h: the CUDA
kernel writes the pinned HOST Likeliest through unified addressing; Vulkan kernels see only the
arena). fmaxf's NaN-IGNORING semantics emulated (GLSL max propagates). Parity (4 shapes incl.
the model's vocab 151936, ties forced at the max): **token INDEX equal every shape (ties
included), prob rel <= 7.0e-8 (exp vs expf)**. The trap chain after that halt found two real
bugs BEHIND it: (1) mmvq_iq4_xs.comp/launcher declared block_iq4_xs as 146 bytes -- the struct
is half+u16+4+128 = 136 (the old parity test transcribed the same wrong stride, so it agreed
with itself: fixed to 136 everywhere, + the model's 512x2048 shapes; IQ4_XS rows read true
bytes now, blk 34/38's NaN gone). (2) gdn_vk.cpp + conv4_silu_vk.cpp over-declared the snap
buffer's span as popcount(snaps)*stride: the last slot writes only its own 3ch / head*dim*dim
payload, so (popcount-1)*stride + payload is the touched extent; the phantom tail crossed an
arena piece once snapshot speculation actually ran (it never did while blk 34 NaN'd -- the
first draft's "all-zero ids" was likeliest reading NaN logits, whose argmax falls to token 0).
Gate (exact command, qids): 18-token prefill 5.1-5.7 tok/s, generated 16 tokens 4.37-5.36
tok/s (length), 10 of 10 guessed right, ids
`220,18,24,16,13,198,48,25,3437,369,220,16,22,2942,220,17` = " 391.\nQ: 17 x 23" --
**'391' present (17x23, the semantic check passes)**; --top-probs 3 confident (0.90/0.83/0.98).
poly traps left: add_bias, softcap, store_q8/_at, store_f16_at, moe_sum, route_rows,
permute_groups, attend_batch, dsa/geglu/hc/kda/dsv4/idx families, gemm_f32 -- NONE demanded by
the gate. ctest 13/13; native_mmvq_vk_parity total now 49500 outputs compared, 0 differ.

## Phase 4.1: deferred-submit batching (2026-10-23)
kernel_launch_sized now records into a PENDING primary command buffer (ONE_TIME_SUBMIT) with one
shared descriptor POOL per pending CB (cap 512 sets; STRATA_VK_MAX_SETS env); flush() = end +
vkQueueSubmit with a VkFence (NO wait); flush_wait() reaps. Flush(+wait) points: stream_sync /
device_sync / event_record, and the TOP of every copy_h2d / copy_d2h / fill_d (the d2h after the
likeliest scratch is the sampler's natural sync point). Pending CBs bounded two ways: 512 sets
(=dispatches) per CB, and kMaxInFlight=8 unreaped CBs (pend_flush waits+reaps the oldest fence).
STRATA_VK_BATCH=0 restores the per-launch path (A/B toggle, compiled in).

THE STAGING HAZARD (why copies stay EAGER): copy_chunk records the stage buffer's ADDRESS, not
its bytes. With deferred submits, a host memcpy into g_stage for the NEXT copy would corrupt a
not-yet-run copy of the PREVIOUS one. Fix chosen: every copy/fill flush_waits pending GPU work
first (and g_stage itself is only ever touched under g_m + after a queue drain). The engine only
copies at token boundaries + staging windows, so the eager copies cost little vs ~600
launches/token. A ring of N stage buffers is the later optimization if copy count grows.

THREE REAL RACES BATCHING EXPOSED (all fixed here; each silent under eager's implicit drain):
1. Arena::alloc captured g_cursor BEFORE arena_add()'s piece-boundary bump -> the caller got a
   dead-tail offset while the region sat at the bumped base (straddle throws + wrong tensors once
   swap threads' allocs raced ahead of recorded launches). Fixed: arena_add returns placed_at.
2. kernel_launch_sized read g_pieces unlocked while background threads push_back (vector realloc).
   Fixed: piece resolution under g_m (leaf order g_km -> g_m -> g_pm; sync paths take g_pm only).
3. THE BIG ONE: consecutive dispatches INSIDE one command buffer carry NO declared dependency --
   RADV overlaps them and storage-buffer hazards scramble the tensor stream (garbage ids, then
   straddle throws from corrupted slot pointers; STRATA_VK_MAX_SETS=1 was bitwise-correct while
   >=2 was not: the bisect that named it). Eager mode was safe because each dispatch was its own
   submit+wait (an implicit full-device barrier). Fixed: vkCmdPipelineBarrier (COMPUTE+TRANSFER,
   full RW memory barrier) after each batched dispatch -- still ~3x faster than submit+wait.
4. vkQueueSubmit/WaitIdle + pool allocate/free from copy threads vs launch-thread flushes: all
   queue/pool ops now under g_pm (begin/end_submit/pend_* take it; one queue underneath anyway).

MEASURED (Qwen3.6-35B-A3B UD-IQ2_XXS, BC-250, 2 runs each, medians; ids IDENTICAL in both modes):
  gate                     BATCH=0            BATCH=1
  16-tok gate prompt       5.0 tok/s          14.6 tok/s
  16-tok gate decode       4.3 tok/s          12.7 tok/s
  128-tok run prompt       5.7 tok/s          14.7 tok/s
  128-tok run decode       7.7 tok/s          24.3 tok/s   <- headline: ~3.2x
Gates: ctest 13/13; 16-tok ids == 220,18,24,16,13,198,48,25,3437,369,220,16,22,2942,220,17 both
modes (3x); 128-tok ids identical BATCH=0 vs BATCH=1 (diff clean); git diff main..HEAD -- src
include == 0.

## Hybrid GSQ bring-up (2026-10-23/24)

Vehicle: models/Qwen3.6-35B-A3B-GSQ-hybrid-GGUF/Qwen3.6-35B-A3B-GSQ-hybrid.gguf (chtisgit GSQ,
arch qwen35moe, 12.2 GB, sha256 8e50912f...b03596 verified). 120 Q2_0 routed-expert tensors,
251 Q8_0, 61 BF16, 301 F32. Same vocab as UD-IQ2_XXS: the /tmp/qids.txt 18-token gate applies.

LOAD FACTS: loads in 29-36 s; 10240 of 10240 routed experts fully in VRAM (8.4 GiB expert
cache; no CPU experts, 0 swaps), BC-250 15.7 GiB. First halt: native_mmvq type 42 (Q2_0 - the
CUDA switch's case 42 -> native_q2_0_mmvq).

Demand table for THIS model (separate list, in demand order):
1. mmvq ggml type 42 (Q2_0), the routed-expert gate/up/down projections.
   PORTED (mmvq_q2_0.comp + native_mmvq_vk.cpp dispatch arm + CMake embed rule): QK=64, QI=2,
   VDR=1, T=2, BLOCKS_PER_ITER=64; SmallK pick n_in/64 < 64 (model n_in 2048 -> nb 32 -> ROWS=4
   arm); 18 B blocks through the wb/w16 assemblers (2-byte-aligned int16 qs load, CUDA :352);
   __byte_perm emulated lane-by-lane (probe: prmt unsigned mode, sn&4 picks src, bit 3 IGNORED,
   so the int16 sign extension never reaches a used selector bit - qe/qo hand-expanded to
   LUT[(q16>>(4n+{0,2}))&3] in both device and host refs, cross-checked against the raw prmt
   chain on 8000 random words). Parity: 5127 outputs across 5 shapes x both arms, 0 differ
   BITWISE - and BOTH contraction arms (fma vs unfused add-mul) match, the int products keep dm*
   sumi exact so NIR contraction cannot show a difference here.
2. Q8_0 attention/LM-head weights: native_mmvq type 8 already ported (Task 2.13) - no demand hit.
2b. ARENA-PCE straddle (runtime law, see below) - fixed in the runtime, not a kernel port.
3. poly::bf16_to_f32 (port 2): demanded only by the 485-token max-context-16384 run (prefill
   batch >= kStageRows routes BF16 dense matrices through lin()'s gemm path). Exact widening
   (out = bitcast(u32(w)<<16)), one kernel, no rounding class: bitwise by construction.
4. poly::gemm_f32 (port 3, HOST launcher - no .comp): the CUDA body is pure cuBLAS; the compat
   surface's ordered-fp32 CPU GEMM (vk_blas.cpp) is real, so the wrapper transcribes whole.
   Honest deviation recorded in the file header: the small-out CUDA perf arm (groups==1 &&
   n_out <= 256, gemm_small_kernel) routes through the same cublasSgemm - that arm's
   accumulation order is not a parity contract anywhere in this port (decode stays on the
   bitwise gemv/mmvq paths; prefill mmq fits() still answers false).

ARENA LAW UPDATE (vk_runtime.cpp arena_add + vk_device.cpp default): the hybrid's 8.4 GiB
single expert-cache alloc CHAINED at the 6 GiB piece default and one expert row straddled the
6 GiB boundary -> named "kernel argument 0 of native_mmvq_q2_0 crosses an arena piece" reject
(offset 9266885600, 589824 B; pieces [..,9267246848] [9267246848,..]). Fix: max_allocation_size
default becomes 16 GiB (the budget itself - arena_add's free_bytes check is the real limit) and
arena_add halve-retries any piece the driver refuses, so chaining is failure-driven, never
mid-region. Piece-chain class stays tested: vk_piece_align now pins STRATA_VK_MAX_ALLOC=512 MiB
itself (the probe's route) instead of inheriting the default.

HALT after ports 1-3 (long-context run, 485 tokens from strata_tokenizer.py, --max-context
16384 --max-new 16): "strata-vk: strata::poly::attend_batch has no Vulkan implementation yet"
(the multi-stage cuBLAS + softmax + q8-cache chain, kernels.cu:2568ff - a Phase-3-class port,
not attempted under this session's 4-port budget).

ATTEND_BATCH PORTED - LONG CONTEXT UNBLOCKED: strata::poly::attend_batch (kernels.cu:2568)
runs on the compat CPU cublas + three new .comp arms (attend_batch_vk.cpp launcher; group /
ungroup one file with -DUNGROUP, softmax-block, scale). The wrapper's chunk loop, the two
cublasGemmStridedBatchedEx calls and the running-max softmax carry over ARG FOR ARG / line for
line; the weights round to f16 before the value product exactly as CUDA's tensor-core design
does (words built with f16_rn_bits + atomicOr into zeroed scratch - the launcher zeroes the
block's weight span per key-block, since a shorter tail block reuses the region). The kq8/vq8
arm traps by name (the q8 workspace's dequant chain); the CUDA dim-64 flash arm needs
tensor_cores(), false on AMD builds, so every shape takes this chain like CUDA's own AMD path.
Parity: poly_attend_vk_parity gains four attend_batch cases (32 rows over 64 keys at dim 128
and 256; 100 rows multi-block chunked; 40 rows n_head=8 with an odd tail m=13): order-reference
rel_l1 2.4-2.7e-8, worst-abs <=1.8e-7, and the cross-check batch-vs-the-ported-single-attend on
the same data rel_l1 1.1-1.5e-7, all under attend's 1e-5 law. Gates: ctest 13/13; the short
hybrid gate UNCHANGED (16 ids '220,18,24,16,13,271,48,25,3437,369,220,16,22,2942,220,17',
18.78 tok/s); the 735-token long gate that halted now completes: "prompt 735 tokens 5.7 tok/s,
generated 16 tokens 10.41 tok/s", ids 271,248068,271,248069,271,1206,10724,393,16,22,1088,14695,
220,17,18,60439 (no further trap reached on this path: store_q8/_at and softcap stay trapped
for the chains that demand them, untested by these gates). Next halt: none observed - pi's
>500-token system prompt now runs.

Phase 3 (attempt 2) - GPU tiled GEMM under the cublas shim (vulkan/src/runtime/mm_gemm.comp +
vk_blas.cpp dispatch): the dense prefill wall (attend_batch's two cublasGemmStridedBatchedEx over
the f16 KV caches + gemm_f32's cublasSgemm/StridedBatched) lands on the card with NO call-site
change. One .comp, four compiled variants (TRANS_A x F16AB glslc defines): 256 threads as 16x16,
64x64 C tile, K-step 16 through two shared f32 tiles (8 KiB LDS), 4x4 microtile per thread,
strided batch folded into the linear group id (kernel_launch_sized dispatches (groups_x,1,1)
only; tilesX/tilesY/batch ride the 64B push-constant block with lda/ldb/ldc/m/n/k/alpha/beta and
int32 element-unit strides - every cublasGemmStridedBatchedEx stride the tree passes fits, the
biggest seen is far under 2^31). f16 operands decode bit-exactly through unpackHalf2x16 of the
isolated 16-bit word (no rounding path, laws intact); C is f32 in every GPU variant (the Hgemm
f16-out arms stay on the CPU reference - nothing on the model path uses them). Dispatcher rules:
NN/TN with f32-f32 or f16-f16 A/B and f32 C go GPU; OP_T-B and exotic strides/alignments fall back
to the ordered-fp32 CPU reference with a COUNTED one-line stderr notice (0 notices across both
model gates). GPU is default; STRATA_VK_BLAS_CPU=1 is the A/B arm; the parity test toggles
in-process through strata_vk_blas_use_gpu(). The launch rides kernel_launch_sized, so it is
queue-ordered under Phase 4.1's per-dispatch barriers exactly like every other poly kernel.
Micro-gate vk_blas_gpu_parity (new ctest): 200x128x300 all three compiled variants + NT-fallback
+ batch 3 + alpha/beta in {0,1} + the attend-shaped f16 TN strided case - GPU vs CPU arm, worst
rel err 0.000e+00 (bitwise identical on every case, incl. the ragged k=300 tail); bench 2048^3:
GPU 862 GFLOPs vs CPU 0.25 GFLOPs (~3400x; a first-pass tile, no double-buffering yet - the
headroom vs the ~1 TFLOP fp32 ceiling is K-panel double-buffer / wider microtiles, deferred).
ctest 14/14. Model gates: canary UNCHANGED - ids '220,18,24,16,13,271,48,25,3437,369,220,16,22,2942,220,17'
(18-token prompt 13.8 tok/s, decode 18.28 tok/s). Long gate (735 tokens): prefill 5.8 -> 32.1
tok/s GPU arm (5.5x; the 50 target needs the KV-block GEMMs at larger m alongside the dense ones -
attend_batch's products dominate now, at 128x16x128 tiles on a 64x64 kernel there is tile waste to
tune next). decode 10.0-11.1 tok/s both arms. pi acceptance GREEN: 'pi works', roundtrip 52.8 s,
server prompt_ms=50304 for the 1596-token system prompt (31.7 tok/s prefill, vs 425.9 s the CPU
shim took on the same request). git diff main..HEAD -- src include == 0 (all edits under vulkan/).

COMPLETION: 16-token gate GREEN on the hybrid (default arena, no env overrides): prompt 18 tok at 12.3-13.3 tok/s, decode 16
tokens at 19.03 / 18.95 / 18.65 / 18.59 tok/s (4 runs incl. the post-gemm_f32 rebuild, ids IDENTICAL all four):
220,18,24,16,13,271,48,25,3437,369,220,16,22,2942,220,17
"experts from VRAM 100.0%, 0 experts swapped, 0 left out, 5 of 5 guessed tokens right".
Gates: ctest 13/13 (parity suite incl. the new q2_0 section); git diff main..HEAD -- src include
== 0 (all edits under vulkan/).

PREFILL TUNING PASS (Phase 3 follow-on). MEASURED FIRST (new env arms: STRATA_VK_GEMM_WALL=1
wall-times every GPU GEMM launch through device_sync, STRATA_VK_MMVQ_SHAPES=1 aggregates
native_mmvq GPU time per (type,n_in,n_out), STRATA_VK_LAUNCH_TIME=1 counts kernel_launch_sized
calls + host ms + flush_wait stalls + per-kernel launch names; all off by default, all inside
vulkan/): the 735-tok prefill (23 s baseline) is NOT GEMM-bound. Time table: GPU GEMMs (f32 dense
proj + f16 attend) 315 ms total WALL; attend_batch whole-call 159 ms (10 calls); mmvq GPU wall
(after drain-in-flush attribution) ~6.4 s at 508k calls; cublas enqueue 12 ms. The wall was the
DISPATCH TAX: 789k launches, ~12 us per-dispatch full-RW barrier each (barriers elided=0 before
this pass), and 508k of those launches were native_mmvq ncols=1 calls — the engine's expert loop
asks native_mmvq_slots first, which returned false for Q2_0 (type 42), the hybrid's 120 routed
experts. Narrow-tile GEMM variants (the brief's suspect 1) are OFF the table: the attend GEMMs
are 0.16 s of a 23 s prefill; n=16 heads never showed up (cols=c*per>=16x8; the real shapes are
m=734,n=1776 / m=512,n=4096 tiles — already wide). Suspect 3 (redundant grouping) checked: the
launcher groups once per chunk (outside the key-block loop), CUDA-faithful — nothing to hoist.
TWO FIXES SHIPPED:
(a) vulkan/runtime: barrier elision by span-cluster. kernel_launch_sized_rw(k,...,rw_mask,...)
takes the kernel's exact WRITE mask (bit i = buffer i written); the runtime keeps the pending
CB's barrier-free cluster (span off/bytes/wr) and skips the post-dispatch vkCmdPipelineBarrier
when the new launch shares NO overlapping span with any cluster member that either side writes
(RAW/WAR/WAW test; read-read overlap — the same weight tensor across launches — is benign).
Any touching launch first CLOSES the cluster with the full barrier (everything before it stays
globally ordered), so queue order and bitwise output are preserved by construction; the CB-set
cap and STRATA_VK_NO_ELIDE=1 (A/B) reset/keep the old law. mmvq launchers pass {w,r; x,r; y,w}
masks (expert fan-out: disjoint y + disjoint x = overlap freely on the SAME weight);
vk_blas gemm_gpu passes C-only. A first version of this test got WAR backwards (cluster READ vs
new WRITE) and flipped the long-gate generation — caught by the canary, rule now: pair hazard =
overlap AND (either side writes).
(b) vulkan/kernels: mmvq_slots_q2_0.comp — native_mmvq_slots for type 42, the q5_k slots
precedent verbatim: slot = group/gx over 24 bindings, the single kernel's exact q2_0_dot (wb/w16
byte assemblers per slot binding), BLOCKS_PER_ITER=64 partition, ROWS=1/4 SmallK pick, the
sh staging + xor-tree reduction per slot. Parity: 3 new cases (3x2048x1024, 8x2048x2048,
2x512x7) vs the single-launch kernel BITWISE (3072+16384+14 outputs, 0 differ); the engine's
per-column fallback and this path answer identically, so the fusion is numerics-free.
RESULT TABLE (same box, same binary family, 735-tok prompt / decode tok/s):
  baseline (2a33f51)        prefill 32.1 (measured 31.7), 789k launches, 0 elided
  +q2_0 slots (elide off)   35.2, 378k launches (508k mmvq -> 59k slots + 23k q8_0)
  +barrier elide            36.0-36.3, 82k barriers elided/run
  485-tok ctx-16384 gate    33.3 tok/s (was ~32); pi 1596-tok system prompt 31.7 -> re-measure:
  short canary prompt 12.2-13.7 -> 13.7 tok/s, decode 18.3-19.0 -> 18.5-19.0 (unchanged class)
Targets (60/70 tok/s) NOT reached: the remaining wall is now flush_wait stall 11.9 s / 31.6k calls
(the engine's own d2h/copy sync points drain the queue: eager-copy law + sampler round-trips),
plus ~12 us host cost per launch x 378k. Both are Phase-4.2-class (recorded copies / persistent
maps, multi-CB overlap), left on the table deliberately: neither is a GEMM-tile problem, and the
two shipped items are bitwise-safe and measured.
GATES: ctest 14/14 (native_mmvq_vk_parity 74k outputs incl. the new q2_0 slots sections);
short canary ids '220,18,24,16,13,271,48,25,3437,369,220,16,22,2942,220,17' -> generated
'18,30,198,32,25,220,18,24,16,13,271,48,25,3437,369,220' IDENTICAL to NO_ELIDE and to the
STRATA_VK_BATCH=0 eager arm; 735-tok long gate generated ids '271,248068,271,248069,271,1206,
10724,393,16,22,1088,14695,220,17,18,60439' IDENTICAL to the 2a33f51 reference run.
NOTE: git diff main..HEAD -- src include is NOT empty by itself: a concurrent session's rename
commit 23615f4 (anystrata) moved src/ targets; this pass touched ONLY files under vulkan/
(git diff 2a33f51..HEAD --stat -- src include == 0).

## q8_0 KV cache — the 64K-context unlock (in progress, 2026-10-27)

Why: law 11 — f16 KV at 64K (2.56 GiB) does not fit this box's pinned budget; q8_0 halves KV
(544 B a row on qwen35moe: 256 codes + 16 scale-bytes per token). The engine's --kv q8_0 path
exists upstream (src/poly/qwen35.cpp:353-360, the same caches CUDA's kvq8 correctness arm runs);
the Vulkan port now carries its write and its prefill-read arm:

* `store_q8` (vulkan/src/poly/kernels/store_q8.comp + store_q8_vk.cpp) — **ported, parity GREEN**:
  4 checks in poly_kernels_vk_parity (codes + scales byte-exact vs the CUDA body over random
  +-60000 inputs, the window's rows-before-pos untouched, and a positive ties-to-EVEN arm:
  rintf(2.5)=2, rintf(3.5)=4 — GLSL round() would give 3). Laws honored: q8_rint mirrors CUDA's
  ties-to-even (NOT GLSL round()); the scale is f16_rn_bits(amax/127); the explicit 1/d the CUDA
  body computes; codes as 8 little-endian-4 uint words and scales two-f16-a-word (glslc's no-byte-
  array law, store_f16's layout). n/32 EVEN is a launcher throw (qwen35moe 256 = nb 8 holds).
* `attend_batch` q8 arm (dequant_q8.comp + the workspace launches in attend_batch_vk.cpp) —
  **wired, building, ctest 14/14; the model-level gate has NOT run yet** (a live serve owns the
  box). CUDA-faithful (kernels.cu:2587-2630): work_rows cap on the key block, per-block
  dequantization of m rows into kwork/vwork, the fp16 GEMMs unchanged; 1D-grid flattening per
  the runtime law; pairs written by even lanes (no word RMW race). A parity case for the arm is
  the next step (run_batch gains a q8 variant: same order reference, f16-rounded codes as input).
* `attend` fused q8 read (2026-10-28, af24cad): ATT_Q8=1 variants of ALL four score reg arms
  and all three value arms, bound at 4/5 (codes: 4 int8 a word, scales: one f16 per 32 values,
  x = float(code)*d as f32 — CUDA-faithful, no f16 step unlike the workspace arm). Parity:
  three q8 cases rel_l1 ~1e-7 vs the order reference AND the naive reference. Span law: exact
  bindings, NO +8 word slack (width % 32 == 0 keeps reads in-row; the slack crossed pieces —
  the f16 cache's u64-read law does NOT carry to q8 words).
* Model gate (--kv q8_0, hybrid, 16K): FULL end-to-end, zero traps — canary ids stable, decode
  11-13 tok/s (q8 read arms add ~30% decode cost at short ctx; prefill unaffected). The earlier
  pitfall (the test's host ref double-transformed the packed scale words while the kernel was
  right) is why the isolate-arms method (q8 keys only / q8 values only) stays in the toolbox.
* store_q8_at: still a trap (no wired card-time row reader under --no-graphs; Phase 4.2's at-
  counters will demand it).
* LAW 11 UNLOCKED: serve at --max-context 65536 --kv q8_0 ran on the box (q8 KV 1.28 GiB vs
  fp16's 2.56): arena fits, all 10,240 experts STILL resident (8.4 GiB), resident RAM ~13.8 GB.
  cublasGemmStridedBatchedEx's ld stays int-safe at 64K rows (width 512 values); 128K would put
  codes' byte offsets past int32 — still machine-limited for other reasons (law 11).

## Phase 5 — the formal correctness gate on the Vulkan engine (2026-10-27/28, this BC-250)

The bench/correctness harness (docs/CORRECTNESS-PLAN.md T1/T2) run teacher-forced against
llama.cpp at the pinned submodule commit (44072a1, Vulkan build, mmap-forced per law 10), model
**chtisgit Qwen3.6-35B-A3B GSQ-hybrid**, engine `anystrata-gguf-vk --no-graphs` (wrapper
anystrata-gguf-vk-nog), arm **base** (--no-draft --lookup 0 --window 1). 12 legs with both
reference arms (short set p00-p09 + code set lru/puzzle), 1,678 scored steps.

| seq | steps | top-1 agree | engine KL (top20, nats) | llama.cpp GPU-vs-CPU floor | T2 ratio | T1 fails |
| --- | --- | --- | --- | --- | --- | --- |
| p00 | 128 | 0.984 | 0.0038 | 0.0019 | 1.93 | 0 |
| p01 | 128 | 0.992 | 0.0010 | 0.0006 | 1.66 | 0 |
| p02 | 128 | 1.000 | 0.0003 | 0.0003 | 1.03 | 0 |
| p03 | 128 | 1.000 | 0.0046 | 0.0024 | 1.94 | 0 |
| p04 | 19 | 1.000 | 0.0018 | 0.0006 | 2.76 | 0 |
| p05 | 128 | 1.000 | 0.0011 | 0.0008 | 1.36 | 0 |
| p06 | 86 | 0.988 | 0.0027 | 0.0023 | 1.17 | 0 |
| p07 | 58 | 0.983 | 0.0005 | 0.0005 | 1.12 | 0 |
| p08 | 128 | 0.992 | 0.0006 | 0.0006 | 1.01 | 0 |
| p09 | 23 | 0.913 | 0.0157 | 0.0087 | 1.81 | 1 |
| lru | 400 | 1.000 | 0.0006 | 0.0007 | 0.81 | 0 |
| puzzle | 324 | 0.988 | 0.0012 | 0.0013 | 0.97 | 0 |

**Verdict.** Step-weighted across the whole set the engine sits at **1.39x the floor — inside
the T2 bar (1.5x)**; on the ten legs with >=58 steps, 1.33x. Per-leg, 7 of 12 pass outright;
the ratios above 1.5 concentrate in legs whose floor is itself tiny (p01-p04: floor 0.0003-
0.0006, absolute KL 0.0003-0.0046 - the fork's A6000 record measured the same shape for its
arms, mean KL 0.6-1.2x there against a different pin). T1 holds at every position with gap
>= 0.5 nats on 1,655 of 1,678 steps; the one p09 flip and the 1-9% top-1 misses are all near
ties (gap < 0.5), as the plan's law predicts. **T0: the draft arm's top-20 bytes are IDENTICAL
to base on all four legs scored (p00-p03) - speculation inert at window 1, bitwise.**
kvq8 arm: the prompt path (attend_batch dequant, ported this session) runs; the decode arm
traps BY NAME (attend()'s fused kq8 read: the ported-queue entry stands) - q8_0 stays
prefill-only until those template arms land (law 11's unlock is half-done).

**Long set on this box (machine-limited, law 10d):** d4k/d16k/r16k engine arms scored against
the CPU reference directly (GPU logits-dump legs not hostable on 15 GB): d4k 256 steps top-1
0.996 KL 0.0198; d16k 256 steps 0.992 / 0.0077; r16k 510 steps 0.984 / 0.0116, zero T1 fails
(2,048 more scored steps at long context). Absolute KLs at long context ride the same order as
the short set; no T2 ratio claimed without the GPU arm. One harness pitfall caught mid-record:
the realtext r16k leg must be teacher-forced on the FIXED answer.ids (the plan's convention for
that set) — forcing it on the hybrid greedy answer (which diverges at token 3) scored a bogus
KL 7.8 before the mismatch was found; both sides re-forced, clean. d31k: not hosted, recorded
as machine-limited like the fork's per-machine gaps.

kvq8 arm at the model level: p00-p02 produced their prompt-path steps through the new
attend_batch q8 dequant arm and halted at attend()'s fused q8 decode read; once that read
landed (af24cad), the FULL arm re-ran clean (2026-10-29, --no-graphs so the delta is ONLY
the q8 cache): all 10 short legs, 1,096 steps:

p00 128 top-1 .992 kl .0054 2.76x | p01 128 .992 .0010 1.59x | p02 128 1.000 .0002 0.97x
p03 128 .984 .0048 2.05x | p04 19 1.000 .0014 2.23x | p05 128 .984 .0011 1.30x
p06 86 1.000 .0028 1.23x | p07 58 .983 .0008 1.65x | p08 128 .992 .0008 1.27x
p09 23 .913 .0141 1.62x     (step-weighted: 1.82x the llama.cpp floor)

T1: ZERO failures across all 1,096 steps (the q8 cache flipped no gap>=0.5 position; the
one base-arm flip does not appear here either). The 1.82x over the floor is the expected
q8_0 representation cost (absolute KL ~0.002 mean - same order as the base arm's 0.003):
the KV quantization is a storage decision, not an arithmetic bug; the store/dequant/fused
reads are all bitwise-verified against the CUDA tree. 64K serve runs (law 11's unlock)
carry this documented trade.

Graphs A/B (Phase 4.2 law): 48-token canary generation, --no-graphs vs graphs: BITWISE
IDENTICAL ids (cmp clean). Graphs are the Vulkan default from b6af60d.

Reference-leg survival notes (all recorded as PORT-NOTES laws 10/11): GPU legs pin 12.2 GB via
TTM (both device tiers), the dump tool needed STRATA_DUMP_MMAP=1 (llama.cpp's AUTO loader copies
the whole GGUF to anon RAM on iGPUs), and every leg runs under choom -n 900 so the leg, never the
desktop, takes the OOM bullet. Three session deaths preceded those laws.


## Phase 4.2 — captured graphs (2026-10-29): GREEN, and the engine now runs with graphs BY DEFAULT

The eager per-launch + per-token-sync path was the last speed wall (PORT-STATUS Phase 4.1:
flush_wait stalls ~11.9s / 31.6k drains). What shipped:
* the runtime's capture layer: BeginCapture opens ONE multi-submit CB; launches record with the
  EAGER full-RW barrier law inside (bit-safe by construction); EndCapture closes; GraphLaunch is
  one vkQueueSubmit whose fence serializes replays AND serves the replay's staged D2H copies;
  chained 128-set pools per graph (1,600 layer graphs, no descriptor bloat).
* cudaMemcpy sites inside a capture become transfer NODES: h2d freezes the host bytes at capture
  time (CUDA memcpy-node semantics — the engine's replay pass writes token positions OUTSIDE any
  graph, so frozen bytes never carry per-token data); d2h lands in per-graph mapped staging and
  is delivered to the host at the next fence drain (the engine reads D2H results only past its
  event/stream syncs); d2d/fill record directly. A capture-body SYNC still aborts loudly.
* vk_graph_capture pins the layer alone (record/replay bits == eager x3, frozen-byte + drain
  semantics, destroy semantics, node counts honest).
* the model gate: hybrid 16-token canary AND the 48-token strict A/B (eager vs graphs,
  grep-exact id lines, cmp clean): BITWISE IDENTICAL - graphs are generation-inert, the fork's
  law holds on this backend; decode 18.5-19 -> 20.9 tok/s (f16 KV, graphs)
  and 21.9 tok/s (q8_0 KV + graphs); 64K ctx + q8_0 + graphs runs at ~100% VRAM experts.
  --no-graphs remains for A/B; graphs are now the engine default on Vulkan.
Pitfalls this leg caught (kept in PORT-NOTES law 12): (a) the D2H capture-time SPEC serviced
before the first replay copies never-written staging — d2h_def/d2h_run split; (b) record_*
helpers re-locking g_m under their callers self-DEADLOCKED (lock law: callers hold g_m, g_pm is
the leaf); (c) engine.cpp's copy2 host-pair classification already routes the router readback
through copy_d2h — the capture branch had to be in copy_h2d/copy_d2h, not only in the compat
strata_memcpy (a compat-only fix leaves the poly launcher path eager-draining mid-capture).


## UMA load-path collapse: the host-mapped arena (2026-10-29)

The histogram of the hybrid bring-up (STRATA_VK_FLUSH_WHY): 30,720 x 288 KiB staged h2d copies
+ every eager drain. RADV exports DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT (types 3/4 in the
device-local heap — the banner's uma=1), so arena pieces now ALLOCATE+MAP host-visible and
copy_h2d/d2h/fill_d take a direct memcpy lane inside a mapped piece (31,336 direct / 0 staged).
Verified: full ctest 15/15; hybrid canary 48 ids BITWISE IDENTICAL mapped vs
STRATA_VK_NO_MAP_ARENA=1; load wall ~30-35 s either way — the SSD->page-cache read dominates
on a warm box (a 300 s gap between runs evicted page cache re-shows it), the removed stage is
pure GPU-copy traffic + ~31k submits saved; the benefit concentrates on expert SWAP-INS at
capacity (the regime the 64K/fp16-arena squeeze creates, and Flash-Next-class models on this
box): each swap-in skips one full DRAM round trip. Laws: PORT-NOTES 13.


## Open bug (recorded 2026-10-29, deterministic): graphs + expert-SWAP regime + speculation -> id-0 collapse

**FIXED 2026-11-05** (branch task-6.0-head-capture, 5f49173: swap-regime head captures record fresh; the
full record with dead ends is at the end of this section — read the FIX SHIPPED paragraph before this
matrix for what the collapse actually was).

Isolation matrix (hybrid, canary ids, 4096 ctx, max-new 32; `--vram-reserve-mib 11000` forces a
2.0 GiB expert cache -> ~35% of experts on CPU with the staging thread live):

| graphs | cache | speculation | result |
| --- | --- | --- | --- |
| on | full (10,240) | on  | OK — bitwise == eager (the 48-id A/B, 132ffc9) |
| off | swap | on  | OK (5.18 tok/s, 20/21 guesses) |
| on  | swap | off | OK (correct ids, run A) |
| off | swap | off | OK, == run A |
| **on** | **swap** | **on** | **COLLAPSE: ids go 0 from token 9, identical across >=4 runs** |

Trace under STRATA_VK_CAPTURE_TRACE: `cap-import=0, cap-axpy=0` — no host upload is recorded
inside ANY capture in this regime, so the frozen-staging-bytes theory is dead; STRATA_VK_NO_ELIDE
does not change it. The staging thread (follow_loop) and the capturing thread share the queue and
the arena; the captured layer passes record slot bindings while swaps move experts - the CUDA
build survives the same interleaving (verified: eager == CUDA semantics here by the fork's own
regression tests), so the delta is in the port's capture or record path, not the engine.
Workaround (safe, verified): run the SWAP regime with `--no-graphs` (eager path is correct,
slow by design: 5-8 tok/s at 35% residency); the FULL-RESIDENCY regime — the only one this box
serves today at 16K/32K/64K-q8 — is unaffected and keeps graphs. Next step when this is picked
up: bisect inside the collapse combo (the spec verify pass records L.card==1 passes whose
captured d_slots_/expert bindings predate a swap that happens between capture and replay).
Narrowing (same day): --no-draft alone still collapses; --window 2 still collapses; --lookup 0
(window+draft defaults) is CLEAN -> the trigger is the prompt-tail LOOKUP guesser (its verify
pass is the first multi-token capture whose recorded expert bindings go stale under a swap).
Lanes ruled out (same combo, each mitigated): STAGED arena copies (NO_MAP_ARENA=1) still
collapse; BATCH=0 (per-launch eager submit, no pending CB) fails DIFFERENTLY - degenerate
token-18 repetition, not zeros - so the capture/replay record itself is implicated, not the
copy lane or the barrier-elision cluster.
SITE PINNED (2026-11-04, STRATA_NO_REPLAY battery, swap regime --vram-reserve-mib 11000,
canary+32): killing ONLY the heads site (site 0: the windowed/step `layer_head` captures,
L.heads/L.rows_heads) restores byte-for-byte-correct output (ids 18,30,198,32,25,220,18,
24,16,13,271,... = recorded line); shared/tail/card_moe/logits kills all still collapse.
So the stale binding lives in the HEAD capture - what it records about the layer ahead of
the router is what a slot trade invalidates (candidates: the recorded copy lane's host
source pointer at capture time, and the router inputs whose arena addresses the head baked
before follow() moved them). The merge 1d474dd is INNOCENT (pre-merge binary collapses
identically; eager control on current main is clean - the earlier "eager collapses" fear
was an instrument bug: the run() helper never passed "$@", so --no-graphs never reached
the engine; caught by re-running with env/args honored).

FIX SHIPPED (2026-11-05, branch task-6.0-head-capture): the stale binding is the WINDOWED HEAD
capture replayed after the window's halves shift rows/snapshots. follow()'s slot writes churn
the caches mostly-fresh (recorded=1623 vs replayed=57 in the collapse combo), and a slot-move
retire did NOT fix it (traced: captures already empty at the slot write) — the mechanism is the
snapshot/row state baked at record time, not the expert address. The fix: while the cache MOVES
(n_swap_ > 0, the swap regime), head launches record fresh instead of replaying (one condition
on the three heads-replay sites); full residency keeps replaying bitwise as before. Measured:
the old collapse combo (swap+graphs+lookup defaults) is byte-for-byte vs the recorded canary at
16 AND 96 tokens; eager-swap, nolookup, and the full-residency serve path all EXACT; the regime
is FASTER than the --no-graphs workaround (7.4 vs 5.2-6.2 tok/s: shared/tail/card captures still
replay) — so the swap-regime recommendation upgrades from --no-graphs to graphs-on.
Dead ends recorded honestly: slot-move retire (empty at that moment), halves-off probe (invalid
env var, never existed), mask arithmetic that read as passes_ staleness (the mask-63 leg differed
from mask-1 only via the logits site; the heads site was the constant).

Candidate fix next session: retire per-layer capture keys when a swap moved that layer's slots
(the slot table is host-visible to the engine; record() lifetime belongs to the engine), or
replay-gate on all_cached() at launch time and drop to eager when the cache moved - matching
what CUDA's recapture cadence does implicitly via head_key/room growth.


## Task 4.2 (plan 2026-10-21): events with REAL GPU timestamps (2026-10-30): GREEN

The graphs half of Phase 4.2 landed 2026-10-29 (record above); the plan's Task 4.2 line —
"events with real GPU timestamps (timestampComputeAndGraphics proven on this box) so bench
timings and the engine's per-stage tables become meaningful" — was NOT done: event_record paid
a full flush_wait per stamp, event_elapsed returned a flat 0 (host-stamp pairs only inside the
compat header), and docs/VULKAN.md still said "Events time 0". Shipped now:

* vk_device.cpp: the timestamp gate is read BEFORE device create — limits.timestampComputeAndGraphics
  AND the chosen queue family's timestampValidBits (BC-250: compute family 64 bits, sparse family 0:
  the per-family check is the gate; the fact is core, NOT a feature bit — enabling nothing is legal,
  writing the first VkPhysicalDeviceFeatures bit name as if it were one does not compile).
  DeviceInfo gains gpu_timestamps + timestamp_period_ns (radv BC-250: 10.0 ns/tick).
* vk_runtime.cpp event layer, two arms:
  - gpu arm (default on this box): event_record appends vkCmdResetQueryPool +
    vkCmdWriteTimestamp(COMPUTE) into the PENDING CB — the stamp sits at its exact stream
    position behind every recorded launch, at ZERO submit cost (the old path paid a full
    queue drain per record: --stage-timing's 512 pairs/token would have re-made the Phase-4.1
    stall wall). event_elapsed drains once and reads both 64-bit ticks; ms = (to-from) *
    timestampPeriod / 1e6. event_synchronize submits/drains the CB carrying the stamp.
    cudaEventDisableTiming events (poly's 16 staging-sync sites: engine.cpp's router readback,
    expert staging) create NO query pool and record NO node — a pure marker, and
    cudaEventCreateWithFlags now honours the flag instead of ignoring it.
  - host arm (kept honest as the fallback for cards without the fact, forced by
    STRATA_VK_NO_GPU_EVENTS=1 for A/B): the old drain+steady_clock behaviour, never a fake
    driver timestamp. A pair mixing arms falls back to the host interval (every record stamps
    steady_clock too), so a foreign-capture eager arm can never yield epoch junk.
  Law 14 respected: while ANOTHER thread captures, this thread records the host arm (never
  batches a timestamp into the shared pending CB beside a foreign capture); a record inside
  one's OWN capture keeps the capture-body sync law and refuses loudly.
* Pinned by execution (the port's law — capability by running, not by listing): ctest
  vk_event_timing (gpu arm) + vk_event_timing_host_arm (env-forced), around a real 256K-store
  dispatch: elapsed > 0 (the pre-4.2 layer was a flat 0), plausible (<10 ms), and <= the
  host-wall interval that wrapped the same span. BC-250 measurement: gpu 0.095 ms vs wall
  0.183 ms for the pair (the driver's ticks cut the submit overhead the host stamp includes).
* Model gate with the new layer in the engine: hybrid canary (the 24-token ChatML prompt,
  re-tokenized from the hybrid file — the old /tmp/qids.txt is gone), graphs vs --no-graphs,
  full residency: ids '248068,271,248069,271,18,24,16,248046' BITWISE IDENTICAL both arms,
  generation unaffected by the timing change (the poly path only ever used DisableTiming
  events; those now skip pools — cheaper, same order). leg.sh engine gate honoured (20 s
  cooldown between legs).
* Task 4.3's unfinished half CLOSED the same day: poly_draft_race_test ported as
  poly_draft_race_test_vk (CUDA's src/poly/draft_race_test.cpp UNCHANGED + the same engine
  sources, a shared VK_POLY_ENGINE_SRC/LIBS set in vulkan/CMakeLists.txt; the port rule holds —
  git diff -- src include == 0). Model gate (hybrid file, RUNS=4, leg.sh engine gate): SKIP rc=3
  with the tree's own honest reason — 'the read gave the draft block 0 call(s) on the layer-major
  path; the race needs two or more (every expert cached...)' — the hybrid model keeps 100% of
  experts resident, exactly the regime where the test declines; the CUDA build's own header says
  SKIP is the correct outcome there. The binary builds and runs end-to-end against the card.
* Honest scope: --stage-timing's per-stage TABLES live in src/program/generate.cpp, which this
  box's vk build does not link (the pack/core engine is not ported to Vulkan); poly has no
  stage-timing CLI. What Task 4.2 owed — a compat event surface returning real device time for
  every consumer in the tree (bench harnesses, session sweep, STRATA_COMMIT_TIMING when a core
  engine runs on Vulkan) — is what landed and is pinned. ctest 17/17.

## Upstream core merge 1d474dd — hybrid streaming + pointer-table MMQ (2026-11-04): VERIFIED INERT HERE

upstream/robertkarl's `hybrid-mmq-tab` branch (29511cf fd4f95f 0c6c01c d2da41b 3d5fc52:
`--stage-rows`/`--stage-overlap` expert streaming while an answer is written; `--mmq-tab`
pointer-table launches of 16 for a layer's cached experts wherever follow() traded their slots)
touches the SHARED core `src/poly/engine.cpp`, so this backend runs the new state machine too.
Cherry-pick sequence conflicted on the fork's diverged docs; the code region applied 3-way clean
as one reviewed patch, upstream's five messages preserved verbatim in the commit body.

Gate on this box (tools/vulkan/post-restart-verify.sh, the standing post-reboot one-command
check): canary 16 ids BITWISE-identical to the recorded line; experts 10,240/10,240 resident,
0 swapped; ctest 17/17; make test e2e green. Decode at full residency 25.4 tok/s (96-token leg)
— inside the existing band: the merge is exactly as promised, inert here. The wins upstream
measured (20.6 -> 26.5-29.3 tok/s decode streaming on a 1060-class card; 94.6 vs 68.1 t/s
prefill with --mmq-tab on a 5090) are DISCRETE-card numbers — this APU's hybrid regime shares
one bus with everything, and the pointer-table kernels are CUDA (STRATA_POLY_MMQ_TAB stays OFF;
the Vulkan Mmq shim answers tab()=false and mul_tab() traps by name, added so the shared core
links). Their numbers are the README's, attributed; this box's own number is this one.

Process note kept on purpose: the first verify run reported "CANARY: DIFFERS" under a swap
storm — an EMPTY capture from the load timeout read as a mismatch. Fixed in da692de (empty =
inconclusive, exit 2); the false alarm is exactly why "compare ids" must never be conflated
with "got no ids". The serve held :18090's engine slot during the storm; one engine at a time
(law 10) applies to verification legs too — stop, leg, restart.

