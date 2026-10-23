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
| native_expert_grouped / native_expert_layout (mode setters host-side) | iq_kernels.hpp | src/kernels/cuda/iq_kernels.cu | native_expert_parity.cpp, native_grouped_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| nvfp4 expert path | iq_kernels.hpp (cpu/nvfp4_*.hpp) | (nvfp4 — NVIDIA-only format) | nvfp4_expert_gpu_parity.cpp | NVIDIA nvfp4 format | n/a (nvfp4 is an NVIDIA-only format; gfx1013 has no path) | n/a |
| native_quantize_q8_1 | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu | (unit: poly_kernels_vk_parity.cpp; in-tree family tests cover the mmvq consumers) | `__shfl` (warp chains) | translate | **ported+parity-green** (vulkan/src/kernels/quantize_q8_1.comp, native_quantize_q8_1_vk.cpp) |
| q8_1 (add_rms_norm_q8_1, swiglu_quantize_q8_1) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| q4_0 / q5_0 / q8_0 (mmvq + f32 pair) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| q4_k / q5_k / q6_k / q2_0 / q2_k / q3_k (mmvq + f32 pair) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| iq4_xs / iq4_nl (mmvq + f32 pair) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_mmvq_slots (types 16/13/14; others: false with nothing launched, exactly CUDA slots_kind==0) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu:3178 + :1710 slots kernel, iq_kernels.cu:3491 mmvq_slots_kernel<16> | native_mmvq_vk_parity.cpp (slots: bitwise y_slots == y_single AND host refs; unported type returns false, y poison untouched) | 24-25 storage bindings (w/x/y per slot + iq2xxs grid), flattened (gx,count) grid, macro per-slot dot | translate | **ported+parity-green** (mmvq_slots_iq2_xxs/q5_k/q6_k.comp + native_mmvq_vk.cpp; pair + ids still todo) |
| native_mmvq_pair / native_mmvq_ids | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu (+ q8_1_il.cuh) | mmvq_multi_parity.cpp, mmvq_il_parity.cpp, q5_projection_parity.cpp | `__shfl`, `__shared__` | translate | todo |
| native_mmvq (iq2_xxs + q5_k + q6_k; other types trap-by-name `native_mmvq type <n>`) | native_mmvq.hpp | src/kernels/cuda/native_mmvq.cu:3110 + iq_kernels.cu launch_mmvq<16> + native_q5_k_mmvq + native_q6_k_mmvq | native_mmvq_vk_parity.cpp (bitwise) | `__shfl` (warp_sum), `__vcmpne4/__vsubss4/dp4a` (scalar-emulated), `iq2xxs_grid` (arena-uploaded table), fp contraction (the CUDA leg builds with --use_fast_math; the vk dot mirrors the fused order) | translate | **ported+parity-green** (vulkan/src/kernels/mmvq_iq2_xxs.comp, mmvq_q5_k.comp, mmvq_q6_k.comp, native_mmvq_vk.cpp; iq-family row: format 16 iq2_xxs, format 13 q5_k, format 14 q6_k. Other types trap by number) |
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
| rope_neox_pair / build_rope_table / rope_neox_apply | rope.hpp | src/kernels/cuda/rope.cu | rope_parity.cpp | none | translate | todo |
| rope_tab_cs / mrope_pos (rope/mrope table setters host-side) | mrope.hpp | (inline / rope.cu) | rope_parity.cpp | none | translate | todo |
| rope_scaling (yarn: rope_yarn_corr_dim / ramp / rope_scaled_angle — inline math) | rope_scaling.hpp | (header-inline) | rope_parity.cpp | none | host-free (compiled-unchanged; CPU/FMA math, GLSL-compatible if device-used) | todo |
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
| poly: layer_norm_kernel, bf16_to_f32_kernel, store_f16_kernel/_at, store_q8_kernel/_at, add_bias_kernel, axpy_kernel, add_scaled_kernel, mul_sigmoid_kernel, softcap_kernel, likeliest_kernel | src/poly/kernels.cu | src/poly/kernels.cu | unit poly_kernels_vk_parity | none | translate | **axpy/add_scaled/mul_sigmoid PORTED (Task 2.7): vulkan/src/poly/kernels/elementwise.comp (EW_VARIANT arms) + elementwise_vk.cpp; parity: axpy/add_scaled bitwise, mul_sigmoid rel 5.0e-7 (exp vs expf, tree tol 1e-5); rest todo** |
| poly: hc_mix/read/post/head/mean kernels | src/poly/kernels.cu | src/poly/kernels.cu | hc_persist_parity.cpp (host-side contract) | `__shfl`, `__shared__` | translate | todo |
| poly: kda_conv/_rows/_state, kda_step/gates/state/out kernels | src/poly/kernels.cu | src/poly/kernels.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | todo |
| poly: gdn_norm/gates/state/state_reg/out kernels | src/poly/kernels.cu | src/poly/kernels.cu | gdn_parity.cpp (host-side contract) | `__shfl`, `__shared__` | translate | **gdn_step PORTED (Task 2.5): gdn_norm.comp/gdn_gates.comp/gdn_state.comp (plain+reg<1,2,4,8>)/gdn_out.comp + gdn_vk.cpp; unit poly_gdn_vk_parity rel_l1 <= 1.6e-7 vs both contraction arms (tree tol 1e-5)** |
| poly: conv4_silu/_rows/_state, shortconv/_rows/_state kernels | src/poly/kernels.cu | src/poly/kernels.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | **conv4_silu (all three kernels: walk/rows/state) ported (Task 2.4): vulkan/src/poly/kernels/conv4_silu.comp, conv4_state.comp, conv4_silu_vk.cpp, unit poly_kernels_vk_parity rel 5.9e-7 (tree tol 1e-5); shortconv todo** |
| poly: swiglu_clamp/_rows, geglu/_rows kernels | src/poly/kernels.cu | src/poly/kernels.cu | unit poly_kernels_vk_parity | none | translate | **swiglu_clamp PORTED (Task 2.7, elementwise.comp): parity rel 7.0e-7 (exp vs expf); swiglu_clamp_rows + geglu todo** |
| poly: dsa_keys/softmax, idx_store/pool/weigh, route_kernel, moe_sum_kernel | src/poly/kernels.cu | src/poly/kernels.cu | none (poly engine tests) | `__shfl`, `__shared__` | translate | todo |
| poly: attn_score/softmax/value/group/ungroup/softmax_block/scale, head_norm_rope kernels | src/poly/kernels.cu | src/poly/kernels.cu | none (poly engine tests) | `__shfl`, `__shared__`, `cp.async` (some paths), `asm volatile` | fp32-fallback for the cp.async/mma-attn paths; ordered-FP32 score/softmax otherwise | todo |
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
| 4 | `strata::kernels::native_mmvq` | native_mmvq / _pair / _slots / _ids — **iq2_xxs (type 16) PORTED (Task 2.2)**; other types trap-by-name |
| 5 | `strata::poly::gemv_f32` | poly: gemv_kernel — **PORTED (Task 2.4)** |
| 6 | `strata::poly::conv4_silu` | poly: conv4_silu/_rows/_state kernels — **PORTED (Task 2.4)** |
| 7 | `strata::poly::gdn_step` | poly: gdn_norm/gates/state/state_reg/out kernels — **PORTED (Task 2.5)** |
| 7b | `strata::kernels::native_mmvq` type 14 (Q6_K) | native_mmvq family — **PORTED (Task 2.6)** |
| 8 | `strata::poly::axpy` | poly: axpy_kernel — **PORTED (Task 2.7)** |
| 9 | `strata::poly::swiglu_clamp` | poly: swiglu_clamp/_rows — **swiglu_clamp PORTED (Task 2.7)**; _rows todo |
| 10 | `strata::poly::mul_sigmoid` | poly: mul_sigmoid_kernel — **PORTED (Task 2.7)** |
| 11 | `strata::kernels::native_mmvq_slots` | native_mmvq / _pair / _slots / _ids |
| 12 | `strata::poly::swiglu_clamp_rows` | poly: swiglu_clamp/_rows |
| 13 | `strata::poly::add_scaled` | poly: add_scaled_kernel — **PORTED (Task 2.7)** |
| 14 | `strata::poly::head_norm_rope` | poly: head_norm_rope kernels |
| 15 | `strata::poly::store_f16` | poly: store_f16_kernel/_at |
| 16 | `strata::poly::attend` | poly: attn_score/softmax/value/… kernels |

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
