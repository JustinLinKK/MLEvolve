# Sweep Unbiased — Cassava Balanced Arch (2026-05-06)

## Setting
- **Workload**: cassava-leaf-disease-classification (12 GB, 18,722 train images)
- **Trace**: W3 — 20 entries balanced across model class
  - 7 CNN: convnext_base ×4, efficientnet_b3 ×2, resnet101 ×1
  - 7 Transformer: vit_base_patch16_224 ×4, swin_small ×2, deit_base ×1
  - 6 MLP-Mixer: mixer_b16 ×3, gmlp_s16 ×2, resmlp_24 ×1
- **Per-job**: subset 4000, epochs 2, bs ∈ {16,24,32,48} per-arch capped (avg ≈ 90-150 s)
- **Submit**: burst (all 20 at t=0)
- **GPU**: RTX 5090 32 GB class card (CUDA stack matching your PyTorch build)
- **Configs**: 12 (B1-B3 baseline serial, T1-T2 parallel default, T4-T7 parallel batch optimized scheduler, T8-T11 torch_mp pool)

## Results — fill from `summary.md` after `plot_results.py`

| ID | Mode | Backend | Probe | Runner | Wall-clock | Speedup vs B1 |
|---|---|---|---|---|---|---|
| B1 | serial_basic | exclusive | off | scheduler | TBD | 1.00x |
| B2 | serial_batch_optimized | exclusive | binary | scheduler | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... |

## Plots
- `plots/gpu_utility_overlay.png` — SM_ACTIVE / DRAM_ACTIVE / FB_MEM / PCIE_RX × 12 configs
- `plots/wallclock_bars.png` — 12 bars + B1 reference
- `plots/speedup_heatmap.png` — backend × probe × runner heatmap
- `plots/arch_sensitivity.png` — per-job time by {CNN, Transformer, Mixer}
- `plots/agent_status_per_config.png` — 12 subplots, stage timeline

## Conclusions (fill after sweep)

### vs previous biased run (2026-05-06 first attempt)
- Previous: 19/20 CNN → T2 stream got 5.65x. Was speedup arch-dependent?
- Now: 7/7/6 balanced → T2 stream speedup = TBD. If still ~5x → backend really helps; if drops → previous was task-pickup bias.

### T4-T7 (parallel_batch_optimized via scheduler)
- Solo profile seed unblocked planner (vs previous deadlock).
- Performance vs T8-T11 (torch_mp same axes): TBD.

### Per arch class
- CNN best backend: TBD
- Transformer best backend: TBD
- Mixer best backend: TBD

## Confidence intervals
- Single seed → no error bars. Future work: rerun with seeds {42, 123, 456} for variance.
