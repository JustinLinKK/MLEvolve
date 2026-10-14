# Hardware catalogue coverage — v8.19

Every row has runtime-gated initial context and parent trial routing. Precision
and fused-optimizer eligibility is determined from its own graph/runtime facts.
NVIDIA targets use CUDA; AMD targets use ROCm. Platform rows refer to one visible member.

| Hardware profile | Backend | Target | Per-GPU memory (MiB) | Native precision eligibility |
| --- | --- | --- | ---: | --- |
| `nvidia.ada_lovelace.geforce_rtx_4060.spec` | CUDA | `sm_89` | 8192 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4060_ti_16gb.spec` | CUDA | `sm_89` | 16384 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4060_ti_8gb.spec` | CUDA | `sm_89` | 8192 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4070_gddr6x.spec` | CUDA | `sm_89` | 12288 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4070_gddr6.spec` | CUDA | `sm_89` | 12288 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4070_super.spec` | CUDA | `sm_89` | 12288 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4070_ti.spec` | CUDA | `sm_89` | 12288 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4070_ti_super.spec` | CUDA | `sm_89` | 16384 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4080.spec` | CUDA | `sm_89` | 16384 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4080_super.spec` | CUDA | `sm_89` | 16384 | BF16/FP16 |
| `nvidia.ada_lovelace.geforce_rtx_4090.spec` | CUDA | `sm_89` | 24576 | BF16/FP16 |
| `nvidia.ada_lovelace.l20.spec` | CUDA | `sm_89` | 49152 | BF16/FP16 |
| `nvidia.ada_lovelace.l4.spec` | CUDA | `sm_89` | 24576 | BF16/FP16 |
| `nvidia.ada_lovelace.l40.spec` | CUDA | `sm_89` | 49152 | BF16/FP16 |
| `nvidia.ada_lovelace.l40s.spec` | CUDA | `sm_89` | 49152 | BF16/FP16 |
| `nvidia.ada_lovelace.rtx_4000_ada_generation.spec` | CUDA | `sm_89` | 20480 | BF16/FP16 |
| `nvidia.ada_lovelace.rtx_4500_ada_generation.spec` | CUDA | `sm_89` | 24576 | BF16/FP16 |
| `nvidia.ada_lovelace.rtx_5000_ada_generation.spec` | CUDA | `sm_89` | 32768 | BF16/FP16 |
| `nvidia.ada_lovelace.rtx_5880_ada_generation.spec` | CUDA | `sm_89` | 49152 | BF16/FP16 |
| `nvidia.ada_lovelace.rtx_6000_ada_generation.spec` | CUDA | `sm_89` | 49152 | BF16/FP16 |
| `nvidia.ampere.a10.spec` | CUDA | `sm_86` | 24576 | BF16/FP16 |
| `nvidia.ampere.a100_pcie_40gb.spec` | CUDA | `sm_80` | 40960 | BF16/FP16 |
| `nvidia.ampere.a100_pcie_80gb.spec` | CUDA | `sm_80` | 81920 | BF16/FP16 |
| `nvidia.ampere.a100_sxm4_40gb.spec` | CUDA | `sm_80` | 40960 | BF16/FP16 |
| `nvidia.ampere.a100_sxm4_80gb.spec` | CUDA | `sm_80` | 81920 | BF16/FP16 |
| `nvidia.ampere.a16.spec` | CUDA | `sm_86` | 65536 | BF16/FP16 |
| `nvidia.ampere.a2.spec` | CUDA | `sm_86` | 16384 | BF16/FP16 |
| `nvidia.ampere.a30.spec` | CUDA | `sm_80` | 24576 | BF16/FP16 |
| `nvidia.ampere.a40.spec` | CUDA | `sm_86` | 49152 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3050.spec` | CUDA | `sm_86` | 8192 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3060.spec` | CUDA | `sm_86` | 12288 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3060_ti.spec` | CUDA | `sm_86` | 8192 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3070.spec` | CUDA | `sm_86` | 8192 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3070_ti.spec` | CUDA | `sm_86` | 8192 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3080.spec` | CUDA | `sm_86` | 10240 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3080_12gb.spec` | CUDA | `sm_86` | 12288 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3080_ti.spec` | CUDA | `sm_86` | 12288 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3090.spec` | CUDA | `sm_86` | 24576 | BF16/FP16 |
| `nvidia.ampere.geforce_rtx_3090_ti.spec` | CUDA | `sm_86` | 24576 | BF16/FP16 |
| `nvidia.ampere.rtx_a2000.spec` | CUDA | `sm_86` | 6144 | BF16/FP16 |
| `nvidia.ampere.rtx_a2000_12gb.spec` | CUDA | `sm_86` | 12288 | BF16/FP16 |
| `nvidia.ampere.rtx_a4000.spec` | CUDA | `sm_86` | 16384 | BF16/FP16 |
| `nvidia.ampere.rtx_a4500.spec` | CUDA | `sm_86` | 20480 | BF16/FP16 |
| `nvidia.ampere.rtx_a5000.spec` | CUDA | `sm_86` | 24576 | BF16/FP16 |
| `nvidia.ampere.rtx_a5500.spec` | CUDA | `sm_86` | 24576 | BF16/FP16 |
| `nvidia.ampere.rtx_a6000.spec` | CUDA | `sm_86` | 49152 | BF16/FP16 |
| `nvidia.blackwell.b100.spec` | CUDA | `sm_100` | 196608 | BF16/FP16 |
| `nvidia.blackwell.b200.spec` | CUDA | `sm_100` | 196608 | BF16/FP16 |
| `nvidia.blackwell.gb200_nvl2.spec` | CUDA | `sm_100` | 190464 | BF16/FP16 |
| `nvidia.blackwell.gb200_nvl4.spec` | CUDA | `sm_100` | 190464 | BF16/FP16 |
| `nvidia.blackwell.gb200_nvl36.spec` | CUDA | `sm_100` | 190464 | BF16/FP16 |
| `nvidia.blackwell.gb200_nvl72.spec` | CUDA | `sm_100` | 190464 | BF16/FP16 |
| `nvidia.blackwell.geforce_rtx_5060.spec` | CUDA | `sm_120` | 8192 | BF16/FP16 |
| `nvidia.blackwell.geforce_rtx_5060_ti_16gb.spec` | CUDA | `sm_120` | 16384 | BF16/FP16 |
| `nvidia.blackwell.geforce_rtx_5060_ti_8gb.spec` | CUDA | `sm_120` | 8192 | BF16/FP16 |
| `nvidia.blackwell.geforce_rtx_5070.spec` | CUDA | `sm_120` | 12288 | BF16/FP16 |
| `nvidia.blackwell.geforce_rtx_5070_ti.spec` | CUDA | `sm_120` | 16384 | BF16/FP16 |
| `nvidia.blackwell.geforce_rtx_5080.spec` | CUDA | `sm_120` | 16384 | BF16/FP16 |
| `nvidia.blackwell.geforce_rtx_5090.spec` | CUDA | `sm_120` | 32768 | BF16/FP16 |
| `nvidia.blackwell.rtx_pro_6000_blackwell.spec` | CUDA | `sm_120` | 98304 | BF16/FP16 |
| `nvidia.hopper.gh200_grace_hopper.spec` | CUDA | `sm_90` | 98304 | BF16/FP16 |
| `nvidia.hopper.h100_nvl_94gb.spec` | CUDA | `sm_90` | 96256 | BF16/FP16 |
| `nvidia.hopper.h100_pcie_80gb.spec` | CUDA | `sm_90` | 81920 | BF16/FP16 |
| `nvidia.hopper.h100_sxm5_80gb.spec` | CUDA | `sm_90` | 81920 | BF16/FP16 |
| `nvidia.hopper.h200_sxm_141gb.spec` | CUDA | `sm_90` | 144384 | BF16/FP16 |
| `amd.rdna3.radeon_rx_7600.spec` | ROCm | `gfx1102` | 8192 | runtime-checked FP16/BF16 |
| `amd.rdna3.radeon_rx_7700.spec` | ROCm | `gfx1101` | 12288 | runtime-checked FP16/BF16 |
| `amd.rdna3.radeon_rx_7800_xt.spec` | ROCm | `gfx1101` | 16384 | runtime-checked FP16/BF16 |
| `amd.rdna3.radeon_rx_7900_gre.spec` | ROCm | `gfx1100` | 16384 | runtime-checked FP16/BF16 |
| `amd.rdna3.radeon_rx_7900_xt.spec` | ROCm | `gfx1100` | 20480 | runtime-checked FP16/BF16 |
| `amd.rdna3.radeon_rx_7900_xtx.spec` | ROCm | `gfx1100` | 24576 | runtime-checked FP16/BF16 |
| `amd.rdna4.radeon_rx_9070.spec` | ROCm | `gfx1201` | 16384 | runtime-checked FP16/BF16 |
| `amd.rdna4.radeon_rx_9070_xt.spec` | ROCm | `gfx1201` | 16384 | runtime-checked FP16/BF16 |
| `nvidia.volta.tesla_v100_32gb.spec` | CUDA | `sm_70` | 32768 | FP16/GradScaler |
| `nvidia.turing.tesla_t4.spec` | CUDA | `sm_75` | 16384 | FP16/GradScaler |
