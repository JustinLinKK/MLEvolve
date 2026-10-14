# L40S random-matrix stress load

## Target

- Pod: `gpu-dev-l40s-1gpu-6fcb7f9bfc-vtlpj` in namespace `ecepxie`
- GPU: one NVIDIA L40S with 46,068 MiB VRAM
- Process: PID `7619` inside the pod

## Workload

The process continuously evaluates BF16 `torch.addmm` on four
`65536 x 65536` random matrices: two operands, one penalty matrix, and one
output matrix. It uses CUDA device 0 with `expandable_segments` allocation.

## Observed result

At launch verification, the workload held 32.0 GiB according to PyTorch and
NVIDIA System Management Interface reported 100% GPU utilization, 33,233 MiB
used VRAM, and 351.98 W board power.

## Stop command

```bash
kubectl exec -n ecepxie gpu-dev-l40s-1gpu-6fcb7f9bfc-vtlpj -- kill -TERM 7619
```
