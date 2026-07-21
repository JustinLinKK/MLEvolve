# Environment

- Branch: hardware-awared
- Commit: 6974d974b0262dd67893249f0cca275e60342bdc
- Codex CLI: codex-cli 0.144.5 at /home/vscode/.local/bin/codex
- Python/Torch:
```
python 3.12.13
executable /opt/conda/bin/python
mp_start_method None
torch 2.12.0+cu130
cuda_available True
torch_cuda 13.0
device_count 1
device0 NVIDIA GeForce RTX 5090
```
- GPU:
```
Mon Jul 20 21:42:12 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.47        CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:01:00.0  On |                  N/A |
|  0%   45C    P5             27W /  575W |    4301MiB /  32607MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
```
- Services:
```
mlevolve-neo4j-profile-1	Up About an hour (healthy)	0.0.0.0:7474->7474/tcp, [::]:7474->7474/tcp, 0.0.0.0:7687->7687/tcp, [::]:7687->7687/tcp
mlevolve-neo4j-hardware-1	Up About an hour (healthy)	0.0.0.0:7475->7474/tcp, [::]:7475->7474/tcp, 0.0.0.0:7688->7687/tcp, [::]:7688->7687/tcp
mlevolve-qdrant-1	Up About an hour	0.0.0.0:6333-6334->6333-6334/tcp, [::]:6333-6334->6333-6334/tcp
awesome_leavitt	Up 59 minutes
```
- MPS control binary:
```

```
