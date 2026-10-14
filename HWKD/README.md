# Shared hardware knowledge component

`hwkg/` comes from `AYPatrL04/MLEvolve` branch `hwkd-impl`, commit
`a25c6413b5181cc9043cae1a150b16606137bc2f`. Its graph file retains
SHA-256 `ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.
See `hwkg/README.md` for the source policy and its evidence limits.

The installed package has one local compatibility change in
`hwkg/device_support.py`: a PyTorch device property whose `gcnArchName` is a
plain NVIDIA device name is treated as CUDA, not ROCm. The source branch used
any nonempty `gcnArchName` as a ROCm signal; on this host that produced
`unknown, ROCmNone` for an NVIDIA GeForce RTX 5070 Ti. The graph and other
37 source files are byte-identical to the pinned commit. The local repair has
a regression test in `tests/test_hwkd_import_path.py`.

## Enable for either agent

Set these environment variables before starting an agent. Use an absolute
path to the installed graph, the exact graph digest, and the graph profile
matching the *one device visible to that process*:

```bash
export MLEVOLVE_HWKG_GRAPH_PATH=/home/downeyflyfan/Research_Projects/AI/Agents/Agents_Scheduler/Lightning-Agent/HWKD/hwkg/graph.json
export MLEVOLVE_HWKG_GRAPH_SHA256=ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1
export MLEVOLVE_HWKG_HARDWARE_ID=nvidia.blackwell.geforce_rtx_5070_ti.spec
export MLEVOLVE_HWKG_EVIDENCE_POLICY=v8
```

The hardware identifier above is only for this workstation's RTX 5070 Ti.
Choose another identifier from `hwkg/SUPPORTED_HARDWARE.md` on another host.
ML-Master additionally requires `LIGHTNING_PLUGINS=hwkd`; MLEvolve uses the
graph-path setting directly. Unset `MLEVOLVE_HWKG_GRAPH_PATH` to disable the
treatment. No graph path means no device probe or injected advice. With a graph
path, a wrong digest or device fails before model generation.

The component requires exactly one visible graphics processing unit through
PyTorch's `torch.cuda` interface. The source package does not establish
multi-device topology, free-memory capacity, or a performance gain. Use a
PyTorch build that supports the visible GPU architecture; on this host, the
agents' Python 3.14 environments include `sm_120`, while the older root
`.venv` does not.
