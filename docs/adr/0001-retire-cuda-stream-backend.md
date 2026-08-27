# ADR 0001: Retire the CUDA-stream scheduler backend

Status: accepted on 2026-08-25.

MLEvolve generated scripts execute as independent child subprocesses. Each
child owns a CUDA context, so a CUDA stream selected in a scheduler parent
cannot govern the child's GPU work. The former `stream`, `cuda_stream`,
`mps_stream`, and `stream_mps` identifiers therefore described an execution
contract the generated-script runner could not provide.

Production execution now has two packed modes: `cuda_process` and
`mps_process`. Both launch one OS process and CUDA context per job;
`mps_process` connects those processes to scheduler-owned NVIDIA MPS. An
`exclusive` run remains available for solo profiling and an explicit safety
fallback. The configured packed mode is selected once and is carried through
planning, profile identity, knowledge retrieval, and prompt assembly.

Historical records using retired identifiers remain readable, but migration
marks their profiles non-selectable. They are never rewritten as process
profiles. Legacy `mps` records are normalized only when stored launch metadata
proves the MPS runtime was used.

Reconsidering in-process streams would require a new runner ABI with structured
training callables in one CUDA context, plus explicit allocator lifetime,
synchronization, isolation, and shared-failure-domain rules. That is a separate
architecture, not a compatibility wrapper around subprocess jobs.

References: [NVIDIA MPS architecture](https://docs.nvidia.com/deploy/mps/architecture.html),
[NVIDIA MPS deployment](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html),
and [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).
