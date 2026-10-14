# Upstream source

- Repository: https://github.com/sjtu-sai-agents/ML-Master
- Branch: `main`
- Snapshot commit: `6a7bf550074d09162fa955adcc4ed81017c0e697`
- Retrieved by Git over SSH on 2026-10-09.

The 123 files at that commit were imported into this directory. Changes under
`plugins/`, the integration hooks in `main_mcts.py`, and the static submission
name helper in `interpreter/interpreter_parallel.py` are local Lightning Agent
changes. The four component sources live in the parent repository root; only
small adapters live in this agent's `plugins/` directory. This directory is
vendored so those changes are tracked by the parent repository.
