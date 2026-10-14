# MLEvolve Agent Workflow

This document explains, step by step, how MLEvolve actually runs during an experiment.

The important thing to keep in mind is that MLEvolve is not a single monolithic agent. It is a search controller plus several specialized LLM-driven stage agents that cooperate over a shared search tree. The main coordinator lives in [run.py](../run.py) and [engine/agent_search.py](../engine/agent_search.py).

## Main Components

- [run.py](../run.py): bootstraps the run, starts the executor, and drives the parallel search loop.
- [engine/agent_search.py](../engine/agent_search.py): central search controller that decides which stage agent to call next.
- [engine/search_node.py](../engine/search_node.py): defines the `SearchNode` tree structure and the `Journal`.
- [engine/node_selection.py](../engine/node_selection.py): chooses where the search should expand next.
- [engine/evaluation.py](../engine/evaluation.py): decides whether a node should keep improving, backpropagate reward, or terminate.
- [engine/executor.py](../engine/executor.py): runs generated code either as a subprocess or through `localml_scheduler`.
- [agents/result_parse_agent.py](../agents/result_parse_agent.py): turns execution logs into structured outcomes such as `is_buggy`, `metric`, and `analysis`.

## Step-By-Step Execution

### 1. Load config and initialize the runtime

The run starts in [run.py](../run.py).

At startup, MLEvolve:

1. Loads the merged config with `load_cfg()`.
2. Sets `torch.hub` if `torch_hub_dir` is configured.
3. Sets the global random seed with `set_global_seed(...)`.
4. Starts structured logging.
5. Starts the hardware monitor so resource usage can be reported even if the run is interrupted.

If `SIGTERM` arrives, MLEvolve performs a controlled shutdown so logs and hardware reports are still written, then exits with the conventional signal exit code.

### 2. Load the task and optional cold-start guidance

Still in [run.py](../run.py), the system:

1. Loads the competition task description with `load_task_desc(cfg)`.
2. Optionally builds cold-start model guidance through [engine/coldstart](../engine/coldstart) when `cfg.coldstart.use_coldstart` is enabled.

This guidance is later injected into prompts so the drafting agent can start from stronger pretrained-model assumptions when appropriate.

### 3. Prepare the workspace

MLEvolve then calls `prep_agent_workspace(cfg)` to build the per-run working directory under `runs/...`.

That workspace becomes the shared location for:

- copied task files
- generated `runfile_*.py` execution scripts
- submission CSVs
- saved best solutions
- top-K candidate snapshots
- optional scheduler runtime artifacts

If the run exits before any real node is completed, the workspace cleanup handler removes the empty run directory.

### 4. Build the search state

The search state is created by constructing:

1. A fresh `Journal` from [engine/search_node.py](../engine/search_node.py)
2. An `AgentSearch` object from [engine/agent_search.py](../engine/agent_search.py)

`AgentSearch` creates a `virtual_root` node with stage `"root"`. This node is not a real solution. It is the anchor for the whole search tree.

`AgentSearch` also initializes several important pieces of state:

- `branch_all_nodes`: every node grouped by branch
- `branch_successful_nodes`: only successful nodes grouped by branch
- `top_candidates`: a global top-K shortlist
- `best_node`: the current best solution
- `global_memory`: optional retrieval memory used by some planning and debug paths

During initialization, [agents/result_parse_agent.py](../agents/result_parse_agent.py) is also asked to determine whether the competition metric should be maximized or minimized. That decision is reused for every future node.

### 5. Build the execution backend

MLEvolve creates an `Interpreter` from [engine/executor.py](../engine/executor.py).

The interpreter can run code in two modes:

- Local subprocess mode: generated Python is written to `runfile_<slot>.py` and executed directly.
- Scheduler bridge mode: generated Python is submitted as a `localml_scheduler` job through [localml_scheduler/client.py](../localml_scheduler/client.py).

In both modes, the executor isolates filenames to avoid collisions across parallel runs. For example, generic `submission.csv` and `best_model.pth` paths are rewritten to node-specific names before execution.

### 6. Phase 1: generate initial drafts before executing them

The outer loop in [run.py](../run.py) begins with a special bootstrap phase.

Instead of generating one draft and immediately executing it, MLEvolve first creates several root drafts with `execute_immediately=False`.

This works like this:

1. `run.py` calls `agent.step(..., execute_immediately=False)`.
2. `AgentSearch.step()` asks [engine/node_selection.py](../engine/node_selection.py) to choose a node.
3. At the beginning of the run, the selected node is the `virtual_root`.
4. `AgentSearch._run_single_step()` dispatches to [agents/draft_agent.py](../agents/draft_agent.py).
5. The draft agent produces a plan and a full Python solution.
6. The new node is marked `pending_execution=True` and returned without running it yet.

This phase exists to increase early diversity. MLEvolve wants multiple different initial directions before exploitation starts.

### 7. How a draft node is created

The draft path in [agents/draft_agent.py](../agents/draft_agent.py) builds the first complete solution for a branch.

Its prompt is assembled from:

- the task description
- the current data preview
- memory of earlier attempts from the root
- a pipeline decision trace that supports the hardware-aware step order `model_design -> datatype_precision -> training_evaluation`
- implementation guidelines
- leakage-prevention instructions
- optional cold-start model guidance
- optional hardware/profile context, used as evidence for compatible tuning rather than as a task override

Depending on config, draft generation can happen in one of two ways:

- direct plan-and-code generation
- stepwise multi-agent generation through the stepwise coder

After generation:

1. A `SearchNode(stage="draft")` is created.
2. `register_node(...)` in [agents/triggers.py](../agents/triggers.py) assigns a new `branch_id`.
3. The original prompt is serialized into `node.prompt_input`.
4. The compact decision is persisted on `node.pipeline_decision`.
5. The node becomes the first real node in that branch.

### 8. Code review runs before execution

Before any new node is executed, [agents/code_review_agent.py](../agents/code_review_agent.py) reviews the code.

Its job is not to redesign the whole solution. It is a guardrail pass that looks for issues such as:

- metric misuse
- leakage risks
- missing or unsafe implementation details

If diff mode is enabled, the reviewer returns SEARCH/REPLACE patches and MLEvolve applies them with the diff patcher. If the review is clean, the original code is kept.

### 9. Phase 2: execute pending drafts and start pipelined search

After the initial draft set is ready, [run.py](../run.py) enters the parallel phase using `ThreadPoolExecutor`.

Two kinds of tasks now run together:

- execution of the pending draft nodes
- full `agent.step(...)` calls that generate and execute additional nodes

Pending draft nodes go through `AgentSearch.execute_deferred_node(...)`, which:

1. calls the executor
2. parses the result
3. validates the executed node
4. appends it to the journal
5. updates best-solution tracking

After each completed future, `run.py` immediately schedules the next task so the thread pool stays full until the step budget is exhausted.

### 10. How MLEvolve chooses the next node to expand

Node selection happens in [engine/node_selection.py](../engine/node_selection.py).

MLEvolve does not use a fixed policy for the whole run. It switches between exploration and exploitation.

The selection flow is:

1. Compute how far through the time budget the run is.
2. Use a soft switch to choose between:
   - exploration through UCT over the tree
   - exploitation through a weighted sample from the global top-K nodes
3. Use piecewise decay to reduce the exploration constant over time.
4. Respect per-node child limits for drafts, improves, and debug attempts.

Later in the run, high-performing branches receive more focused attention through Top-K exploitation, but the search can still fall back to UCT when needed.

### 11. How MLEvolve decides which stage agent to call

Once a parent node is selected, [engine/agent_search.py](../engine/agent_search.py) decides which specialized stage agent should act next.

The dispatch logic is:

1. If the parent is the root and still has room, call the draft agent.
2. If the parent is the root and draft expansion is saturated, optionally call the aggregation agent.
3. If the parent node is buggy or invalid, call the debug agent.
4. If the parent is successful but the branch is stagnant, choose evolution or fusion.
5. Otherwise, call the normal improve agent.

The available stage agents are:

- [agents/draft_agent.py](../agents/draft_agent.py): create a brand-new branch seed
- [agents/improve_agent.py](../agents/improve_agent.py): make a focused improvement to a successful node
- [agents/debug_agent.py](../agents/debug_agent.py): repair a buggy node
- [agents/evolution_agent.py](../agents/evolution_agent.py): improve using the branch’s own historical trajectory
- [agents/fusion_agent.py](../agents/fusion_agent.py): borrow ideas from other successful branches
- [agents/aggregation_agent.py](../agents/aggregation_agent.py): create a new root-level fusion draft from multiple good branches

Every code-producing stage receives the pipeline decision contract when
`agent.pipeline_decision_enabled=true`. Child nodes inherit their parent's trace
by default, and stages that reconsider the solution generate and persist an
updated trace before writing code.

### 12. What each stage agent actually uses as context

The stage agents differ mainly in what context they expose to the model.

Draft agent:

- task description
- data preview
- root memory of existing drafts
- optional cold-start guidance

Improve agent:

- parent code
- parent execution output
- sibling attempt memory from the same parent
- parent pipeline decision, updated only where execution feedback or the new plan contradicts it
- plateau-aware prompting
- optional diff-based implementation

Debug agent:

- full buggy code
- execution traceback and terminal output
- root-cause analysis from the previous parse step
- optional similar successful fixes from global memory

Evolution agent:

- everything the improve agent sees
- plus the branch’s root-to-current trajectory so it can learn from what already worked and failed

Fusion agent:

- current solution
- one or more strong solutions from other branches
- instructions to adapt useful ideas selectively rather than copy them blindly

Aggregation agent:

- top branch representatives from multiple successful branches
- instructions to synthesize a fresh root-level solution instead of merely editing an existing branch

### 13. How generated code is executed

Execution is handled by [engine/executor.py](../engine/executor.py).

For each run:

1. The code is rewritten to use node-specific submission and checkpoint filenames.
2. CPU affinity is assigned to the execution slot when supported by the OS.
3. The code is written to a temporary `runfile_<slot>.py`.
4. The script is executed either:
   - directly as a subprocess, or
   - through `localml_scheduler` when the scheduler bridge is enabled

In scheduler mode, the executor also prepares scheduler-side metadata such as:

- resource requirements
- optional batch probing
- optional runtime probing
- packing eligibility
- node metadata for tracing

The executor returns a structured `ExecutionResult` containing terminal output, execution time, exception type, and exception details.

### 14. How execution results are parsed

After execution, [agents/result_parse_agent.py](../agents/result_parse_agent.py) converts raw logs into structured search state.

This step:

1. Stores stdout and stderr on the node.
2. Asks an LLM to summarize the run and extract:
   - whether it failed
   - the validation metric
   - a short analysis
   - optionally a code summary for memory
3. Checks whether the expected submission file exists.
4. Validates submission format and content quality.
5. Verifies that the returned metric direction matches the precomputed maximize/minimize decision.
6. Optionally runs a data leakage check when a metric is suspiciously perfect.
7. Saves successful nodes into global memory when that feature is enabled.

If any of these checks fail, the node is marked buggy and its metric is reset to the worst possible value.

### 15. Extra post-execution validation

After parsing, [engine/execution.py](../engine/execution.py) performs extra rule-based validation.

Examples:

- If the submission file does not exist, the node is buggy.
- If the task is maximize-style and the metric is exactly `0.0`, the node is treated as a failure that likely needs debugging.

Successful executed nodes are also registered into `branch_successful_nodes` for later fusion, aggregation, and branch ranking.

### 16. How MLEvolve decides whether to keep improving a branch

This decision is handled in [engine/evaluation.py](../engine/evaluation.py).

For each executed node, MLEvolve checks:

1. Did a debug attempt fix a previously buggy parent?
2. Did the new metric improve on the branch’s local best?
3. Has the branch failed to improve too many times in a row?
4. Is it time to force backpropagation because the search is late in its time budget?

Based on those checks, the node is marked as one of:

- continue improving
- backpropagate reward and return control upward
- terminal, meaning this branch should stop expanding from here

This is what makes the search tree adaptive rather than a simple linear chain of edits.

### 17. How rewards are propagated through the tree

When a node should backpropagate, [engine/evaluation.py](../engine/evaluation.py) pushes reward back through its ancestors.

Backpropagation updates:

- `visits`
- `total_reward`
- draft locks
- debug-success flags
- branch continuation state

These values are later used by UCT selection, so every completed node influences future search decisions.

### 18. How the best solution and top-K shortlist are maintained

[engine/solution_manager.py](../engine/solution_manager.py) maintains the current best outputs.

After a valid executed node finishes:

1. It may enter the global top-K pool.
2. Per-branch diversity limits are applied so one branch does not dominate the shortlist.
3. If the node beats the current best, its code and submission are copied to:
   - `best_solution/`
   - `best_submission/`
4. The broader top-K artifacts are copied into `top_solution/top1`, `top_solution/top2`, and so on.

This makes the final run directory directly inspectable even before the full journal is opened.

### 19. How branch-level strategies change over time

MLEvolve becomes more strategic as the run progresses.

Some important adaptive behaviors are:

- exploration constant decay in UCT
- soft switching from global exploration to Top-K exploitation
- patience counters that detect branch stagnation
- evolution mode when the system should learn from a branch’s own history
- fusion mode when it should borrow ideas from other successful branches
- aggregation mode when root-level diversity has produced enough strong branches to synthesize a new one

This is why the system behaves differently early, mid, and late in a run.

### 20. How the run ends

The outer loop in [run.py](../run.py) continues until the configured step budget is reached.

After that, MLEvolve:

1. saves the journal state with `save_run(...)`
2. prints a tree view of the journal
3. shuts down executor resources
4. stops the optional scheduler service
5. stops the hardware monitor

If the run is interrupted, subprocesses and scheduler jobs are explicitly terminated so the workspace is left in a clean state.

### 21. Suggested next step: hardware-aware training optimization

The next high-value improvement is to let MLEvolve use hardware knowledge before it writes or edits training code, not only after code is executed.

The full report is now split into [mlevolve_hardware_aware_optimization.md](./mlevolve_hardware_aware_optimization.md). It covers the scheduler graph MCP feedback loop, the Qdrant hardware feature vector database, the combined `HardwareOptimizationBrief`, stage-specific agent behavior, and the read-only MCP tools that expose this context to other agent frameworks.

## One-Node Lifecycle Summary

A single node usually moves through this exact lifecycle:

1. A parent node is selected.
2. A stage agent generates a new plan and code.
3. The node is registered into the tree and a branch.
4. Code review optionally patches the code.
5. The executor runs the code.
6. The parse agent converts logs into `metric`, `analysis`, and `is_buggy`.
7. Validation checks the submission and metric consistency.
8. Evaluation decides whether to continue, backpropagate, debug, fuse, evolve, or terminate.
9. The journal, best solution, and top-K shortlist are updated.

## Mental Model

The easiest way to think about MLEvolve is:

- `run.py` is the conductor.
- `AgentSearch` is the search brain.
- stage agents propose code changes.
- the executor tests those changes in the real environment.
- the parse and evaluation modules convert outcomes into search feedback.
- the journal and branch structures let the system remember what it has already tried.

That closed loop is the core of how MLEvolve improves solutions step by step instead of generating only a single script.
