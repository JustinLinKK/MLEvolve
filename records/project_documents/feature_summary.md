# MLEvolve Hardware-Aware Branch

## Logic-Only Analysis of Time-Saving Features and Slowdown Risks

> This report intentionally ignores all experiment results. It evaluates the current implementation only from its architecture and control logic.

## TL;DR

The branch follows a sensible overall strategy:

> Spend a relatively small amount of CPU, database, or LLM time to avoid much larger costs from failed GPU jobs, inefficient training, repeated debugging, poor scheduling, and duplicated reasoning.

The largest potential time reductions come from:

1. preventing invalid candidates from reaching the GPU;
2. early stopping and finite execution limits;
3. hardware-aware precision, batch-size, and dataloader choices;
4. safe parallel training-job packing;
5. reusing profiles, documentation, and lessons from earlier work;
6. repairing only the responsible pipeline stage instead of regenerating everything.

The largest slowdown risks come from:

1. enabling the scheduler while packing is disabled or runtime estimates are unavailable;
2. running cold batch probes and colocation trials for short jobs;
3. performing hardware retrieval, review, and repair for every candidate;
4. repeatedly adding long hardware, memory, and documentation sections to LLM prompts;
5. allowing long or unbounded candidate execution;
6. using inaccurate hardware guidance that produces a faster but lower-quality model and therefore requires more search.

## Rating legend

- **Very high:** Can materially change total task wall time.
- **High:** Important in common workloads.
- **Medium:** Useful under the right conditions.
- **Low:** Small or indirect effect.
- **Conditional:** Implemented, but disabled by default or dependent on evidence/configuration.

---

## 1. GPU execution and scheduling

| Feature | Contribution | Current state | Improvement potential | Slowdown risk | Rationale |
|---|---|---|---|---|---|
| Immediate pipeline submission | Direct | Active with scheduler | High | Low | A generated candidate is submitted immediately instead of waiting for several initial candidates. This reduces early GPU idle time. |
| Parallel training-job execution | Direct | Implemented; packing disabled by default | Very high | High | Multiple small jobs can share one GPU and reduce makespan. Heavy interference can make every job slower. |
| Time-aware queue ordering | Direct | Active but dependent on runtime estimates | High | Medium | Shorter remaining jobs can finish earlier, improving time-to-first-useful-result. Incorrect or missing estimates can produce poor ordering. |
| Incremental concurrency | Direct | Implemented | High | Medium | Starts with one job and admits newcomers individually instead of launching a large group blindly. Admission decisions add overhead. |
| Colocation slowdown verification | Direct | Implemented | High | High | Tests whether adding another job really improves combined completion time. The trial consumes GPU time and temporarily disturbs the running job. |
| Colocation profile reuse | Direct | Implemented | High when warm | Low | Reuses previously measured pair behavior instead of repeating live trials. Stale or incorrectly matched profiles can cause harmful placement. |
| Predicted VRAM admission | Direct | Implemented | High | Medium | Rejects combinations expected to exceed safe GPU memory, preventing OOM failures and retries. Conservative predictions can underutilize the GPU. |
| Live memory backpressure | Direct | Active | High | Low | Stops new admissions when measured memory approaches the limit and resumes later. It improves safety but may delay runnable jobs. |
| Canonical backend selection | Direct/indirect | Active | Medium | Low | Uses one authoritative process backend throughout scheduling and knowledge retrieval, avoiding contradictory backend decisions. |
| CUDA-process packing | Direct | Conditional | High | Medium | Runs independent CUDA subprocesses concurrently without relying on parent-controlled CUDA streams. Context switching and duplicated memory can reduce benefit. |
| NVIDIA MPS process packing | Direct | Conditional | High | High | MPS can improve process sharing and compute allocation. It adds service-management, compatibility, isolation, and deployment complexity. |
| Exclusive fallback | Reliability | Active | Medium | Medium | Runs jobs alone when evidence or backend support is insufficient. It prevents unsafe packing but may remove all concurrency gains. |
| CUDA-stream removal | Indirect | Active | Medium | Low | Removes a backend that cannot reliably control independent child-process CUDA work, reducing invalid scheduling and backend-specific debugging. |
| Five-option batch-size search | Direct | Active when a batch control is detected | High | High | Evaluates nearby power-of-two batch sizes to find a faster safe configuration. Cold probes can cost more than they save for short jobs. |
| Batch-probe cache | Direct | Implemented | High when warm | Low | Reuses a previously safe batch result instead of probing again. Reuse must match model, shape, hardware, software, and backend. |
| Model-family/startpoint profile reuse | Direct | Partially active | Medium | Medium | Related candidates can inherit conservative batch information from an earlier model. Broad family matching can produce poor settings. |
| Runtime-profile learning | Direct/enabling | Implemented | High | Medium | Learns per-epoch and total-runtime information for future scheduling. Cold operation may still depend on probes or exclusive fallback. |
| Batch-aware runtime prediction | Direct | Implemented | High | High | Compares estimated total training time across batch choices rather than simply maximizing batch size. Prediction error can select a slower choice. |
| Optional ML resource predictor | Direct/enabling | Conditional | High | High | Can estimate VRAM and runtime for unseen scripts without full profiling. Transfer error across GPU types, model structures, or datatypes can cause bad decisions. |
| Early stopping | Direct | Enabled but dependent on progress reporting | Very high | Medium | Stops candidates whose metric no longer improves. Poor patience settings or noisy metrics can terminate promising candidates prematurely. |
| Best-checkpoint preservation | Direct/reliability | Implemented | Medium | Low | Preserves the best state instead of losing a useful model after later degradation. Checkpoint I/O can slow training. |
| Runtime telemetry | Enabling | Active | Medium | Low–medium | Supplies utilization, memory, and progress evidence for safer decisions. Polling and logging consume some host and driver resources. |
| CPU allocation and subprocess isolation | Reliability | Active | Medium | Low | Reduces CPU oversubscription and CUDA/fork issues. Conservative CPU partitioning may leave host cores idle. |
| Per-candidate file isolation | Reliability | Active | Medium | Low | Prevents parallel jobs from overwriting each other's models and submissions, avoiding corrupted results and reruns. |

---

## 2. Hardware-aware code generation

| Feature | Contribution | Current state | Improvement potential | Slowdown risk | Rationale |
|---|---|---|---|---|---|
| Hardware-aware model selection | Direct/indirect | Active in hardware-aware mode | High | Medium | Encourages model families that fit and train efficiently on the current GPU. Overweighting hardware efficiency can hurt model quality and require more search. |
| Dedicated precision-design stage | Direct/indirect | Active in hardware-aware mode | High | Medium–high | Selects FP32, TF32, FP16, BF16, AMP, or supported Transformer Engine paths before training. It adds another specialized reasoning stage and can introduce compatibility failures. |
| Hardware-aware batch and accumulation design | Direct | Active | High | Medium | Chooses physical batch size and gradient accumulation with hardware limits in mind. Bad choices can reduce throughput or cause OOM. |
| Hardware-aware dataloader design | Direct | Active | Medium | Medium | Adjusts workers, pinned memory, persistent workers, and nonblocking transfers to reduce input stalls. Too many workers can create CPU/RAM contention. |
| Hardware-aware checkpoint cadence | Direct/reliability | Prompt-driven | Medium | Medium | Reduces lost work after interruption or timeout. Excessive checkpointing adds storage and synchronization time. |
| Generated OOM and timeout fallbacks | Indirect | Prompt-driven | High | Low–medium | Allows a candidate to retry with smaller batch, accumulation, lower resolution, or fewer epochs instead of creating another debug node. More fallback code increases complexity. |
| Hardware-aware model brief | Indirect | Active | Medium–high | Medium | Compares suitable model families before code generation, preventing obviously impractical designs. Retrieval and prompt construction add latency. |
| Stage ownership | Indirect | Active | Medium | Medium | Separates model design, precision, and training responsibilities, reducing conflicting choices. Coordination increases prompt size and model calls. |
| Cross-stage decision contract | Indirect | Active | Medium–high | Low–medium | Carries selected model, precision, and training intent across stages, reducing accidental redesign and inconsistent code. |
| Cross-stage note board | Indirect | Active | Medium | Low–medium | Records what changed and why so later stages preserve the same optimization target. It adds tokens to later prompts. |
| Hardware-specific static validation | Indirect | Active | High | Low | Detects invalid precision or hardware combinations before GPU execution. This is cheap compared with a failed training run. |
| Exact backend-aware knowledge filtering | Indirect | Active | High | Low–medium | Uses evidence from the effective backend, reducing failures caused by applying MPS-specific advice to ordinary CUDA processes or vice versa. Strict filtering can reduce available evidence. |

---

## 3. Bug prevention, debugging, and search efficiency

| Feature | Contribution | Current state | Improvement potential | Slowdown risk | Rationale |
|---|---|---|---|---|---|
| Pre-execution code review | Indirect | Active | Very high | High | Can prevent a long GPU job with metric, leakage, shape, precision, or integration errors. Every valid candidate still pays review latency. |
| Deterministic precision checks | Indirect | Active | High | Low | Detects hardware-policy violations without an additional model call. This has a strong overhead-to-benefit ratio. |
| Stage-aware issue classification | Indirect | Active | High | Medium | Assigns each issue to model design, precision, training, integration, or an unknown owner, avoiding whole-script regeneration. Classification uses an LLM. |
| Selective stage repair | Indirect | Active | High | Medium–high | Regenerates only the responsible stage. Multiple repair and re-review rounds can become expensive. |
| Parallel independent repairs | Indirect | Conditional | Medium–high | Medium | Repairs independent issues concurrently, reducing repair wall time. It increases request load and may create patch conflicts. |
| Critical-issue execution rejection | Indirect | Active | Very high | Medium | Prevents GPU execution when critical issues remain unresolved. False positives can reject a useful candidate. |
| Fail-open review behavior | Reliability | Active | Medium | Medium | A reviewer outage does not stop the search. It also allows potentially avoidable bad candidates to reach the GPU. |
| CUDA-aware debug routing | Indirect | Active when CUDA docs are enabled | High | Low | Sends likely CUDA incidents to CUDA documentation while excluding syntax, metric, dataset, and filesystem problems. |
| Debug-agent loop | Indirect | Active | High | High | Repairs failed candidates rather than abandoning all earlier work. Large debug-depth limits can repeatedly spend time on an unrecoverable branch. |
| Stagnation detection | Indirect | Active | High | Medium | Stops repeatedly applying ordinary improvements to an unproductive branch and switches strategy. Incorrect thresholds can switch too early or too late. |
| Top-candidate exploitation | Indirect | Active | Medium–high | Medium | Focuses more effort on candidates already showing promise, improving time-to-quality. It may reduce useful exploration. |
| Evolution after stagnation | Indirect | Active | Medium | High | Makes a larger change when incremental improvement stalls. The new generation and training may lose a stable baseline. |
| Cross-branch fusion | Indirect | Active | Medium | High | Reuses complementary ideas instead of rediscovering them. Fusion and validation are expensive and may introduce integration bugs. |
| Multi-branch aggregation | Indirect | Active | Medium | High | Consolidates knowledge from several branches. It is only worthwhile when the branches contain complementary evidence. |
| Metric-direction detection | Reliability | Active | High | Low | Prevents the search from optimizing in the wrong direction. |
| Exact metric and submission guidance | Indirect | Active | High | Medium | Reduces invalid solutions and wasted training. Detailed validation instructions increase prompt size. |
| Data-leakage checking | Indirect | Active | High | Medium | Prevents invalid high-scoring candidates from consuming later search steps. Review adds model latency. |

---

## 4. Knowledge retrieval and caching

| Feature | Contribution | Current state | Improvement potential | Slowdown risk | Rationale |
|---|---|---|---|---|---|
| Hardware knowledge graph | Indirect | Active in hardware-aware mode | High | Medium | Supplies GPU, CUDA, framework, precision, and backend facts so agents avoid incompatible designs. Graph lookup and startup warming add latency. |
| Hardware optimization vector retrieval | Indirect | Active when local database is available | High | Medium–high | Retrieves existing optimization knowledge rather than asking the LLM to rediscover it. Embedding and vector queries can add tail latency. |
| Hardware-context compaction | Indirect | Active | Medium | Low | Limits evidence before sending it to the LLM, reducing input tokens and irrelevant information. Excessive compaction can remove decisive facts. |
| Context storage on search nodes | Indirect | Implemented | Medium–high | Low | Allows later reasoning to reuse the candidate's evidence instead of retrieving it again. Frequent refreshes reduce the benefit. |
| Optional Redis query cache | Indirect | Implemented but disabled by default | Medium–high when warm | Low | Shares graph, vector, and documentation results across processes and runs. It produces no benefit while disabled. |
| In-run CUDA-document memo | Indirect | Implemented; service disabled by default | High when repeated | Very low | Reuses identical documentation inside one run without Redis or network access. |
| RAM TTL/LRU CUDA-document cache | Indirect | Implemented; service disabled by default | High when warm | Very low | Serves repeated CUDA guidance from local memory. |
| Redis CUDA-document cache | Indirect | Implemented but Redis disabled by default | Medium–high | Low | Shares verified documentation across runs and processes. Its value depends on repeated incidents with matching applicability. |
| Persistent CUDA-document store | Indirect | Conditional | Medium–high | Medium | Allows later runs to reuse curated NVIDIA evidence without a remote call. A synchronous local lookup still adds latency on a cache miss. |
| Negative caching | Indirect | Implemented | Medium | Low | Avoids repeatedly retrying malformed, unauthorized, irrelevant, or unavailable requests. Temporary failures can remain suppressed until expiry. |
| Stale-while-refresh | Indirect | Implemented | Medium | Low | Returns existing evidence immediately and refreshes it in the background. Stale guidance must remain version- and backend-gated. |
| Singleflight request suppression | Indirect | Implemented | Medium | Low–medium | Prevents many agents from requesting identical evidence simultaneously. Cold-miss waiters may receive no evidence when the short wait budget expires. |
| Global search memory | Indirect | Active by default | High | Medium–high | Reuses lessons from earlier candidates, reducing repeated mistakes. GPU-based embeddings can compete with training for memory and compute. |
| Cold-start model guidance | Indirect | Active by default | Medium–high | Medium | Directs initial candidates toward known model families instead of spending nodes on weak starts. A poor prior can slow convergence. |
| Pretrained-model cache | Direct | Implemented but effectively disabled in the current example | High for repeated models | Medium | Can avoid repeated model loading and initialization. RAM usage may reduce packing capacity. |

---

## 5. NVIDIA CUDA MCP integration

| Feature | Contribution | Current state | Improvement potential | Slowdown risk | Rationale |
|---|---|---|---|---|---|
| Debug-only blocking lookup | Indirect | Implemented; disabled by default | High | Medium | A verified CUDA answer can avoid several debug and retry cycles. Blocking is limited to eligible debug actions. |
| Nonblocking documentation prefetch | Indirect | Conditional | Medium | Low–medium | Fetches common topics before they are needed without blocking the agent. It consumes threads, network capacity, and rate-limit tokens. |
| Local-first lookup order | Indirect | Implemented | High | Low | Checks in-run, RAM, Redis, and persistent local evidence before NVIDIA, minimizing remote latency. |
| CUDA incident filtering | Indirect | Implemented | High | Very low | Prevents documentation calls for syntax, import, filesystem, data, or metric errors. |
| Source-code exclusion and error redaction | Reliability | Active | Low direct benefit | Very low | Avoids transmitting candidate code, paths, secrets, or workload identifiers, reducing security and deployment risk. |
| Hardware/software applicability keys | Reliability | Active | Medium | Low | Separates evidence by GPU, driver, CUDA, framework, and backend, avoiding failures from incompatible advice. |
| Persistent MCP session | Direct/indirect | Implemented | Medium | Low–medium | Reuses one connection and tool schema instead of reconnecting for every request. Authentication and discovery still cost time on a cold start. |
| Remote timeout | Reliability | Active when enabled | High | Low | Bounds the maximum delay from a slow NVIDIA service. A tight timeout may discard a useful response. |
| Rate limiter and circuit breaker | Reliability | Active when enabled | High | Low | Prevents repeated remote failures from delaying many agents. It can temporarily suppress useful calls. |
| Async recipe curation | Indirect | Conditional | Medium | Low–medium | Converts remote results into reusable structured knowledge without blocking the active agent. Queued work may be cancelled during shutdown. |

---

## 6. Measurement and operational support

| Feature | Contribution | Current state | Improvement potential | Slowdown risk | Rationale |
|---|---|---|---|---|---|
| Separate experiment modes | Enabling | Active | No direct speedup | Low | Separates original, scheduler-only, and hardware-aware behavior so slow components can be identified. |
| Pipeline event database | Enabling | Active | Medium indirectly | Low–medium | Makes agent, review, scheduler, and job timing traceable. Frequent writes add some I/O. |
| Comparison metrics | Enabling | Active | Medium indirectly | Low | Records wall time, execution, queueing, probing, packing, failures, and reviewer activity. It does not shorten the current run. |
| Hardware utilization monitoring | Enabling | Active | Medium indirectly | Low | Reveals GPU idle gaps and bottlenecks that can be optimized later. |
| Adaptive monitoring compression | Operational | Active | Low | Very low | Prevents long monitoring files from growing without bound. |
| Graceful cancellation | Reliability | Active | Medium | Low | Stops scheduler and subprocess work after shutdown instead of leaking GPU jobs. |
| Recoverable-job support | Reliability | Implemented but disabled by default | Medium | Low–medium | Can resume interrupted work instead of restarting from zero. Incorrect recovery state can create inconsistent results. |
| Backend/configuration migration | Reliability | Active | Medium | Low | Rejects retired or contradictory configurations before they waste GPU time. |

---

## 7. Major system-level slowdown risks

These are the most important logical risks when all components are combined.

| Risk | Degree | Why it can increase total time |
|---|---|---|
| Scheduler enabled while packing is disabled | Very high | The system pays scheduling, probing, telemetry, database, and service-management overhead while executing jobs exclusively. |
| Runtime-estimation inputs are incomplete | Very high | Time-aware ordering and batch comparison cannot operate reliably, causing exclusive fallback or poor decisions. |
| Candidate timeout is absent or unbounded | Critical | A stalled script can consume the entire allocation, and scheduler execution may expect a timeout value. |
| Long overall search budget | High | Faster individual candidates do not guarantee a shorter task if the search is allowed to run much longer. |
| Hardware retrieval for too many roles | High | Planner, coder, reviewer, repair, fusion, and improvement calls can all pay database and prompt costs. |
| Review and repair for every candidate | High | Valid candidates still pay classification and possible re-review before GPU execution. |
| Cold batch and colocation probes | High | Profiling can cost more than it saves for short candidates. |
| GPU-based memory embeddings | Medium–high | Retrieval may consume VRAM and compute concurrently with training. |
| Disabled model cache | Medium | Repeated loading and initialization cannot benefit from warm reuse. |
| Disabled Redis cache | Medium | Cross-process and cross-run reuse is unavailable. |
| Duplicate startup prewarming | Medium | Similar hardware context may be loaded more than once during initialization. |
| Synchronous vector lookup on cache miss | Medium | A slow local database can delay prompt construction even when remote access is nonblocking. |
| MCP prewarm and rate-limit mismatch | Medium | Several startup topics compete for a small burst and may not all be retrieved or retried. |
| Incomplete remote authentication lifecycle | Medium | The live service can remain unavailable when a pre-established token is missing or expires. |
| Prompt growth | Medium–high | Hardware evidence, CUDA docs, pipeline contracts, memory, previous code, and note boards all increase LLM input latency and cost. |
| Overconservative safety gates | Medium | OOM and contention are avoided, but the GPU may remain underutilized. |
| Incorrect hardware advice | High | A faster but lower-quality candidate may need additional search nodes to reach the required target. |

---

## 8. Recommended importance order

For reducing total time-to-useful-solution, the implemented mechanisms should be valued in this order:

1. **Prevent invalid GPU executions** through deterministic checks, review, critical rejection, and targeted repair.
2. **Stop unproductive work** through early stopping, finite candidate limits, total-run limits, and no-progress limits.
3. **Generate hardware-efficient training code** using appropriate precision, batch size, accumulation, and dataloader settings.
4. **Pack jobs safely** only when runtime, memory, and slowdown evidence are sufficiently reliable.
5. **Reuse profiles and caches** so cold probing and retrieval do not recur for every candidate or run.

6. **Use time-aware ordering** after runtime-estimate coverage becomes reliable.

7. **Use CUDA documentation mainly for eligible debug incidents**, with local caches checked before a remote request.
8. **Gate expensive search strategies** such as repair rounds, fusion, evolution, and aggregation according to expected benefit.
9. **Use observability to remove overhead**, even though observability itself does not directly accelerate the current job.

## Final assessment

The branch contains many valid time-saving mechanisms, including several indirect ones that reduce buggy-node generation, repeated debugging, invalid precision choices, OOM failures, and duplicated reasoning.

The central design principle should be:

```text
Run an optimization action only when:

probability of preventing failure or shortening training
    × expected saved GPU/debug time
    > action latency + additional GPU work
```

The system is most likely to become slower when every optimization layer is applied unconditionally. The strongest design is therefore an adaptive one: use cheap deterministic checks broadly, reuse warm evidence aggressively, and reserve expensive LLM review, live probing, colocation trials, and remote CUDA-document retrieval for candidates where their expected savings clearly exceed their overhead.
