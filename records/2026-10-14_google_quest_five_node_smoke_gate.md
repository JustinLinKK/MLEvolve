# Google Quest five-node ML-Master smoke: pre-run gate (2026-10-14)

## Requested settings

- Task: `google-quest-challenge`; five ML-Master nodes on the local NVIDIA GeForce RTX 5070 Ti.
- Code generation and feedback: authenticated DeepSeek API; `deepseek-flash` was available from the account's model listing and passed a live generation request and a live structured review tool call.
- Plugins: preflight, scheduler, A10-trained PerfSeer predictor, and hardware knowledge (HWKD). The predictor is a smoke-only cross-device estimate, not a calibrated RTX 5070 Ti performance claim.
- Preserve the competition's exact public files; do not substitute a mirror or dummy data for the five-node experiment.

## Verified component results

- Before the local graphics-processing-unit smoke, `nvidia-smi` showed 6 MiB used of 16,303 MiB and 0% utilization; the tiny `TinyNet` workload was safe to start.
- Hardware knowledge returned 298 characters of draft-stage guidance. Its Neo4j evidence mirror at `127.0.0.1:7687` was unavailable, so it used the SQLite primary.
- Preflight admitted the candidate and persisted a report.
- The A10-trained PerfSeer artifact returned all six targets: predicted train utilization 12.72%, train memory 795.56 MiB, train time 3.27; predicted inference utilization 10.81%, inference memory 784.44 MiB, inference time 1.22. These are predictions, not observed GPU measurements.
- The scheduler executed the candidate on the actual RTX 5070 Ti, returned `CUDA_OK`, and recorded 1.667 seconds of execution with no exception. It then stopped its service. Temporary smoke files were removed.
- The ML-Master predictor plugin requires explicit `LIGHTNING_PREDICTOR_ENTRY` and `LIGHTNING_PREDICTOR_INPUT_SHAPES`; the controlled `TinyNet` smoke supplied them. Arbitrarily generated Google Quest candidates do not yet have a guaranteed compatible model entry or shape. On continuation, the manager was changed to attach a successful six-target A10 prediction to the scheduler job's `perfseer_a10_advisory` metadata. A failing-before/passing-after integration test proves this handoff, and a second real-GPU smoke observed that metadata on the submitted job while preflight, hardware guidance, and CUDA execution all succeeded. The prediction deliberately does not alter hard resource admission because it is not calibrated to the RTX 5070 Ti. Thus this is a scheduler-visible advisory, not proof that predictions change placement decisions.
- Focused tests for the handoff and runner completion passed (`3 passed`). A broader scheduler bridge test could not be collected: its legacy import `engine.preflight` no longer exists after the repository move. This unrelated stale test was not repaired as part of the smoke gate.
- A runner completion regression test first reproduced an empty-futures wait after the final node, then passed after changing the main loop condition from `<= total_steps` to `< total_steps`. One-node and five-node cases both passed (`2 passed`). This fixes a hang that would otherwise follow a completed five-node run.

## Original five-node gate before Kaggle acceptance

- A no-training configuration probe validated `agent.steps=5`, serial search (`parallel_search_num=1` to avoid overlapping unknown training workloads on the 16 GiB card), both DeepSeek stages set to `deepseek-flash`, and explicit `data_dir`, `dataset_dir`, `start_cpu_id=0`, and `cpu_number=4`. The default `dataset_dir: null` fails structured configuration validation, so it must be overridden at launch. With a synthetic test key, OmegaConf resolved `${oc.env:DEEPSEEK_API_KEY}` for both stages while saving only the interpolation string, not the key. Temporary probe directories were removed.
- The authenticated Kaggle API listed the three public competition files, but downloading `google-quest-challenge` returned HTTP 403. A read-only competition listing reports `user_has_entered=False`; Kaggle's competition data page requires the signed-in account to agree to its rules before data access. No exact train/test/submission CSVs were found in the local AI project tree.
- Expected public MD5 checksums in MLE-Bench metadata: `train.csv` `f41aa13133d70b94bfa53801762937aa`; `test.csv` `06f482b597547c3c975ff1fc59314011`; `sample_submission.csv` `ba75781e8b1a5936533c4de82e515c98`.
- At that point, ML-Master required a real `data_dir`, so no five-node Google Quest run had been launched. The account holder subsequently accepted the rules and the run below used the directly downloaded files.

## Resumption after Kaggle acceptance

- The authenticated Kaggle listing now reports `user_has_entered=True`. The official Kaggle command downloaded `google-quest-challenge.zip` (4.85 MiB) into `.cache/google-quest-challenge`; the archive contains exactly `train.csv`, `test.csv`, and `sample_submission.csv` and extracted without errors.
- Parsed row counts are 6,079 training examples, 476 test examples, and 476 sample submissions. The downloaded files' MD5 values are `186d65e7e0c82b6f8180b4543cbc7109`, `ef6a35b0a7416dafdc19c94071bf2cd3`, and `30c6536124830baa5bd6692500ad151b`, respectively. These differ from the historical MLE-Bench manifest above; the files are used as directly downloaded from the authorized Kaggle competition, not silently replaced or claimed to match that manifest.
- Immediately before launch, the local RTX 5070 Ti reported 6 MiB of 16,303 MiB used and 0% utilization; no existing `main_mcts.py` process was found. The available central-processing-unit affinity includes cores 0–19, so the planned four-core allocation is valid.

## Five-node execution and results

- Run `google_quest_5node_20261014` used five serial search nodes; `deepseek-flash` handled code generation and feedback, with the preflight, scheduler, predictor, and HWKD plugins enabled. The script timeout was 1,800 seconds per node. It copied the official input files into its run workspace, leaving the downloaded originals unchanged; post-run MD5 values matched the pre-run values above. The saved configuration contains only `${oc.env:DEEPSEEK_API_KEY}` references, not the credential.
- Preflight admitted all five generated candidates with empty diagnostic lists. The scheduler dispatched five exclusive jobs, recorded three completed worker scripts and two failed worker scripts, and shut down its service after the run. Hardware knowledge guidance for the RTX 5070 Ti appears in the model prompt; its optional Neo4j mirror was unavailable and SQLite primary was used. The A10-trained predictor was enabled, but none of these arbitrary generated scripts provided a known model entry and input shape, so it warned and made no per-node prediction. The prior controlled `TinyNet` predictor-to-scheduler GPU smoke remains the evidence for that integration.
- Node `18a5d500`: transformer loading failed before training because local `transformers` 5.7.0 met incompatible `kernels` 0.15.2 (`Either a revision or a version must be specified`). The version mismatch is consistent with [Hugging Face issue #46291](https://github.com/huggingface/transformers/issues/46291). Installed `kernels` 0.12.3 only in the ML-Master virtual environment and verified `BertModel` imports afterward; no system package was changed.
- Node `5c95e49b`: the transformer's attempted monkey patch did not cover the later `FuncRepository` error. A fallback TF-IDF/Ridge model produced a CSV and local validation Spearman 0.2979, but the agent classified the node as buggy and discarded its metric.
- Node `0abf63c4`: transformer import passed after the dependency repair, but DataLoader workers could not pickle a locally defined `QADataset`. A fallback TF-IDF/Ridge model produced a CSV and local validation Spearman 0.3081; the agent again classified the node as buggy and discarded its metric.
- Node `9262073b`: generated transformer training failed with multiprocessing forkserver `ConnectionResetError`; no submission was produced. The immediate cause beyond that traceback was not established.
- Node `1afef2c2`: a CPU-based term-frequency–inverse-document-frequency (TF-IDF) plus Ridge model completed. Its grouped local hold-out mean column-wise Spearman correlation was 0.28249. Its 476-row, 31-column submission matches the official sample's column order and question IDs; all numeric outputs are finite and within [0,1]. This is a local validation result, not a Kaggle hidden-test score.
- Measured GPU memory during candidate execution remained near 234 MiB with 0% utilization; these five generated candidates did not demonstrate sustained GPU training. The separate controlled `TinyNet` CUDA smoke did prove execution on the RTX 5070 Ti.
- After the fifth node had been saved, the main process exited with code 1 while printing the search tree: the virtual root's missing metric was formatted as a float. Its best solution, submission, worker results, and logs remained intact, and no runner process was left. Updated both tree renderers to label missing metrics `unscored`; a new regression test plus the existing runner/predictor tests passed (`4 passed`). The five nodes were not rerun solely to reprint the tree.

## Conclusion

The local end-to-end search, DeepSeek calls, preflight admission, scheduler dispatch, hardware guidance, and valid final submission worked, with the caveats above. The five-node experiment did not establish per-node A10 predictor applicability or actual GPU training. Candidate failures and the search-tree cleanup error are preserved as experiment abnormalities, not hidden by the successful final CSV.
