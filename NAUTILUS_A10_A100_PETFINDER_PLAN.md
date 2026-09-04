# Nautilus A100-Agent / A10-Scheduler PetFinder Run

1. Verify the live A100 vLLM endpoint and the idle A10 execution pod.
2. Synchronize the current committed Scheduler code to the persistent A10 workspace without deleting prior artifacts.
3. Launch a new 30-node PetFinder Branch-Profile Scheduler run with no fixed parallel-job cap, dynamic physical-A10 telemetry, a bounded per-completion output budget, and an unbounded vLLM server context.
4. Monitor generation and execution; repair only demonstrated bugs, retaining logs and journals.
5. Compare equal valid-node counts against the recorded baseline, generate the required combined Gantt/metric figure, record the result, and push the experiment artifacts.
