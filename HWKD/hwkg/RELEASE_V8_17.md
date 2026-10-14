# v8.17: initial hardware context has its own provenance and count

When a native draft prompt lacks host identity, the component's own initial card
now states the visible device name, compute capability, installed Torch/CUDA
runtime, and total device memory. These values come from local runtime inspection,
not a preset dataset description. The compact precision and optimizer hypotheses
remain optional, unmeasured choices; they do not prescribe a model or recipe.

Audit fields separate header characters from advice characters and count both in
the total delivered context. Existing host identity still avoids repeated facts.
The original provider parameters, response, parser, retries, search, executor,
scoring, candidates, and baseline remain unchanged. The card is at most340chars.

Validation consists of425 offline component tests plus4 subtests, one separately
completed existing GPU identity test, and14 native/prompt checks. These fixtures
and identity checks establish compatibility, not task accuracy or performance.
The neutral CIFAR document is a regression fixture only. A new user-selected
Cassava public train/validation comparison is pending; its shared task must not
contain host-specific presets. Historical hardware-informed CIFAR baseline scores
cannot serve as a matched control for this new task/context experiment.

Public validation is available to both agents for model selection. Final review
must separately check training/validation overlap, label misuse, output integrity,
actual training and resource budgets. Adaptive public validation is not an
independent hidden-test score. No native v8.17 performance gain is claimed.
