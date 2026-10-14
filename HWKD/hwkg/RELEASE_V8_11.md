# HWKD v8.11 shared precision aliases

Component-only repair from actual frozen native source: shared AMP_DTYPE changes
affect both train and evaluation despite unchanged autocast call syntax. Scoped
alias signatures and conservative dtype resolution now expose the difference.
Feedback excludes evaluation precision changes from source-only time/quality
comparison eligibility. General binding rules remain conservative under dynamic
values and shadowing; no candidate execution/evaluation/API changes.

Native performance retest is pending at publication. Calibration bundles remain
unchanged. v8.10 isolated residency/fused diagnostics found local step gains,
but no projected net saving after the declared5s context reserve; task quality
and full-pipeline gains remain unverified. No general API fix or unmeasured
hardware performance improvement is claimed.

Verified before publication:292 componenttests+4subtests,12 native compatibility
tests,194 protected hashes,76 exact upstream files and12 equivalent hooks pass.
17 new regression cases include actual-source shared alias, train-only alias,
scaler/confounded feedback and preserved node fields.
