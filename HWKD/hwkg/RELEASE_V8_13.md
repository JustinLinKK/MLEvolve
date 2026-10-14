# HWKD v8.13 feedback work-scope exclusion

Selected recipe syntax missed a standalone training-window length change in
saved public source. Synthetic time/metric regression then blocked a hardware
profile even though the compared work was unmatched. Trial-memory now excludes
known standalone gradient-window/epoch-reset sources on either side from
its time/quality comparison. Source scope and comparison exclusions are audited
separately from hardware suppression reasons. Failure and precision warnings
remain independent; unknown/embedded syntax still cannot certify useful work.

No native node/reward, generated source, scoring/search, API/SDK/parser/executor
or shared adapter changes. The scope screen is narrow, not a dataset/epoch-budget
or runtime grader. Calibration bundles unchanged; no new task/microbenchmark
scores. Native scored performance validation remains pending at publication.

Validation: 326 component tests plus 4 subtests, 14 targeted feedback regressions
and 12 native compatibility checks passed. Protected194/upstream76/equivalent12
hook checks and unchanged public input/model identities are checked separately
before publication. Synthetic regression values are never task measurements.
