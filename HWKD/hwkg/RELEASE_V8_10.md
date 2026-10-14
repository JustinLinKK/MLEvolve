# HWKD v8.10 counter-source validation

Component-only repair based on two actual native source/log failures: tuple-bound
batch was unavailable for counter arithmetic, and FP16 GradScaler attempts with
a constant zero skip count appeared eligible for matched-work projections.
The parser resolves only unambiguous declarative values; scaler counter evidence
is conservatively screened and remains unverified self-report. No baseline,
candidate, API/request/SDK, parser, execution, scoring or search changes.

Calibration bundles are unchanged. Validation results are recorded locally in
local-tests/hwkg-v810-component-20261013/VALIDATION.json. Native performance
validation for v8.10 is pending at publication. v8.9 usability recovered1/4
CUDA candidates, accuracy80.50%; all advice was0chars and no HWKD gain was shown.
The fix protects evidence quality, and is not a generation or speedup claim.

Verified before publication:275 component tests+4subtests,12 native compatibility
tests,194 protected hashes,76 exact upstream files and12 equivalent hooks pass.
Actual source/log replay confirms the two failures are guarded. No performance
advantage is claimed.
