# Arch Sensitivity Summary

Short sensitivity check for co-locating two training jobs.

| Config ID | Composition | Mode | Total wall time (s) | Replay elapsed (s) | Completed jobs | Packed dispatches | Speedup vs same-composition serial |
|---|---|---|---|---|---|---|---|
|  | same exact model | serial | 29.473 | 27.979 | 2 | 0 | - |
|  | same arch, different models | serial | 25.395 | 23.981 | 2 | 0 | - |
|  | different arch | serial | 27.318 | 25.984 | 2 | 0 | - |
|  | same exact model | packed stream | 17.312 | 16.015 | 2 | 1 | 1.70x |
|  | same arch, different models | packed stream | 17.313 | 15.985 | 2 | 1 | 1.47x |
|  | different arch | packed stream | 17.280 | 15.977 | 2 | 1 | 1.58x |

Interpretation:
- : two copies of 
- :  + 
- :  + 

Packed cases use  +  with batch search disabled so the comparison stays focused on co-location sensitivity rather than batch-size tuning.
