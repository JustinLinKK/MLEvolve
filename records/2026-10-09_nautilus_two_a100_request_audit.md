# Nautilus two-A100 request audit (2026-10-09)

## Requested settings

- Name: `Lightning-Agent-2-A100` (Kubernetes resource name would be `lightning-agent-2-a100`).
- Resources: two A100 graphics processing units (GPUs), based on `Research_Projects/AI/Files/Server/nautilus-default.yaml`.
- Proposed workload: indefinite synthetic matrix multiplication, with a ten-minute rest every hour, to occupy GPUs rather than complete an experiment.

## Verified result

- No Pod or Job was submitted by this audit. Local `kubectl` still requires interactive re-authentication, but `kubectl` over the `Justin-Linux` Secure Shell alias successfully reaches Nautilus context `nautilus`, namespace `ecepxie`.
- The remote read-only query returned `NotFound` for `lightning-agent-2-a100`. Namespace A100 request quota was `2/4`, leaving two quota units at the time of the query. This does not prove that any specific node has two free A100 GPUs.
- The `Justin-Linux` Secure Shell (SSH) alias is reachable, but resolves to host `Justin-WS`. On that host, `nvidia-smi` and `lspci` are absent. A direct read of `/sys/bus/pci/devices/*/vendor` found no NVIDIA (vendor `0x10de`) devices; `/dev/nvidia[0-9]*` has zero entries and `/proc/driver/nvidia/gpus` is absent. There is no verified A100 available there; no workload was started.
- The default template is an interactive Pod with `sleep infinity`, one generic GPU request, and `opportunistic2` priority. It does not itself specify a two-A100 computation.
- The current [Nautilus cluster policy](https://nrp.ai/documentation/userdocs/start/policies/) says to release reserved resources when computation is done, allows an uncontrolled interactive Pod for at most six hours, bans Jobs that end in sleep, and permits multiple GPUs only when actual work can use them. Indefinite synthetic activity solely to reserve two A100s is not a compliant research workload.

## Conclusion

Do not submit an occupancy workload. A real, bounded two-GPU experiment needs a defined computation and renewed Nautilus authentication; it should release both GPUs upon completion.
