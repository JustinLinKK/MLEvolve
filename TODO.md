# TODO

## Agent System Plugin

- Feedback to Agent through MCP

- Try all kinds of agent systems: **MLEvolve**, 

## Graph DB and Data accumulation on Cluster

- Test on different hardwares

## Pack Size Limit Setting

## Log Database

- combination(tasks) -> Preheat time -> Stay time in GPU -> When out

- tasks: probe method, others ...

## Data Collection

> SQL DB 

- Overall prompt: 

```yaml
recommended_patterns:
  - Use torch.amp autocast.
  - DataLoader pin_memory=True; .to(device, non_blocking=True)
  - torch.set_float32_matmul_precision("high")
  - Keep batch size, precision, accumulation in top-level config
avoid_patterns:
  - Do not hard-code unsupported precision modes.
```

- one chunk, one `yaml`
