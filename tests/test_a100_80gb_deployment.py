from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "deployments" / "mlevolve-a100-80gb.yaml"


def test_agent_can_use_full_80gb_a100s_but_cannot_fall_back_to_40gb_or_mig() -> None:
    deployment = yaml.safe_load(MANIFEST.read_text())
    pod_spec = deployment["spec"]["template"]["spec"]

    node_affinity = pod_spec["affinity"]["nodeAffinity"]
    assert "preferredDuringSchedulingIgnoredDuringExecution" not in node_affinity
    expressions = node_affinity["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ][0]["matchExpressions"]
    product_constraint = next(
        expression
        for expression in expressions
        if expression["key"] == "nvidia.com/gpu.product"
    )
    assert product_constraint == {
        "key": "nvidia.com/gpu.product",
        "operator": "In",
        "values": ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"],
    }

    resources = pod_spec["containers"][0]["resources"]
    assert resources["requests"]["nvidia.com/a100"] == "1"
    assert resources["limits"]["nvidia.com/a100"] == "1"
