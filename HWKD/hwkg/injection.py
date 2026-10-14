"""A small, opt-in HWKG prompt adapter for the upstream agents."""

import ast
import hashlib
import importlib.util
import json
import os
import re
import textwrap
from pathlib import Path


_FEATURE_IDS = ("feat:amp", "feat:fp16", "feat:bf16")
DEFAULT_POLICY = "v8"
_MAX_CHARS = 1600
_TOKENIZER_PACKAGES = ("sentencepiece", "tiktoken")
_IMAGE_TASK = re.compile(r"\b(?:images?|photos?|pictures?|jpe?gs?|pngs?|vision)\b", re.I)
_LANGUAGE_MODEL_TASK = re.compile(
    r"\b(?:language model(?:ing)?|next[- ]token|autoregressive|token prediction)\b", re.I
)
_LONG_TEXT = re.compile(r"\b(?:documents?|corpus|sequences?|long context)\b", re.I)
_TEXT_CLASSIFICATION = re.compile(
    r"\b(?:tweets?|sentiment)\b|text.{0,40}classif|classif.{0,40}text", re.I
)
_EVIDENCE_SOURCES = {
    "padding": "https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/data_collator",
    "transfer": "https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
    "precision": "https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html",
    "validation": "https://scikit-learn.org/stable/modules/classification_threshold.html",
}


def _evidence_guidance(stage, task_description, candidate_code, device, missing_packages):
    """Rank conditional interventions, not unsupported hardware-to-quality claims."""
    try:
        tree = ast.parse(candidate_code)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
             if isinstance(n, (ast.Attribute, ast.Name))}
    strings = {n.value.lower() for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    keywords = {n.arg for n in ast.walk(tree) if isinstance(n, ast.keyword)}
    text_task = bool(_TEXT_CLASSIFICATION.search(str(task_description or "")))
    torch_model = bool(names & {"AutoModel", "AutoModelForSequenceClassification", "Module"})
    neural = torch_model or (stage == "draft" and text_task)
    precision, deberta = _candidate_precision(candidate_code)
    cards = []
    if neural and text_task and ("max_length" in strings or not candidate_code):
        cards.append((300, "padding",
            "For a tokenized text model: cache tokenization once; collate with per-batch padding "
            "(all input streams/masks/labels aligned). Keep truncation and sample order unchanged. "
            "Try padding to a multiple of 8 only if measured faster."))
    if neural and (precision or not candidate_code or "autocast" not in names):
        cards.append((290 if precision else 220, "precision",
            "AMP is optional, not mandatory: keep FP32 parameters; try BF16 autocast only if "
            "torch.cuda.is_bf16_supported(). Check one forward/backward/step for finite loss/gradients; "
            "otherwise use FP32. Do not blanket-enable FP16 for DeBERTa."))
    if neural and candidate_code and "DataLoader" in names and "non_blocking" not in keywords:
        cards.append((240, "transfer",
            "For CPU-to-CUDA batches: test DataLoader(pin_memory=True) with to(device, non_blocking=True); "
            "tune workers within CPU quota. Avoid per-step loss.item() synchronization; aggregate detached "
            "losses. Pinned copies alone do not prove compute overlap."))
    has_threshold = any("threshold" in name.lower() or name.lower() in {"thr", "best_thr"}
                        for name in names)
    if text_task and not has_threshold:
        cards.append((250 if not candidate_code else 180, "validation",
            "For binary F1: optionally choose a threshold on held-out training validation or OOF "
            "probabilities, never test/private labels. Keep splits fixed; report F1 at 0.5 and tuned F1. "
            "Same-split tuning gain is optimistic, not proof of generalization."))
    lines = [f"Evidence-policy v2; device {device.name}; total {device.total_memory // 1024**2} MiB (not free).",
             "Keep task/model strategy; change one bottleneck at a time. These are conditional trials, not proven gains."]
    if missing_packages:
        lines.append("Missing tokenizer dependencies: " + ", ".join(missing_packages) +
                     "; check model/tokenizer loading before training.")
    measurement = (
        "Log chosen/rejected advice, actual optimizer/dtype/batch size. For a short normal-training window, "
        "synchronize CUDA only at timing boundaries; log examples/s, non-padding tokens/s, peak allocated "
        "memory and total wall time. Compare fixed-split F1; GPU busy% alone is not efficiency."
    )
    selected = []
    omitted = []
    for _, key, advice in sorted(cards, key=lambda c: (-c[0], c[1])):
        line = f"[{key}] {advice}"
        if len(selected) < 3 and len("\n".join(lines + [line, measurement])) <= _MAX_CHARS:
            selected.append(key)
            lines.append(line)
        else:
            omitted.append(key)
    lines.append(measurement)
    return "\n".join(lines), {
        "policy": "evidence-v2", "advice_ids": selected, "omitted_advice_ids": omitted,
        "evidence_sources": {key: _EVIDENCE_SOURCES[key] for key in selected},
        "candidate_code_sha256": hashlib.sha256(candidate_code.encode()).hexdigest(),
        "text_classification": text_task, "neural_candidate_or_conditional_draft": neural,
        "precision_guard": "precision" in selected,
        "deberta_guard": "precision" in selected and deberta,
    }


def _visible_cuda_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("HWKG requires PyTorch to verify the visible GPU") from exc

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("HWKG requires exactly one visible CUDA GPU")
    return torch.cuda.get_device_properties(0)


def _normalized_gpu_name(name):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^(?:nvidia|amd)\s+", "", name.casefold()))


def _matches_visible_device(hardware, device):
    from .device_support import capability, architecture
    if not hardware["id"].startswith(("hw:nvidia.", "hw:amd.")):
        return False
    props = hardware["properties"]
    visible_capability = capability(device)
    if hardware['id'].startswith('hw:amd.') != architecture(device).startswith('gfx'):
        return False
    memory_mib = device.total_memory / (1024 * 1024)
    expected_memory = props.get('vram_per_gpu_MB',props["vram_MB"])
    names = [props['name'],*props.get('aliases',[])]
    normalized = _normalized_gpu_name(device.name)
    matches_name = normalized in {_normalized_gpu_name(n) for n in names}
    # Memory distinguishes catalogue capacity variants whose driver name omits
    # the suffix. Bus variants stay unverified unless uniquely identifiable.
    if not matches_name:
        trim=lambda n:_normalized_gpu_name(re.sub(r'\s+\d+\s*GB$', '', n, flags=re.I))
        matches_name=trim(device.name)==trim(props['name'])
    if not matches_name and props.get('gpu_count',1)>1:
        # GB200 platform profiles apply to one visible member, never aggregate
        # HBM or an implicitly launched distributed job.
        matches_name=('gb200' in normalized and 'gb200' in hardware['id'])
    return (
        matches_name
        and visible_capability in props["compute_capabilities"]
        and 0.85 * expected_memory <= memory_mib <= 1.15 * expected_memory
    )


def _verify_visible_hardware(graph, hardware):
    from .device_support import capability as device_capability
    device = _visible_cuda_device()
    capability = device_capability(device)
    memory_mib = device.total_memory / (1024 * 1024)
    if hardware['properties'].get('gpu_count',1)>1:
        matches=[hardware] if _matches_visible_device(hardware,device) else []
    else:
        matches = [node for node in graph["nodes"] if node.get('properties',{}).get('gpu_count',1)==1
                   and _matches_visible_device(node, device)]
    family_only=(os.environ.get('MLEVOLVE_HWKG_EVIDENCE_POLICY',DEFAULT_POLICY)=='v8'
                 and len(matches)>1 and any(n['id']==hardware['id'] for n in matches)
                 and all(n['properties']['name']==hardware['properties']['name']
                         and n['properties']['compute_capabilities']==hardware['properties']['compute_capabilities']
                         and n['properties']['vram_MB']==hardware['properties']['vram_MB'] for n in matches))
    if not family_only and (len(matches) != 1 or matches[0]["id"] != hardware["id"]):
        raise RuntimeError(
            f"HWKG visible GPU {device.name} ({capability}, {memory_mib:.0f} MiB) "
            f"does not uniquely match pinned graph node {hardware['id']}"
        )
    return device, capability, memory_mib


def _has_hidden_matrix_model(candidate_code):
    if not candidate_code:
        return False
    try:
        tree = ast.parse(candidate_code)
    except SyntaxError:
        return False

    def name(node):
        if isinstance(node, ast.Attribute):
            return node.attr
        return node.id if isinstance(node, ast.Name) else ""

    def constructors(node):
        return [name(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(name(base) == "Module" for base in node.bases):
            layers = constructors(node)
            if layers.count("Linear") >= 2 or any(
                layer in layers for layer in ("TransformerEncoderLayer", "TransformerDecoderLayer")
            ):
                return True
        if isinstance(node, ast.Call) and name(node.func) == "Sequential":
            if constructors(node).count("Linear") >= 2:
                return True
    return False


def _muon_available():
    helper = Path(__file__).resolve().parents[1] / "utils" / "muon_optimizer.py"
    if helper.is_file():
        return True
    try:
        import torch
    except ImportError:
        return False
    return callable(getattr(torch.optim, "Muon", None))


def _candidate_precision(candidate_code):
    if not candidate_code:
        return False, False
    try:
        tree = ast.parse(candidate_code)
    except SyntaxError:
        return False, False
    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    }
    has_precision = bool(names & {"autocast", "GradScaler", "half", "float16", "bfloat16"})
    has_deberta = any(
        isinstance(node, ast.Constant) and isinstance(node.value, str)
        and "deberta" in node.value.casefold()
        for node in ast.walk(tree)
    )
    return has_precision, has_deberta


def _missing_tokenizer_packages():
    return [name for name in _TOKENIZER_PACKAGES if importlib.util.find_spec(name) is None]


def _exploratory_relevance(feature_id, task_description, candidate_code):
    task_description = str(task_description or "")
    if feature_id == "feat:nvimagecodec_gpu_decode":
        return bool(_IMAGE_TASK.search(task_description)), "no image-task evidence"
    if feature_id == "feat:muon_optimizer":
        if not _has_hidden_matrix_model(candidate_code):
            return False, "no explicit hidden-matrix model in candidate code"
        if not _muon_available():
            return False, "no Muon implementation available in this environment"
    if feature_id == "feat:cut_cross_entropy":
        return bool(_LANGUAGE_MODEL_TASK.search(task_description)), "no language-model task evidence"
    if feature_id == "feat:dataset_decomposition":
        relevant = _LANGUAGE_MODEL_TASK.search(task_description) and _LONG_TEXT.search(task_description)
        return bool(relevant), "no long-sequence language-model task evidence"
    return True, ""


def hardware_guidance(stage, task_description="", candidate_code="", **stage_context):
    """Return graph-backed advice, or None when the treatment is disabled."""
    graph_path = os.environ.get("MLEVOLVE_HWKG_GRAPH_PATH")
    if not graph_path:
        return None

    policy = os.environ.get("MLEVOLVE_HWKG_EVIDENCE_POLICY", DEFAULT_POLICY)
    if policy not in {"legacy", "v2", "v4", "v5", "v6", "v7", "v8"}:
        raise ValueError(f"Unknown HWKG evidence policy: {policy}")
    expected_sha = os.environ["MLEVOLVE_HWKG_GRAPH_SHA256"]
    hardware_id = os.environ["MLEVOLVE_HWKG_HARDWARE_ID"]
    raw = Path(graph_path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("HWKG graph SHA-256 does not match the pinned experiment graph")

    graph = json.loads(raw)
    node_id = f"hw:{hardware_id}"
    hardware = next((node for node in graph["nodes"] if node["id"] == node_id), None)
    if hardware is None:
        raise ValueError(f"HWKG hardware node not found: {node_id}")
    device, capability, memory_mib = _verify_visible_hardware(graph, hardware)

    if policy in {"v6", "v7", "v8"}:
        from .host_policy import guidance_v6, runtime_facts
        if policy == 'v7':
            from .adaptive_policy import guidance_v7
        if policy == 'v8':
            from .feedback_policy import guidance_v8
        selected_guidance = guidance_v8 if policy == 'v8' else guidance_v7 if policy == 'v7' else guidance_v6
        edges = {e["to"]: e for e in graph["edges"] if e["from"] == node_id}
        features = {n["id"]: n for n in graph["nodes"] if n["id"].startswith("feat:")}
        runtime = runtime_facts()
        siblings=[n for n in graph['nodes'] if n.get('properties',{}).get('gpu_count',1)==1
                  and _matches_visible_device(n,device)]
        runtime['hardware_profile_identity_verified']=len(siblings)==1 and hardware['properties'].get('gpu_count',1)==1
        runtime['matching_graph_profile_ids']=[n['id'] for n in siblings]
        if len(siblings)>1:
            # Equal driver identities can represent GDDR6/GDDR6X variants. Use
            # common feature eligibility, without borrowing variant bandwidth
            # or a calibration that cannot identify the actual memory variant.
            hardware={**hardware,'properties':{**hardware['properties'],
                'memory_bandwidth_GBps':None,'vram_type':'unverified driver-identical variant'}}
            sets=[{e['to'] for e in graph['edges'] if e['from']==n['id']
                   and e.get('properties',{}).get('verified') is True
                   and e.get('properties',{}).get('support_level')=='supported'} for n in siblings]
            shared=set.intersection(*sets)
            edges={k:v for k,v in edges.items() if k in shared}
        if policy == 'v8':
            from .calibration import calibration_flags
            runtime['calibration_flags'] = calibration_flags()
        text, evidence = selected_guidance(stage, task_description, candidate_code, device, hardware,
                                    features, edges, runtime, _MAX_CHARS,
                                    optimization_mode=os.environ.get("MLEVOLVE_HWKG_OPTIMIZATION_MODE", "normal"),
                                    **stage_context)
        directory = os.environ.get("MLEVOLVE_HWKG_AUDIT_DIR")
        if directory:
            path = Path(directory); path.mkdir(parents=True, exist_ok=True)
            record = {"event":"context_built", "stage":stage, "graph_sha256":actual_sha,
                      "visible_gpu_name":device.name, "visible_compute_capability":capability,
                      "visible_memory_mib":round(memory_mib), "chars":len(text or ''),
                      "context_sha256":hashlib.sha256((text or '').encode()).hexdigest(),
                      "context_text":text, "graph_profile_gpu_count":hardware['properties'].get('gpu_count',1),
                      "visible_device_scope":"one GPU; platform topology and aggregate memory not verified",
                      **evidence}
            with (path/"prompt-audit.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True)+"\n")
        return text

    props = hardware["properties"]
    lines = [
        f"Graph SHA-256: {actual_sha}; hardware node: {node_id}.",
        f"Device: {props['name']}; compute capability: {', '.join(props['compute_capabilities'])}; "
        f"VRAM: {props['vram_MB']} MB.",
        f"Runtime-visible GPU memory: {memory_mib:.0f} MiB total (not free); use this for memory planning.",
        "Choose model and data strategy for the task first; do not replace them merely for GPU features. "
        "Hardware facts are not acceptance rules or evidence of an accuracy gain.",
    ]
    selected = []
    exploratory = []
    background = []
    background_reasons = {}
    missing_packages = _missing_tokenizer_packages()
    if missing_packages:
        lines.append(
            "Missing optional tokenizer packages in this runtime: "
            + ", ".join(missing_packages)
            + ". Verify a pretrained tokenizer loads before full training; use a task-appropriate "
            "compatible model if its dependency is unavailable."
        )

    edges = {edge["to"]: edge for edge in graph["edges"] if edge["from"] == node_id}
    precision_requested, has_deberta = _candidate_precision(candidate_code)
    if precision_requested:
        selected = [feature_id for feature_id in _FEATURE_IDS
                    if (edge := edges.get(feature_id))
                    and edge["properties"].get("support_level") == "supported"
                    and edge["properties"].get("verified")]
        if selected:
            lines.append(
                f"{device.name} supports {', '.join(selected)}; support does not establish safety for this model. "
                "Before full training, check one forward/backward/optimizer step in the chosen dtype "
                "for finite loss and gradients. Keep FP32 parameters and an FP32 fallback; "
                "benchmark speed and task quality against FP32."
            )
        if has_deberta:
            lines.append(
                "DeBERTa attention has previously overflowed under FP16 autocast here. "
                "Do not blanket-enable FP16 for it; validate BF16 on one batch or use FP32."
            )

    if os.environ.get("MLEVOLVE_HWKG_EXPLORATORY") == "1":
        feature_nodes = {node["id"]: node for node in graph["nodes"] if node["id"].startswith("feat:")}
        for feature_id, edge in sorted(edges.items()):
            feature = edge["properties"]
            if feature.get("verified") or feature.get("support_level") not in ("supported", "experimental"):
                continue
            if feature.get("support_level") == "supported" and feature.get("recommended"):
                relevant, reason = _exploratory_relevance(feature_id, task_description, candidate_code)
                if relevant:
                    exploratory.append(feature_id)
                    continue
                background_reasons[feature_id] = reason
            background.append(feature_id)

    if exploratory:
        lines.append(
            "Lower-confidence, unverified possibilities for this GPU only: not defaults or proven "
            "speedups. Check task fit and installed dependencies, then benchmark against a baseline; "
            "graph examples do not imply their helpers exist in this repository."
        )
        for feature_id in exploratory:
            feature = feature_nodes.get(feature_id, {}).get("properties", {})
            summary = feature.get("description") or feature.get("design_summary") or feature_id
            summary = textwrap.shorten(" ".join(summary.split()), width=150, placeholder="...")
            pattern = feature.get("recommended_patterns", [])
            advice = f" Optional trial: {textwrap.shorten(pattern[0], width=130, placeholder='...')}" if pattern else ""
            lines.append(f"{feature_id} [unverified]: {summary}{advice}")
    if background:
        lines.append(
            "Other unverified or experimental GPU-linked entries (background only, not advice): "
            + ", ".join(background) + "."
        )

    text = "\n".join(lines)
    evidence = {}
    if policy in {"v4", "v5"}:
        if os.environ.get("MLEVOLVE_HWKG_EXPLORATORY") == "1":
            raise ValueError("Hardware policy must not be mixed with exploratory graph advice")
        if policy == "v5":
            from .staged_policy import guidance_v5
            text, evidence = guidance_v5(stage, task_description, candidate_code, device, edges, _MAX_CHARS, **stage_context)
        else:
            from .policy import guidance_v4
            text, evidence = guidance_v4(stage, task_description, candidate_code, device, edges, _MAX_CHARS)
        selected = evidence["verified_hardware_feature_ids"]
        exploratory, background, background_reasons = [], [], {}
    if policy == "v2":
        if os.environ.get("MLEVOLVE_HWKG_EXPLORATORY") == "1":
            raise ValueError("Evidence policy must not be mixed with exploratory graph advice")
        text, evidence = _evidence_guidance(
            stage, task_description, candidate_code, device, missing_packages
        )
        selected, exploratory, background, background_reasons = [], [], [], {}
    if len(text) > _MAX_CHARS:
        raise ValueError("HWKG prompt exceeds the fixed character budget")

    audit_dir = os.environ.get("MLEVOLVE_HWKG_AUDIT_DIR")
    if audit_dir:
        path = Path(audit_dir)
        path.mkdir(parents=True, exist_ok=True)
        record = {
            "event": "context_built",
            "stage": stage,
            "graph_sha256": actual_sha,
            "hardware_node": node_id,
            "visible_gpu_name": device.name,
            "visible_compute_capability": capability,
            "visible_memory_mib": round(memory_mib),
            "selected_feature_ids": selected,
            "exploratory_feature_ids": exploratory,
            "background_unverified_feature_ids": background,
            "background_reasons": background_reasons,
            "missing_tokenizer_packages": missing_packages,
            "precision_guard": precision_requested,
            "deberta_guard": precision_requested and has_deberta,
            "chars": len(text),
            "context_sha256": hashlib.sha256(text.encode()).hexdigest(),
            **evidence,
        }
        if evidence:
            record["context_text"] = text
        with (path / "prompt-audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return text
