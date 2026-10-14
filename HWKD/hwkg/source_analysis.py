"""Scoped syntax observations, never a claim that a path actually executed."""
import ast
from .source_bindings import declarative_bindings


def call_name(node):
    return getattr(node.func, 'attr', getattr(node.func, 'id', ''))


def scopes(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # Module statements exclude function bodies, which have their own scope.
    module = ast.Module(body=[n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], type_ignores=[])
    owners = {'<module>': module, **functions}
    direct = {k: list(ast.walk(n)) for k, n in owners.items()}
    training = {k for k, nodes in direct.items() if any(isinstance(n, ast.Call) and call_name(n) == 'backward' for n in nodes)}
    evaluation = {k for k, n in functions.items() if (
        any(word in k.lower() for word in ('predict', 'eval', 'valid', 'infer', 'test'))
        or any(isinstance(d, ast.Call) and call_name(d) in ('no_grad', 'inference_mode') for d in n.decorator_list))}
    def reachable(roots):
        found = set(roots)
        pending = list(roots)
        while pending:
            key = pending.pop()
            for n in direct[key]:
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions and n.func.id not in found:
                    found.add(n.func.id); pending.append(n.func.id)
        return found
    return tree, owners, direct, training, reachable(training), reachable(evaluation)


def precision_scope(code):
    parsed = scopes(code)
    if parsed is None:
        return {'training': None, 'evaluation': None, 'training_scaler': None, 'syntax_only': True}
    tree, _, direct, _, train, evaluation = parsed
    bindings = declarative_bindings(tree)
    torch_names = {'torch'}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            torch_names.update(item.asname or 'torch' for item in n.names if item.name == 'torch')
    # A local/module shadow may refer to a different object. Stay unknown.
    shadowed = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    shadowed.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
    shadowed.update(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            shadowed.update(item.asname or item.name.split('.')[0] for item in n.names if item.name != 'torch')
        elif isinstance(n, ast.ImportFrom):
            shadowed.update(item.asname or item.name for item in n.names)
    torch_names -= shadowed
    def dtype_value(node):
        seen = set()
        while isinstance(node, ast.Name) and node.id in bindings:
            if node.id in seen: return None
            seen.add(node.id); node = bindings[node.id]
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in torch_names and node.attr in ('float16', 'bfloat16', 'float32')):
            return node.attr
        return None
    def signature(keys):
        rows = []
        for key in sorted(keys):
            for n in direct[key]:
                if isinstance(n, ast.Call) and call_name(n) in ('autocast', 'half', 'bfloat16', 'GradScaler', 'set_autocast_dtype'):
                    # GradScaler is a training update mechanism, not eval precision.
                    if call_name(n) != 'GradScaler':
                        rows.append(ast.dump(n, include_attributes=False))
                        # The call AST can stay identical while a shared module
                        # dtype/enabled alias changes both train and evaluation.
                        values = [k.value for k in n.keywords if k.arg in ('dtype', 'enabled')]
                        if call_name(n) == 'set_autocast_dtype': values.extend(n.args)
                        pending = {x.id for value in values for x in ast.walk(value) if isinstance(x, ast.Name)}
                        seen = set()
                        while pending:
                            name = pending.pop()
                            if name in seen: continue
                            seen.add(name)
                            if name in bindings:
                                value = bindings[name]
                                rows.append('precision_binding:' + name + ':' + ast.dump(value, include_attributes=False))
                                pending.update(x.id for x in ast.walk(value) if isinstance(x, ast.Name))
        return sorted(rows)
    # Scalers are normally constructed outside train_epoch; observe construction
    # and usage separately. Disabled/unused constructors are not active evidence.
    scaler_names = {}
    for n in ast.walk(parsed[0]):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and call_name(n.value) == 'GradScaler':
            enabled = next((k.value for k in n.value.keywords if k.arg == 'enabled'), None)
            state = not (isinstance(enabled, ast.Constant) and enabled.value is False)
            for t in n.targets:
                if isinstance(t, ast.Name): scaler_names[t.id] = state
    used = {n.func.value.id for k in train for n in direct[k] if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            and n.func.attr in ('scale', 'step', 'update')}
    active = [k for k in used if scaler_names.get(k) is True]
    dtype = set()
    for k in train:
        for n in direct[k]:
            if isinstance(n, ast.Call) and call_name(n) == 'autocast':
                for kw in n.keywords:
                    if kw.arg == 'dtype':
                        value = dtype_value(kw.value)
                        if value: dtype.add(value)
    return {'training': signature(train), 'evaluation': signature(evaluation),
            'training_dtypes': sorted(dtype), 'training_scaler': sorted(active),
            'bf16_with_active_scaler': dtype == {'bfloat16'} and bool(active), 'syntax_only': True}


def training_scope_uncertain(code):
    parsed = scopes(code)
    if parsed is None: return True
    tree, owners, direct, roots, _, _ = parsed
    if not roots: return True
    # A torchvision constructor alone does not establish the measured ordinary
    # model path after instance/class forward replacement or execution hooks.
    # This is a conservative source screen, not complete alias/dataflow proof.
    for n in ast.walk(tree):
        if (isinstance(n, ast.Attribute) and n.attr in {'forward', '__call__'}
                and isinstance(n.ctx, (ast.Store, ast.Del))): return True
        if isinstance(n, ast.Call):
            name = call_name(n)
            if name in {'register_forward_hook', 'register_forward_pre_hook',
                        'register_backward_hook', 'register_full_backward_hook',
                        'register_full_backward_pre_hook'}: return True
            position = 1 if name in {'setattr', 'delattr'} else 0 if name in {'__setattr__', '__delattr__'} else None
            if position is not None and len(n.args) > position:
                attr = n.args[position]
                if isinstance(attr, ast.Constant) and attr.value in {'forward', '__call__'}: return True
    # Freezing remains conservative. Logging detach and eval-only no_grad no
    # longer poison an otherwise direct full-model training recipe.
    if any(isinstance(n, ast.Call) and call_name(n) == 'requires_grad_' for n in ast.walk(tree)): return True
    if any(isinstance(n, ast.Attribute) and n.attr == 'requires_grad' and isinstance(n.ctx, ast.Store) for n in ast.walk(tree)): return True
    for key in roots:
        root = owners[key]
        if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(d, ast.Call) and call_name(d) in ('no_grad', 'inference_mode') for d in root.decorator_list): return True
        for n in direct[key]:
            if isinstance(n, ast.With) and any(isinstance(i.context_expr, ast.Call) and call_name(i.context_expr) in ('no_grad', 'inference_mode') for i in n.items):
                # A no_grad forward/loss in the backward scope invalidates scope;
                # metrics/checkpoint operations on detached tensors do not.
                if any(isinstance(c, ast.Call) and (call_name(c) in ('cross_entropy', 'CrossEntropyLoss')
                       or isinstance(c.func, ast.Name) and c.func.id in ('model', 'net')) for s in n.body for c in ast.walk(s)): return True
            if isinstance(n, ast.Assign) and any(isinstance(c, ast.Call) and call_name(c) == 'detach' for c in ast.walk(n.value)):
                targets = {t.id for t in n.targets if isinstance(t, ast.Name)}
                if targets & {'loss', 'logits', 'outputs', 'out', 'pred', 'prediction'}: return True
    return False
