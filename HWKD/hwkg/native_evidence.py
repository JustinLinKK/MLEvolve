"""Read native parent logs without altering the executor or trusting printed claims.

Counter checks establish arithmetic consistency, not learning quality or a
verified hardware bottleneck. All accepted log measurements remain self-reported.
"""
import ast
import hashlib
import json
import math
import re
from .source_analysis import precision_scope, training_scope_uncertain
from .source_bindings import declarative_constants
from .window_scope import training_window_scope


def number(value, positive=False):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if positive else value >= 0))
    except OverflowError: return False


def source_spec(code):
    result = {'model': None, 'optimizer': None, 'batch': None, 'input_shape': None,
              'epochs': None, 'forwards_per_update': None, 'training_scope': 'unknown',
              'optimizer_options': None, 'weights': None, 'classifier_classes': None,
              'loss': None, 'runtime_flags_mutated': False, 'runtime_overrides': {}}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result
    constants = declarative_constants(tree)
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
             and isinstance(n.ctx, (ast.Store, ast.Del))}
    def unique(names):
        values = {constants[n] for n in names if type(constants.get(n)) is int and constants[n] > 0}
        # Two recognized settings with one unresolved value are ambiguous too.
        if any(n in bound and (type(constants.get(n)) is not int or constants[n] <= 0) for n in names):
            return None
        return next(iter(values)) if len(values) == 1 else None
    result['batch'] = unique(('BATCH', 'BATCH_SIZE', 'BS'))
    size = unique(('SIZE', 'IMAGE_SIZE', 'IMG_SIZE', 'RESOLUTION'))
    if size: result['input_shape'] = [3, size, size]
    result['epochs'] = unique(('EPOCHS', 'MAX_EPOCHS', 'NUM_EPOCHS'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    imports = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for item in n.names: imports[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom):
            for item in n.names: imports[item.asname or item.name] = (n.module or '') + '.' + item.name
    def origin(node):
        if isinstance(node, ast.Name): return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute): return origin(node.value) + '.' + node.attr
        return ''
    names = {getattr(n.func, 'attr', getattr(n.func, 'id', '')) for n in calls}
    models = {origin(n.func).split('.')[-1] for n in calls if origin(n.func) in
              {'torchvision.models.resnet18', 'torchvision.models.resnet34', 'torchvision.models.resnet50'}}
    # Custom wrappers/heads have different work even when they contain a ResNet.
    custom = any(isinstance(n, ast.ClassDef) and any(
        getattr(b, 'attr', getattr(b, 'id', '')) == 'Module' for b in n.bases) for n in ast.walk(tree))
    if len(models) == 1 and not custom: result['model'] = 'torchvision.' + next(iter(models))
    opts = names & {'SGD', 'Adam', 'AdamW'}
    if len(opts) == 1 and 'SAM' not in names: result['optimizer'] = next(iter(opts))
    def literal(node):
        if isinstance(node, ast.Name) and node.id in constants: return constants[node.id]
        return ast.literal_eval(node)
    # Accept a consistent literal resize plus RGB allocation, rather than
    # borrowing dimensions from a raw CIFAR array or an unrelated scalar.
    sizes = []
    for call in calls:
        if origin(call.func) == 'torch.nn.functional.interpolate':
            value = next((k.value for k in call.keywords if k.arg == 'size'), None)
            try:
                shape = literal(value)
                if type(shape) is int and shape > 0:
                    sizes.append(shape)
                elif (isinstance(shape, (tuple, list)) and len(shape) == 2
                      and all(type(v) is int and v > 0 for v in shape) and shape[0] == shape[1]):
                    sizes.append(shape[0])
                else:
                    sizes.append(None)
            except (ValueError, TypeError):
                sizes.append(None)
    allocations = []
    for call in calls:
        if origin(call.func) in ('torch.empty', 'torch.zeros') and call.args:
            shape = call.args[0]
            if isinstance(shape, (ast.Tuple, ast.List)) and len(shape.elts) == 4:
                try:
                    c, h, w = [literal(v) for v in shape.elts[1:]]
                    if type(c) is int and c == 3:
                        allocations.append(h if type(h) is int and type(w) is int and h == w and h > 0 else None)
                except (ValueError, TypeError):
                    allocations.append(None)
    if sizes and (None in sizes or len(set(sizes)) != 1
                  or (size is not None and sizes[0] != size)):
        result['input_shape'] = None
    elif size is None and sizes and allocations and None not in allocations and set(sizes) == set(allocations):
        result['input_shape'] = [3, sizes[0], sizes[0]]
    # Calibration deliberately matches a narrow direct recipe. Parameter groups,
    # aliases/wrappers and dynamic settings stay unknown instead of borrowing a
    # timing from a different optimizer or classifier.
    model_calls = [n for n in calls if origin(n.func) in
                   {'torchvision.models.resnet18','torchvision.models.resnet34','torchvision.models.resnet50'}]
    if len(model_calls) == 1:
        weight = next((k.value for k in model_calls[0].keywords if k.arg == 'weights'), None)
        if isinstance(weight, ast.Constant) and weight.value is None: result['weights'] = 'none'
        elif isinstance(weight, ast.Attribute) and ast.unparse(weight).endswith('ResNet50_Weights.IMAGENET1K_V2'):
            result['weights'] = 'ResNet50_Weights.IMAGENET1K_V2'
    classifiers = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == 'fc' for t in n.targets)]
    if len(classifiers) == 1 and isinstance(classifiers[0], ast.Call):
        call = classifiers[0]
        if getattr(call.func, 'attr', getattr(call.func, 'id', '')) == 'Linear' and len(call.args) == 2:
            try:
                classes = literal(call.args[1])
                if type(classes) is int and classes > 0: result['classifier_classes'] = classes
            except (ValueError, TypeError): pass
    optimizer_calls = [n for n in calls if origin(n.func) == 'torch.optim.AdamW']
    if result['optimizer'] == 'AdamW' and len(optimizer_calls) == 1:
        call = optimizer_calls[0]
        if (len(call.args) == 1 and isinstance(call.args[0], ast.Call)
                and isinstance(call.args[0].func, ast.Attribute) and call.args[0].func.attr == 'parameters'
                and not call.args[0].args and not call.args[0].keywords):
            defaults = {'lr': .001, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': .01,
                        'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False,
                        'differentiable': False, 'fused': None}
            try:
                for k in call.keywords:
                    if k.arg not in defaults: raise ValueError('unknown optimizer argument')
                    value = literal(k.value)
                    defaults[k.arg] = list(value) if isinstance(value, tuple) else value
                result['optimizer_options'] = defaults
            except (ValueError, TypeError): pass
    losses = [n for n in calls if getattr(n.func, 'attr', getattr(n.func, 'id', '')) in
              {'cross_entropy', 'CrossEntropyLoss'}]
    if len(losses) == 1:
        loss = losses[0]
        from .loss_scope import used_for_backward
        direct = ((origin(loss.func) == 'torch.nn.functional.cross_entropy' and len(loss.args) == 2)
                  or (origin(loss.func) == 'torch.nn.CrossEntropyLoss' and not loss.args))
        if (direct and used_for_backward(tree,loss)
                and all(k.arg == 'label_smoothing' for k in loss.keywords) and len(loss.keywords) <= 1):
            try:
                smoothing = literal(loss.keywords[0].value) if loss.keywords else 0.
                if type(smoothing) in (float, int) and math.isfinite(smoothing) and 0 <= smoothing <= 1:
                    result['loss'] = ('cross_entropy_default' if smoothing == 0 else
                                      {'kind':'cross_entropy', 'label_smoothing':float(smoothing), 'reduction':'mean'})
            except (ValueError, TypeError):
                pass
    if {'first_step', 'second_step'} <= names:
        result['forwards_per_update'] = 2
    elif not custom and 'backward' in names and 'step' in names:
        result['forwards_per_update'] = 1
    # Conservative syntax screen only; requires_grad mutations or detach paths
    # leave full-model scope unestablished, rather than guessing from timings.
    uncertain = training_scope_uncertain(code)
    result['runtime_flags_mutated'] = (bool(names & {'set_float32_matmul_precision', 'use_deterministic_algorithms'})
        or any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
               and n.attr in {'benchmark', 'deterministic', 'allow_tf32', 'fp32_precision'} for n in ast.walk(tree)))
    flags = {'torch.backends.cudnn.benchmark': 'cudnn_benchmark',
             'torch.backends.cudnn.deterministic': 'cudnn_deterministic',
             'torch.backends.cudnn.allow_tf32': 'cudnn_allow_tf32',
             'torch.backends.cuda.matmul.allow_tf32': 'matmul_allow_tf32'}
    mutations = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                 and n.attr in {'benchmark', 'deterministic', 'allow_tf32', 'fp32_precision'}]
    if result['runtime_flags_mutated']:
        recognized = []
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                key = flags.get(origin(n.targets[0]))
                if key and isinstance(n.value, ast.Constant) and type(n.value.value) is bool:
                    recognized.append(n.targets[0]); result['runtime_overrides'][key] = n.value.value
        if (len(recognized) != len(mutations) or len(result['runtime_overrides']) != len(recognized)
                or names & {'set_float32_matmul_precision', 'use_deterministic_algorithms'}):
            result['runtime_overrides'] = None
    if result['model'] and not uncertain: result['training_scope'] = 'full_model_source_screen'
    return result


def scaler_update_accounting(code):
    """Conservatively screen explicit scale-delta counting, never certify work.

    GradScaler.step can skip an optimizer update. A literal skipped_updates=0
    is not a success counter. Even a recognized counter remains a self-report;
    this function does not establish window resets or runtime execution.
    """
    try: parsed = ast.parse(code)
    except SyntaxError: return 'unknown_source'
    scope = precision_scope(code)
    scalers = scope.get('training_scaler') or []
    if not scalers:
        aliases = {'GradScaler'}
        for n in ast.walk(parsed):
            if isinstance(n, ast.ImportFrom):
                aliases.update(item.asname or item.name for item in n.names if item.name == 'GradScaler')
        constructors = [n for n in ast.walk(parsed) if isinstance(n, ast.Call)
                        and getattr(n.func, 'attr', getattr(n.func, 'id', '')) in aliases]
        if any(not any(k.arg == 'enabled' and isinstance(k.value, ast.Constant)
                       and k.value.value is False for k in n.keywords) for n in constructors):
            return 'unknown_scaler_binding'
        if not constructors and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                    and n.func.attr in ('get_scale', 'unscale_', 'scale') for n in ast.walk(parsed)):
            return 'unknown_scaler_binding'
        return 'no_active_scaler_source_screen'
    # Only recognize the narrow, explicitly linked scale comparison + counter
    # idiom. Aliased/dynamic mechanisms stay unknown instead of trusting zero.
    reads = {}
    for n in ast.walk(parsed):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id in scalers
                and n.value.func.attr == 'get_scale' and not n.value.args and not n.value.keywords):
            reads[n.targets[0].id] = n.value.func.value.id
    skipped_fields = [v for n in ast.walk(parsed) if isinstance(n, ast.Dict)
                      for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == 'skipped_updates']
    successful_fields = [v for n in ast.walk(parsed) if isinstance(n, ast.Dict)
                         for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == 'optimizer_steps']
    if (len(scalers) != 1 or len(skipped_fields) != 1 or len(successful_fields) != 1
            or not isinstance(skipped_fields[0], ast.Name) or not isinstance(successful_fields[0], ast.Name)):
        return 'successful_updates_unestablished'
    for n in ast.walk(parsed):
        test = n.test if isinstance(n, ast.If) else None
        if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Lt)
                and isinstance(test.left, ast.Name) and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Name)
                and test.left.id != test.comparators[0].id
                and reads.get(test.left.id) == reads.get(test.comparators[0].id) == scalers[0]):
            def increments(body, name):
                return any(isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name)
                           and x.target.id == name and isinstance(x.op, ast.Add)
                           and isinstance(x.value, ast.Constant) and type(x.value.value) is int
                           and x.value.value == 1 for x in body)
            if increments(n.body, skipped_fields[0].id) and increments(n.orelse, successful_fields[0].id):
                return 'scale_delta_counter_syntax_only'
    return 'successful_updates_unestablished'


def native_feedback(code, text, device, *, parent_id=None, execution_succeeded=None):
    result = {'trust': 'unknown', 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
              'candidate_id': parent_id, 'gpu': {'name': device.name, 'capability': [device.major, device.minor]},
              'source_spec': source_spec(code), 'window': None, 'reasons': [],
              'verified_throughput': False, 'bottleneck_verified': False}
    if execution_succeeded is not True:
        result['reasons'].append('parent execution success not established'); return result
    records = []
    malformed = False
    for line in str(text)[-131072:].splitlines():
        for prefix in ('THROUGHPUT_JSON ', 'HWKG_RUNTIME_JSON '):
            if line.startswith(prefix):
                try:
                    row = json.loads(line[len(prefix):])
                    if not isinstance(row, dict): raise ValueError()
                    records.append(row)
                except (ValueError, TypeError, RecursionError): malformed = True
    if malformed or len(records) != 1:
        result['reasons'].append('missing, malformed or ambiguous native windows'); return result
    row = records[0]
    if row.get('code_sha256', result['source_sha256']) != result['source_sha256']:
        result['reasons'].append('native source identity mismatch'); return result
    if row.get('candidate_id', parent_id) != parent_id:
        result['reasons'].append('native candidate identity mismatch'); return result
    gpu = row.get('gpu')
    if gpu is not None and gpu != result['gpu']:
        result['reasons'].append('native GPU identity mismatch'); return result
    updates, examples, start, end = [row.get(k) for k in
        ('optimizer_steps', 'completed_examples', 'window_start_step', 'window_end_step')]
    skipped = row.get('skipped_updates', 0)
    wall, rate = row.get('wall_seconds'), row.get('examples_per_second')
    if (any(type(v) is not int or not number(v) for v in (updates, examples, start, end, skipped))
            or updates < 20 or examples <= 0 or end-start not in (updates+skipped, updates+skipped-1)
            or not number(wall, True) or not number(rate, True)
            or not math.isclose(rate, examples/wall, rel_tol=.015, abs_tol=.1)):
        result['reasons'].append('inconsistent native window counters'); return result
    dtype = row.get('autocast_dtype', row.get('actual_autocast_dtype'))
    if dtype not in ('bfloat16', 'torch.bfloat16', 'float32', 'torch.float32', 'disabled', 'float16', 'torch.float16'):
        result['reasons'].append('unknown native dtype'); return result
    layouts = {'channels_last': 'channels_last', 'torch.channels_last': 'channels_last',
               'contiguous_format': 'contiguous', 'torch.contiguous_format': 'contiguous', 'contiguous': 'contiguous',
               'channels_first (contiguous NCHW)': 'contiguous', 'channels_first (NCHW)': 'contiguous'}
    # Exact native source spelling seen in the v8.14 ordinary NCHW parent.
    # Do not infer channels_last from an ambiguous "non-contiguous" label.
    layouts['channels_first(NCHW,contiguous)'] = 'contiguous'
    model_layout, input_layout = [layouts.get(row.get(k)) if isinstance(row.get(k),str) else None
                                 for k in ('model_memory_format', 'input_memory_format')]
    if not model_layout or not input_layout:
        result['reasons'].append('unknown native layouts'); return result
    spec = result['source_spec']
    if spec['batch'] and examples != updates * spec['batch']:
        result['reasons'].append('native examples disagree with source batch'); return result
    result['trust'] = 'source_bound_self_report'
    result['window'] = {'updates': updates, 'examples': examples, 'wall_seconds': wall,
        'examples_per_second': rate, 'autocast_dtype': dtype, 'model_layout': model_layout,
        'input_layout': input_layout, 'skipped_updates': skipped,
        'window_index_convention': 'exclusive' if end-start == updates+skipped else 'inclusive',
        'forwards_per_update_source_screen': spec['forwards_per_update'],
        'training_scope_source_screen': spec['training_scope'],
        'peak_allocated_bytes': row.get('peak_allocated_bytes') if type(row.get('peak_allocated_bytes')) is int else None}
    if spec['forwards_per_update'] != 1 or spec['training_scope'] != 'full_model_source_screen':
        result['reasons'].append('work scope cannot support a full-model single-forward comparison')
    if row.get('training_scope') not in (None, 'full_model'):
        result['reasons'].append('reported partial or mixed training')
    if skipped: result['reasons'].append('skipped updates reported')
    timing_scope = training_window_scope(code)
    result['window']['normal_training_source_screen'] = timing_scope
    if timing_scope['status'] == 'standalone_gradient_window_before_epoch_iterator_reset_source_screen':
        result['reasons'].append('native timing window is a standalone gradient pass before epoch iterator reset')
    if dtype in ('float16', 'torch.float16'):
        accounting = scaler_update_accounting(code)
        result['window']['successful_update_accounting'] = accounting
        if accounting not in ('no_active_scaler_source_screen', 'scale_delta_counter_syntax_only'):
            result['reasons'].append('native FP16 GradScaler successful-work counters not established by source')
    return result


def auxiliary_recipe_signature(code):
    """EMA/SWA/checkpoint syntax coverage, without certifying equivalence.

    Include known averaging classes/helpers and their bound calls. Checkpoint
    load/select changes are separate from hardware layout/precision reporting.
    Dynamic aliases and unknown control/data flow remain uncertified.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    def averaging(name):
        return bool(re.search(
            r'(?:^|_)(?:ema|swa)(?:_|$)|(?:^|[a-z_])(?:EMA|SWA)(?:$|[A-Z_])'
            r'|AveragedModel|ExponentialMovingAverage', name))
    def name(node):
        return getattr(node, 'attr', getattr(node, 'id', ''))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
    def average_call(node):
        return averaging(aliases.get(name(node.func), name(node.func)))
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and average_call(n.value):
            bound.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call) and average_call(n.value):
            if isinstance(n.target, ast.Name):
                bound.add(n.target.id)
    ema = []
    checkpoint = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and averaging(n.name):
            ema.append(ast.dump(n, include_attributes=False))
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(averaging(name(t)) or (isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name) and t.value.id in bound) for t in targets):
                ema.append(ast.dump(n, include_attributes=False))
        elif isinstance(n, ast.Call):
            is_bound = (isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                        and n.func.value.id in bound)
            if average_call(n) or is_bound:
                ema.append(ast.dump(n, include_attributes=False))
            if name(n.func) in ('load_state_dict', 'copy_to', 'save', 'load'):
                checkpoint.append(ast.dump(n, include_attributes=False))
    return {'averaging': sorted(ema), 'checkpoint': sorted(checkpoint)}


def evaluation_work_signature(code):
    """Named inference helpers: conservative syntax coverage, not complete dataflow."""
    try: tree=ast.parse(code)
    except SyntaxError: return None
    return sorted(ast.dump(n,include_attributes=False) for n in ast.walk(tree)
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and re.search(
            r'(?:^|_)(?:predict|inference|infer|evaluate|validate|validation|eval|tta)(?:_|$)',n.name,re.I))


def recipe_changes(before, after):
    """Detect relevant syntax changes; do not rewrite or reject upstream candidates."""
    def signature(code):
        try: tree = ast.parse(code)
        except SyntaxError: return None
        selected = []
        selected_calls = []
        settings = {'EPOCHS', 'MAX_EPOCHS', 'NUM_EPOCHS', 'BATCH', 'BATCH_SIZE',
                    'BS', 'SIZE', 'LR', 'WEIGHT_DECAY'}
        custom_modules = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                          and any(getattr(b, 'attr', getattr(b, 'id', '')) == 'Module'
                                  for b in n.bases)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                name = getattr(n.func, 'attr', getattr(n.func, 'id', ''))
                if (name in {'Adam', 'AdamW', 'SGD', 'SAM', 'resnet18', 'resnet34', 'resnet50'}
                        or name in custom_modules
                        or 'Loss' in name or name in {'cross_entropy', 'interpolate', 'Resize', 'RandomCrop',
                            'RandomResizedCrop', 'RandomHorizontalFlip', 'ColorJitter', 'RandAugment',
                            'LambdaLR', 'CosineAnnealingLR', 'OneCycleLR'}):
                    selected.append(ast.dump(n, include_attributes=False))
                    selected_calls.append(n)
            # A custom loss is also an nn.Module. Its formula can change without
            # changing the constructor call, so include the source definition.
            # This is a conservative syntax screen, not module equivalence.
            if isinstance(n, ast.ClassDef) and any(
                    getattr(b, 'attr', getattr(b, 'id', '')) == 'Module' for b in n.bases):
                selected.append(ast.dump(n, include_attributes=False))
        bindings = {}
        def bind(target, value):
            if isinstance(target, ast.Name):
                bindings.setdefault(target.id, []).append(value)
            elif (isinstance(target, (ast.Tuple, ast.List))
                  and isinstance(value, (ast.Tuple, ast.List))
                  and len(target.elts) == len(value.elts)):
                for t, v in zip(target.elts, value.elts): bind(t, v)
        for n in tree.body:
            if isinstance(n, ast.Assign):
                for target in n.targets: bind(target, n.value)
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                bind(n.target, n.value)
        # Follow module assignments referenced by model/optimizer/loss/scheduler
        # calls. LR constants in parameter groups and tuple BATCH assignments
        # otherwise have identical call ASTs despite a different recipe.
        pending = settings | {n.id for c in selected_calls for n in ast.walk(c)
                              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited: continue
            visited.add(name)
            for value in bindings.get(name, []):
                # Follow declarative settings, not mutable model/tensor objects
                # or executable helpers. Otherwise a model layout conversion
                # would be misclassified as an optimizer/quality recipe change.
                if any(isinstance(n, (ast.Call, ast.ListComp, ast.SetComp,
                                      ast.DictComp, ast.GeneratorExp, ast.Lambda,
                                      ast.NamedExpr, ast.Await, ast.Yield))
                       for n in ast.walk(value)): continue
                selected.append('binding:' + name + ':' + ast.dump(value, include_attributes=False))
                pending.update(n.id for n in ast.walk(value)
                               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        return sorted(selected)
    a, b = signature(before), signature(after)
    aa, ab=auxiliary_recipe_signature(before),auxiliary_recipe_signature(after)
    ema_changed=None if not before or aa is None or ab is None else aa['averaging']!=ab['averaging']
    checkpoint_changed=None if not before or aa is None or ab is None else aa['checkpoint']!=ab['checkpoint']
    ea,eb=evaluation_work_signature(before),evaluation_work_signature(after)
    eval_work_changed=None if not before or ea is None or eb is None else ea!=eb
    pa, pb = precision_scope(before), precision_scope(after)
    return {'recipe_syntax_changed': None if a is None or b is None or not before else a != b or bool(ema_changed) or bool(checkpoint_changed) or bool(eval_work_changed),
            'averaging_recipe_syntax_changed':ema_changed,'checkpoint_selection_syntax_changed':checkpoint_changed,
            'evaluation_work_syntax_changed':eval_work_changed,
            'recipe_audit_coverage':'selected model/optimizer/loss/augmentation/budget/scheduler, referenced module assignments/tuple settings, custom Module definitions, EMA/SWA/checkpoint and named inference helper syntax; not complete semantic/dataflow equivalence',
            'training_precision_changed': None if not before or pa['training'] is None or pb['training'] is None else pa['training'] != pb['training'],
            'evaluation_precision_changed': None if not before or pa['evaluation'] is None or pb['evaluation'] is None else pa['evaluation'] != pb['evaluation'],
            'bf16_with_active_scaler': pb.get('bf16_with_active_scaler', False),
            'precision_scope_before': pa, 'precision_scope_after': pb,
            'semantic_preservation_certified': False}
