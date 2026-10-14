"""Narrow native timing-window source screen, never an epoch-budget grader.

The log's arithmetic cannot establish ordinary useful training. Detect a
standalone gradient window followed by a fresh iterator in epoch training.
Dynamic/wrapped/ambiguous forms remain unknown; no source is executed or edited.
"""
import ast
from collections import Counter


def training_window_scope(code):
    result = {'status': 'unknown', 'syntax_only': True,
              'runtime_verified': False, 'epoch_budget_verified': False}
    try: tree = ast.parse(code)
    except SyntaxError: return result

    def nodes(root):
        pending = [root]
        while pending:
            n = pending.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): continue
            yield n
            pending.extend(ast.iter_child_nodes(n))

    # Trace only unique, unshadowed direct function calls to backward work.
    definitions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    counts = Counter(n.name for n in definitions)
    shadowed = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    shadowed.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
    shadowed.update(n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            shadowed.update(item.asname or item.name.split('.')[0] for item in n.names)
    functions = {n.name: ast.Module(body=n.body, type_ignores=[]) for n in definitions
                 if counts[n.name] == 1 and n.name not in shadowed}
    helpers = set()
    def gradient(root):
        return any(isinstance(n, ast.Call) and (
            isinstance(n.func, ast.Attribute) and n.func.attr == 'backward'
            or isinstance(n.func, ast.Name) and n.func.id in helpers) for n in nodes(root))
    while True:
        found = {name for name, body in functions.items() if gradient(body)}
        if found <= helpers: break
        helpers.update(found)

    dictionaries = [n for n in ast.walk(tree) if isinstance(n, ast.Dict) and
                    {'window_start_step', 'window_end_step'} <= {
                        k.value for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}]
    if len(dictionaries) != 1: return result
    fields = {k.value: v for k, v in zip(dictionaries[0].keys, dictionaries[0].values)
              if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    start, end = fields['window_start_step'], fields['window_end_step']
    if not isinstance(start, ast.Name) or not isinstance(end, ast.Name) or start.id == end.id: return result
    parents = {child: n for n in ast.walk(tree) for child in ast.iter_child_nodes(n)}
    def ancestors(node):
        while node in parents:
            node = parents[node]
            yield node
    def owner(node):
        return next((n for n in ancestors(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), tree)
    record_owner = owner(dictionaries[0])
    def assignments(name):
        return [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and owner(n) is record_owner
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                and not (isinstance(n.value, ast.Constant) and n.value.value is None)]
    starts, ends = assignments(start.id), assignments(end.id)
    if len(starts) != 1 or len(ends) != 1: return result

    def epoch_loop(node):
        if not isinstance(node, (ast.For, ast.While)) or not gradient(node): return False
        marker = (isinstance(node, ast.For) and any(
            isinstance(n, ast.Name) and 'epoch' in n.id.lower() for n in ast.walk(node.target)))
        marker = marker or (isinstance(node, ast.For) and any(
            isinstance(n, ast.Name) and 'epoch' in n.id.lower() for n in ast.walk(node.iter)))
        marker = marker or any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
                              and 'epoch' in n.target.id.lower() for n in nodes(node))
        return marker
    shared = [n for n in ancestors(starts[0]) if n in set(ancestors(ends[0])) and epoch_loop(n)]
    if shared:
        result.update(status='embedded_epoch_window_source_screen', epoch_loop_line=shared[0].lineno)
        return result

    # The affirmative standalone pattern needs module boundaries, real-looking
    # gradient calls between them, and a later fresh training iterator. Merely
    # a zero start-step or an unrelated timer is not grounds for rejection.
    if record_owner is not tree or starts[0] not in tree.body or ends[0] not in tree.body: return result
    a, b = tree.body.index(starts[0]), tree.body.index(ends[0])
    if a >= b: return result
    interval = ast.Module(body=tree.body[a+1:b], type_ignores=[])
    if not gradient(interval): return result
    iterators = {n.args[0].id for n in nodes(interval) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'next'
                 and len(n.args) == 1 and isinstance(n.args[0], ast.Name)}
    if not iterators: return result
    for stmt in tree.body[b+1:]:
        for loop in nodes(stmt):
            if not epoch_loop(loop): continue
            resets = [n for n in nodes(loop) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id in iterators for t in n.targets)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == 'iter' and len(n.value.args) == 1
                      and not any(isinstance(x, ast.Name) and x.id in iterators for x in ast.walk(n.value.args[0]))]
            if resets:
                result.update(status='standalone_gradient_window_before_epoch_iterator_reset_source_screen',
                              window_start_line=starts[0].lineno, window_end_line=ends[0].lineno,
                              later_epoch_loop_line=loop.lineno, iterator_reset_line=resets[0].lineno)
                return result
    return result
