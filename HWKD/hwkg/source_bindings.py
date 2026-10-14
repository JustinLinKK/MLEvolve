"""Conservative reusable component AST bindings, never execute candidate code."""
import ast


def declarative_bindings(tree):
    """Unambiguous direct module bindings; no candidate execution or semantic proof."""
    writes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            writes[node.id] = writes.get(node.id, 0) + 1
        elif isinstance(node, ast.arg):
            writes[node.arg] = writes.get(node.arg, 0) + 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            writes[node.name] = writes.get(node.name, 0) + 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for item in node.names:
                name = item.asname or item.name.split('.')[0]
                writes[name] = writes.get(name, 0) + 1
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            writes[node.name] = writes.get(node.name, 0) + 1
        elif isinstance(node, ast.MatchMapping) and node.rest:
            writes[node.rest] = writes.get(node.rest, 0) + 1
    bindings = {}
    def bind(target, value):
        if isinstance(target, ast.Name):
            if writes.get(target.id) == 1: bindings[target.id] = value
        elif (isinstance(target, (ast.Tuple, ast.List))
              and isinstance(value, (ast.Tuple, ast.List))
              and len(target.elts) == len(value.elts)):
            for left, right in zip(target.elts, value.elts): bind(left, right)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets: bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
    return bindings


def declarative_constants(tree):
    """Read immutable module literals/aliases; dynamic or shadowed values unknown."""
    bindings = declarative_bindings(tree)
    values = {}
    def literal(node, pending):
        if isinstance(node, ast.Name):
            if node.id in values: return values[node.id]
            if node.id not in bindings or node.id in pending: raise ValueError('unresolved alias')
            return literal(bindings[node.id], pending | {node.id})
        if isinstance(node, ast.Tuple):
            items = [literal(item, pending) for item in node.elts]
            return tuple(items)
        value = ast.literal_eval(node)
        if value is None or type(value) in (bool, int, float, str): return value
        raise ValueError('not a scalar setting')
    for name, node in bindings.items():
        try: values[name] = literal(node, {name})
        except (ValueError, TypeError, RecursionError): pass
    return values
