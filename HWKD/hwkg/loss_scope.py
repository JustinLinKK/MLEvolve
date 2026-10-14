"""Narrow active-loss source screen for calibration metadata, not semantics."""
import ast


def used_for_backward(tree, loss_call):
    assignments={}
    writes={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Del)):
            writes[n.id]=writes.get(n.id,0)+1
        elif isinstance(n,ast.arg): writes[n.arg]=writes.get(n.arg,0)+1
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name):assignments.setdefault(t.id,[]).append(n.value)
        elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.value:
            assignments.setdefault(n.target.id,[]).append(n.value)
    bindings={k:v[0] for k,v in assignments.items() if len(v)==1 and writes.get(k)==1}
    gradients=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='backward':
            value=n.func.value
            if (isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute)
                    and value.func.attr=='scale' and len(value.args)==1):value=value.args[0]
            if not isinstance(value,ast.Name):return False
            gradients.append(value.id)
    if not gradients:return False
    owners={k for k,v in bindings.items() if v is loss_call}
    def matches(name,seen):
        if name in seen or name not in bindings:return False
        value=bindings[name]
        if isinstance(value,ast.Name):return matches(value.id,seen|{name})
        if value is loss_call:return True
        return (isinstance(value,ast.Call) and isinstance(value.func,ast.Name)
                and value.func.id in owners and len(value.args)==2 and not value.keywords)
    return all(matches(k,set()) for k in gradients)
