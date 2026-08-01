"""
This script reads hardware_knowledge_graph.json, and emits hardware_knowledge_graph.cypher.
> cd path/to/schema
> python build_cypher_from_json.py
"""

import json
from pathlib import Path

def emit_cypher(nodes, edges) -> str:
    lines = [
        "// Independent of graph_schema.yaml (the job-evidence graph).",
        "CREATE CONSTRAINT hardware_id IF NOT EXISTS FOR (h:Hardware) REQUIRE h.hardware_id IS UNIQUE;",
        "CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.feature_id IS UNIQUE;",
        "",
    ]
    for n in nodes:
        key = "hardware_id" if n['label'] == 'Hardware' else "feature_id"
        props = json.dumps(n['properties'], ensure_ascii=False)
        lines.append(
            f"MERGE (n:{n['label']} {{{key}: {json.dumps(n['properties'][key])}}}) "
            f"SET n += apoc.convert.fromJsonMap('{props.replace(chr(39), chr(92) + chr(39))}');"
        )
    lines.append("")
    for e in edges:
        fkey = "hardware_id" if e['from'].startswith('hw:') else "feature_id"
        fval = e['from'].split(':', 1)[1]
        tval = e['to'].split(':', 1)[1]
        flabel = "Hardware" if e['from'].startswith('hw:') else "Feature"
        props = json.dumps(e['properties'], ensure_ascii=False)
        lines.append(
            f"MATCH (a:{flabel} {{{fkey}: {json.dumps(fval)}}}), "
            f"(b:Feature {{feature_id: {json.dumps(tval)}}}) "
            f"MERGE (a)-[r:{e['type']}]->(b) "
            f"SET r += apoc.convert.fromJsonMap('{props.replace(chr(39), chr(92) + chr(39))}');"
        )
    return "\n".join(lines) + "\n"

g=json.loads(Path('hardware_knowledge_graph.json').read_text(encoding='utf-8'))

Path('hardware_knowledge_graph.cypher').write_text(emit_cypher(g['nodes'], g['edges']), encoding='utf-8')