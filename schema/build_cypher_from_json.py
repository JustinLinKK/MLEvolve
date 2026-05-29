"""
This script reads hardware_knowledge_graph.json, and emits hardware_knowledge_graph.cypher.
> cd path/to/schema
> python build_cypher_from_json.py
"""

import json
from pathlib import Path
from build_graph_db import emit_cypher;

g=json.loads(Path('hardware_knowledge_graph.json').read_text(encoding='utf-8'))

Path('hardware_knowledge_graph.cypher').write_text(emit_cypher(g['nodes'], g['edges']), encoding='utf-8')