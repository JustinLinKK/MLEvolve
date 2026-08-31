from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def test_oneshot_sends_one_valid_chat_completion_request() -> None:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = Path(__file__).parents[1] / "deployments" / "keep_qwen_l40s_busy.sh"
    env = {
        **os.environ,
        "ENDPOINT": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        "MODEL_NAME": "test-qwen",
        "MAX_TOKENS": "17",
        "ONESHOT": "1",
    }

    try:
        result = subprocess.run(
            ["bash", str(script)],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert len(requests) == 1
    assert requests[0]["model"] == "test-qwen"
    assert requests[0]["max_tokens"] == 17
    assert requests[0]["stream"] is False

