from __future__ import annotations

import json
import os
import signal
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


def test_continuous_mode_defaults_to_sixteen_concurrent_requests() -> None:
    active = 0
    peak_active = 0
    lock = threading.Lock()
    sixteen_active = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            nonlocal active, peak_active
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            with lock:
                active += 1
                peak_active = max(peak_active, active)
                if active >= 16:
                    sixteen_active.set()
            release.wait(timeout=5)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            with lock:
                active -= 1

        def log_message(self, _format: str, *_args: object) -> None:
            return

    class ConcurrentServer(ThreadingHTTPServer):
        request_queue_size = 32

    server = ConcurrentServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = Path(__file__).parents[1] / "deployments" / "keep_qwen_l40s_busy.sh"
    env = {
        **os.environ,
        "ENDPOINT": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        "MODEL_NAME": "test-qwen",
        "MAX_TOKENS": "17",
    }
    process = subprocess.Popen(
        ["bash", str(script)],
        env=env,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        assert sixteen_active.wait(timeout=5)
        assert peak_active >= 16
    finally:
        release.set()
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
