import subprocess
import sys
from pathlib import Path


def test_importing_llm_does_not_eagerly_load_unused_provider_sdks():
    repo = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, llm; print('openai' in sys.modules, 'google.genai' in sys.modules)",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "False False"
