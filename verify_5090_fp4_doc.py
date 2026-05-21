"""Cross-verify the RTX 5090 record's fp4 claim against the official NVIDIA page.

Fetches the URL stored in 5090's source_refs[*] that documents fp4
(or feature_docs.yaml fp4 entry), then asserts that the live page
text contains every claim the yaml makes:

  required substrings (case-insensitive):
    - "fp4"
    - "rtx 5090"  or  "geforce rtx 5090"  or  "rtx&trade; 5090"
    - "tensor core"  AND ("5th" or "fifth")

If a required substring is missing from the live page, FAIL. Also FAIL
if the URL is unreachable.
"""
from __future__ import annotations
import html as html_mod
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
A_PATH = REPO / "schema" / "hardware_feature_records" / "nvidia.blackwell.geforce_rtx_5090.spec.yaml"
FD_PATH = REPO / "schema" / "feature_docs.yaml"
USER_AGENT = "Mozilla/5.0 (verify-5090-fp4-doc)"


def fetch_text(url: str) -> tuple[str, int]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(
            req, timeout=20, context=ssl.create_default_context()
        ) as r:
            if r.status != 200:
                return "", r.status
            raw = r.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return "", exc.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return "", 0
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text).lower()
    return text, 200


def main() -> int:
    fails: list[str] = []

    # gather URLs claimed for fp4
    urls: list[tuple[str, str]] = []
    rec = yaml.safe_load(A_PATH.read_text())
    for sr in rec.get("source_refs") or []:
        if "fp4" in (sr.get("documents_features") or []):
            urls.append((sr["url"], f"5090.source_refs ({sr.get('title')})"))

    fd = yaml.safe_load(FD_PATH.read_text())
    fp4_entry = (fd.get("feature_docs") or {}).get("fp4")
    if fp4_entry:
        urls.append((fp4_entry["url"], f"feature_docs.yaml fp4 ({fp4_entry.get('title')})"))

    if not urls:
        print("FAIL no URL claims fp4 in 5090 record or feature_docs.yaml")
        return 1

    print(f"checking {len(urls)} URL(s) claimed to document fp4 + 5090:")
    for url, src in urls:
        print(f"\n--- {src} ---\n  url: {url}")
        text, status = fetch_text(url)
        if status != 200:
            fails.append(f"  HTTP {status}: {url}")
            print(f"  FAIL HTTP {status}")
            continue

        # required substrings
        checks: list[tuple[str, bool]] = []
        checks.append(("contains 'fp4'", "fp4" in text))
        checks.append((
            "contains 'rtx 5090' (any form)",
            any(s in text for s in ("rtx 5090", "rtx™ 5090", "geforce rtx 5090"))
            or ("rtx" in text and "5090" in text)
        ))
        checks.append((
            "contains 'tensor core' + ('5th' or 'fifth')",
            ("tensor core" in text) and (("5th" in text) or ("fifth" in text))
        ))
        for label, ok in checks:
            print(f"  {'OK  ' if ok else 'FAIL'} {label}")
            if not ok:
                fails.append(f"{src}: missing '{label}'")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== 5090 fp4 doc 100% matches yaml ({len(urls)} URL(s) verified) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
