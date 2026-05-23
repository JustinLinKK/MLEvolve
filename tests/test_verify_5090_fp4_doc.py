from __future__ import annotations

import html as html_mod
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
RTX_5090_RECORD = REPO / "schema" / "hardware_feature_records" / "nvidia.blackwell.geforce_rtx_5090.spec.yaml"
FEATURE_DOCS = REPO / "schema" / "feature_docs.yaml"
USER_AGENT = "Mozilla/5.0 (verify-5090-fp4-doc)"


def claimed_fp4_urls() -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    record = yaml.safe_load(RTX_5090_RECORD.read_text(encoding="utf-8"))
    for source_ref in record.get("source_refs") or []:
        if "fp4" in (source_ref.get("documents_features") or []):
            urls.append((source_ref["url"], f"5090.source_refs ({source_ref.get('title')})"))

    feature_docs = yaml.safe_load(FEATURE_DOCS.read_text(encoding="utf-8"))
    fp4_entry = (feature_docs.get("feature_docs") or {}).get("fp4")
    if fp4_entry:
        urls.append((fp4_entry["url"], f"feature_docs.yaml fp4 ({fp4_entry.get('title')})"))
    return urls


def normalize_page_text(raw_html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).lower()


def fp4_claim_failures(text: str) -> list[str]:
    checks = [
        ("contains 'fp4'", "fp4" in text),
        (
            "contains 'rtx 5090' (any form)",
            any(value in text for value in ("rtx 5090", "rtx™ 5090", "geforce rtx 5090"))
            or ("rtx" in text and "5090" in text),
        ),
        (
            "contains 'tensor core' + ('5th' or 'fifth')",
            ("tensor core" in text) and (("5th" in text) or ("fifth" in text)),
        ),
    ]
    return [label for label, ok in checks if not ok]


def fetch_text(url: str) -> tuple[str, int]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=20, context=ssl.create_default_context()) as response:
            if response.status != 200:
                return "", int(response.status)
            raw = response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return "", int(exc.code)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return "", 0
    return normalize_page_text(raw), 200


def test_5090_record_or_feature_docs_claim_fp4_url() -> None:
    urls = claimed_fp4_urls()

    assert urls
    assert any("rtx-5090" in url for url, _ in urls)


def test_fp4_claim_matcher_accepts_expected_page_terms() -> None:
    text = normalize_page_text(
        """
        <html><body>
        GeForce RTX 5090 with FP4 support and fifth-generation Tensor Cores.
        </body></html>
        """
    )

    assert fp4_claim_failures(text) == []


def test_fp4_claim_matcher_reports_missing_terms() -> None:
    text = normalize_page_text("<html><body>RTX 5090 product page</body></html>")

    failures = fp4_claim_failures(text)

    assert "contains 'fp4'" in failures
    assert "contains 'tensor core' + ('5th' or 'fifth')" in failures


@pytest.mark.skipif(
    os.getenv("MLEVOLVE_RUN_NETWORK_TESTS") != "1",
    reason="live vendor documentation verification is opt-in",
)
def test_live_5090_fp4_documentation_matches_claims() -> None:
    failures: list[str] = []
    for url, source in claimed_fp4_urls():
        text, status = fetch_text(url)
        if status != 200:
            failures.append(f"{source}: HTTP {status} for {url}")
            continue
        for missing in fp4_claim_failures(text):
            failures.append(f"{source}: missing {missing}")

    assert not failures
