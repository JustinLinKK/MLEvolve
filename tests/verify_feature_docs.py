"""Verifier for schema/feature_docs.yaml.

Checks:
  1. Every feature tag appearing in any Schema A record's `features` list has
     an entry in feature_docs.
  2. No orphan feature_docs entries (entries that no record uses) -- WARN only.
  3. Each entry has the required fields: title, url, source_type,
     retrieved_or_verified_date.
  4. URLs are reachable. HEAD request via urllib (no external deps).
     - offline / DNS failure -> WARN (do not fail)
     - HTTP 4xx (except 403 when entry sets blocks_automated_check: true) -> FAIL
     - HTTP 5xx -> FAIL
     - HTTP 2xx / 3xx -> OK
"""
from __future__ import annotations

import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from schema_io import REPO, collect_field

FEATURE_DOCS_PATH = REPO / "schema" / "feature_docs.yaml"
REQUIRED_FIELDS = ("title", "url", "source_type", "retrieved_or_verified_date")
ALLOWED_SOURCE_TYPES = {"vendor_reference", "official_doc", "academic_paper"}
HEAD_TIMEOUT_SEC = 8
USER_AGENT = "Mozilla/5.0 (verify_feature_docs)"


def load_feature_docs() -> dict[str, dict[str, Any]]:
    if not FEATURE_DOCS_PATH.exists():
        print(f"FAIL: {FEATURE_DOCS_PATH} does not exist")
        sys.exit(2)
    blob = yaml.safe_load(FEATURE_DOCS_PATH.read_text())
    if not isinstance(blob, dict) or "feature_docs" not in blob:
        print(f"FAIL: {FEATURE_DOCS_PATH} missing top-level 'feature_docs' key")
        sys.exit(2)
    return blob["feature_docs"] or {}


def head_check(url: str) -> tuple[str, int | None]:
    """Return (status_label, http_code).

    status_label is one of: ok, http_error, offline.
    """
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HEAD_TIMEOUT_SEC) as resp:  # noqa: S310
            return "ok", resp.status
    except urllib.error.HTTPError as e:
        # HEAD often forbidden; try GET with Range to minimize bytes
        try:
            req2 = urllib.request.Request(
                url,
                method="GET",
                headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
            )
            with urllib.request.urlopen(req2, timeout=HEAD_TIMEOUT_SEC) as resp:  # noqa: S310
                return "ok", resp.status
        except urllib.error.HTTPError as e2:
            return "http_error", e2.code
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
            return "offline", None
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        return "offline", None


def main() -> int:
    feature_docs = load_feature_docs()
    schema_a_tags = collect_field("hardware_feature_records", "features")

    failures: list[str] = []
    warnings: list[str] = []

    # (1) every Schema A tag has an entry
    missing = sorted(schema_a_tags - set(feature_docs.keys()))
    for tag in missing:
        failures.append(f"missing entry for feature tag '{tag}'")

    # (2) orphan entries (not used by any Schema A record)
    orphans = sorted(set(feature_docs.keys()) - schema_a_tags)
    for tag in orphans:
        warnings.append(f"orphan feature_docs entry '{tag}' (no Schema A record uses it)")

    # (3) required fields and source_type sanity
    for tag, entry in feature_docs.items():
        if not isinstance(entry, dict):
            failures.append(f"'{tag}' is not a mapping")
            continue
        for field in REQUIRED_FIELDS:
            if field not in entry or not entry[field]:
                failures.append(f"'{tag}' missing required field '{field}'")
        st = entry.get("source_type")
        if st and st not in ALLOWED_SOURCE_TYPES:
            failures.append(
                f"'{tag}' has invalid source_type '{st}' "
                f"(allowed: {sorted(ALLOWED_SOURCE_TYPES)})"
            )

    # (4) URL reachability
    total = len(feature_docs)
    reachable = 0
    any_offline = False
    print("URL reachability check:")
    for tag in sorted(feature_docs.keys()):
        entry = feature_docs[tag]
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not url:
            continue
        status, code = head_check(url)
        blocks_auto = bool(entry.get("blocks_automated_check"))
        if status == "ok":
            reachable += 1
            print(f"  [{code}]  {tag:20s}  {url}")
        elif status == "offline":
            any_offline = True
            warnings.append(f"'{tag}' offline / DNS failure for {url}")
            print(f"  [---]  {tag:20s}  {url}  (offline)")
        else:  # http_error
            if code == 403 and blocks_auto:
                warnings.append(
                    f"'{tag}' returns 403 to automated check but is marked "
                    f"blocks_automated_check=true (verify manually)"
                )
                reachable += 1  # treat as ok for percentage purposes
                print(f"  [403*] {tag:20s}  {url}  (anti-bot, manual-verify)")
            else:
                failures.append(f"'{tag}' URL returned HTTP {code}: {url}")
                print(f"  [{code}]  {tag:20s}  {url}  FAIL")

    print()
    print("Schema A feature tags found:", len(schema_a_tags))
    print("feature_docs entries:       ", len(feature_docs))
    if missing:
        print("MISSING tags (no doc URL):  ", missing)
    if orphans:
        print("ORPHAN entries (unused):    ", orphans)

    for w in warnings:
        print(f"WARN: {w}")
    for f in failures:
        print(f"FAIL: {f}")

    # Final summary line per spec
    pct = int(round(100 * reachable / total)) if total else 0
    tags_covered = len(schema_a_tags) - len(missing)
    print(
        f"=== feature_docs {pct}% valid "
        f"({tags_covered}/{len(schema_a_tags)} tags, "
        f"{reachable}/{total} URLs reachable) ==="
    )

    if failures and not any_offline:
        return 1
    if failures and any_offline:
        # if everything that failed was offline, do not hard-fail
        hard = [f for f in failures if "offline" not in f.lower()]
        return 1 if hard else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
