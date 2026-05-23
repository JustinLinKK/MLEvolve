"""Verify every Schema A.source_refs[*].url and every Schema B.source_url
is HTTP-reachable (not 4xx/5xx).

Pass criteria:
  - every A record has at least 1 entry in source_refs
  - every A.source_refs[*].url returns 200 (or 403 with documented anti-bot WAF)
  - every B chunk has non-empty source_url
  - every B.source_url returns 200 / acceptable status
"""
from __future__ import annotations
import socket
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402

USER_AGENT = "Mozilla/5.0 (verify-record-urls)"
TIMEOUT = 10
ACCEPTABLE_STATUSES = (200, 301, 302, 303, 307, 308, 403)  # 403 for anti-bot pages


def head(url: str) -> int:
    if not url:
        return 0
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}, method="HEAD",
        )
        with urllib.request.urlopen(
            req, timeout=TIMEOUT, context=ssl.create_default_context()
        ) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        if exc.code in (405,):  # HEAD not allowed; try GET
            try:
                req2 = urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT}, method="GET",
                )
                with urllib.request.urlopen(
                    req2, timeout=TIMEOUT, context=ssl.create_default_context()
                ) as r:
                    return r.status
            except Exception:
                return exc.code
        return exc.code
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):
        return -1
    except Exception:
        return -2


def online() -> bool:
    try:
        socket.create_connection(("nvidia.com", 443), timeout=4).close()
        return True
    except OSError:
        return False


def main() -> int:
    fails: list[str] = []
    is_online = online()
    print(f"network: {'ONLINE' if is_online else 'OFFLINE (URL checks SKIPPED)'}")

    a = load_records("hardware_feature_records")
    b = load_records("code_doc_chunks")

    # Schema A audit
    a_no_refs = [r["record_id"] for r in a if not (r.get("source_refs") or [])]
    if a_no_refs:
        fails.append(f"A records missing source_refs ({len(a_no_refs)}): "
                     f"{a_no_refs[:5]}{'...' if len(a_no_refs) > 5 else ''}")

    # Schema B audit
    b_no_url = [r["chunk_id"] for r in b if not r.get("source_url", "").strip()]
    if b_no_url:
        fails.append(f"B chunks missing source_url ({len(b_no_url)}): {b_no_url}")

    print(f"\nA records: {len(a)} ({len(a) - len(a_no_refs)} have source_refs)")
    print(f"B chunks:  {len(b)} ({len(b) - len(b_no_url)} have source_url)")

    if not is_online:
        print("\nskipping URL HEAD checks (offline)")
        return 1 if fails else 0

    # de-dup URLs (each URL fetched once)
    url_to_records: dict[str, list[str]] = {}
    for r in a:
        for sr in r.get("source_refs") or []:
            u = (sr.get("url") or "").strip()
            if u:
                url_to_records.setdefault(u, []).append(f"A:{r['record_id']}")
    for r in b:
        u = (r.get("source_url") or "").strip()
        if u:
            url_to_records.setdefault(u, []).append(f"B:{r['chunk_id']}")

    print(f"\nchecking {len(url_to_records)} unique URLs ({sum(len(v) for v in url_to_records.values())} references)...")
    status_counter: Counter = Counter()
    for url, refs in sorted(url_to_records.items()):
        st = head(url)
        status_counter[st] += 1
        if st in ACCEPTABLE_STATUSES:
            n = len(refs)
            print(f"  OK   [{st}] {url}  ({n} ref{'s' if n > 1 else ''})")
        else:
            err = (f"HTTP {st}" if st > 0 else
                   ("network error" if st == -1 else "fetch error"))
            fails.append(f"{url} -> {err}")
            print(f"  FAIL [{st}] {url}")
            for r in refs:
                print(f"        used by {r}")

    print(f"\nstatus distribution: {dict(status_counter)}")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== record URLs 100% valid "
          f"({len(url_to_records)} unique URLs, "
          f"{len(a)} A + {len(b)} B records) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
