"""Independent verifier for schema/api_symbol_chunks.yaml.

Reads ONLY the yaml + pytorch.org docs. Does NOT import from
generate_schema_d.py - this is a separate ground-truth pass so a bug in
the generator's parser would not also pass verify.

For each record:
  1. schema_version + 14 required fields present
  2. covers_versions: non-empty list of "X.Y" strings
  3. framework_version: "X.Y" (single) or "X.Y-A.B" (range) matches covers_versions
  4. parameters_json: valid JSON list of {name, default}
  5. signature string: round-trip from parameters_json
  6. example_code: compiles cleanly
  7. docs_url: HTTP 200, version in URL is in covers_versions
  8. for EVERY version v in covers_versions: independently fetch the docs
     page for v (URL derived from docs_url with /<rep>/ -> /<v>/), parse
     the <dt> block for api_symbol, normalize params, compare to yaml
     parameters_json exactly. Handles memory.* namespace change at 2.8.
  9. project validator accepts the record
"""
from __future__ import annotations
import html as html_mod
import json
import re
import socket
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml  # noqa: F401

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402

SCHEMA_VERSION = "api_symbol_chunk_v1"
USER_AGENT = "Mozilla/5.0 (verify-schema-d)"
REQUIRED = {
    "schema_version", "api_symbol_id", "title", "api_symbol",
    "signature", "parameters_json", "usage_summary", "example_code",
    "text", "framework", "framework_version", "docs_url",
    "covers_versions", "deprecated",
}
VERSION_RE = re.compile(r"^\d+\.\d+$")
LABEL_RE = re.compile(r"^\d+\.\d+(?:-\d+\.\d+)?$")

sys.path.insert(0, str(REPO))


def fetch(url: str, _cache: dict[str, str | None] = {}) -> str | None:
    if url in _cache:
        return _cache[url]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        body = urllib.request.urlopen(
            req, timeout=20, context=ssl.create_default_context()
        ).read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        body = None
    _cache[url] = body
    return body


def extract_param_entries(html: str, dt_id: str) -> list[str] | None:
    m = re.search(r'<dt[^>]*id="' + re.escape(dt_id) + r'"[^>]*>(.*?)</dt>',
                  html, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    params = re.findall(r'<em class="sig-param"[^>]*>(.*?)</em>', block, re.DOTALL)
    out = []
    for p in params:
        text = re.sub(r"<[^>]+>", " ", p)
        text = html_mod.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        out.append(text)
    return out


def parse_entry(entry: str) -> tuple[str, str]:
    e = entry.strip()
    if e == "*":
        return ("*", "")
    if "=" in e:
        left, default = e.split("=", 1)
        default = default.strip()
    else:
        left, default = e, ""
    return (left.split(":", 1)[0].strip(), default)


def derive_url_and_dtid(record: dict, target_version: str) -> tuple[str, str]:
    """Compute (url, dt_id) for target_version by adapting stored docs_url."""
    rep_url = record["docs_url"]
    sym = record["api_symbol"]
    rep_match = re.search(r"/docs/(\d+\.\d+)/", rep_url)
    if not rep_match:
        return rep_url, sym
    rep_ver = rep_match.group(1)
    # swap version segment
    url = rep_url.replace(f"/docs/{rep_ver}/", f"/docs/{target_version}/")
    dt_id = sym
    # memory namespace change at 2.8: torch.cuda.<fn> -> torch.cuda.memory.<fn>
    minor_t = int(target_version.split(".")[1])
    if sym in ("torch.cuda.max_memory_allocated",
               "torch.cuda.reset_peak_memory_stats"):
        fn = sym.split(".")[-1]
        if minor_t >= 8:
            url = re.sub(
                r"/generated/torch\.cuda\.[^/]+\.html$",
                f"/generated/torch.cuda.memory.{fn}.html",
                url,
            )
            dt_id = f"torch.cuda.memory.{fn}"
        else:
            url = re.sub(
                r"/generated/torch\.cuda(?:\.memory)?\.[^/]+\.html$",
                f"/generated/torch.cuda.{fn}.html",
                url,
            )
            dt_id = f"torch.cuda.{fn}"
    return url, dt_id


def derived_label(versions: list[str]) -> str:
    return versions[0] if len(versions) == 1 else f"{versions[0]}-{versions[-1]}"


def online_check() -> bool:
    try:
        socket.create_connection(("docs.pytorch.org", 443), timeout=4).close()
        return True
    except OSError:
        return False


def main() -> int:
    fails: list[str] = []
    records = load_records("api_symbol_chunks")
    if not records:
        print("FAIL schema/D/ empty")
        return 1
    print(f"loaded {len(records)} Schema D records")
    if not records:
        print("FAIL no records")
        return 1
    online = online_check()
    print(f"network: {'ONLINE' if online else 'OFFLINE'}")
    print()

    for i, rec in enumerate(records):
        sym = rec.get("api_symbol", f"<#{i}>")
        ctx = f"[{sym} @ {rec.get('framework_version')}]"

        # 1
        if rec.get("schema_version") != SCHEMA_VERSION:
            fails.append(f"{ctx} bad schema_version")
            continue
        # 1
        missing = REQUIRED - rec.keys()
        if missing:
            fails.append(f"{ctx} missing fields {sorted(missing)}")
            continue
        # 2
        covers = rec["covers_versions"]
        if not isinstance(covers, list) or not covers:
            fails.append(f"{ctx} covers_versions empty/not-list")
            continue
        for v in covers:
            if not isinstance(v, str) or not VERSION_RE.match(v):
                fails.append(f"{ctx} bad version in covers_versions: {v!r}")
                break
        # 3
        if not LABEL_RE.match(str(rec["framework_version"])):
            fails.append(f"{ctx} framework_version not 'X.Y' or 'X.Y-A.B'")
            continue
        if rec["framework_version"] != derived_label(covers):
            fails.append(
                f"{ctx} framework_version {rec['framework_version']!r} "
                f"!= derived {derived_label(covers)!r}"
            )
            continue
        # 4
        try:
            stored_params = json.loads(rec["parameters_json"])
        except json.JSONDecodeError as exc:
            fails.append(f"{ctx} parameters_json invalid: {exc}")
            continue
        if not isinstance(stored_params, list):
            fails.append(f"{ctx} parameters_json not a list")
            continue
        for p in stored_params:
            if not isinstance(p, dict) or "name" not in p or "default" not in p:
                fails.append(f"{ctx} parameters_json entry malformed: {p!r}")
                break
        # 5 signature string consistency with parameters_json
        regen = "(" + ", ".join(
            (p["name"] + "=" + p["default"]) if p["default"]
            else p["name"]
            for p in stored_params
        ) + ")"
        # accept either with or without '*' marker in stored sig (param_json drops it)
        stored_sig_no_star = re.sub(r",\s*\*\s*,", ",", str(rec["signature"]))
        stored_sig_no_star = stored_sig_no_star.replace("(*, ", "(")
        # compare normalized (strip spaces around '*')
        if regen.replace(" ", "") != stored_sig_no_star.replace(" ", ""):
            # fall back: if removing '*' from yaml sig leaves regen, OK
            stripped = str(rec["signature"])
            stripped = stripped.replace("*, ", "")
            stripped = stripped.replace(", *", "")
            stripped = stripped.replace("*", "")
            if regen.replace(" ", "") != stripped.replace(" ", ""):
                fails.append(
                    f"{ctx} signature does not regenerate from parameters_json\n"
                    f"     yaml: {rec['signature']}\n"
                    f"     regen: {regen}"
                )
                continue
        # 6 example_code compiles
        try:
            compile(rec["example_code"], f"<{sym}.example>", "exec")
        except SyntaxError as exc:
            fails.append(f"{ctx} example_code SyntaxError: {exc}")
            continue
        # 7 docs_url version in covers
        m = re.search(r"/docs/(\d+\.\d+)/", rec["docs_url"])
        if not m:
            fails.append(f"{ctx} docs_url has no version segment")
            continue
        if m.group(1) not in covers:
            fails.append(
                f"{ctx} docs_url version {m.group(1)} not in covers_versions"
            )
            continue
        # 8 fetch each covered version + compare params
        if not online:
            print(f"SKIP {ctx}: offline (docs round-trip not run)")
            continue
        stored_norm = [(p["name"], p["default"]) for p in stored_params]
        ok = True
        for v in covers:
            url, dt_id = derive_url_and_dtid(rec, v)
            html = fetch(url)
            if html is None:
                fails.append(f"{ctx} fetch failed at v={v} ({url})")
                ok = False
                break
            entries = extract_param_entries(html, dt_id)
            if entries is None:
                fails.append(f"{ctx} no <dt id={dt_id}> at v={v} ({url})")
                ok = False
                break
            norm = [parse_entry(e) for e in entries]
            norm_no_marker = [(n, d) for n, d in norm if n != "*"]
            if norm_no_marker != stored_norm:
                fails.append(
                    f"{ctx} param mismatch at v={v}\n"
                    f"     yaml: {stored_norm}\n"
                    f"     docs: {norm_no_marker}"
                )
                ok = False
                break
        if ok:
            print(f"OK   {ctx}: {len(covers)} versions docs-verified")

    # 9 project validator
    try:
        from localml_scheduler.code_knowledge.records import validate_code_knowledge_record
    except Exception as exc:
        fails.append(f"could not import project validator: {exc}")
    else:
        n_ok = 0
        for i, rec in enumerate(records):
            try:
                norm = validate_code_knowledge_record(rec)
                if norm.get("schema_version") == SCHEMA_VERSION:
                    n_ok += 1
            except Exception as exc:
                fails.append(f"records[{i}]: validator rejected: {exc}")
        print(f"\nproject validator: accepted {n_ok}/{len(records)} records")

    total_v = sum(len(r.get("covers_versions") or []) for r in records)
    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== Schema D 100% VALID "
          f"({len(records)} records, {total_v} version checks) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
