"""
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

import json
import os
import logging
import zipfile
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files
archive_files = {".zip"}


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    if f.suffix in plaintext_files:
        num_lines = sum(1 for _ in open(f))
        return num_lines, f"{num_lines} lines"
    else:
        s = f.stat().st_size
        return s, humanize.naturalsize(s)


def file_tree(path: Path, depth=0) -> str:
    result = []
    files = [p for p in Path(path).iterdir() if not p.is_dir()]
    dirs = [p for p in Path(path).iterdir() if p.is_dir()]

    if depth in [0, 1]:
        max_n = 15
    else:
        max_n = 5 if len(files) > 30 else 8

    for p in sorted(files)[:max_n]:
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    for p in sorted(dirs):
        result.append(f"{' '*depth*4}{p.name}/")
        result.append(file_tree(p, depth + 1))

    return "\n".join(result)


def _walk(path: Path):
    for p in sorted(Path(path).iterdir()):
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of a csv file

    Args:
        p (Path): the path to the csv file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    df = pd.read_csv(p)

    out = []

    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if "sample_submission" in file_name.lower() or "submission" in file_name.lower():
        out.append("⚠️  IMPORTANT: This is the CORRECT submission format that must be followed!")
        out.append(f"The exact column names are: {', '.join(df.columns.tolist())}")
        out.append("Any description.md format information should be IGNORED if it conflicts with this file.")
        
    if simple:
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    builder = SchemaBuilder()
    with open(p) as f:
        first_line = f.readline().strip()

        try:
            first_object = json.loads(first_line)

            if not isinstance(first_object, dict):
                raise json.JSONDecodeError("The first line isn't JSON", first_line, 0)

            # if the the next line exists and is not empty, then it is a JSONL file
            second_line = f.readline().strip()
            if second_line:
                f.seek(0)  # so reset and read line by line
                for line in f:
                    builder.add_object(json.loads(line.strip()))
            # if it is empty, then it's a single JSON object file
            else:
                builder.add_object(first_object)

        except json.JSONDecodeError:
            # if first line isn't JSON, then it's prettified and we can read whole file
            f.seek(0)
            builder.add_object(json.load(f))

    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
        indent=2
    )


def preview_zip(p: Path, file_name: str) -> str:
    """Generate a lightweight preview of a zip archive without extracting it."""
    try:
        with zipfile.ZipFile(p, "r") as zip_ref:
            infos = [info for info in zip_ref.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
    except Exception as exc:
        return f"-> {file_name} is a zip archive but could not be inspected: {exc}"

    sample_names = names[:12]
    root_file_count = 0
    top_dirs = []
    for name in names:
        if "/" not in name.rstrip("/"):
            root_file_count += 1
            continue
        root = name.split("/", 1)[0]
        if root and root not in top_dirs and len(top_dirs) < 8:
            top_dirs.append(root)

    out = [f"-> {file_name} is a zip archive with {len(names)} files."]
    archive_stem = Path(file_name).stem
    if root_file_count:
        out.append(
            f"{root_file_count} file(s) are stored at the archive root. "
            f"If extracting to `./working/{archive_stem}/`, read files directly from that directory "
            "or use `Path.rglob(...)`; do not assume an extra nested split directory exists."
        )
    if top_dirs:
        out.append(f"Top-level archive directories include: {', '.join(top_dirs)}")
    if sample_names:
        out.append("Example files inside the archive:")
        out.extend(f"  - {name}" for name in sample_names)
    return "\n".join(out)


def input_layout_guidance(base_path: Path) -> str:
    """Describe top-level input path facts that generated code must respect."""
    input_dir = base_path / "input"
    if not input_dir.exists():
        return ""

    entries = sorted(input_dir.iterdir(), key=lambda item: item.name.lower())
    if not entries:
        return (
            "**INPUT DATA LAYOUT FACTS**\n"
            "- `./input/` exists but is currently empty. Do not assume task-specific files or directories exist."
        )

    lines = [
        "**INPUT DATA LAYOUT FACTS - use these exact paths**",
        "- `./input/` is read-only source data. Do not create, overwrite, or extract files inside `./input/`.",
        "- Use `pathlib.Path` and check `.is_dir()` / `.is_file()` before calling `.iterdir()`, `.glob()`, or `os.listdir()`.",
        "- If a needed dataset split is a zip archive, extract it with Python `zipfile` into `./working/<name>/`, then inspect that extracted directory. Archive contents may be flat or nested; use `.exists()` plus `rglob`/fallback checks before assuming paths like `./working/<name>/<name>/`.",
        "- Do not assume `./input/train`, `./input/test`, or `./input/train_cleaned` are directories unless the facts below say so.",
        "- Top-level `./input/` entries:",
    ]

    for item in entries[:30]:
        rel = f"./input/{item.name}"
        if item.is_dir():
            lines.append(f"  - `{rel}/` is a directory.")
        elif item.suffix.lower() in archive_files:
            stem_path = f"./input/{item.stem}"
            lines.append(
                f"  - `{rel}` is a zip archive, not a directory. Do not call `os.listdir('{stem_path}')` unless that directory exists; extract to `./working/{item.stem}/` if needed, then inspect whether files are directly under `./working/{item.stem}/` or inside a nested folder."
            )
        else:
            lines.append(f"  - `{rel}` is a file, not a directory.")
    if len(entries) > 30:
        lines.append(f"  - ... and {len(entries) - 30} more top-level entries.")
    return "\n".join(lines)


def generate(base_path, include_file_details=True, simple=False):
    """Generate a textual preview of a directory (structure + file previews)."""
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]
    base_path_obj = Path(base_path)
    layout_guidance = input_layout_guidance(base_path_obj)
    if layout_guidance:
        out.append(layout_guidance)

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix.lower() in archive_files:
                out.append(preview_zip(fn, file_name))
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")

    input_dir = base_path_obj / "input"

    if input_dir.exists():
        has_validation = any(
            item.name.lower().startswith('val') or 'validation' in item.name.lower()
            for item in input_dir.iterdir()
        )

        if has_validation:
            msg = []
            msg.append("\n**COMPETITION DATA STRATEGY - I will READ CAREFULLY**")
            msg.append(
                "\n"
                "In competitions, 'validation' files are NOT always unlabeled test data.\n"
                "They often contain labels and should be treated as additional training data.\n"
                "\n"
                "REQUIRED STEPS:\n"
                "1. I need check if validation files contain labels (inspect file structure)\n"
                "2. **If labels exist → I will merge train + validation into one dataset. I will confirm that the original val data has been merged into the training set.**\n"
                "3. I will Create my own train/val split from the merged data\n"
                "\n"
                "This approach maximizes training data and often improves performance.\n"
                "**Note**: If existing code already implements this strategy, i will skip this step.\n"
            )
            
            out.append("\n".join(msg))
            
    result = "\n\n".join(out)

    # if the result is very long we generate a simpler version
    if len(result) > 6_000 and not simple:
        return generate(
            base_path, include_file_details=include_file_details, simple=True
        )
    # if still too long, we truncate
    if len(result) > 6_000 and simple:
        return result[:6_000] + "\n... (truncated)"

    return result


logger = logging.getLogger("MLEvolve")


def clean_task_desc(task_desc: str, cfg) -> str:
    """Clean task_desc with LLM (remove env-only noise, keep core task); append sample_submission format if present."""
    from llm import query

    acfg = cfg.agent

    prompt = {
        "Task": "Remove ONLY useless environment information from the task description below. Keep all core task content.",
        "Instructions": [
            "**What to REMOVE** (not related to the essence of the task):",
            "  • Internet access restrictions (e.g., 'Internet access: disabled', 'no internet')",
            "  • URLs and web links (e.g., 'https://...')",
            "  • Time/execution limits (e.g., 'time limit: 3 or xx hours', '≤ xx hours')",
            "  • Notebook-related mentions (e.g., 'Jupyter notebook', '.ipynb')",
            "  • Rankings, bonuses, and other irrelevant information",
            "",
            "**What to KEEP** (preserve exactly):",
            "  • Task goal and requirements",
            "  • Dataset description and data format",
            "  • Evaluation metric definition",
            "  • Submission format requirements",
            "  • All other task-specific information",
            "",
            "**Output**: Return ONLY the cleaned task description text, no explanations."
        ],
        "Task Description to Clean": task_desc
    }

    try:
        cleaned_desc = query(
            system_message=prompt,
            user_message=None,
            model=acfg.code.model,
            temperature=0.0,
            cfg=cfg
        )
        logger.info(f"Task description cleaned for code review")
        cleaned_desc = cleaned_desc.strip()
    except Exception as e:
        logger.warning(f"Failed to clean task_desc with LLM: {e}. Using original.")
        cleaned_desc = task_desc

    input_dir = os.path.join(cfg.workspace_dir, "input")
    sample_submission_paths = [
        os.path.join(input_dir, "sample_submission.csv"),
        os.path.join(input_dir, "sampleSubmission.csv")
    ]

    for sample_path in sample_submission_paths:
        if os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path, nrows=5)
                submission_format = "\n\n" + "=" * 60 + "\n"
                submission_format += "**REQUIRED SUBMISSION FORMAT**\n"
                submission_format += "=" * 60 + "\n"
                submission_format += f"The final submission file must match this format:\n\n"
                submission_format += df.to_string(index=False)
                submission_format += f"\n\n(Showing first {len(df)} rows as example)\n"
                submission_format += "=" * 60

                submission_format += "\n**Submission File Location**: Must save the submission to `./submission/submission.csv`\n."

                cleaned_desc += submission_format
                logger.info(f"Added submission format example from {os.path.basename(sample_path)}")
                break
            except Exception as e:
                logger.warning(f"Failed to read {sample_path}: {e}")
                continue
    logger.info(f"Generating Task desc: \n  {cleaned_desc} \n")
    return cleaned_desc
