"""Static preflight for declared model nodes and dataset contracts.

The module has five sections: dataset formats, TensorFlow Record parsing,
competition fixtures, source rules, and command-line entry points. ``check``
applies the controlled-corpus rules; ``check_source_syntax_only`` retains the
original source gate, while ``check_source`` adds bounded Pyright analysis.
Neither function imports or executes candidate code.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import csv
import io
import json
import math
import os
import select
import shutil
import struct
import subprocess
import sys
import time
import wave
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
BUILTINS = set(dir(builtins))
BUILTINS.add("__file__")
FUNCTIONAL = {
    "relu",
    "gelu",
    "avg_pool2d",
    "interpolate",
    "elu",
    "scaled_dot_product_attention",
    "cross_entropy",
    "softmax",
    "one_hot",
}


# Dataset contracts and format-specific validators.
@dataclass(frozen=True)
class DatasetContract:
    kind: str
    classes: int = 2
    min_rows: int = 25
    feature_columns: tuple[str, ...] = ()
    label_column: str = "label"
    text_column: str = "text"
    image_size: tuple[int, int] = (16, 16)
    image_mode: str = "RGB"
    sample_rate: int = 8000
    channels: int = 1
    sample_width: int = 2
    frame_count: int = 256


Checker = Callable[[Path, DatasetContract], set[str]]
CHECKERS: dict[str, Checker] = {}


def register_checker(kind: str):
    def register(checker: Checker) -> Checker:
        if kind in CHECKERS:
            raise ValueError(f"duplicate dataset checker: {kind}")
        CHECKERS[kind] = checker
        return checker

    return register


def dataset_issues(path: Path, contract: DatasetContract, cache: dict) -> set[str]:
    """Validate one dataset under its declared contract, caching within a run."""
    path = Path(path)
    key = (path.resolve(), contract)
    if key not in cache:
        try:
            checker = CHECKERS[contract.kind]
        except KeyError as exc:
            raise ValueError(f"unsupported dataset kind: {contract.kind}") from exc
        cache[key] = checker(path, contract)
    return cache[key]


class DatasetValidator:
    """Validate one declared dataset using a format-specific check."""

    def __init__(self, path: Path, contract: DatasetContract):
        self.path = Path(path)
        self.contract = contract

    def _label(self, raw, issues: set[str], prefix: str) -> None:
        if raw is None or raw == "":
            issues.add(
                "missing_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )
            return
        if isinstance(raw, str) and raw.lower() in ("inf", "+inf", "-inf", "infinity"):
            issues.add(
                "inf_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )
            return
        if isinstance(raw, (bool, float)):
            issues.add(
                "invalid_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )
            return
        try:
            label = int(raw)
            if not 0 <= label < self.contract.classes:
                issues.add(
                    "out_of_range_label_csv"
                    if prefix == "tabular"
                    else f"{prefix}_bad_label"
                )
        except (TypeError, ValueError):
            issues.add(
                "invalid_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )

    def _scan_csv(self, path: Path, kind: str, required: tuple[str, ...], on_row):
        if not path.is_file():
            return (
                {"missing_file" if kind == "tabular" else f"{kind}_missing_index"},
                0,
                [],
            )
        issues = set()
        rows = 0
        fields = []
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                if len(fields) != len(set(fields)):
                    issues.add(
                        "duplicate_csv_column"
                        if kind == "tabular"
                        else f"{kind}_duplicate_index_column"
                    )
                for field in required:
                    if field not in fields:
                        issues.add(
                            ("wrong_column" if field == "label" else "wrong_feature")
                            if kind == "tabular"
                            else f"{kind}_wrong_index_column"
                        )
                for row in reader:
                    rows += 1
                    on_row(row, issues)
        except (OSError, UnicodeError, csv.Error):
            issues.add(
                "unreadable_csv"
                if kind == "tabular"
                else (
                    "text_invalid_utf8"
                    if kind == "text"
                    else f"{kind}_unreadable_index"
                )
            )
        return issues, rows, fields

    def check_tabular_csv(self) -> set[str]:
        def on_row(row, issues):
            for column in self.contract.feature_columns:
                raw = row.get(column)
                if raw is None or raw == "":
                    issues.add("missing_value_csv")
                    continue
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    issues.add("nonnumeric_csv")
                    continue
                if math.isnan(number):
                    issues.add("nan_csv")
                elif math.isinf(number):
                    issues.add("inf_csv")
            self._label(row.get(self.contract.label_column), issues, "tabular")

        issues, rows, _ = self._scan_csv(
            self.path,
            "tabular",
            (*self.contract.feature_columns, self.contract.label_column),
            on_row,
        )
        if rows < self.contract.min_rows and "missing_file" not in issues:
            issues.add("too_few_rows")
        return issues

    def _asset(self, root: Path, relative, issues: set[str], kind: str) -> Path | None:
        if not isinstance(relative, str) or not relative:
            issues.add(f"{kind}_missing_asset")
            return None
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
            issues.add(f"{kind}_missing_asset")
            return None
        return candidate

    def _image_file(self, path: Path, issues: set[str]) -> None:
        try:
            with Image.open(path) as image:
                if image.size != self.contract.image_size:
                    issues.add("image_wrong_size")
                if image.mode != self.contract.image_mode:
                    issues.add("image_wrong_mode")
                image.verify()
        except (OSError, ValueError, UnidentifiedImageError):
            issues.add("image_corrupt_asset")

    def check_image_index(self) -> set[str]:
        files = set()

        def on_row(row, issues):
            self._label(row.get(self.contract.label_column), issues, "image")
            asset = self._asset(self.path.parent, row.get("file"), issues, "image")
            if asset is not None:
                if asset in files:
                    issues.add("image_duplicate_asset")
                files.add(asset)
                self._image_file(asset, issues)

        issues, rows, _ = self._scan_csv(
            self.path, "image", ("file", self.contract.label_column), on_row
        )
        if rows and rows < self.contract.min_rows:
            issues.add("image_wrong_row_count")
        return issues

    def check_image_directory(self) -> set[str]:
        issues = set()
        if not self.path.is_dir():
            return {"image_missing_directory"}
        if any(
            child.name not in {str(label) for label in range(self.contract.classes)}
            for child in self.path.iterdir()
        ):
            issues.add("image_unexpected_asset")
        count = 0
        for label in range(self.contract.classes):
            class_dir = self.path / str(label)
            if not class_dir.is_dir():
                issues.add("image_missing_class_directory")
                continue
            for asset in class_dir.iterdir():
                if not asset.is_file() or asset.suffix.lower() not in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                ):
                    issues.add("image_unexpected_asset")
                    continue
                count += 1
                self._image_file(asset, issues)
        if count < self.contract.min_rows:
            issues.add("image_wrong_row_count")
        return issues

    def check_text_csv(self) -> set[str]:
        def on_row(row, issues):
            raw = row.get(self.contract.text_column)
            if not isinstance(raw, str) or not raw.strip():
                issues.add("text_missing_value")
            self._label(row.get(self.contract.label_column), issues, "text")

        issues, rows, _ = self._scan_csv(
            self.path,
            "text",
            (self.contract.text_column, self.contract.label_column),
            on_row,
        )
        if rows and rows < self.contract.min_rows:
            issues.add("text_wrong_row_count")
        return issues

    def check_text_jsonl(self) -> set[str]:
        if not self.path.is_file():
            return {"text_missing_file"}
        issues = set()
        rows = 0
        try:
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    rows += 1
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        issues.add("text_invalid_jsonl")
                        continue
                    if not isinstance(record, dict):
                        issues.add("text_invalid_jsonl")
                        continue
                    raw = record.get(self.contract.text_column)
                    if not isinstance(raw, str) or not raw.strip():
                        issues.add("text_missing_value")
                    self._label(record.get(self.contract.label_column), issues, "text")
        except (OSError, UnicodeError):
            issues.add("text_invalid_utf8")
        if rows < self.contract.min_rows:
            issues.add("text_wrong_row_count")
        return issues

    def _audio_file(self, path: Path, issues: set[str]) -> None:
        try:
            with wave.open(str(path), "rb") as handle:
                if handle.getframerate() != self.contract.sample_rate:
                    issues.add("audio_bad_rate")
                if handle.getnchannels() != self.contract.channels:
                    issues.add("audio_stereo")
                if handle.getsampwidth() != self.contract.sample_width:
                    issues.add("audio_bad_width")
                if handle.getnframes() != self.contract.frame_count:
                    issues.add("audio_truncated")
        except (OSError, ValueError, EOFError, wave.Error):
            issues.add("audio_corrupt")

    def check_audio_index(self) -> set[str]:
        def on_row(row, issues):
            self._label(row.get(self.contract.label_column), issues, "audio")
            asset = self._asset(self.path.parent, row.get("file"), issues, "audio")
            if asset is not None:
                self._audio_file(asset, issues)

        issues, rows, _ = self._scan_csv(
            self.path, "audio", ("file", self.contract.label_column), on_row
        )
        if rows and rows < self.contract.min_rows:
            issues.add("audio_wrong_row_count")
        return issues

    def check_audio_directory(self) -> set[str]:
        issues = set()
        if not self.path.is_dir():
            return {"audio_missing_directory"}
        if any(
            child.name not in {str(label) for label in range(self.contract.classes)}
            for child in self.path.iterdir()
        ):
            issues.add("audio_unexpected_asset")
        count = 0
        for label in range(self.contract.classes):
            class_dir = self.path / str(label)
            if not class_dir.is_dir():
                issues.add("audio_missing_class_directory")
                continue
            for asset in class_dir.iterdir():
                if not asset.is_file() or asset.suffix.lower() != ".wav":
                    issues.add("audio_unexpected_asset")
                    continue
                count += 1
                self._audio_file(asset, issues)
        if count < self.contract.min_rows:
            issues.add("audio_wrong_row_count")
        return issues


CHECKERS.update(
    {
        "tabular_csv": lambda path, contract: DatasetValidator(
            path, contract
        ).check_tabular_csv(),
        "image_csv_index": lambda path, contract: DatasetValidator(
            path, contract
        ).check_image_index(),
        "image_directory": lambda path, contract: DatasetValidator(
            path, contract
        ).check_image_directory(),
        "text_csv": lambda path, contract: DatasetValidator(
            path, contract
        ).check_text_csv(),
        "text_jsonl": lambda path, contract: DatasetValidator(
            path, contract
        ).check_text_jsonl(),
        "audio_csv_index": lambda path, contract: DatasetValidator(
            path, contract
        ).check_audio_index(),
        "audio_directory": lambda path, contract: DatasetValidator(
            path, contract
        ).check_audio_directory(),
    }
)

check_tabular_csv = CHECKERS["tabular_csv"]
check_image_index = CHECKERS["image_csv_index"]
check_image_directory = CHECKERS["image_directory"]
check_text_csv = CHECKERS["text_csv"]
check_text_jsonl = CHECKERS["text_jsonl"]
check_audio_index = CHECKERS["audio_csv_index"]
check_audio_directory = CHECKERS["audio_directory"]


_MASK = 0xFFFFFFFF
# TensorFlow Record framing and minimal Example decoding.
_POLYNOMIAL = 0x82F63B78
_MAX_RECORD_BYTES = 128 * 1024 * 1024


def _table():
    values = []
    for start in range(256):
        value = start
        for _ in range(8):
            value = (value >> 1) ^ (_POLYNOMIAL if value & 1 else 0)
        values.append(value)
    return tuple(values)


_CRC_TABLE = _table()


def masked_crc32c(data):
    crc = _MASK
    for byte in data:
        crc = _CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= _MASK
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & _MASK


def records(path: Path):
    """Yield complete payloads; reject truncated or checksum-failing records."""
    with path.open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError("truncated TFRecord length")
            length_crc = handle.read(4)
            if len(length_crc) != 4 or struct.unpack("<I", length_crc)[
                0
            ] != masked_crc32c(length_bytes):
                raise ValueError("invalid TFRecord length checksum")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > _MAX_RECORD_BYTES:
                raise ValueError("oversized TFRecord")
            payload = handle.read(length)
            data_crc = handle.read(4)
            if len(payload) != length or len(data_crc) != 4:
                raise ValueError("truncated TFRecord payload")
            if struct.unpack("<I", data_crc)[0] != masked_crc32c(payload):
                raise ValueError("invalid TFRecord payload checksum")
            yield payload


def _varint(data, offset):
    result = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
    raise ValueError("oversized protobuf varint")


def _fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("invalid protobuf field")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 2:
            length, offset = _varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf bytes")
            value = data[offset:end]
            offset = end
        elif wire in (1, 5):
            end = offset + (8 if wire == 1 else 4)
            if end > len(data):
                raise ValueError("truncated protobuf fixed field")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError("unsupported protobuf wire type")
        yield number, wire, value


def _only(fields, number, wire):
    matches = [
        value for field, kind, value in fields if field == number and kind == wire
    ]
    if len(matches) != 1:
        raise ValueError("missing or repeated protobuf feature")
    return matches[0]


def cassava_example(payload):
    """Read image bytes and integer target from a tf.train.Example payload."""
    features_blob = _only(list(_fields(payload)), 1, 2)
    features = {}
    for number, wire, entry in _fields(features_blob):
        if (number, wire) != (1, 2):
            continue
        parts = list(_fields(entry))
        key = _only(parts, 1, 2).decode("utf-8")
        if key in features:
            raise ValueError("duplicate protobuf feature")
        features[key] = _only(parts, 2, 2)
    image_feature = _only(list(_fields(features["image"])), 1, 2)
    image = _only(list(_fields(image_feature)), 1, 2)
    target_feature = _only(list(_fields(features["target"])), 3, 2)
    target_fields = list(_fields(target_feature))
    unpacked = [
        value for number, wire, value in target_fields if (number, wire) == (1, 0)
    ]
    packed = [
        value for number, wire, value in target_fields if (number, wire) == (1, 2)
    ]
    if packed:
        if len(packed) != 1 or unpacked:
            raise ValueError("repeated target")
        offset = 0
        unpacked = []
        while offset < len(packed[0]):
            value, offset = _varint(packed[0], offset)
            unpacked.append(value)
    if len(unpacked) != 1:
        raise ValueError("missing or repeated target")
    target = unpacked[0]
    if target >= 1 << 63:
        target -= 1 << 64
    return image, target


# Competition-specific dataset and source checks.
TASKS = {
    "petfinder-pawpularity-score": ("Id", "Pawpularity", "regression", 1),
    "cassava-leaf-disease-classification": ("image_id", "label", "classification", 5),
    "uw-madison-gi-tract-image-segmentation": (
        "id",
        "segmentation",
        "segmentation",
        768,
    ),
    "aptos2019-blindness-detection": ("id_code", "diagnosis", "classification", 5),
    "seti-breakthrough-listen": ("id", "target", "classification", 2),
    "freesound-audio-tagging-2019": ("fname", "labels", "multilabel", 3),
    "google-quest-challenge": (
        "qa_id",
        "question_asker_intent_understanding",
        "regression",
        30,
    ),
    "jigsaw-toxic-comment-classification-challenge": ("id", "toxic", "multilabel", 6),
    "chaii-hindi-and-tamil-question-answering": ("id", "answer_start", "regression", 2),
}
PHOTO = "Focus Eyes Face Near Action Accessory Group Collage Human Occlusion Info Blur".split()
ORGANS = ("large_bowel", "small_bowel", "stomach")
JIGSAW = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

QUEST_FIRST = "question_asker_intent_understanding"


class MissingZipCSV(Exception):
    pass


@contextmanager
def _csv_stream(path):
    if path.suffix.lower() != ".zip":
        with path.open(newline="", encoding="utf-8") as handle:
            yield handle
        return
    with zipfile.ZipFile(path) as archive:
        csv_names = [
            item.filename
            for item in archive.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".csv")
        ]
        preferred = path.name[:-4]
        if preferred in csv_names:
            member = preferred
        elif len(csv_names) == 1:
            member = csv_names[0]
        else:
            raise MissingZipCSV(path)
        with archive.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
                yield handle


def assignment(tree, key):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == key for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
    return None


def _number(raw, low, high):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and low <= value <= high


def _kaggle_asset(root, relative, issues):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        issues.add("missing_asset")
        return None
    return path


def _image(path, issues, grayscale=False):
    try:
        with Image.open(path) as image:
            if (
                min(image.size) < 1
                or (grayscale and image.mode not in ("I;16", "I;16B", "I"))
                or (not grayscale and image.mode != "RGB")
            ):
                issues.add("bad_asset_shape_or_type")
            dimensions = image.size
            image.load()
            return dimensions
    except (OSError, ValueError, UnidentifiedImageError):
        issues.add("corrupt_asset")
        return None


def _image_in_archive(root, archive_name, filename, issues):
    archive_path = root / archive_name
    if not archive_path.is_file():
        issues.add("missing_asset")
        return
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = [
                item
                for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).name == filename
            ]
            if len(matches) != 1:
                issues.add("missing_asset" if not matches else "ambiguous_asset")
                return
            with archive.open(matches[0]) as raw:
                _image(io.BytesIO(raw.read()), issues)
    except zipfile.BadZipFile:
        issues.add("zip_corrupt")
    except (OSError, ValueError):
        issues.add("corrupt_asset")


def _rle(raw, size, issues):
    if raw in (None, ""):
        return
    try:
        values = [int(v) for v in raw.split()]
    except (AttributeError, ValueError):
        issues.add("invalid_target")
        return
    if len(values) % 2 or any(
        start < 1 or length < 1 or start + length - 1 > size
        for start, length in zip(values[::2], values[1::2])
    ):
        issues.add("invalid_target")


def _audio_info(root, folder, filename, issues):
    import soundfile as sf

    path = root / folder / filename
    try:
        if path.is_file():
            info = sf.info(path)
        else:
            archive_path = root / f"{folder}.zip"
            if not archive_path.is_file():
                issues.add("missing_asset")
                return
            with zipfile.ZipFile(archive_path) as archive:
                matches = [
                    item
                    for item in archive.infolist()
                    if not item.is_dir() and Path(item.filename).name == filename
                ]
                if len(matches) != 1:
                    issues.add("missing_asset" if not matches else "ambiguous_asset")
                    return
                with archive.open(matches[0]) as raw:
                    info = sf.info(io.BytesIO(raw.read()))
        if info.frames < 1 or info.samplerate < 1 or info.channels not in (1, 2):
            issues.add("bad_asset_shape_or_type")
    except zipfile.BadZipFile:
        issues.add("zip_corrupt")
    except (OSError, ValueError, RuntimeError):
        issues.add("corrupt_asset")


def _cassava_tfrecord_issues(index, issues):
    shards = sorted(
        path
        for path in index.iterdir()
        if path.is_file() and path.suffix.lower() in (".tfrec", ".tfrecord")
    )
    if not shards:
        issues.add("missing_file")
        return
    count = 0
    for shard in shards:
        try:
            for payload in records(shard):
                count += 1
                try:
                    image, target = cassava_example(payload)
                except (KeyError, UnicodeError, ValueError):
                    issues.add("invalid_record_schema")
                    continue
                if target not in range(5):
                    issues.add("invalid_target")
                _image(io.BytesIO(image), issues)
        except (OSError, ValueError):
            issues.add("corrupt_tfrecord")
    if count < 6:
        issues.add("too_few_rows")


class CompetitionValidator:
    """Check one competition index and its referenced assets."""

    def __init__(self, task, index, root, cache):
        self.task = task
        self.index = index
        self.root = root
        self.cache = cache
        self.issues = set()
        self.required = set()
        self.vocabulary = set()
        self.organ_seen = {}
        self.target_field = None

    def _schema(self):
        """Return required columns and optional label vocabulary."""
        input_field, target_field, _, _ = TASKS[self.task]
        required = {input_field, target_field}
        vocabulary = set()
        if self.task == "petfinder-pawpularity-score":
            required.update(PHOTO)
        elif self.task == "uw-madison-gi-tract-image-segmentation":
            required.add("class")
        elif self.task == "google-quest-challenge":
            required.update(("question_title", "question_body", "answer"))
            sample = self.root / "sample_submission.csv"
            if sample.is_file():
                with sample.open(newline="", encoding="utf-8") as handle:
                    targets = (csv.reader(handle).__next__())[1:]
                required.update(targets)
                if len(targets) != 30 or targets[0] != QUEST_FIRST:
                    self.issues.add("wrong_target_schema")
            else:
                self.issues.add("missing_target_schema")
        elif self.task == "jigsaw-toxic-comment-classification-challenge":
            required.update(JIGSAW)
            required.add("comment_text")
        elif self.task == "chaii-hindi-and-tamil-question-answering":
            required.update(("context", "question", "answer_text", "language"))
        if self.task == "freesound-audio-tagging-2019":
            try:
                with (self.root / "sample_submission.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    vocabulary = set(next(csv.reader(handle))[1:])
            except (OSError, UnicodeError, StopIteration):
                self.issues.add("missing_target_schema")
        return required, vocabulary

    def _check_petfinder_row(self, row, ident):
        if not _number(row.get(self.target_field), 0, 100) or any(
            not _number(row.get(column), 0, 1) for column in PHOTO
        ):
            self.issues.add("invalid_target")
        path = _kaggle_asset(self.root, f"train/{ident}.jpg", self.issues)
        if path:
            _image(path, self.issues)

    def _check_cassava_row(self, row, ident):
        target = row.get(self.target_field)
        if not _number(target, 0, 4) or not str(target).isdigit():
            self.issues.add("invalid_target")
        path = _kaggle_asset(self.root, f"train_images/{ident}", self.issues)
        if path:
            _image(path, self.issues)

    def _check_gi_tract_row(self, row, ident):
        organ = row.get("class")
        if organ not in ORGANS:
            self.issues.add("invalid_target")
        self.organ_seen.setdefault(ident, set()).add(organ)
        try:
            case, day, _, slice_id = str(ident).split("_")
            scans = list(
                (self.root / "train" / case / f"{case}_{day}" / "scans").glob(
                    f"slice_{slice_id}_*.png"
                )
            )
        except ValueError:
            self.issues.add("missing_asset")
            return
        if len(scans) != 1:
            self.issues.add("missing_asset" if not scans else "ambiguous_asset")
            return
        dimensions = _image(scans[0], self.issues, grayscale=True)
        if dimensions:
            _rle(row.get(self.target_field), dimensions[0] * dimensions[1], self.issues)

    def _check_aptos_row(self, row, ident):
        target = row.get(self.target_field)
        if not str(target).isdigit() or not _number(target, 0, 4):
            self.issues.add("invalid_target")
        path = self.root / "train_images" / f"{ident}.png"
        if path.is_file():
            _image(path, self.issues)
        else:
            _image_in_archive(self.root, "train.zip", f"{ident}.png", self.issues)

    def _check_seti_row(self, row, ident):
        if str(row.get(self.target_field)) not in ("0", "1"):
            self.issues.add("invalid_target")
        path = _kaggle_asset(
            self.root, f"train/{str(ident)[0]}/{ident}.npy", self.issues
        )
        if path:
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != (6, 273, 256) or array.dtype != np.float16:
                    self.issues.add("bad_asset_shape_or_type")
            except (OSError, ValueError, EOFError):
                self.issues.add("corrupt_asset")

    def _check_freesound_row(self, row, ident):
        labels = str(row.get(self.target_field, "")).split(",")
        if not labels or any(label not in self.vocabulary for label in labels):
            self.issues.add("invalid_target")
        folder = (
            "train_noisy"
            if self.index.name.startswith("train_noisy")
            else "train_curated"
        )
        _audio_info(self.root, folder, str(ident), self.issues)

    def _check_quest_row(self, row, ident):
        if any(
            not str(row.get(column, "")).strip()
            for column in ("question_title", "question_body", "answer")
        ):
            self.issues.add("missing_text")
        if any(
            not _number(row.get(column), 0, 1)
            for column in self.required
            if column.startswith(("question_", "answer_"))
            and column not in ("question_title", "question_body", "answer")
        ):
            self.issues.add("invalid_target")

    def _check_jigsaw_row(self, row, ident):
        if not str(row.get("comment_text", "")).strip():
            self.issues.add("missing_text")
        if any(row.get(column) not in ("0", "1") for column in JIGSAW):
            self.issues.add("invalid_target")

    def _check_chaii_row(self, row, ident):
        context = row.get("context") or ""
        question = row.get("question") or ""
        answer = row.get("answer_text") or ""
        if not context.strip() or not question.strip() or not answer.strip():
            self.issues.add("missing_text")
        if row.get("language") not in ("hindi", "tamil"):
            self.issues.add("invalid_language")
        try:
            start = int(row.get(self.target_field))
            if start < 0 or (
                context[start : start + len(answer)] != answer and answer not in context
            ):
                self.issues.add("invalid_target")
        except (ValueError, TypeError):
            self.issues.add("invalid_target")

    _ROW_CHECKERS = {
        "petfinder-pawpularity-score": _check_petfinder_row,
        "cassava-leaf-disease-classification": _check_cassava_row,
        "uw-madison-gi-tract-image-segmentation": _check_gi_tract_row,
        "aptos2019-blindness-detection": _check_aptos_row,
        "seti-breakthrough-listen": _check_seti_row,
        "freesound-audio-tagging-2019": _check_freesound_row,
        "google-quest-challenge": _check_quest_row,
        "jigsaw-toxic-comment-classification-challenge": _check_jigsaw_row,
        "chaii-hindi-and-tamil-question-answering": _check_chaii_row,
    }

    def _rows(self):
        """Stream rows and report index read failures as findings."""
        try:
            with _csv_stream(self.index) as handle:
                yield from csv.DictReader(handle)
        except MissingZipCSV:
            self.issues.add("zip_missing_csv")
        except zipfile.BadZipFile:
            self.issues.add("zip_corrupt")
        except (OSError, UnicodeError, csv.Error):
            self.issues.add("unreadable_index")

    def check(self):
        key = (self.task, self.index.resolve(), self.root.resolve())
        if key in self.cache:
            return self.cache[key]
        if self.task == "cassava-leaf-disease-classification" and self.index.is_dir():
            _cassava_tfrecord_issues(self.index, self.issues)
            self.cache[key] = self.issues
            return self.issues
        if not self.index.is_file():
            return {"missing_file"}
        try:
            with _csv_stream(self.index) as handle:
                fields = csv.DictReader(handle).fieldnames or []
        except MissingZipCSV:
            return {"zip_missing_csv"}
        except zipfile.BadZipFile:
            return {"zip_corrupt"}
        except (OSError, UnicodeError, csv.Error):
            return {"unreadable_index"}

        input_field, self.target_field, _, _ = TASKS[self.task]
        self.required, self.vocabulary = self._schema()
        if len(fields) != len(set(fields)):
            self.issues.add("duplicate_column")
        if not self.required.issubset(fields):
            self.issues.add("wrong_header")

        check_row = self._ROW_CHECKERS[self.task]
        seen = set()
        row_count = 0
        for row in self._rows():
            row_count += 1
            ident = row.get(input_field)
            unique_key = (
                (ident, row.get("class"))
                if self.task == "uw-madison-gi-tract-image-segmentation"
                else ident
            )
            if unique_key in seen:
                self.issues.add("duplicate_id")
            seen.add(unique_key)
            if ident in (None, ""):
                self.issues.add(
                    "missing_text"
                    if self.task
                    in (
                        "google-quest-challenge",
                        "jigsaw-toxic-comment-classification-challenge",
                        "chaii-hindi-and-tamil-question-answering",
                    )
                    else "missing_asset"
                )
                continue
            check_row(self, row, ident)

        if self.task == "uw-madison-gi-tract-image-segmentation" and any(
            value != set(ORGANS) for value in self.organ_seen.values()
        ):
            self.issues.add("missing_organ_mask")
        if row_count < 6:
            self.issues.add("too_few_rows")
        self.cache[key] = self.issues
        return self.issues


def kaggle_dataset_issues(task, index, root, cache):
    """Compatibility entry point for competition dataset validation."""
    return CompetitionValidator(task, index, root, cache).check()


def check_kaggle(path, tree, cache):
    task = assignment(tree, "TASK")
    issues = set()
    if task not in TASKS:
        return ["unknown_task"]
    expected_input, expected_target, expected_type, expected_dim = TASKS[task]
    if assignment(tree, "INPUT_FIELD") != expected_input:
        issues.add("wrong_input_field")
    if assignment(tree, "TARGET_FIELD") != expected_target:
        issues.add("wrong_target_field")
    if (
        assignment(tree, "TASK_TYPE") != expected_type
        or assignment(tree, "OUT_DIM") != expected_dim
    ):
        issues.add("wrong_task_shape")
    index = assignment(tree, "DATA_INDEX")
    root = assignment(tree, "ASSET_ROOT")
    base = Path(__file__).resolve().parent
    if not isinstance(index, str) or not isinstance(root, str):
        issues.add("missing_file")
    else:
        index_path = (base / index).resolve()
        root_path = (base / root).resolve()
        if not index_path.is_relative_to(base) or not root_path.is_relative_to(base):
            issues.add("unsafe_path")
        else:
            issues.update(kaggle_dataset_issues(task, index_path, root_path, cache))
    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        ),
        None,
    )
    if run is None:
        return sorted(issues | {"missing_pipeline"})
    for node in ast.walk(run):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "loss" for t in node.targets
        ):
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "detach"
                for n in ast.walk(node.value)
            ):
                issues.add("detached_loss")
        if isinstance(node, ast.With) and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "no_grad"
            for item in node.items
            for n in ast.walk(item.context_expr)
        ):
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "model"
                and any(
                    isinstance(arg, ast.Name) and arg.id == "xtrain" for arg in n.args
                )
                for n in ast.walk(node)
            ):
                issues.add("no_grad_train")
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Tuple)
            and {elt.id for elt in t.elts if isinstance(elt, ast.Name)}
            == {"xval", "yval"}
            for t in node.targets
        ):
            if "features[:4]" in ast.unparse(
                node.value
            ) or "targets[:4]" in ast.unparse(node.value):
                issues.add("validation_on_train")
    return sorted(issues)


# Image, text, and audio source contracts.
def constants(tree):
    values = {}
    for n in tree.body:
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    return values


def check_modality(path, tree, cache):
    values = constants(tree)
    modality = values.get("MODALITY")
    issues = set()
    if modality not in ("image", "text", "audio"):
        return ["unknown_modality"]
    index_name = values.get("INDEX_FILE")
    if not isinstance(index_name, str) or Path(index_name).name != index_name:
        issues.add(f"{modality}_missing_index")
    else:
        index = ROOT / "data" / index_name
        kind = {
            "image": "image_csv_index",
            "text": "text_csv",
            "audio": "audio_csv_index",
        }[modality]
        issues.update(dataset_issues(index, DatasetContract(kind, min_rows=32), cache))

    # Canonicalized syntax ignores quote style and ordinary whitespace.
    code = ast.unparse(tree).replace('"', "'").replace(" ", "").replace("\n", "")
    if modality == "image":
        if "image.convert('L')" in code:
            issues.add("image_grayscale")
        if "dtype=np.uint8" in code:
            issues.add("image_uint8_tensor")
        elif "np.asarray(image,dtype=np.float32)/255.0" not in code:
            issues.add("image_unnormalized")
        if "permute(2,0,1)" not in code:
            issues.add("image_channel_order")
        if "table['class']" in code:
            issues.add("image_wrong_label_column")
        if "root/'absent'" in code:
            issues.add("image_wrong_root")
        if "resize((8,8))" in code:
            issues.add("image_bad_resize")
        if "[::-1].copy()" in code:
            issues.add("image_label_permutation")
        if values.get("IMAGE_SIZE") != (16, 16):
            issues.add("image_wrong_size_contract")
    elif modality == "text":
        if "VOCAB.get(word,len(VOCAB))" in code:
            issues.add("text_oov_bad_id")
        if "ids+[-1]*" in code:
            issues.add("text_negative_pad")
        if "str(text).split()" in code:
            issues.add("text_no_lower")
        if "split(',')" in code:
            issues.add("text_wrong_split")
        if (
            "str(text).lower().split()" in code
            and "str(text).lower().split()[:MAX_TOKENS]" not in code
        ):
            issues.add("text_no_truncate")
        if "dtype=torch.float32" in code:
            issues.add("text_float_ids")
        if "mask=(x==0)" in code:
            issues.add("text_invert_mask")
        if "table['sentence']" in code:
            issues.add("text_wrong_column")
        if "[::-1].copy()" in code:
            issues.add("text_label_permutation")
        if values.get("MAX_TOKENS") != 8:
            issues.add("text_wrong_length_contract")
    else:
        for key, expected, bug in (
            ("SAMPLE_RATE", 8000, "audio_wrong_rate_constant"),
            ("CHANNELS", 1, "audio_wrong_channels_constant"),
            ("SAMPLE_WIDTH", 2, "audio_wrong_width_constant"),
            ("FRAME_COUNT", 256, "audio_wrong_frames_constant"),
        ):
            if values.get(key) != expected:
                issues.add(bug)
        if "dtype='>i2'" in code:
            issues.add("audio_endian")
        if "/32768.0" not in code:
            issues.add("audio_no_normalize")
        if "table['class']" in code:
            issues.add("audio_wrong_label_column")
        if "[::-1].copy()" in code:
            issues.add("audio_label_permutation")
    return sorted(issues)


# Shared abstract-syntax-tree helpers and controlled model-node rules.
def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(target_names(e) for e in node.elts))
    return set()


def assigned(tree, variable):
    if not hasattr(tree, "_assignment_index"):
        index = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, ast.AnnAssign):
                targets = [n.target]
            else:
                continue
            for target in targets:
                for target_name in target_names(target):
                    index.setdefault(target_name, []).append(n)
        tree._assignment_index = index
    return tree._assignment_index.get(variable, [])


def value(statement):
    return statement.value


def unparse(node):
    return ast.unparse(node) if node is not None else ""


def has_call(node, call_name):
    return any(
        isinstance(n, ast.Call) and name(n.func) == call_name for n in ast.walk(node)
    )


def literal_assignment(tree, variable):
    matches = assigned(tree, variable)
    if not matches:
        return None
    try:
        return ast.literal_eval(value(matches[0]))
    except (ValueError, TypeError, MemoryError):
        return None


def local_unknowns(function, global_names):
    local = {
        a.arg
        for a in function.args.posonlyargs
        + function.args.args
        + function.args.kwonlyargs
    }
    if function.args.vararg:
        local.add(function.args.vararg.arg)
    if function.args.kwarg:
        local.add(function.args.kwarg.arg)
    nodes = list(ast.walk(function))
    local.update(
        n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )
    unknown = set()
    for n in nodes:
        if (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id not in local | global_names | BUILTINS
        ):
            unknown.add(n.id)
    return unknown


def check_source_syntax_only(source: str, filename: str = "<candidate>") -> dict:
    """Original source gate, retained for compatibility and comparison.

    The corpus-specific model and dataset rules below require a declared task
    contract and must not be applied to unrelated ML-Master/MLEvolve scripts.
    """
    try:
        ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return {
            "status": "rejected",
            "admitted": False,
            "assurance": "syntax_only",
            "issues": ["syntax_error"],
            "diagnostics": [
                {"code": "syntax_error", "message": str(exc), "line": exc.lineno}
            ],
        }
    return {
        "status": "syntax_valid",
        "admitted": True,
        "assurance": "syntax_only",
        "issues": [],
        "diagnostics": [],
    }


class PyrightSourceChecker:
    """Analyze one in-memory candidate over the language-server protocol."""

    REJECT_CODES = {
        "reportMissingImports",
        "reportUndefinedVariable",
        "reportUnboundVariable",
        "reportPossiblyUnboundVariable",
        "reportArgumentType",
        "reportCallIssue",
        "reportAttributeAccessIssue",
    }

    def __init__(self, executable: str, workspace: Path, timeout: float):
        self.executable = executable
        self.workspace = workspace
        self.timeout = timeout
        self.buffer = b""

    @staticmethod
    def _send(process, message: dict) -> None:
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        process.stdin.flush()

    def _receive(self, process, deadline: float) -> dict:
        while True:
            header_end = self.buffer.find(b"\r\n\r\n")
            if header_end >= 0:
                headers = self.buffer[:header_end].split(b"\r\n")
                lengths = [
                    int(line.split(b":", 1)[1].strip())
                    for line in headers
                    if line.lower().startswith(b"content-length:")
                ]
                if len(lengths) != 1 or not 0 <= lengths[0] <= 8 * 1024 * 1024:
                    raise ValueError("invalid language-server message length")
                end = header_end + 4 + lengths[0]
                if len(self.buffer) >= end:
                    body, self.buffer = self.buffer[header_end + 4:end], self.buffer[end:]
                    return json.loads(body)
            if len(self.buffer) > 8 * 1024 * 1024:
                raise ValueError("language-server response too large")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pyright analysis exceeded its time limit")
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("Pyright analysis exceeded its time limit")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Pyright language server exited before diagnostics")
            self.buffer += chunk

    def check(self, source: str, filename: str) -> list[dict]:
        deadline = time.monotonic() + self.timeout
        name = Path(filename).name
        if not name.endswith(".py") or name.startswith("<"):
            name = "candidate.py"
        uri = (self.workspace / name).as_uri()
        process = subprocess.Popen(
            [self.executable, "--stdio"],
            cwd=self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        try:
            self._send(process, {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": self.workspace.as_uri(),
                    "workspaceFolders": [{
                        "uri": self.workspace.as_uri(),
                        "name": self.workspace.name,
                    }],
                    "capabilities": {
                        "textDocument": {"publishDiagnostics": {}},
                        "workspace": {"configuration": True},
                    },
                },
            })
            while True:
                message = self._receive(process, deadline)
                if message.get("id") == 1:
                    if "error" in message:
                        raise RuntimeError(str(message["error"]))
                    break
                self._reply_to_server_request(process, message)
            self._send(process, {"jsonrpc": "2.0", "method": "initialized", "params": {}})
            self._send(process, {
                "jsonrpc": "2.0", "method": "textDocument/didOpen",
                "params": {"textDocument": {
                    "uri": uri, "languageId": "python", "version": 1, "text": source,
                }},
            })
            while True:
                message = self._receive(process, deadline)
                if message.get("method") == "textDocument/publishDiagnostics":
                    params = message.get("params", {})
                    if params.get("uri") == uri:
                        return params.get("diagnostics", [])
                self._reply_to_server_request(process, message)
        finally:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)

    @staticmethod
    def _reply_to_server_request(process, message: dict) -> None:
        if "id" not in message or "method" not in message:
            return
        result = None
        if message["method"] == "workspace/configuration":
            result = []
            for item in message.get("params", {}).get("items", []):
                section = item.get("section")
                if section == "python":
                    result.append({"pythonPath": sys.executable})
                elif section == "python.analysis":
                    result.append({"typeCheckingMode": "basic", "diagnosticMode": "openFilesOnly"})
                else:
                    result.append({})
        PyrightSourceChecker._send(process, {
            "jsonrpc": "2.0", "id": message["id"], "result": result,
        })


def _incomplete_source_analysis(report: dict, status: str, message: str) -> dict:
    return {
        **report,
        "analysis_status": status,
        "diagnostics": [{
            "code": "pyright_unavailable" if status == "unavailable" else "pyright_timeout",
            "message": message,
            "line": None,
            "severity": 2,
        }],
    }


def check_source(
    source: str,
    filename: str = "<candidate>",
    *,
    pyright_executable: str | None = None,
    workspace: Path | None = None,
    analysis_timeout: float = 8.0,
) -> dict:
    """Check source without running it, retaining the original syntax gate."""
    report = check_source_syntax_only(source, filename)
    if not report["admitted"]:
        return report
    executable = pyright_executable or os.environ.get("PYRIGHT_LANGSERVER")
    if not executable:
        executable = shutil.which("pyright-langserver")
    if not executable:
        bundled = ROOT.parent / ".cache/pyright/node_modules/.bin/pyright-langserver"
        executable = str(bundled) if bundled.is_file() else None
    if not executable:
        return _incomplete_source_analysis(
            report, "unavailable", "Pyright language server is not installed"
        )
    workspace = Path(workspace or Path.cwd()).resolve()
    try:
        raw = PyrightSourceChecker(executable, workspace, analysis_timeout).check(source, filename)
    except TimeoutError as exc:
        return _incomplete_source_analysis(report, "timeout", str(exc))
    except (OSError, RuntimeError, ValueError) as exc:
        return _incomplete_source_analysis(report, "unavailable", str(exc))
    diagnostics = []
    for item in raw:
        code = item.get("code", "pyright_diagnostic")
        if isinstance(code, dict):
            code = code.get("value", "pyright_diagnostic")
        diagnostics.append({
            "code": str(code),
            "message": item.get("message", ""),
            "line": item.get("range", {}).get("start", {}).get("line", 0) + 1,
            "severity": item.get("severity"),
        })
    errors = [
        item for item in diagnostics
        if item["severity"] == 1
        and item["code"] in PyrightSourceChecker.REJECT_CODES
    ]
    return {
        "status": "rejected" if errors else "static_checked",
        "admitted": not errors,
        "assurance": "syntax_and_pyright",
        "issues": sorted({item["code"] for item in errors}),
        "diagnostics": diagnostics,
        "analysis_status": "complete",
    }


class NodeValidator:
    """Validate one declared source node without executing it."""

    def __init__(self, path, cache):
        self.path = path
        self.cache = cache
        self.issues = set()

    def _check_names(self, tree):
        """Find unresolved source names before inspecting corpus-specific rules."""
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        globals_ = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        globals_.update(
            name
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            for name in target_names(target)
        )
        for node in tree.body:
            if isinstance(node, ast.Import):
                globals_.update(
                    alias.asname or alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                globals_.update(alias.asname or alias.name for alias in node.names)
        for function in functions:
            if local_unknowns(function, globals_):
                self.issues.add("undefined_variable")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and name(node.func.value) == "F"
            ):
                if node.func.attr not in FUNCTIONAL:
                    self.issues.add("undefined_function")
        return functions

    def _check_node_data(self, tree):
        """Check the controlled node's declared table and tensor shape."""
        data_file = literal_assignment(tree, "DATA_FILE")
        if literal_assignment(tree, "LABEL_COLUMN") != "label":
            self.issues.add("wrong_column")
        if literal_assignment(tree, "IMAGE_SHAPE") != (1, 16, 16):
            self.issues.add("bad_reshape")
        if literal_assignment(tree, "NUM_CLASSES") != 2:
            self.issues.add("wrong_output_classes")
        if not isinstance(data_file, str) or Path(data_file).name != data_file:
            self.issues.add("missing_file")
            return
        feature_expr = next(
            (value(node) for node in assigned(tree, "FEATURE_COLUMNS")), None
        )
        expected_features = 256 if "range(256)" in unparse(feature_expr) else 257
        if expected_features != 256:
            self.issues.add("wrong_feature")
        contract = DatasetContract(
            "tabular_csv",
            classes=2,
            feature_columns=tuple(
                f"pixel_{index}" for index in range(expected_features)
            ),
        )
        self.issues.update(
            dataset_issues(ROOT / "data" / data_file, contract, self.cache)
        )

    def _check_node_loader(self, load):
        """Check data conversion in load_data()."""
        labels = assigned(load, "labels")
        if not labels or "to_numpy(dtype=np.int64" not in unparse(value(labels[0])):
            text = unparse(value(labels[0])) if labels else ""
            if "nan" in text:
                self.issues.add("nan_to_int")
            elif "inf" in text:
                self.issues.add("inf_to_int")
            else:
                self.issues.add("invalid_label_conversion")
        images = assigned(load, "images")
        image_text = unparse(value(images[0])) if images else ""
        if "torch.from_numpy(features)" not in image_text:
            self.issues.add("numpy_to_model")
        if "reshape(-1, 1, 16, 16)" not in image_text:
            self.issues.add("bad_reshape")
        targets = assigned(load, "targets")
        target_text = unparse(value(targets[0])) if targets else ""
        if "np.asarray" in target_text:
            self.issues.add("torch_to_numpy")
        elif ".float()" in target_text:
            self.issues.add("float_targets")
        elif target_text != "torch.from_numpy(labels)":
            self.issues.add("invalid_target_conversion")

    def _check_training_setup(self, run):
        """Check device placement and the training loss expression."""
        model_assignment = assigned(run, "model")
        if (
            not model_assignment
            or unparse(value(model_assignment[0])) != "Model().to(device)"
        ):
            self.issues.add("model_on_cpu")
        train_move = next(
            (
                node
                for node in run.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and target_names(node.targets[0]) == {"xtrain", "ytrain"}
                and ".to(" in unparse(value(node))
            ),
            None,
        )
        if train_move is None or "xtrain.to(device)" not in unparse(value(train_move)):
            self.issues.add("inputs_on_cpu")
        if train_move is None or "ytrain.to(device)" not in unparse(value(train_move)):
            self.issues.add("targets_on_cpu")
        losses = assigned(run, "loss")
        loss_text = unparse(value(losses[0])) if losses else ""
        if ".detach()" in loss_text:
            self.issues.add("detached_loss")
        elif "torch.tensor(" in loss_text or ".item()" in loss_text:
            self.issues.add("reboxed_loss")
        elif "F.one_hot" in loss_text:
            self.issues.add("one_hot_targets")
        elif loss_text != "F.cross_entropy(logits, ytrain)":
            if "cross_entropy_typo" not in loss_text:
                self.issues.add("invalid_train_loss")
        logits = assigned(run, "logits")
        if logits and "F.softmax" in unparse(value(logits[0])):
            self.issues.add("softmax_before_loss")

    def _check_gradient_flow(self, run):
        """Check training calls that can disconnect or disable gradients."""
        for node in ast.walk(run):
            if isinstance(node, ast.Call) and name(node.func) == "torch.autograd":
                self.issues.add("invalid_autograd_context")
            if (
                isinstance(node, ast.Call)
                and name(node.func) == "torch.set_grad_enabled"
                and node.args
                and unparse(node.args[0]) == "False"
            ):
                self.issues.add("global_grad_disabled")
            if isinstance(node, ast.With) and has_call(node, "torch.no_grad"):
                if any(
                    isinstance(child, ast.Call)
                    and name(child.func) == "model"
                    and any(unparse(arg) == "xtrain" for arg in child.args)
                    for child in ast.walk(node)
                ):
                    self.issues.add("no_grad_train")
            if (
                isinstance(node, ast.Call)
                and name(node.func) == "requires_grad_"
                and node.args
                and unparse(node.args[0]) == "False"
            ):
                self.issues.add("all_parameters_frozen")
            if (
                isinstance(node, ast.Call)
                and name(node.func) == "parameter.requires_grad_"
                and node.args
                and unparse(node.args[0]) == "False"
            ):
                self.issues.add("all_parameters_frozen")

    def _check_validation(self, run):
        """Check validation split and target usage."""
        val_split = next(
            (
                node
                for node in run.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and target_names(node.targets[0]) == {"xval", "yval"}
            ),
            None,
        )
        if (
            val_split is None
            or "images[24:]" not in unparse(value(val_split))
            or "targets[24:]" not in unparse(value(val_split))
        ):
            self.issues.add("validation_on_train")
        val_losses = assigned(run, "val_loss")
        if not val_losses or "F.cross_entropy(val_logits, yval)" != unparse(
            value(val_losses[0])
        ):
            self.issues.add("wrong_validation_target")

    def _check_model_head(self, tree):
        """Check the classifier output width."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == "head"
                for target in node.targets
            ):
                if (
                    isinstance(node.value, ast.Call)
                    and name(node.value.func) == "nn.Linear"
                    and len(node.value.args) >= 2
                ):
                    if (
                        not isinstance(node.value.args[1], ast.Constant)
                        or node.value.args[1].value != 2
                    ):
                        self.issues.add("wrong_output_classes")

    def _check_optimizer_steps(self, run):
        """Check gradient reset and backward/step ordering."""
        if "optimizer.zero_grad(" not in unparse(run):
            self.issues.add("no_zero_grad")
        backward_line = next(
            (
                node.lineno
                for node in ast.walk(run)
                if isinstance(node, ast.Call) and name(node.func) == "loss.backward"
            ),
            None,
        )
        step_line = next(
            (
                node.lineno
                for node in ast.walk(run)
                if isinstance(node, ast.Call) and name(node.func) == "optimizer.step"
            ),
            None,
        )
        if backward_line is None:
            self.issues.add("missing_backward")
        elif step_line is None or step_line < backward_line:
            self.issues.add("optimizer_before_backward")

    def check(self):
        try:
            source = self.path.read_text()
            tree = ast.parse(source, filename=str(self.path))
        except (OSError, UnicodeError, SyntaxError):
            return ["unreadable_or_syntax_error"]

        if literal_assignment(tree, "TASK") is not None:
            return check_kaggle(self.path, tree, self.cache)
        if literal_assignment(tree, "MODALITY") is not None:
            return check_modality(self.path, tree, self.cache)

        functions = self._check_names(tree)
        self._check_node_data(tree)
        load = next(
            (function for function in functions if function.name == "load_data"), None
        )
        run = next((function for function in functions if function.name == "run"), None)
        if load is None or run is None:
            return sorted(self.issues | {"missing_pipeline"})

        self._check_node_loader(load)
        self._check_training_setup(run)
        self._check_gradient_flow(run)
        self._check_validation(run)
        self._check_model_head(tree)
        self._check_optimizer_steps(run)
        return sorted(self.issues)


def check(path, cache):
    """Compatibility entry point for a declared source node."""
    return NodeValidator(path, cache).check()


class PreflightApp:
    """Parse command-line requests and report bounded validation results."""

    def _run_bounded_competition(self):
        """Run a competition audit in a child process with a hard deadline."""
        args = self.args
        child = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--competition",
            args.competition,
            "--dataset",
            str(args.dataset),
            "--json",
        ]
        if args.asset_root is not None:
            child.extend(("--asset-root", str(args.asset_root)))
        started = time.perf_counter()
        try:
            process = subprocess.run(
                child,
                capture_output=True,
                text=True,
                check=False,
                timeout=args.max_seconds,
            )
        except subprocess.TimeoutExpired:
            report = {
                "competition": args.competition,
                "dataset": str(args.dataset),
                "status": "incomplete",
                "issues": ["audit_incomplete"],
                "assurance": "configured_checks_only",
                "elapsed_seconds": time.perf_counter() - started,
            }
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(
                    f"{args.competition}: INCOMPLETE after {report['elapsed_seconds']:.4f}s"
                )
            raise SystemExit(2)
        try:
            report = json.loads(process.stdout)
            if process.returncode not in (0, 1) or not isinstance(
                report.get("issues"), list
            ):
                raise ValueError("invalid child result")
        except (json.JSONDecodeError, ValueError, AttributeError):
            report = {
                "competition": args.competition,
                "dataset": str(args.dataset),
                "status": "incomplete",
                "issues": ["audit_error"],
                "assurance": "configured_checks_only",
            }
            process = None
        report["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "incomplete" if process is None else "complete"
        report["assurance"] = "configured_checks_only"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            suffix = (
                "INCOMPLETE"
                if process is None
                else ", ".join(report["issues"])
                if report["issues"]
                else "clean"
            )
            print(f"{args.competition}: {suffix} ({report['elapsed_seconds']:.4f}s)")
        raise SystemExit(2 if process is None else process.returncode)

    def _run_competition(self):
        """Validate one competition index directly."""
        args = self.args
        asset_root = args.asset_root or args.dataset.parent
        start = time.perf_counter()
        issues = sorted(
            kaggle_dataset_issues(args.competition, args.dataset, asset_root, {})
        )
        report = {
            "competition": args.competition,
            "dataset": str(args.dataset),
            "asset_root": str(asset_root),
            "issues": issues,
            "status": "complete",
            "assurance": "configured_checks_only",
            "elapsed_seconds": time.perf_counter() - start,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"{args.competition}: {', '.join(issues) if issues else 'clean'} "
                f"({report['elapsed_seconds']:.4f}s)"
            )
        raise SystemExit(1 if issues else 0)

    def _run_dataset_contract(self):
        """Validate a dataset without a model node."""
        args = self.args
        try:
            payload = json.loads(args.contract.read_text())
            for key in ("feature_columns", "image_size"):
                if key in payload:
                    payload[key] = tuple(payload[key])
            contract = DatasetContract(**payload)
            start = time.perf_counter()
            issues = sorted(dataset_issues(args.dataset, contract, {}))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            self.parser.error(str(exc))
        report = {
            "dataset": str(args.dataset),
            "kind": contract.kind,
            "issues": issues,
            "elapsed_seconds": time.perf_counter() - start,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"{args.dataset}: {', '.join(issues) if issues else 'clean'} "
                f"({report['elapsed_seconds']:.4f}s)"
            )
        raise SystemExit(1 if issues else 0)

    def _run_nodes(self):
        """Check requested nodes or the controlled corpus manifest."""
        args = self.args
        paths = args.paths or [
            ROOT / entry["path"]
            for entry in json.loads((ROOT / "manifest.json").read_text())
        ]
        start = time.perf_counter()
        cache = {}
        findings = {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): check(
                path, cache
            )
            for path in paths
        }
        elapsed = time.perf_counter() - start
        report = {
            "nodes": len(paths),
            "flagged": sum(bool(issues) for issues in findings.values()),
            "elapsed_seconds": elapsed,
            "issues": findings,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"checked={report['nodes']} flagged={report['flagged']} elapsed={elapsed:.4f}s"
            )
            for path, issues in findings.items():
                if issues:
                    print(f"{path}: {', '.join(issues)}")
        raise SystemExit(1 if report["flagged"] else 0)

    def run(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "paths",
            nargs="*",
            type=Path,
            help="node files; defaults to every manifest node",
        )
        parser.add_argument(
            "--json", action="store_true", help="print all findings as JSON"
        )
        parser.add_argument(
            "--dataset",
            type=Path,
            help="validate a dataset directly, without a model node",
        )
        parser.add_argument("--contract", type=Path, help="JSON contract for --dataset")
        parser.add_argument(
            "--competition",
            help="Kaggle competition slug for direct dataset validation",
        )
        parser.add_argument(
            "--asset-root", type=Path, help="competition root containing asset folders"
        )
        parser.add_argument(
            "--max-seconds",
            type=float,
            help="time limit for a full competition scan; returns incomplete if exceeded",
        )
        self.parser = parser
        args = self.args = parser.parse_args()
        if args.max_seconds is not None:
            if (
                args.competition is None
                or args.dataset is None
                or args.max_seconds <= 0
                or args.contract is not None
                or args.paths
            ):
                parser.error(
                    "--max-seconds requires --competition and --dataset, without --contract or node paths"
                )
            self._run_bounded_competition()

        if args.competition is not None:
            if args.dataset is None or args.contract is not None or args.paths:
                parser.error(
                    "--competition requires --dataset, without --contract or node paths"
                )
            if args.competition not in TASKS:
                parser.error(f"unknown competition: {args.competition}")
            self._run_competition()

        if args.asset_root is not None:
            parser.error("--asset-root requires --competition")
        if args.dataset is not None or args.contract is not None:
            if args.dataset is None or args.contract is None or args.paths:
                parser.error(
                    "--dataset and --contract must be used together, without node paths"
                )
            self._run_dataset_contract()
        self._run_nodes()


def main():
    """Compatibility command-line entry point."""
    PreflightApp().run()


if __name__ == "__main__":
    main()
