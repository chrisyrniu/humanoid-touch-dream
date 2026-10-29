#!/usr/bin/env python3
"""Read-only validator for Humanoid Touch Dream G1 datasets.

The validator deliberately has no repair mode.  It opens HDF5 files read-only,
checks the historical HTD G1 schema, and can write a JSON report outside the
dataset tree.  Dataset migrations belong in a separate tool so an audit cannot
accidentally change its input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


TOOL_VERSION = "0.1.0"
REPORT_VERSION = 1
LEGACY_COORDINATE = "htd_g1_legacy_v1"
EXPECTED_EMBODIMENT = "g1_inspire_wb"

LEGACY_BODY_MASK = np.ones(6, dtype=np.bool_)
CORRECTED_BODY_MASK = np.array([False, False, True, True, True, True])

IMAGE_SHAPES = {
    "observations/images/main": (480, 1280, 3),
    "observations/images/wrist_left": (480, 640, 3),
    "observations/images/wrist_right": (480, 640, 3),
}

MASKS = {
    "masks/act_body": CORRECTED_BODY_MASK,
    "masks/act_eef": np.ones(12, dtype=np.bool_),
    "masks/act_gripper": np.ones(2, dtype=np.bool_),
    "masks/img_main": np.ones(1, dtype=np.bool_),
    "masks/img_wrist": np.ones(2, dtype=np.bool_),
    "masks/proprio_eef": np.ones(12, dtype=np.bool_),
    "masks/proprio_gripper": np.ones(2, dtype=np.bool_),
    "masks/proprio_other": np.ones(2, dtype=np.bool_),
}

CORE_TEMPORAL_DIMS: dict[str, int | None] = {
    "actions/body": 6,
    "actions/delta_body": 6,
    "actions/delta_eef": 12,
    "actions/delta_gripper": 2,
    "actions/eef": 12,
    "actions/eef_joints_target": 150,
    "actions/eef_to_body": 12,
    "actions/gripper": 2,
    "actions/velocity_commands": 3,
    "camera_poses/head_camera_to_init": 6,
    "observations/head_cam_timestamp": None,
    "observations/proprioceptions/body_proprio": 6,
    "observations/proprioceptions/eef_proprio": 12,
    "observations/proprioceptions/eef_to_body_proprio": 12,
    "observations/proprioceptions/gripper_proprio": 2,
    "observations/proprioceptions/other/eef_joint_proprio": 24,
    "observations/proprioceptions/other/eef_link_proprio": 30,
    "observations/proprioceptions/other/imu_proprio": 9,
    "observations/proprioceptions/other/robot_joint_proprio": 58,
}

G1_POLICY_DIMS = {
    "actions/eef_link": 30,
    "actions/left_eef_link": 15,
    "actions/left_eef_to_body": 6,
    "actions/right_eef_link": 15,
    "actions/right_eef_to_body": 6,
    "observations/proprioceptions/left_eef_finger_force_proprio": 6,
    "observations/proprioceptions/left_eef_joint_pos_proprio": 6,
    "observations/proprioceptions/left_eef_link_proprio": 15,
    "observations/proprioceptions/left_eef_to_body_proprio": 6,
    "observations/proprioceptions/right_eef_finger_force_proprio": 6,
    "observations/proprioceptions/right_eef_joint_pos_proprio": 6,
    "observations/proprioceptions/right_eef_link_proprio": 15,
    "observations/proprioceptions/right_eef_to_body_proprio": 6,
    "observations/proprioceptions/robot_joint_pos_proprio": 29,
    "observations/proprioceptions/robot_joint_vel_proprio": 29,
}

TACTILE_PARTS = ("pinky", "ring", "middle", "index", "thumb", "palm")
TACTILE_DIMS = {
    "dftp": (185, 185, 185, 185, 210, 112),
    "f1": (5, 5, 5, 5, 5, 12),
}
TOUCH_WIDTH_TO_PROFILE_STAGE = {
    1062: ("dftp", "collector"),
    29: ("f1", "collector"),
    37: ("f1", "policy"),
}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _finding(
    report: dict[str, Any],
    severity: str,
    check_id: str,
    message: str,
    *,
    hdf5_path: str | None = None,
    expected: Any = None,
    actual: Any = None,
    count: int | None = None,
    indices: Sequence[int] | None = None,
    remediation: str = "none",
) -> None:
    item: dict[str, Any] = {
        "severity": severity,
        "check_id": check_id,
        "message": message,
        "remediation": remediation,
    }
    if hdf5_path is not None:
        item["hdf5_path"] = "/" + hdf5_path.lstrip("/")
    if expected is not None:
        item["expected"] = _json_value(expected)
    if actual is not None:
        item["actual"] = _json_value(actual)
    if count is not None:
        item["count"] = int(count)
    if indices is not None:
        item["first_indices"] = [int(i) for i in indices[:20]]
    report.setdefault("findings", []).append(item)


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def discover_hdf5(input_path: Path) -> tuple[Path, list[Path]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in {".hdf5", ".h5"}:
            raise ValueError(f"input file is not HDF5: {input_path}")
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise ValueError(f"input path does not exist: {input_path}")

    discovered: list[Path] = []
    for current, dirnames, filenames in os.walk(input_path, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not Path(current, d).is_symlink())
        for filename in sorted(filenames):
            if filename.lower().endswith((".hdf5", ".h5")):
                candidate = Path(current, filename)
                if candidate.is_symlink():
                    raise ValueError(f"symlinked HDF5 files are not allowed in a dataset scan: {candidate}")
                discovered.append(candidate.resolve())
    if not discovered:
        raise ValueError(f"no HDF5 files found under: {input_path}")

    return input_path, sorted(discovered)


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.resolve().open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("manifest must be a JSON object with a files list")
    return manifest


def _manifest_value(manifest: dict[str, Any], key: str) -> Any:
    value = manifest.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _validate_manifest_inventory(
    corpus_report: dict[str, Any],
    root: Path,
    files: Sequence[Path],
    manifest: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}

    entries: dict[str, dict[str, Any]] = {}
    splits: dict[str, set[str]] = {}
    for raw in manifest["files"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("relative_path"), str):
            _finding(corpus_report, "error", "manifest_entry_invalid", "Manifest file entry lacks relative_path")
            continue
        relative = Path(raw["relative_path"]).as_posix()
        if relative in entries:
            _finding(
                corpus_report,
                "error",
                "manifest_duplicate_path",
                f"Manifest lists {relative} more than once",
                actual=relative,
            )
        entries[relative] = raw
        split = str(raw.get("split", "unspecified"))
        splits.setdefault(split, set()).add(relative)

    split_names = sorted(splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(splits[left] & splits[right])
            if overlap:
                _finding(
                    corpus_report,
                    "error",
                    "manifest_split_overlap",
                    f"Manifest splits {left} and {right} overlap",
                    count=len(overlap),
                    actual=overlap[:20],
                )

    actual = {path.relative_to(root).as_posix() for path in files}
    expected = set(entries)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        _finding(
            corpus_report,
            "error",
            "manifest_files_missing",
            "Files listed by the manifest are missing from the input",
            count=len(missing),
            actual=missing[:20],
        )
    if extra:
        _finding(
            corpus_report,
            "error",
            "manifest_unlisted_files",
            "Input contains HDF5 files not listed by the manifest",
            count=len(extra),
            actual=extra[:20],
        )
    return entries


def _shape_matches(dataset: h5py.Dataset, frames: int, width: int | None) -> bool:
    if width is None:
        return dataset.shape == (frames,)
    return dataset.shape == (frames, width)


def _check_required_temporal(
    h5_file: h5py.File,
    report: dict[str, Any],
    frames: int,
    specs: dict[str, int | None],
) -> None:
    for path, width in specs.items():
        if path not in h5_file:
            _finding(report, "error", "required_dataset_missing", "Required dataset is missing", hdf5_path=path)
            continue
        dataset = h5_file[path]
        if not isinstance(dataset, h5py.Dataset):
            _finding(
                report,
                "error",
                "required_dataset_not_dataset",
                "Required path is not an HDF5 dataset",
                hdf5_path=path,
            )
            continue
        if not _shape_matches(dataset, frames, width):
            expected = [frames] if width is None else [frames, width]
            _finding(
                report,
                "error",
                "dataset_shape_invalid",
                "Dataset has the wrong shape",
                hdf5_path=path,
                expected=expected,
                actual=list(dataset.shape),
            )
        if dataset.dtype.kind not in "fiu" or dataset.dtype.kind == "b":
            _finding(
                report,
                "error",
                "numeric_dtype_invalid",
                "Required trajectory dataset must have a numeric non-boolean dtype",
                hdf5_path=path,
                actual=str(dataset.dtype),
            )


def _check_all_temporal_lengths(h5_file: h5py.File, report: dict[str, Any], frames: int) -> None:
    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset) or name.startswith("masks/"):
            return
        if name.startswith(("actions/", "camera_poses/", "observations/")):
            if not obj.shape or obj.shape[0] != frames:
                _finding(
                    report,
                    "error",
                    "temporal_length_mismatch",
                    "Temporal dataset length does not match the episode",
                    hdf5_path=name,
                    expected=frames,
                    actual=list(obj.shape),
                )

    h5_file.visititems(visit)


def _check_masks(
    h5_file: h5py.File,
    report: dict[str, Any],
    expected_body_mask: str,
    camera_profile: str = "head-and-wrists",
) -> str:
    for path, expected in MASKS.items():
        if path == "masks/img_wrist" and camera_profile == "head-only":
            expected = np.zeros(2, dtype=bool)
        if path not in h5_file:
            _finding(report, "error", "required_mask_missing", "Required mask is missing", hdf5_path=path)
            continue
        dataset = h5_file[path]
        if not isinstance(dataset, h5py.Dataset):
            _finding(report, "error", "mask_not_dataset", "Mask path is not a dataset", hdf5_path=path)
            continue
        actual = np.asarray(dataset[...])
        if dataset.dtype != np.dtype(np.bool_):
            _finding(
                report,
                "error",
                "mask_dtype_invalid",
                "Mask dtype must be bool",
                hdf5_path=path,
                expected="bool",
                actual=str(dataset.dtype),
            )
        if not np.array_equal(actual, expected):
            _finding(
                report,
                "error",
                "mask_value_invalid",
                "Mask values do not match the requested HTD G1 contract",
                hdf5_path=path,
                expected=expected,
                actual=actual,
            )

    body_path = "masks/proprio_body"
    if body_path not in h5_file:
        _finding(report, "error", "required_mask_missing", "Required body proprio mask is missing", hdf5_path=body_path)
        return "missing"
    dataset = h5_file[body_path]
    if not isinstance(dataset, h5py.Dataset):
        _finding(report, "error", "mask_not_dataset", "Body mask path is not a dataset", hdf5_path=body_path)
        return "invalid"
    actual = np.asarray(dataset[...])
    if dataset.dtype != np.dtype(np.bool_):
        _finding(
            report,
            "error",
            "mask_dtype_invalid",
            "Body proprio mask dtype must be bool",
            hdf5_path=body_path,
            expected="bool",
            actual=str(dataset.dtype),
        )
    if np.array_equal(actual, LEGACY_BODY_MASK):
        contract = "legacy"
    elif np.array_equal(actual, CORRECTED_BODY_MASK):
        contract = "corrected"
    else:
        contract = "invalid"
        _finding(
            report,
            "error",
            "proprio_body_mask_invalid",
            "Body proprio mask matches neither supported contract",
            hdf5_path=body_path,
            expected={"legacy": LEGACY_BODY_MASK, "corrected": CORRECTED_BODY_MASK},
            actual=actual,
            remediation="checkpoint_contract",
        )
    if expected_body_mask != "any" and contract != expected_body_mask:
        _finding(
            report,
            "error",
            "proprio_body_mask_contract_mismatch",
            "Body proprio mask does not match the requested checkpoint/data contract",
            hdf5_path=body_path,
            expected=expected_body_mask,
            actual=contract,
            remediation="checkpoint_contract",
        )
    return contract


def _profile_split_paths(profile: str) -> dict[str, int]:
    specs: dict[str, int] = {}
    for side in ("left", "right"):
        for part, width in zip(TACTILE_PARTS, TACTILE_DIMS[profile]):
            specs[f"observations/tactile/{side}_{part}_tactile"] = width
    return specs


def _detect_profile_and_stage(
    h5_file: h5py.File,
    report: dict[str, Any],
    requested_profile: str,
    requested_stage: str,
) -> tuple[str, str]:
    touch_paths = [
        "observations/proprioceptions/other/left_touch",
        "observations/proprioceptions/other/right_touch",
    ]
    widths: list[int] = []
    for path in touch_paths:
        if path not in h5_file:
            _finding(report, "error", "touch_dataset_missing", "Whole-hand tactile dataset is missing", hdf5_path=path)
            continue
        dataset = h5_file[path]
        if not isinstance(dataset, h5py.Dataset):
            _finding(
                report,
                "error",
                "touch_not_dataset",
                "Whole-hand tactile path is not a dataset",
                hdf5_path=path,
            )
        elif len(dataset.shape) != 2:
            _finding(
                report,
                "error",
                "touch_shape_invalid",
                "Whole-hand tactile dataset must be rank 2",
                hdf5_path=path,
                actual=list(dataset.shape),
            )
        else:
            widths.append(int(dataset.shape[1]))
            if dataset.dtype.kind not in "fiu":
                _finding(
                    report,
                    "error",
                    "touch_dtype_invalid",
                    "Whole-hand tactile dataset must have a numeric dtype",
                    hdf5_path=path,
                    actual=str(dataset.dtype),
                )

    if len(widths) != 2 or widths[0] != widths[1]:
        _finding(
            report,
            "error",
            "touch_profile_conflict",
            "Left and right whole-hand tactile widths disagree or are unavailable",
            actual=widths,
        )
        return "unknown", "invalid"

    detected = TOUCH_WIDTH_TO_PROFILE_STAGE.get(widths[0])
    if detected is None:
        _finding(
            report,
            "error",
            "touch_width_unknown",
            "Whole-hand tactile width is not a supported HTD profile",
            expected=sorted(TOUCH_WIDTH_TO_PROFILE_STAGE),
            actual=widths[0],
        )
        return "unknown", "invalid"
    profile, width_stage = detected

    if requested_profile != "auto" and requested_profile != profile:
        _finding(
            report,
            "error",
            "hand_profile_mismatch",
            "Detected hand profile does not match --profile",
            expected=requested_profile,
            actual=profile,
        )

    metadata_profile = h5_file.attrs.get("hand_type")
    if metadata_profile is None:
        _finding(
            report,
            "warning",
            "hand_profile_metadata_missing",
            f"hand_type is absent; shape evidence identifies {profile}",
            actual=profile,
            remediation="field_migration",
        )
    else:
        metadata_profile = str(_json_value(metadata_profile)).lower()
        if metadata_profile != profile:
            _finding(
                report,
                "error",
                "hand_profile_metadata_conflict",
                "hand_type metadata conflicts with tactile dimensions",
                expected=profile,
                actual=metadata_profile,
            )

    tactile_format = h5_file.attrs.get("tactile_format")
    if tactile_format is not None:
        tactile_format_text = str(_json_value(tactile_format)).lower()
        conflicting_marker = "f1" if profile == "dftp" else "dftp"
        if conflicting_marker in tactile_format_text:
            _finding(
                report,
                "error",
                "tactile_format_metadata_conflict",
                "tactile_format metadata conflicts with tactile dimensions",
                expected=profile,
                actual=tactile_format_text,
            )

    tactile_specs = _profile_split_paths(profile)
    tactile_present = sum(path in h5_file for path in tactile_specs)
    g1_present = sum(path in h5_file for path in G1_POLICY_DIMS)
    tactile_complete = tactile_present == len(tactile_specs)
    g1_complete = g1_present == len(G1_POLICY_DIMS)
    tactile_absent = tactile_present == 0
    g1_absent = g1_present == 0

    policy_width = profile == "dftp" or widths[0] == 37
    if tactile_complete and g1_complete and policy_width:
        stage = "policy"
    elif tactile_absent and (g1_absent or g1_complete) and width_stage == "collector":
        stage = "collector"
    else:
        stage = "invalid_partial"
        _finding(
            report,
            "error",
            "partial_processing",
            "Dataset contains an incomplete mixture of collector and policy-stage fields",
            expected={"g1_split_paths": len(G1_POLICY_DIMS), "tactile_split_paths": len(tactile_specs)},
            actual={"g1_split_paths": g1_present, "tactile_split_paths": tactile_present},
            remediation="field_migration",
        )

    if profile == "f1" and widths[0] == 37 and stage != "policy":
        _finding(
            report,
            "error",
            "f1_37d_splits_missing",
            "Processed 37-D F1 tactile data requires all 5/5/5/5/5/12 split datasets",
            remediation="field_migration",
        )

    if requested_stage != "auto" and stage != requested_stage:
        _finding(
            report,
            "error",
            "processing_stage_mismatch",
            "Detected processing stage does not match --stage",
            expected=requested_stage,
            actual=stage,
        )
    return profile, stage


def _check_attrs(
    h5_file: h5py.File,
    report: dict[str, Any],
    assume_legacy_coordinate: bool,
    expected_frequency: float,
) -> None:
    attrs = h5_file.attrs
    embodiment = _json_value(attrs.get("embodiment"))
    if embodiment != EXPECTED_EMBODIMENT:
        _finding(
            report,
            "error",
            "embodiment_invalid",
            "HTD release data must use the G1 whole-body embodiment",
            expected=EXPECTED_EMBODIMENT,
            actual=embodiment,
        )

    frequency = _json_value(attrs.get("frequency"))
    if not isinstance(frequency, (int, float)) or not math.isfinite(float(frequency)) or frequency <= 0:
        _finding(report, "error", "frequency_invalid", "frequency must be a positive finite number", actual=frequency)
    elif not math.isclose(float(frequency), expected_frequency, rel_tol=0.0, abs_tol=1e-9):
        _finding(
            report,
            "error",
            "frequency_contract_mismatch",
            "Episode frequency does not match the requested HTD data contract",
            expected=expected_frequency,
            actual=frequency,
        )

    compress = _json_value(attrs.get("compress"))
    if not isinstance(compress, bool):
        _finding(report, "error", "compress_attribute_invalid", "compress must be a boolean", actual=compress)

    sim = _json_value(attrs.get("sim", False))
    if bool(sim):
        _finding(report, "error", "simulation_data_unsupported", "This validator contract is for real HTD data", actual=sim)

    coordinate = _json_value(attrs.get("coordinate_convention"))
    if coordinate is None:
        if assume_legacy_coordinate:
            report["coordinate_convention"] = {
                "value": LEGACY_COORDINATE,
                "source": "explicit_cli_provenance_assertion",
            }
            _finding(
                report,
                "warning",
                "coordinate_metadata_missing",
                f"coordinate_convention is absent; caller explicitly asserted {LEGACY_COORDINATE}",
                remediation="field_migration",
            )
        else:
            _finding(
                report,
                "error",
                "coordinate_convention_unresolved",
                "coordinate_convention is absent; pass --assume-legacy-coordinate only with provenance evidence",
                expected=LEGACY_COORDINATE,
                remediation="checkpoint_contract",
            )
    elif coordinate != LEGACY_COORDINATE:
        _finding(
            report,
            "error",
            "coordinate_convention_unsupported",
            "HTD v1 accepts only its historical G1 coordinate convention",
            expected=LEGACY_COORDINATE,
            actual=coordinate,
        )
    else:
        report["coordinate_convention"] = {"value": coordinate, "source": "hdf5_attribute"}

    missing = [
        name
        for name in ("schema_version", "coordinate_convention", "hand_type", "tactile_format")
        if name not in attrs
    ]
    if missing:
        _finding(
            report,
            "warning",
            "release_metadata_missing",
            "Legacy file lacks release metadata; record provenance in a sidecar manifest",
            actual=missing,
            remediation="field_migration",
        )


def _iter_slices(length: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def _check_finite(h5_file: h5py.File, report: dict[str, Any], frames: int, chunk_size: int) -> None:
    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset) or name.startswith("masks/"):
            return
        if not name.startswith(("actions/", "camera_poses/", "observations/")):
            return
        if name.startswith("observations/images/") or obj.dtype.kind not in "fiu":
            return
        if not obj.shape or obj.shape[0] != frames:
            return
        nonfinite = 0
        first_rows: list[int] = []
        for selection in _iter_slices(frames, chunk_size):
            values = np.asarray(obj[selection])
            if values.dtype.kind == "f":
                bad = ~np.isfinite(values)
                nonfinite += int(np.count_nonzero(bad))
                if np.any(bad) and len(first_rows) < 20:
                    row_mask = bad if bad.ndim == 1 else np.any(bad, axis=tuple(range(1, bad.ndim)))
                    first_rows.extend((np.flatnonzero(row_mask) + selection.start).tolist())
        if nonfinite:
            _finding(
                report,
                "error",
                "numeric_nonfinite",
                "Dataset contains NaN or infinity",
                hdf5_path=name,
                count=nonfinite,
                indices=first_rows,
            )

    h5_file.visititems(visit)


def _compare_exact(
    h5_file: h5py.File,
    report: dict[str, Any],
    target_path: str,
    source_paths: Sequence[str],
    frames: int,
    chunk_size: int,
    *,
    transform: Any = None,
) -> None:
    paths = [target_path, *source_paths]
    if any(path not in h5_file for path in paths):
        return
    if any(not isinstance(h5_file[path], h5py.Dataset) for path in paths):
        return
    mismatch_rows: list[int] = []
    mismatch_values = 0
    for selection in _iter_slices(frames, chunk_size):
        target = np.asarray(h5_file[target_path][selection])
        sources = [np.asarray(h5_file[path][selection]) for path in source_paths]
        try:
            expected = transform(*sources) if transform is not None else np.concatenate(sources, axis=1)
        except (ValueError, TypeError, IndexError) as exc:
            _finding(
                report,
                "error",
                "derived_source_shape_invalid",
                f"Could not derive expected data from source arrays: {exc}",
                hdf5_path=target_path,
                expected={"derived_from": ["/" + path for path in source_paths]},
                remediation="field_migration",
            )
            return
        if target.shape != expected.shape:
            _finding(
                report,
                "error",
                "derived_dataset_shape_mismatch",
                "Derived dataset shape does not match its retained source",
                hdf5_path=target_path,
                expected=[frames, *expected.shape[1:]],
                actual=list(h5_file[target_path].shape),
                remediation="field_migration",
            )
            return
        unequal = target != expected
        mismatch_values += int(np.count_nonzero(unequal))
        if np.any(unequal) and len(mismatch_rows) < 20:
            row_mask = unequal if unequal.ndim == 1 else np.any(unequal, axis=tuple(range(1, unequal.ndim)))
            mismatch_rows.extend((np.flatnonzero(row_mask) + selection.start).tolist())
    if mismatch_values:
        _finding(
            report,
            "error",
            "derived_dataset_mismatch",
            "Derived dataset does not exactly match its retained source",
            hdf5_path=target_path,
            count=mismatch_values,
            indices=mismatch_rows,
            expected={"derived_from": ["/" + path for path in source_paths]},
            remediation="field_migration",
        )


def _check_policy_consistency(
    h5_file: h5py.File,
    report: dict[str, Any],
    profile: str,
    frames: int,
    chunk_size: int,
) -> None:
    _compare_exact(
        h5_file,
        report,
        "actions/eef_to_body",
        ["actions/right_eef_to_body", "actions/left_eef_to_body"],
        frames,
        chunk_size,
    )
    _compare_exact(
        h5_file,
        report,
        "actions/eef_link",
        ["actions/right_eef_link", "actions/left_eef_link"],
        frames,
        chunk_size,
    )
    _compare_exact(
        h5_file,
        report,
        "observations/proprioceptions/eef_to_body_proprio",
        [
            "observations/proprioceptions/right_eef_to_body_proprio",
            "observations/proprioceptions/left_eef_to_body_proprio",
        ],
        frames,
        chunk_size,
    )
    _compare_exact(
        h5_file,
        report,
        "observations/proprioceptions/other/eef_link_proprio",
        [
            "observations/proprioceptions/right_eef_link_proprio",
            "observations/proprioceptions/left_eef_link_proprio",
        ],
        frames,
        chunk_size,
    )
    _compare_exact(
        h5_file,
        report,
        "observations/proprioceptions/other/eef_joint_proprio",
        [
            "observations/proprioceptions/right_eef_joint_pos_proprio",
            "observations/proprioceptions/left_eef_joint_pos_proprio",
            "observations/proprioceptions/right_eef_finger_force_proprio",
            "observations/proprioceptions/left_eef_finger_force_proprio",
        ],
        frames,
        chunk_size,
    )
    _compare_exact(
        h5_file,
        report,
        "observations/proprioceptions/other/robot_joint_proprio",
        [
            "observations/proprioceptions/robot_joint_pos_proprio",
            "observations/proprioceptions/robot_joint_vel_proprio",
        ],
        frames,
        chunk_size,
    )

    def link_from_target(values: np.ndarray, side: int) -> np.ndarray:
        reshaped = values.reshape(-1, 2, 25, 3)
        return np.concatenate([reshaped[:, side, index, :] for index in (4, 9, 14, 19, 24)], axis=1)

    _compare_exact(
        h5_file,
        report,
        "actions/right_eef_link",
        ["actions/eef_joints_target"],
        frames,
        chunk_size,
        transform=lambda values: link_from_target(values, 0),
    )
    _compare_exact(
        h5_file,
        report,
        "actions/left_eef_link",
        ["actions/eef_joints_target"],
        frames,
        chunk_size,
        transform=lambda values: link_from_target(values, 1),
    )

    split_paths = _profile_split_paths(profile)
    for path, width in split_paths.items():
        if path not in h5_file or not isinstance(h5_file[path], h5py.Dataset):
            continue
        if not _shape_matches(h5_file[path], frames, width):
            _finding(
                report,
                "error",
                "tactile_split_shape_invalid",
                "Tactile split has the wrong shape for the detected profile",
                hdf5_path=path,
                expected=[frames, width],
                actual=list(h5_file[path].shape),
            )
    for side in ("left", "right"):
        source_paths = [f"observations/tactile/{side}_{part}_tactile" for part in TACTILE_PARTS]
        _compare_exact(
            h5_file,
            report,
            f"observations/proprioceptions/other/{side}_touch",
            source_paths,
            frames,
            chunk_size,
        )


def _check_tactile_semantics(
    h5_file: h5py.File,
    report: dict[str, Any],
    profile: str,
    frames: int,
    chunk_size: int,
) -> None:
    for side in ("left", "right"):
        path = f"observations/proprioceptions/other/{side}_touch"
        if path not in h5_file:
            continue
        dataset = h5_file[path]
        if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "fiu":
            continue
        if dataset.dtype != np.dtype(np.float32):
            _finding(
                report,
                "warning",
                "touch_dtype_unexpected",
                "Historical processed tactile arrays use float32",
                hdf5_path=path,
                expected="float32",
                actual=str(dataset.dtype),
            )
        if profile == "dftp":
            low_count = high_count = 0
            first_rows: list[int] = []
            for selection in _iter_slices(frames, chunk_size):
                values = np.asarray(dataset[selection])
                bad = (values < -1e-6) | (values > 1.0 + 1e-6)
                low_count += int(np.count_nonzero(values < -1e-6))
                high_count += int(np.count_nonzero(values > 1.0 + 1e-6))
                if np.any(bad) and len(first_rows) < 20:
                    first_rows.extend((np.flatnonzero(np.any(bad, axis=1)) + selection.start).tolist())
            if low_count or high_count:
                _finding(
                    report,
                    "error",
                    "dftp_touch_range_invalid",
                    "DFTP normalized tactile values must lie in [0, 1]",
                    hdf5_path=path,
                    count=low_count + high_count,
                    indices=first_rows,
                    actual={"below_zero": low_count, "above_one": high_count},
                )
        elif dataset.shape == (frames, 37):
            invalid_norms = invalid_proximity = 0
            norm_rows: list[int] = []
            proximity_rows: list[int] = []
            direction_bases = (0, 5, 10, 15, 20, 25, 29, 33)
            proximity_indices = (4, 9, 14, 19, 24)
            for selection in _iter_slices(frames, chunk_size):
                values = np.asarray(dataset[selection])
                norms = np.stack(
                    [values[:, base + 2] ** 2 + values[:, base + 3] ** 2 for base in direction_bases],
                    axis=1,
                )
                bad_norm = ~(np.isclose(norms, 0.0, atol=2e-4) | np.isclose(norms, 1.0, atol=2e-4))
                proximity = values[:, proximity_indices]
                bad_proximity = (proximity < -1e-6) | (proximity > 1.0 + 1e-6)
                invalid_norms += int(np.count_nonzero(bad_norm))
                invalid_proximity += int(np.count_nonzero(bad_proximity))
                if np.any(bad_norm) and len(norm_rows) < 20:
                    norm_rows.extend((np.flatnonzero(np.any(bad_norm, axis=1)) + selection.start).tolist())
                if np.any(bad_proximity) and len(proximity_rows) < 20:
                    proximity_rows.extend(
                        (np.flatnonzero(np.any(bad_proximity, axis=1)) + selection.start).tolist()
                    )
            if invalid_norms:
                _finding(
                    report,
                    "error",
                    "f1_direction_encoding_invalid",
                    "F1 direction sine/cosine pairs must have squared norm zero or one",
                    hdf5_path=path,
                    count=invalid_norms,
                    indices=norm_rows,
                )
            if invalid_proximity:
                _finding(
                    report,
                    "error",
                    "f1_proximity_range_invalid",
                    "F1 proximity values must lie in [0, 1]",
                    hdf5_path=path,
                    count=invalid_proximity,
                    indices=proximity_rows,
                )


def _check_timestamps(h5_file: h5py.File, report: dict[str, Any], frames: int, gap_factor: float) -> None:
    path = "observations/head_cam_timestamp"
    if (
        path not in h5_file
        or not isinstance(h5_file[path], h5py.Dataset)
        or h5_file[path].shape != (frames,)
        or h5_file[path].dtype.kind not in "fiu"
    ):
        return
    values = np.asarray(h5_file[path][...], dtype=np.float64)
    if frames < 2 or not np.all(np.isfinite(values)):
        return
    deltas = np.diff(values)
    negative = np.flatnonzero(deltas < 0) + 1
    duplicate = np.flatnonzero(deltas == 0) + 1
    positive = deltas[deltas > 0]
    try:
        frequency = float(h5_file.attrs.get("frequency", 0))
    except (TypeError, ValueError):
        frequency = 0.0
    expected_period = 1.0 / frequency if frequency > 0 else None
    gap_threshold = gap_factor * expected_period if expected_period else None
    gaps = np.flatnonzero(deltas > gap_threshold) + 1 if gap_threshold else np.array([], dtype=np.int64)
    report["timestamps"] = {
        "duplicate_count": int(len(duplicate)),
        "negative_count": int(len(negative)),
        "gap_count": int(len(gaps)),
        "median_positive_period": float(np.median(positive)) if len(positive) else None,
        "max_delta": float(np.max(deltas)) if len(deltas) else None,
        "gap_threshold": gap_threshold,
    }
    if len(negative):
        _finding(
            report,
            "error",
            "timestamp_decreases",
            "Head-camera timestamps decrease",
            hdf5_path=path,
            count=len(negative),
            indices=negative.tolist(),
        )
    if len(duplicate):
        _finding(
            report,
            "warning",
            "timestamp_duplicate",
            "Head-camera timestamps repeat",
            hdf5_path=path,
            count=len(duplicate),
            indices=duplicate.tolist(),
        )
    if len(gaps):
        _finding(
            report,
            "warning",
            "timestamp_gap",
            "Head-camera timestamp gap exceeds the configured multiple of the nominal period",
            hdf5_path=path,
            count=len(gaps),
            indices=gaps.tolist(),
            expected={"maximum_seconds": gap_threshold},
        )


def _image_indices(frames: int, mode: str, sample_count: int) -> list[int]:
    if mode == "none" or frames == 0:
        return []
    if mode == "all" or sample_count >= frames:
        return list(range(frames))
    return sorted(set(np.linspace(0, frames - 1, num=max(1, sample_count), dtype=np.int64).tolist()))


def _decode_image_payload(payload: Any) -> tuple[np.ndarray | None, str | None]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("OpenCV is required when image validation is enabled") from exc
    if isinstance(payload, bytes):
        encoded = np.frombuffer(payload, dtype=np.uint8)
    else:
        encoded = np.asarray(payload, dtype=np.uint8).reshape(-1)
    if len(encoded) < 4:
        return None, "empty_or_too_short"
    is_jpeg = encoded[0] == 0xFF and encoded[1] == 0xD8
    is_png = np.array_equal(encoded[:4], np.array([0x89, 0x50, 0x4E, 0x47], dtype=np.uint8))
    if not (is_jpeg or is_png):
        return None, "unknown_magic"
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return None, "decode_failed"
    return decoded, None


def _check_images(
    h5_file: h5py.File,
    report: dict[str, Any],
    frames: int,
    mode: str,
    sample_count: int,
    camera_profile: str = "head-and-wrists",
) -> None:
    checked = 0
    required_shapes = IMAGE_SHAPES if camera_profile == "head-and-wrists" else {
        "observations/images/main": IMAGE_SHAPES["observations/images/main"]
    }
    report["images"] = {"mode": mode, "checked": 0, "camera_profile": camera_profile,
                        "required": len(required_shapes),
                        "available": sum(path in h5_file for path in required_shapes)}
    declared_profile = _json_value(h5_file.attrs.get("camera_profile"))
    if declared_profile is not None and declared_profile != camera_profile:
        _finding(report, "error", "camera_profile_metadata_mismatch",
                 "Camera profile metadata does not match the requested contract",
                 expected=camera_profile, actual=declared_profile)
    raw_compress = h5_file.attrs.get("compress", False)
    if not isinstance(raw_compress, (bool, np.bool_)):
        report["images"]["skipped_reason"] = "invalid_compress_attribute"
        return
    compress = bool(raw_compress)
    for path, expected_shape in required_shapes.items():
        if path not in h5_file:
            _finding(report, "error", "image_dataset_missing", "Required camera dataset is missing", hdf5_path=path)
            continue
        dataset = h5_file[path]
        if not isinstance(dataset, h5py.Dataset):
            _finding(report, "error", "image_not_dataset", "Camera path is not a dataset", hdf5_path=path)
            continue
        if compress:
            if dataset.shape != (frames,) or dataset.dtype.kind != "O":
                _finding(
                    report,
                    "error",
                    "compressed_image_layout_invalid",
                    "Compressed images must be rank-1 variable-length byte arrays",
                    hdf5_path=path,
                    expected=[frames],
                    actual={"shape": list(dataset.shape), "dtype": str(dataset.dtype)},
                )
                continue
            bad_indices: list[int] = []
            reasons: Counter[str] = Counter()
            wrong_shapes: Counter[str] = Counter()
            for index in _image_indices(frames, mode, sample_count):
                decoded, reason = _decode_image_payload(dataset[index])
                checked += 1
                if reason is not None:
                    bad_indices.append(index)
                    reasons[reason] += 1
                elif tuple(decoded.shape) != expected_shape:
                    bad_indices.append(index)
                    wrong_shapes[str(tuple(decoded.shape))] += 1
            if bad_indices:
                _finding(
                    report,
                    "error",
                    "image_decode_or_shape_invalid",
                    "Encoded image is invalid or decodes to the wrong shape",
                    hdf5_path=path,
                    count=len(bad_indices),
                    indices=bad_indices,
                    expected=list(expected_shape),
                    actual={"decode_errors": dict(reasons), "wrong_shapes": dict(wrong_shapes)},
                )
        else:
            expected = (frames, *expected_shape)
            if dataset.shape != expected or dataset.dtype != np.dtype(np.uint8):
                _finding(
                    report,
                    "error",
                    "raw_image_layout_invalid",
                    "Uncompressed images must be uint8 arrays with the canonical shape",
                    hdf5_path=path,
                    expected=list(expected),
                    actual={"shape": list(dataset.shape), "dtype": str(dataset.dtype)},
                )
                continue
            unreadable: list[int] = []
            for index in _image_indices(frames, mode, sample_count):
                try:
                    image = np.asarray(dataset[index])
                    if image.shape != expected_shape:
                        unreadable.append(index)
                except (OSError, ValueError, TypeError):
                    unreadable.append(index)
                checked += 1
            if unreadable:
                _finding(
                    report,
                    "error",
                    "raw_image_read_failed",
                    "Uncompressed image frame could not be read with its canonical shape",
                    hdf5_path=path,
                    count=len(unreadable),
                    indices=unreadable,
                )
    report["images"]["checked"] = checked


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _rx(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_error(delta: np.ndarray, target: np.ndarray) -> float:
    error = target.T @ delta
    sine = 0.5 * np.linalg.norm(
        [
            error[2, 1] - error[1, 2],
            error[0, 2] - error[2, 0],
            error[1, 0] - error[0, 1],
        ]
    )
    cosine = float(np.clip((np.trace(error) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.atan2(float(sine), cosine))


def detect_repeated_dds_transforms(
    h5_file: h5py.File,
    frames: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Detect exact cached-state repeats with an additional hand-frame remap.

    This identifies the historical in-place DDS-array mutation signature.  It
    does not infer corruption from a merely similar pose and never joins files.
    """

    required = [
        "observations/proprioceptions/body_proprio",
        "observations/proprioceptions/eef_proprio",
        "observations/proprioceptions/eef_to_body_proprio",
        "observations/proprioceptions/other/robot_joint_proprio",
        "observations/proprioceptions/other/imu_proprio",
        "actions/velocity_commands",
    ]
    if frames < 2 or any(path not in h5_file for path in required):
        return []
    expected_shapes = ((frames, 6), (frames, 12), (frames, 12), (frames, 58), (frames, 9), (frames, 3))
    datasets = [h5_file[path] for path in required]
    if any(
        not isinstance(dataset, h5py.Dataset)
        or dataset.shape != expected_shape
        or dataset.dtype.kind not in "fiu"
        for dataset, expected_shape in zip(datasets, expected_shapes)
    ):
        return []
    body, eef, relative, robot, imu, velocity = [np.asarray(dataset[...]) for dataset in datasets]

    same = np.ones(frames - 1, dtype=np.bool_)
    for array in (
        body,
        eef[:, [0, 1, 2, 6, 7, 8]],
        relative[:, [0, 1, 2, 6, 7, 8]],
        robot,
        imu,
        velocity,
    ):
        finite = np.all(np.isfinite(array), axis=1)
        same &= finite[:-1] & finite[1:] & np.all(array[:-1] == array[1:], axis=1)

    right_base = _rx(math.pi / 2.0)
    left_base = _rx(-math.pi / 2.0)
    hits: list[dict[str, Any]] = []
    for current in (np.flatnonzero(same) + 1).tolist():
        previous = current - 1
        deltas = [
            _rpy_to_matrix(eef[previous, 3:6]).T @ _rpy_to_matrix(eef[current, 3:6]),
            _rpy_to_matrix(eef[previous, 9:12]).T @ _rpy_to_matrix(eef[current, 9:12]),
            _rpy_to_matrix(relative[previous, 3:6]).T @ _rpy_to_matrix(relative[current, 3:6]),
            _rpy_to_matrix(relative[previous, 9:12]).T @ _rpy_to_matrix(relative[current, 9:12]),
        ]
        targets = [right_base, left_base, right_base, left_base]
        errors = [_rotation_error(delta, target) for delta, target in zip(deltas, targets)]
        maximum_error = max(errors)
        if maximum_error <= tolerance:
            hits.append(
                {
                    "previous_frame": previous,
                    "current_frame": current,
                    "remap_power": 1,
                    "rotation_errors_rad": errors,
                    "max_rotation_error_rad": maximum_error,
                    "confidence": "exact_cached_state_high_confidence",
                }
            )
    return hits


def _check_quality_warnings(h5_file: h5py.File, report: dict[str, Any], frames: int, chunk_size: int) -> None:
    for path, check_id, message in (
        (
            "observations/proprioceptions/gripper_proprio",
            "placeholder_gripper",
            "Gripper proprio is all zero while its mask is valid; canonical HTD configs do not consume it",
        ),
        (
            "actions/velocity_commands",
            "stationary_velocity_subset",
            "All velocity commands are zero; this episode does not cover locomotion commands",
        ),
    ):
        if (
            path not in h5_file
            or not isinstance(h5_file[path], h5py.Dataset)
            or not h5_file[path].shape
            or h5_file[path].shape[0] != frames
        ):
            continue
        all_zero = True
        for selection in _iter_slices(frames, chunk_size):
            if np.any(np.asarray(h5_file[path][selection]) != 0):
                all_zero = False
                break
        if all_zero:
            _finding(report, "warning", check_id, message, hdf5_path=path)


def validate_file(
    path: Path,
    root: Path,
    args: argparse.Namespace,
    manifest_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    before = _source_state(path)
    before_hash = _sha256(path) if args.hash_files or args.prove_read_only else None
    report: dict[str, Any] = {
        "path": str(path),
        "relative_path": path.relative_to(root).as_posix(),
        "source_before": before,
        "findings": [],
    }
    if before_hash is not None:
        report["sha256"] = before_hash

    if manifest_entry is not None:
        expected_size = manifest_entry.get("size_bytes")
        if expected_size is not None and int(expected_size) != before["size_bytes"]:
            _finding(
                report,
                "error",
                "manifest_size_mismatch",
                "File size does not match the manifest",
                expected=int(expected_size),
                actual=before["size_bytes"],
            )
        expected_hash = manifest_entry.get("sha256")
        if expected_hash is not None and before_hash is not None and str(expected_hash) != before_hash:
            _finding(
                report,
                "error",
                "manifest_sha256_mismatch",
                "File SHA-256 does not match the manifest",
                expected=str(expected_hash),
                actual=before_hash,
            )
        report["manifest_split"] = manifest_entry.get("split")

    try:
        with h5py.File(path, "r") as h5_file:
            report["attributes"] = {key: _json_value(value) for key, value in sorted(h5_file.attrs.items())}
            _check_attrs(h5_file, report, args.assume_legacy_coordinate, args.expected_frequency)

            timestamp_path = "observations/head_cam_timestamp"
            if (
                timestamp_path not in h5_file
                or not isinstance(h5_file[timestamp_path], h5py.Dataset)
                or len(h5_file[timestamp_path].shape) != 1
            ):
                frames = 0
                _finding(
                    report,
                    "error",
                    "episode_length_unavailable",
                    "Cannot determine episode length from head_cam_timestamp",
                    hdf5_path=timestamp_path,
                )
            else:
                frames = int(h5_file[timestamp_path].shape[0])
                if frames <= 0:
                    _finding(report, "error", "episode_empty", "Episode contains no frames")
            report["frames"] = frames
            if manifest_entry is not None and manifest_entry.get("frames") is not None:
                expected_frames = int(manifest_entry["frames"])
                if frames != expected_frames:
                    _finding(
                        report,
                        "error",
                        "manifest_frame_count_mismatch",
                        "Episode frame count does not match the manifest",
                        expected=expected_frames,
                        actual=frames,
                    )

            profile, stage = _detect_profile_and_stage(
                h5_file, report, args.profile, args.stage
            )
            report["hand_profile"] = profile
            report["processing_stage"] = stage
            camera_profile = getattr(args, "camera_profile", "head-and-wrists")
            body_contract = _check_masks(h5_file, report, args.expected_body_mask, camera_profile)
            report["body_mask_contract"] = body_contract

            if frames > 0:
                _check_required_temporal(h5_file, report, frames, CORE_TEMPORAL_DIMS)
                if stage == "policy" or args.stage == "policy":
                    _check_required_temporal(h5_file, report, frames, G1_POLICY_DIMS)
                    if profile in TACTILE_DIMS:
                        _check_required_temporal(h5_file, report, frames, _profile_split_paths(profile))
                _check_all_temporal_lengths(h5_file, report, frames)
                _check_finite(h5_file, report, frames, args.chunk_size)
                if stage == "policy" and profile in TACTILE_DIMS:
                    _check_policy_consistency(h5_file, report, profile, frames, args.chunk_size)
                if profile in TACTILE_DIMS:
                    _check_tactile_semantics(h5_file, report, profile, frames, args.chunk_size)
                _check_timestamps(h5_file, report, frames, args.timestamp_gap_factor)
                _check_images(h5_file, report, frames, args.images, args.image_samples, camera_profile)
                hits = detect_repeated_dds_transforms(h5_file, frames, args.dds_rotation_tolerance)
                report["repeated_dds_transform"] = {
                    "count": len(hits),
                    "hits": hits,
                    "rotation_tolerance_rad": args.dds_rotation_tolerance,
                }
                if hits:
                    _finding(
                        report,
                        "error",
                        "repeated_dds_transform",
                        "Exact cached DDS state received an additional in-place EEF frame transform",
                        count=len(hits),
                        indices=[hit["current_frame"] for hit in hits],
                        actual=hits,
                        remediation="frame_repair_or_quarantine",
                    )
                _check_quality_warnings(h5_file, report, frames, args.chunk_size)
    except OSError as exc:
        _finding(report, "error", "hdf5_open_failed", f"Could not read HDF5 file: {exc}")

    after = _source_state(path)
    report["source_after"] = after
    unchanged = before == after
    if args.prove_read_only:
        after_hash = _sha256(path)
        report["sha256_after"] = after_hash
        unchanged = unchanged and before_hash == after_hash
    report["source_unchanged"] = unchanged
    if not unchanged:
        _finding(
            report,
            "error",
            "source_changed_during_validation",
            "Input file metadata or content changed during validation",
            expected={"state": before, "sha256": before_hash},
            actual={"state": after, "sha256": report.get("sha256_after")},
        )
    return report


def _summarize(report: dict[str, Any]) -> None:
    findings = [
        finding
        for file_report in report["files"]
        for finding in file_report.get("findings", [])
    ] + report.get("corpus_findings", [])
    severity_counts = Counter(finding["severity"] for finding in findings)
    profiles = Counter(file_report.get("hand_profile", "unknown") for file_report in report["files"])
    stages = Counter(file_report.get("processing_stage", "unknown") for file_report in report["files"])
    masks = Counter(file_report.get("body_mask_contract", "unknown") for file_report in report["files"])
    dds_hits = sum(file_report.get("repeated_dds_transform", {}).get("count", 0) for file_report in report["files"])
    images_checked = sum(file_report.get("images", {}).get("checked", 0) for file_report in report["files"])
    report["summary"] = {
        "files": len(report["files"]),
        "frames": sum(int(file_report.get("frames", 0)) for file_report in report["files"]),
        "errors": severity_counts["error"],
        "warnings": severity_counts["warning"],
        "infos": severity_counts["info"],
        "profiles": dict(sorted(profiles.items())),
        "processing_stages": dict(sorted(stages.items())),
        "body_mask_contracts": dict(sorted(masks.items())),
        "repeated_dds_transform_hits": dds_hits,
        "images_checked": images_checked,
        "all_sources_unchanged": all(file_report.get("source_unchanged", False) for file_report in report["files"]),
    }


def validate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    root, files = discover_hdf5(input_path)
    manifest = _load_manifest(Path(args.manifest) if args.manifest else None)
    corpus_report: dict[str, Any] = {"findings": []}
    entries = _validate_manifest_inventory(corpus_report, root, files, manifest)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "tool": {"name": "validate_htd_dataset", "version": TOOL_VERSION},
        "input_root": str(root),
        "requested_contract": {
            "embodiment": EXPECTED_EMBODIMENT,
            "profile": args.profile,
            "stage": args.stage,
            "camera_profile": getattr(args, "camera_profile", "head-and-wrists"),
            "coordinate_convention": LEGACY_COORDINATE,
            "body_mask": args.expected_body_mask,
            "frequency_hz": args.expected_frequency,
        },
        "read_only": True,
        "files": [],
        "corpus_findings": corpus_report["findings"],
    }
    corpus_sink = {"findings": report["corpus_findings"]}

    physical_paths: dict[tuple[int, int], list[str]] = {}
    for path in files:
        stat = path.stat()
        identity = (int(stat.st_dev), int(stat.st_ino))
        physical_paths.setdefault(identity, []).append(path.relative_to(root).as_posix())
    for logical_paths in physical_paths.values():
        if len(logical_paths) > 1:
            _finding(
                corpus_sink,
                "error",
                "duplicate_physical_episode",
                "Multiple logical episode paths refer to the same physical HDF5 file",
                count=len(logical_paths),
                actual=sorted(logical_paths),
            )
    if manifest is not None:
        manifest_coordinate = _manifest_value(manifest, "coordinate_convention")
        manifest_profile = _manifest_value(manifest, "hand_profile")
        report["manifest"] = {
            "path": str(Path(args.manifest).resolve()),
            "dataset_id": manifest.get("dataset_id"),
            "manifest_version": manifest.get("manifest_version"),
            "coordinate_convention": manifest_coordinate,
            "hand_profile": manifest_profile,
        }
        if manifest_coordinate is not None and manifest_coordinate != LEGACY_COORDINATE:
            _finding(
                corpus_sink,
                "error",
                "manifest_coordinate_conflict",
                "Manifest declares a coordinate convention outside the HTD v1 contract",
                expected=LEGACY_COORDINATE,
                actual=manifest_coordinate,
            )
        if manifest_profile is not None and (
            not isinstance(manifest_profile, str) or manifest_profile not in {"dftp", "f1"}
        ):
            _finding(
                corpus_sink,
                "error",
                "manifest_hand_profile_invalid",
                "Manifest hand profile is not supported",
                expected=["dftp", "f1"],
                actual=manifest_profile,
            )
        if not (args.hash_files or args.prove_read_only) and any(
            entry.get("sha256") for entry in entries.values()
        ):
            _finding(
                corpus_sink,
                "warning",
                "manifest_hashes_not_checked",
                "Manifest has SHA-256 values; pass --hash-files to verify them",
            )

    for index, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        if not args.quiet:
            print(f"[{index}/{len(files)}] {relative}", file=sys.stderr, flush=True)
        report["files"].append(validate_file(path, root, args, entries.get(relative)))

    profiles = {
        profile
        for file_report in report["files"]
        if isinstance((profile := file_report.get("hand_profile")), str) and profile != "unknown"
    }
    if len(profiles) > 1:
        _finding(
            corpus_sink,
            "error",
            "corpus_mixed_hand_profiles",
            "A release dataset may not mix DFTP and F1 files",
            actual=sorted(profiles),
        )
    if manifest is not None:
        manifest_profile = _manifest_value(manifest, "hand_profile")
        if (
            isinstance(manifest_profile, str)
            and manifest_profile in {"dftp", "f1"}
            and profiles
            and profiles != {manifest_profile}
        ):
            _finding(
                corpus_sink,
                "error",
                "manifest_hand_profile_conflict",
                "Manifest hand profile conflicts with detected episode data",
                expected=manifest_profile,
                actual=sorted(profiles),
            )
    if profiles == {"dftp"} and (
        manifest is None or _manifest_value(manifest, "dftp_hand_order") is None
    ):
        _finding(
            corpus_sink,
            "warning",
            "dftp_hand_order_provenance_unresolved",
            "Dataset provenance does not identify the historical DFTP hand-proprio ordering",
            remediation="checkpoint_contract",
        )
    if manifest is None or not any(
        key in manifest for key in ("collector_commit", "collection_code_commit", "saver_revision")
    ):
        _finding(
            corpus_sink,
            "warning",
            "collector_saver_provenance_unresolved",
            "Dataset provenance does not identify the collector/saver code revision",
            remediation="checkpoint_contract",
        )
    masks = {
        contract
        for file_report in report["files"]
        if isinstance((contract := file_report.get("body_mask_contract")), str)
        and contract not in {"missing", "invalid"}
    }
    if len(masks) > 1:
        _finding(
            corpus_sink,
            "error",
            "corpus_mixed_body_mask_contracts",
            "Files use different body-mask/checkpoint contracts",
            actual=sorted(masks),
            remediation="checkpoint_contract",
        )
    _summarize(report)
    return _json_value(report)


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validator for historical HTD G1 DFTP/F1 HDF5 datasets"
    )
    parser.add_argument("input", help="HDF5 file or directory to validate")
    parser.add_argument("--manifest", help="Optional HTD JSON manifest with an exact file inventory")
    parser.add_argument("--profile", choices=("auto", "dftp", "f1"), default="auto")
    parser.add_argument("--stage", choices=("auto", "collector", "policy"), default="policy")
    parser.add_argument("--camera-profile", choices=("head-and-wrists", "head-only"),
                        default="head-and-wrists",
                        help="Required views and image masks; Tea uses head-only")
    parser.add_argument(
        "--assume-legacy-coordinate",
        action="store_true",
        help=f"Assert provenance for files missing coordinate_convention={LEGACY_COORDINATE}",
    )
    parser.add_argument(
        "--expected-body-mask",
        choices=("any", "legacy", "corrected"),
        default="any",
        help="Checkpoint/data body-mask contract",
    )
    parser.add_argument("--images", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--image-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--expected-frequency", type=float, default=30.0)
    parser.add_argument("--timestamp-gap-factor", type=float, default=2.5)
    parser.add_argument("--dds-rotation-tolerance", type=float, default=1e-6)
    parser.add_argument("--hash-files", action="store_true", help="Compute and verify SHA-256 when available")
    parser.add_argument(
        "--prove-read-only",
        action="store_true",
        help="Hash every input both before and after validation (expensive)",
    )
    parser.add_argument("--report", help="Write deterministic JSON report outside the input directory")
    parser.add_argument("--fail-on", choices=("error", "warning", "never"), default="error")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.image_samples <= 0:
        raise ValueError("--image-samples must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not math.isfinite(args.expected_frequency) or args.expected_frequency <= 0:
        raise ValueError("--expected-frequency must be positive and finite")
    if not math.isfinite(args.timestamp_gap_factor) or args.timestamp_gap_factor <= 0:
        raise ValueError("--timestamp-gap-factor must be positive and finite")
    if not math.isfinite(args.dds_rotation_tolerance) or args.dds_rotation_tolerance <= 0:
        raise ValueError("--dds-rotation-tolerance must be positive and finite")
    if args.report:
        input_path = Path(args.input).resolve()
        input_root = input_path if input_path.is_dir() else input_path.parent
        report_path = Path(args.report).resolve()
        if _is_relative_to(report_path, input_root):
            raise ValueError("--report must be outside the input dataset directory")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_args(args)
        report = validate_dataset(args)
        if args.report:
            _write_json_atomic(Path(args.report), report)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"validator failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    summary = report["summary"]
    print(
        "HTD dataset validation: "
        f"files={summary['files']} frames={summary['frames']} "
        f"errors={summary['errors']} warnings={summary['warnings']} "
        f"dds_hits={summary['repeated_dds_transform_hits']} "
        f"images_checked={summary['images_checked']} "
        f"sources_unchanged={summary['all_sources_unchanged']}"
    )
    if args.report:
        print(f"Report: {Path(args.report).resolve()}")

    if args.fail_on == "never":
        return 0
    if summary["errors"]:
        return 1
    if args.fail_on == "warning" and summary["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
