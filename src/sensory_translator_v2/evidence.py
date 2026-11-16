"""Atomic evidence writers for JSON reports, NPZ numeric traces, and manifests."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write strict finite JSON atomically (NaN/Infinity forbidden)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(encoded, encoding="utf-8")
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target


def write_npz_atomic(path: str | Path, arrays: dict[str, Any]) -> Path:
    """Write uncompressed numeric NPZ arrays atomically without object pickles."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp.npz")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        clean: dict[str, Any] = {}
        for key, value in arrays.items():
            arr = np.asarray(value)
            if arr.dtype == object:
                raise ValueError(f"Object dtype forbidden in NPZ evidence key {key!r}")
            clean[key] = arr
        np.savez(tmp_path, **clean)
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target


def write_run_artifacts(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    protocol_rate: dict[str, Any],
    protocol_physics: dict[str, Any],
    traces_npz: dict[str, Any],
) -> dict[str, str]:
    """Persist standard run evidence files into an exclusive output folder."""
    manifest_rel = "manifest.json"
    rate_rel = "motor_rate_protocol.json"
    phys_rel = "motor_rate_protocol_physics.json"
    traces_rel = "physical_traces.npz"

    write_json_atomic(output_dir / rate_rel, protocol_rate)
    write_json_atomic(output_dir / phys_rel, protocol_physics)
    write_npz_atomic(output_dir / traces_rel, traces_npz)
    write_json_atomic(output_dir / manifest_rel, manifest)
    return {
        "manifest": manifest_rel,
        "motor_rate_protocol": rate_rel,
        "motor_rate_protocol_physics": phys_rel,
        "physical_traces": traces_rel,
    }
