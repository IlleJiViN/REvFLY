"""Length-prefixed binary-safe JSON + contiguous ndarray IPC protocol for persistent workers."""

from __future__ import annotations

import json
import os
import select
import struct
import time
from typing import Any, BinaryIO

import numpy as np

from ..contracts import BackendSessionError, BackendTimeoutError

PROTOCOL_VERSION = 1
_HEADER_STRUCT = struct.Struct(">IQ")  # 4-byte JSON len, 8-byte binary blob len
MAX_FRAME_BYTES = 128 * 1024 * 1024


def encode_message(header: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> bytes:
    """Serialize metadata header dict and zero-pickle float64 arrays into a single frame."""
    meta = dict(header)
    chunks: list[bytes] = []
    offset = 0
    array_specs: dict[str, Any] = {}
    if arrays:
        for name in sorted(arrays.keys()):
            arr = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.float64))
            raw = arr.tobytes()
            array_specs[name] = {
                "shape": list(arr.shape),
                "dtype": "float64",
                "offset": offset,
                "nbytes": len(raw),
            }
            chunks.append(raw)
            offset += len(raw)
    if array_specs:
        meta["arrays"] = array_specs
    header_bytes = json.dumps(meta, allow_nan=False, separators=(",", ":")).encode("utf-8")
    blob = b"".join(chunks)
    if len(header_bytes) + len(blob) > MAX_FRAME_BYTES:
        raise BackendSessionError("IPC frame exceeds maximum allowed size")
    return _HEADER_STRUCT.pack(len(header_bytes), len(blob)) + header_bytes + blob


def decode_arrays(header: dict[str, Any], blob: bytes) -> dict[str, np.ndarray]:
    """Extract contiguous writable float64 arrays from decoded IPC payload."""
    result: dict[str, np.ndarray] = {}
    specs = header.get("arrays", {})
    if not isinstance(specs, dict):
        return result
    view = memoryview(blob)
    for name, spec in specs.items():
        offset = int(spec["offset"])
        nbytes = int(spec["nbytes"])
        shape = tuple(int(dim) for dim in spec["shape"])
        if offset < 0 or nbytes < 0 or offset + nbytes > len(blob):
            raise BackendSessionError(f"Corrupt IPC array bounds for {name!r}")
        arr = np.frombuffer(view[offset : offset + nbytes], dtype=np.float64).reshape(shape).copy()
        result[name] = arr
    return result


def write_frame(stream: BinaryIO, header: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> None:
    """Write a complete IPC frame and flush immediately."""
    frame = encode_message(header, arrays)
    stream.write(frame)
    stream.flush()


def read_frame_fd(fd: int, *, timeout_s: float | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    """Read a complete IPC frame from raw OS fd with strict timeout (returns None on clean EOF)."""
    deadline = None if timeout_s is None else (time.monotonic() + timeout_s)

    def _read_exact(length: int, *, allow_eof_at_start: bool = False) -> bytes | None:
        buf = bytearray()
        while len(buf) < length:
            wait = None
            if deadline is not None:
                wait = deadline - time.monotonic()
                if wait <= 0.0:
                    raise BackendTimeoutError(f"Worker IPC timed out waiting for {length} bytes")
            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                raise BackendTimeoutError(f"Worker IPC timed out waiting for {length} bytes")
            chunk = os.read(fd, length - len(buf))
            if not chunk:
                if allow_eof_at_start and len(buf) == 0:
                    return None
                raise BackendSessionError("Worker process closed IPC pipe unexpectedly (truncated frame)")
            buf.extend(chunk)
        return bytes(buf)

    prefix = _read_exact(_HEADER_STRUCT.size, allow_eof_at_start=True)
    if prefix is None:
        return None
    header_len, blob_len = _HEADER_STRUCT.unpack(prefix)
    if header_len == 0 or header_len + blob_len > MAX_FRAME_BYTES:
        raise BackendSessionError(f"Invalid IPC frame lengths: header={header_len}, blob={blob_len}")
    header_bytes = _read_exact(header_len)
    assert header_bytes is not None
    blob_bytes = _read_exact(blob_len) if blob_len > 0 else b""
    assert blob_bytes is not None
    header = json.loads(header_bytes.decode("utf-8"))
    arrays = decode_arrays(header, blob_bytes)
    return header, arrays
