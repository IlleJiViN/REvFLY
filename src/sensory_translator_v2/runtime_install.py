"""Install a local, content-addressed V2 runtime bundle."""

from __future__ import annotations

import ctypes
import json
import os
import re
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

from . import assets
from .assets import default_runtime_asset_root
from .contracts import AssetValidationError

WORKBENCH_FILES = frozenset({
    "apps/templates/rate_control.html",
    "static/rate_control.css",
    "static/rate_control.js",
})
_MAX_ENTRIES = 4096
_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
_MAX_FILE_BYTES = 128 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _archive_path(name: str) -> PurePosixPath:
    parts = name.removesuffix("/").split("/")
    if (
        len(parts) > 24
        or any(part in {"", ".", ".."} for part in parts)
        or any(char in name for char in ("\\", ":"))
        or any(ord(char) < 32 for char in name)
    ):
        raise AssetValidationError(f"Unsafe path in ZIP or checksum list: {name!r}")
    return PurePosixPath(*parts)


def _bundle_file(path: PurePosixPath) -> bool:
    if path.as_posix() in WORKBENCH_FILES or path.as_posix() == "manifest.json":
        return True
    parts = path.parts
    return (
        len(parts) == 3
        and parts[0] == "objects"
        and parts[1] == parts[2][:2]
        and _DIGEST.fullmatch(parts[2]) is not None
    )


def _copy_file(handle: BinaryIO, path: Path, expected_size: int) -> None:
    if not 0 <= expected_size <= _MAX_FILE_BYTES:
        raise AssetValidationError(f"Bundle file exceeds the allowed size: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with path.open("xb") as output:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > expected_size:
                raise AssetValidationError(f"Bundle file changed size while being copied: {path.name}")
            output.write(chunk)
    if size != expected_size:
        raise AssetValidationError(f"Bundle file changed size while being copied: {path.name}")


def _checksum_list(contents: bytes) -> dict[PurePosixPath, str]:
    checksums = {}
    for line in contents.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if match is None:
            raise AssetValidationError("Invalid SHA256SUMS line")
        digest, name = match.groups()
        path = _archive_path(name)
        if path in checksums:
            raise AssetValidationError(f"Duplicate SHA256SUMS entry: {name}")
        checksums[path] = digest.lower()
    return checksums


def _check_release_file(
    checksums: dict[PurePosixPath, str] | None, listed_path: PurePosixPath, staged_path: Path,
) -> None:
    if checksums is not None and assets.sha256_file(staged_path) != checksums.get(listed_path):
        raise AssetValidationError(f"Release SHA256SUMS mismatch or missing entry for {listed_path}")


def _stage_zip(source: Path, stage: Path) -> bool:
    if source.stat().st_size > _MAX_BUNDLE_BYTES:
        raise AssetValidationError("Runtime ZIP exceeds the allowed size")
    with ZipFile(source) as archive:
        info_list = archive.infolist()
        if len(info_list) > _MAX_ENTRIES or sum(info.file_size for info in info_list) > _MAX_BUNDLE_BYTES:
            raise AssetValidationError("Runtime ZIP exceeds the expanded size or entry limit")

        entries = {}
        for info in info_list:
            path = _archive_path(info.orig_filename)
            mode = stat.S_IFMT(info.external_attr >> 16)
            expected = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
            if mode not in {0, expected} or (info.create_system == 0 and info.external_attr & 0x400):
                raise AssetValidationError(f"Runtime ZIP contains a symbolic link or special file: {path}")
            if info.flag_bits & 1:
                raise AssetValidationError(f"Runtime ZIP contains an encrypted entry: {path}")
            if info.file_size > _MAX_FILE_BYTES:
                raise AssetValidationError(f"Runtime ZIP file exceeds the allowed size: {path}")
            if path in entries:
                raise AssetValidationError(f"Duplicate runtime ZIP entry: {path}")
            entries[path] = info
        for path in entries:
            if any(parent in entries and not entries[parent].is_dir() for parent in path.parents):
                raise AssetValidationError(f"Runtime ZIP file conflicts with a directory: {path}")

        files = {path: info for path, info in entries.items() if not info.is_dir()}
        manifests = {path.parent for path in files if path.name == "manifest.json"}
        object_roots = {parent.parent for path in files for parent in path.parents if parent.name == "objects"}
        roots = manifests & object_roots
        if len(roots) != 1:
            raise AssetValidationError("Runtime ZIP must contain exactly one bundle manifest and objects tree")
        root = roots.pop()
        checksum_path = next((
            path for path in (PurePosixPath("SHA256SUMS"), root.parent / "SHA256SUMS", root / "SHA256SUMS")
            if path in files
        ), None)
        checksums = None
        if checksum_path is not None:
            if files[checksum_path].file_size > _MAX_METADATA_BYTES:
                raise AssetValidationError("Release SHA256SUMS exceeds the allowed size")
            checksums = _checksum_list(archive.read(files[checksum_path]))

        for path, info in files.items():
            if not path.is_relative_to(root):
                continue
            relative = path.relative_to(root)
            if not _bundle_file(relative):
                continue
            target = stage.joinpath(*relative.parts)
            with archive.open(info) as handle:
                _copy_file(handle, target, info.file_size)
            if checksum_path is not None:
                _check_release_file(checksums, path.relative_to(checksum_path.parent), target)
        return checksums is not None


def _regular_files(root: Path) -> dict[PurePosixPath, tuple[Path, int]]:
    files = {}
    total_size = 0
    count = 0
    directories = [root]
    while directories:
        for path in directories.pop().iterdir():
            info = path.lstat()
            count += 1
            if count > _MAX_ENTRIES:
                raise AssetValidationError("Bundle folder exceeds the allowed entry count")
            if stat.S_ISDIR(info.st_mode) and not _is_link(info):
                directories.append(path)
            elif stat.S_ISREG(info.st_mode):
                total_size += info.st_size
                if info.st_size > _MAX_FILE_BYTES or total_size > _MAX_BUNDLE_BYTES:
                    raise AssetValidationError("Bundle folder exceeds the allowed size")
                files[PurePosixPath(path.relative_to(root).as_posix())] = (path, info.st_size)
            else:
                raise AssetValidationError(f"Bundle folder contains a symbolic link or special file: {path}")
    return files


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _path_is_link(path: Path) -> bool:
    try:
        return _is_link(path.lstat())
    except (FileNotFoundError, NotADirectoryError):
        return False


def _stage_folder(source: Path, stage: Path) -> bool:
    root = source
    if not (root / "manifest.json").exists() and not (root / "manifest.json").is_symlink():
        root = source / "bundle"
    if _path_is_link(root) or not root.is_dir():
        raise AssetValidationError(f"Source does not contain a runtime bundle folder: {source}")
    files = _regular_files(root)
    checksum_path = source / "SHA256SUMS"
    checksums = None
    if checksum_path.exists() or checksum_path.is_symlink():
        info = checksum_path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_METADATA_BYTES:
            raise AssetValidationError("Release SHA256SUMS must be a bounded regular file")
        checksums = _checksum_list(checksum_path.read_bytes())

    for relative, (path, size) in files.items():
        if not _bundle_file(relative):
            continue
        target = stage.joinpath(*relative.parts)
        with path.open("rb") as handle:
            _copy_file(handle, target, size)
        _check_release_file(checksums, PurePosixPath(path.relative_to(source).as_posix()), target)
    return checksums is not None


def _validate_manifest_layout(root: Path) -> None:
    """Reject malformed object digests and paths before the catalog resolves them."""
    manifest = root / "manifest.json"
    if not manifest.is_file() or not (root / "objects").is_dir():
        raise AssetValidationError("Runtime bundle needs manifest.json and its objects tree")
    if manifest.stat().st_size > _MAX_METADATA_BYTES:
        raise AssetValidationError("Runtime bundle manifest exceeds the allowed size")

    def digest(value: Any) -> None:
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise AssetValidationError(f"Malformed content-addressed object SHA256: {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise AssetValidationError("Duplicate JSON keys in runtime bundle manifest")
        return obj

    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        digest(raw["raw_xml"]["sha256"])
        digest(raw["teacher"]["sha256"])
        for name, value in raw["stl_meshes"].items():
            if not re.fullmatch(r"meshes/stl/[A-Za-z0-9_.-]+\.stl", name):
                raise AssetValidationError(f"Unsafe runtime mesh path: {name!r}")
            digest(value)
        for entry in raw["models"].values():
            digest(entry["checkpoint_sha256"])
            for key in ("preregistered_config_sha256", "selection_sha256"):
                if entry.get(key) is not None:
                    digest(entry[key])
        if raw.get("multijoint_checkpoint") is not None:
            digest(raw["multijoint_checkpoint"]["sha256"])
    except (KeyError, AttributeError, TypeError, json.JSONDecodeError) as exc:
        raise AssetValidationError(f"Invalid runtime bundle manifest: {exc}") from exc


def _verify_staged_bundle(root: Path) -> dict[str, Any]:
    _validate_manifest_layout(root)
    catalog = assets.load_asset_catalog(asset_root=root, verify_hashes=True)
    if not catalog.models and catalog.resolve_multijoint_model() is None:
        raise AssetValidationError("Runtime bundle contains no rate model checkpoint")
    for spec in assets.KNOWN_V1_MODELS:
        model = catalog.models.get(spec["model_id"])
        if model is not None and model.checkpoint_sha256 != spec["expected_checkpoint_sha256"]:
            raise AssetValidationError(f"Registered v1 checkpoint SHA256 mismatch for {model.model_id}")
    embedded_meshes = {
        Path(mesh.attrib["file"]).as_posix()
        for mesh in ET.parse(catalog.xml_path).getroot().findall(".//asset/mesh[@file]")
    }
    if embedded_meshes != catalog.mesh_paths.keys():
        raise AssetValidationError("Runtime bundle mesh names do not match the frozen FlyMimic XML")
    return assets.verify_asset_catalog(catalog)


def _fingerprint(root: Path) -> dict[PurePosixPath, str]:
    files = _regular_files(root)
    return {relative: assets.sha256_file(path) for relative, (path, _) in files.items()}


def _exchange_directories(staged: Path, destination: Path) -> None:
    """Atomically exchange populated directories on supported filesystems."""
    libc = ctypes.CDLL(None, use_errno=True)
    paths = (os.fsencode(staged), os.fsencode(destination))
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        exchange = libc.renameat2
        exchange.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        result = exchange(-100, paths[0], -100, paths[1], 2)  # AT_FDCWD, RENAME_EXCHANGE
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        exchange = libc.renamex_np
        exchange.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = exchange(*paths, 2)  # RENAME_SWAP
    else:
        raise AssetValidationError("Atomic bundle replacement is unavailable on this platform. Choose a new destination.")
    if result != 0:
        reason = os.strerror(ctypes.get_errno())
        raise AssetValidationError(f"Atomic bundle replacement failed: {reason}")


def install_runtime_bundle(
    source: str | Path, *, destination: str | Path | None = None, replace: bool = False,
) -> dict[str, Any]:
    """Verify and install a local release ZIP or extracted folder before changing the destination."""
    source_path = Path(source).expanduser()
    destination_path = (default_runtime_asset_root() if destination is None
                        else Path(destination).expanduser())
    try:
        if _path_is_link(source_path) or _path_is_link(destination_path):
            raise AssetValidationError("Runtime source and destination cannot be symbolic links")
        source_path, destination_path = source_path.resolve(), destination_path.resolve()
        source_kind = "folder" if source_path.is_dir() else "zip"
        if source_kind == "zip" and not source_path.is_file():
            raise AssetValidationError(f"Runtime source must be a local ZIP file or bundle folder: {source_path}")
        if source_path != destination_path and (
            source_path.is_relative_to(destination_path)
            or (source_kind == "folder" and destination_path.is_relative_to(source_path))
        ):
            raise AssetValidationError("Runtime source and destination cannot contain one another")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{destination_path.name}-install-", dir=destination_path.parent,
            ignore_cleanup_errors=True,
        ) as tmp:
            stage = Path(tmp) / "bundle"
            stage.mkdir(mode=0o700)
            stage_bundle = _stage_folder if source_kind == "folder" else _stage_zip
            checksums_verified = stage_bundle(source_path, stage)
            report = _verify_staged_bundle(stage)
            staged_files = _fingerprint(stage)
            for relative, actual in staged_files.items():
                if relative.parts[0] == "objects" and actual != relative.name:
                    raise AssetValidationError(f"Content-addressed object hash mismatch: {relative}")

            lock = destination_path.parent / f".{destination_path.name}.install-lock"
            try:
                lock.mkdir()
            except FileExistsError as exc:
                raise AssetValidationError(f"Runtime install already in progress at {lock}") from exc
            try:
                if _path_is_link(destination_path):
                    raise AssetValidationError("Runtime destination cannot be a symbolic link")
                if destination_path.exists() and not destination_path.is_dir():
                    raise AssetValidationError(f"Runtime destination is not a directory: {destination_path}")
                if destination_path.is_dir() and any(destination_path.iterdir()):
                    existing_files = _fingerprint(destination_path)
                    if existing_files == staged_files:
                        status = "unchanged"
                    elif not replace:
                        raise AssetValidationError(f"Runtime destination already exists: {destination_path}. Use --replace to update it.")
                    elif (
                        PurePosixPath("manifest.json") not in existing_files
                        or not any(path.parts[0] == "objects" for path in existing_files)
                        or any(not _bundle_file(path) for path in existing_files)
                    ):
                        raise AssetValidationError(f"Runtime destination contains files outside a managed bundle: {destination_path}")
                    else:
                        _exchange_directories(stage, destination_path)
                        status = "replaced"
                else:
                    stage.replace(destination_path)
                    status = "installed"
            finally:
                lock.rmdir()
    except AssetValidationError:
        raise
    except (OSError, EOFError, UnicodeError, BadZipFile, RuntimeError, ET.ParseError, zlib.error) as exc:
        raise AssetValidationError(f"Could not install runtime bundle from {source_path}: {exc}") from exc

    for model in (*report["models"].values(), report.get("multijoint_checkpoint", {})):
        if "checkpoint_path" in model:
            model["checkpoint_path"] = str(destination_path / Path(model["checkpoint_path"]).relative_to(stage))
    report.update(
        status=status,
        asset_root=str(destination_path),
        install_source=str(source_path),
        install_source_kind=source_kind,
        workbench_available=all(PurePosixPath(name) in staged_files for name in WORKBENCH_FILES),
        release_bundle_checksums_verified=checksums_verified,
    )
    return report
