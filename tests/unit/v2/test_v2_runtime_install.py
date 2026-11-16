"""Local bundle installation and preservation of an existing install."""

import json
import stat
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from sensory_translator_v2 import assets, runtime_install
from sensory_translator_v2.contracts import AssetValidationError


@pytest.fixture
def exported_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "exported"

    def store(payload):
        digest = assets.sha256_bytes(payload)
        path = bundle / "objects" / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return digest

    meshes = {f"meshes/stl/mesh{i}.stl": store(f"mesh {i}".encode()) for i in range(71)}
    mesh_xml = "".join(f'<mesh file="./{path}"/>' for path in meshes)
    xml = store(f"<mujoco><asset>{mesh_xml}</asset></mujoco>".encode())
    teacher = store(b"teacher fixture")
    monkeypatch.setattr(assets, "EXPECTED_XML_SHA256", xml)
    monkeypatch.setattr(assets, "EXPECTED_TEACHER_SHA256", teacher)
    model = store(b"multi-joint checkpoint fixture")
    monkeypatch.setitem(assets.KNOWN_MULTIJOINT_CHECKPOINTS, "registered-fixture", model)
    (bundle / "manifest.json").write_text(json.dumps({
        "schema_version": assets.BUNDLE_SCHEMA,
        "raw_xml": {"sha256": xml},
        "teacher": {"sha256": teacher},
        "stl_meshes": meshes,
        "mesh_tree_sha256": assets.canonical_json_sha256(sorted(meshes.items())),
        "models": {},
        "multijoint_checkpoint": {"model_id": "registered-fixture", "sha256": model},
    }))
    return bundle, model


def _write_workbench(bundle):
    for filename in runtime_install.WORKBENCH_FILES:
        path = bundle / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)


def _release_zip(bundle, destination, *, tamper=None):
    entries = {"README.md": b"Release instructions", "sensory_translator_v2.whl": b"Wheel fixture"}
    entries.update({
        f"bundle/{path.relative_to(bundle).as_posix()}": path.read_bytes()
        for path in bundle.rglob("*") if path.is_file()
    })
    sums = "".join(f"{assets.sha256_bytes(content)}  {name}\n" for name, content in sorted(entries.items()))
    entries["SHA256SUMS"] = sums.encode()
    if tamper is not None:
        entries[tamper] = b"Changed file with original checksum"
    with ZipFile(destination, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def test_outer_release_zip_verifies_copied_entries_and_reinstalls_cleanly(tmp_path, exported_bundle, monkeypatch):
    source, digest = exported_bundle
    _write_workbench(source)
    archive = tmp_path / "release.zip"
    _release_zip(source, archive)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))

    report = runtime_install.install_runtime_bundle(archive)
    destination = assets.default_runtime_asset_root()
    assert destination == tmp_path / "share/sensory-translator-v2/bundle"
    assert report["asset_root"] == str(destination)
    assert report["status"] == "installed"
    assert report["workbench_available"] is True
    assert report["release_bundle_checksums_verified"] is True
    assert report["multijoint_checkpoint"]["checkpoint_path"] == str(destination / "objects" / digest[:2] / digest)
    assert assets.verify_asset_catalog(assets.load_asset_catalog(asset_root=destination))["verified"]
    assert not (destination / "README.md").exists()
    assert not (destination / "sensory_translator_v2.whl").exists()
    assert runtime_install.install_runtime_bundle(archive)["status"] == "unchanged"


def test_folder_optional_workbench_and_atomic_replacement(tmp_path, exported_bundle):
    source, _ = exported_bundle
    destination = tmp_path / "installed"
    report = runtime_install.install_runtime_bundle(source, destination=destination)
    assert report["status"] == "installed"
    assert report["workbench_available"] is False
    assert report["release_bundle_checksums_verified"] is False

    _write_workbench(source)
    with pytest.raises(AssetValidationError, match="Use --replace"):
        runtime_install.install_runtime_bundle(source, destination=destination)
    assert not (destination / "static").exists()

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        replaced = runtime_install.install_runtime_bundle(source, destination=destination, replace=True)
        assert replaced["status"] == "replaced"
        assert replaced["workbench_available"] is True
        assert report["manifest_sha256"] == replaced["manifest_sha256"]
        assert (destination / "static/rate_control.js").is_file()


def test_validation_failure_preserves_previous_install(tmp_path, exported_bundle):
    source, digest = exported_bundle
    _write_workbench(source)
    destination = tmp_path / "installed"
    runtime_install.install_runtime_bundle(source, destination=destination)
    manifest_before = (destination / "manifest.json").read_bytes()
    ui_path = "static/rate_control.js"
    archive = tmp_path / "broken-release.zip"
    _release_zip(source, archive, tamper=f"bundle/{ui_path}")
    with pytest.raises(AssetValidationError, match="Release SHA256SUMS mismatch"):
        runtime_install.install_runtime_bundle(archive, destination=destination, replace=True)
    assert (destination / ui_path).read_text() == ui_path

    checkpoint = source / "objects" / digest[:2] / digest
    checkpoint.write_bytes(b"bad checkpoint")
    with pytest.raises(AssetValidationError, match="Hash mismatch for multi-joint checkpoint"):
        runtime_install.install_runtime_bundle(source, destination=destination, replace=True)
    assert (destination / "manifest.json").read_bytes() == manifest_before
    assert assets.sha256_file(destination / checkpoint.relative_to(source)) == digest


@pytest.mark.parametrize("malformed", ["traversal", "symlink", "oversize"])
def test_unsafe_zip_entries_are_rejected_even_outside_bundle(tmp_path, exported_bundle, monkeypatch, malformed):
    source, _ = exported_bundle
    archive_path = tmp_path / "unsafe.zip"
    _release_zip(source, archive_path)
    with ZipFile(archive_path, "a") as archive:
        if malformed == "traversal":
            archive.writestr("../unsafe.txt", b"unsafe")
        elif malformed == "symlink":
            symlink = ZipInfo("unsafe-link")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(symlink, "/etc/passwd")
        else:
            monkeypatch.setattr(runtime_install, "_MAX_FILE_BYTES", 16 * 1024)
            archive.writestr("unrelated.txt", b"0" * (17 * 1024))

    destination = tmp_path / "installed"
    with pytest.raises(AssetValidationError, match="[Uu]nsafe path|symbolic link|allowed size"):
        runtime_install.install_runtime_bundle(archive_path, destination=destination)
    assert not destination.exists()


def test_symlink_folder_and_unmanaged_destination_are_rejected(tmp_path, exported_bundle):
    source, _ = exported_bundle
    destination = tmp_path / "installed"
    destination.mkdir()
    unrelated = destination / "do-not-remove.txt"
    unrelated.write_text("unrelated")
    with pytest.raises(AssetValidationError, match="outside a managed bundle"):
        runtime_install.install_runtime_bundle(source, destination=destination, replace=True)
    assert unrelated.read_text() == "unrelated"

    (source / "link").symlink_to(unrelated)
    with pytest.raises(AssetValidationError, match="symbolic link"):
        runtime_install.install_runtime_bundle(source, destination=destination, replace=True)
