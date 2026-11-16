"""Content-addressed asset store, v1 compatibility adapter, and XML profile transforms."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    AssetValidationError,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_XML_SHA256,
    PROFILES,
    require_profile,
)

ENV_ASSET_ROOT = "SENSORY_TRANSLATOR_V2_ASSET_ROOT"
ENV_RUNTIME_ROOT = "SENSORY_TRANSLATOR_V2_RUNTIME_ROOT"
ENV_PHYSICS_PYTHON = "SENSORY_TRANSLATOR_V2_PHYSICS_PYTHON"
ENV_SCRIPTS_ROOT = "SENSORY_TRANSLATOR_V2_SCRIPTS_ROOT"

BUNDLE_SCHEMA = "sensory_translator_v2_asset_bundle_v1"

V1_XML_RELPATH = Path("external/FlyMimic/flymimic/assets/models/best_combined_arm_cvt3.xml")
V1_MESH_DIR_RELPATH = Path("external/FlyMimic/flymimic/assets/models/meshes/stl")
V1_TEACHER_RELPATH = Path("results/flymimic_teacher/demo_20261027_v1/teacher_rollout.npz")

KNOWN_V1_MODELS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "20261111_v2_preclip",
        "rel_dir": "results/conditional_rate_ffn/20261111_v2_preclip",
        "profile": "original_v1",
        "expected_checkpoint_sha256": "7b88ba8e9f9946c28869fb571c27074058d419d34ffbeb1e9fc70cfea5c736cf",
        "default_for_profile": "original_v1",
        "description": "Preclip conditional rate FFN trained on 7 offline trajectories",
    },
    {
        "model_id": "20261113_v5_trust",
        "rel_dir": "results/rate_ffn_e2e/20261113_v5_trust",
        "profile": "adjacent_static_parent_v1",
        "expected_checkpoint_sha256": "52f96d066bedc0445f9f41a35e29895df33fb796ffe46fd3d3a67b8056ee42c0",
        "default_for_profile": None,
        "description": "End-to-end physical VJP rate FFN (v5 trust-region stage)",
    },
    {
        "model_id": "20261114_v6_longrun",
        "rel_dir": "results/rate_ffn_e2e/20261114_v6_longrun",
        "profile": "adjacent_static_parent_v1",
        "expected_checkpoint_sha256": "5a84401a4e966a3eb81df14197cbe065200f3c66ba1933da929e3a1539320f2c",
        "default_for_profile": "adjacent_static_parent_v1",
        "description": "Resumed end-to-end physical VJP rate FFN (112 accepted updates)",
    },
)

KNOWN_V2_MODELS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "20261115_v2_fastbridge_lowenergy",
        "rel_dir": "results/v2_models/20261115_v2_fastbridge_lowenergy",
        "profile": "adjacent_static_parent_v1",
        "expected_checkpoint_sha256": "4a60a9ba41a4c19f27efd5486a27e94387f0d6b622e5ac9b2f44a1dd9e939a8f",
        "default_for_profile": None,
        "description": "Stage 3 V2 fast-bridge (tau=2.0ms, K=105) low-energy E2E VJP rate FFN",
    },
)

V2_MODELS_RELPATH = Path("results/v2_models")
MULTIJOINT_MODEL_ID = "20261116_v2_multijoint15_resnorm"
MULTIJOINT_CHECKPOINT_RELPATH = V2_MODELS_RELPATH / MULTIJOINT_MODEL_ID / "best_model.npz"
MULTIJOINT_CHECKPOINT_SHA256 = "45879505e182e205b3554903a25a7e89362304ada2a307f65e1a01731f5a3585"
# Checkout resolution tries the current checkpoint first and supports older pinned bundles.
KNOWN_MULTIJOINT_CHECKPOINTS = {
    MULTIJOINT_MODEL_ID: MULTIJOINT_CHECKPOINT_SHA256,
    "20261115_v2_multijoint15_fastbridge": "8548c7c7aa8227c4a695518327d576b2c6cf26afe7e549e0c5bc1510d01424b9",
}

ADJACENT_EXCLUDED_BODY_PAIRS: tuple[tuple[str, str], ...] = (
    ("Thorax", "LFCoxa"),
    ("Thorax", "RFCoxa"),
)


def sha256_bytes(payload: bytes) -> str:
    """Return lowercase hexadecimal SHA256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return lowercase hexadecimal SHA256 digest of a file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise AssetValidationError(f"Asset file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash canonical UTF-8 JSON without machine-dependent formatting."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def profile_transform_spec(profile: str) -> dict[str, Any]:
    """Return the versioned deterministic rule describing a physics profile transform."""
    require_profile(profile)
    if profile == "original_v1":
        spec = {
            "schema": "sensory_translator_v2_profile_rule_v1",
            "profile": "original_v1",
            "rule_version": 1,
            "enable_multiccd": False,
            "excluded_body_pairs": [],
            "xml_rewrite": "identity",
        }
    elif profile == "multiccd_v1":
        spec = {
            "schema": "sensory_translator_v2_profile_rule_v1",
            "profile": "multiccd_v1",
            "rule_version": 1,
            "enable_multiccd": True,
            "excluded_body_pairs": [],
            "xml_rewrite": "identity_with_runtime_multiccd_flag",
        }
    else:
        spec = {
            "schema": "sensory_translator_v2_profile_rule_v1",
            "profile": "adjacent_static_parent_v1",
            "rule_version": 1,
            "enable_multiccd": True,
            "excluded_body_pairs": [list(pair) for pair in ADJACENT_EXCLUDED_BODY_PAIRS],
            "xml_rewrite": "inject_contact_exclude_pairs",
        }
    spec["profile_rule_sha256"] = canonical_json_sha256(spec)
    return spec


def transform_xml_for_profile(
    raw_xml_text: str,
    profile: str,
    *,
    mesh_dir_override: str | None = None,
) -> str:
    """Deterministically materialize XML text for a profile without altering raw assets."""
    require_profile(profile)
    needs_rewrite = profile == "adjacent_static_parent_v1" or mesh_dir_override is not None
    if not needs_rewrite:
        return raw_xml_text

    root = ET.fromstring(raw_xml_text)
    if mesh_dir_override is not None:
        base_dir = Path(mesh_dir_override)
        for mesh in root.findall(".//asset/mesh"):
            file_attr = mesh.attrib.get("file")
            if file_attr:
                mesh_name = Path(file_attr).name
                mesh.attrib["file"] = str(base_dir / mesh_name)

    if profile == "adjacent_static_parent_v1":
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        existing = {
            (el.attrib.get("body1"), el.attrib.get("body2"))
            for el in contact.findall("exclude")
        }
        for body1, body2 in ADJACENT_EXCLUDED_BODY_PAIRS:
            if (body1, body2) not in existing and (body2, body1) not in existing:
                ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    return ET.tostring(root, encoding="unicode")


@dataclass(frozen=True)
class ModelEntry:
    """Resolved rate policy checkpoint entry in an asset catalog."""

    model_id: str
    profile: str
    checkpoint_path: Path
    checkpoint_sha256: str
    preregistered_config_path: Path | None = None
    preregistered_config_sha256: str | None = None
    selection_path: Path | None = None
    selection_sha256: str | None = None
    default_for_profile: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "profile": self.profile,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "preregistered_config_sha256": self.preregistered_config_sha256,
            "selection_sha256": self.selection_sha256,
            "default_for_profile": self.default_for_profile,
            "description": self.description,
        }


@dataclass(frozen=True)
class AssetCatalog:
    """Immutable verified view of all assets from either a v2 bundle or v1 checkout."""

    source_kind: str  # "v2_bundle" | "v1_checkout"
    asset_root: Path
    runtime_root: Path
    physics_python: Path
    scripts_root: Path | None
    xml_path: Path
    raw_xml_sha256: str
    mesh_paths: dict[str, Path]  # relative key (e.g. "meshes/stl/Thorax.stl") -> path
    mesh_hashes: dict[str, str]
    mesh_tree_sha256: str
    teacher_path: Path
    teacher_sha256: str
    models: dict[str, ModelEntry]
    manifest_sha256: str
    provenance: dict[str, Any] = field(default_factory=dict)
    _bundled_multijoint_model: ModelEntry | None = None

    def default_model_for_profile(self, profile: str) -> ModelEntry:
        require_profile(profile)
        for model in self.models.values():
            if model.default_for_profile == profile:
                return model
        for model in self.models.values():
            if model.profile == profile:
                return model
        # If no model was trained specifically for multiccd_v1, fall back deterministically.
        if "20261114_v6_longrun" in self.models:
            return self.models["20261114_v6_longrun"]
        if self.models:
            return next(iter(self.models.values()))
        raise AssetValidationError("Asset catalog contains no registered rate models")

    @property
    def v2_models(self) -> dict[str, ModelEntry]:
        """Discover optional V2-trained models in checkout workspaces with strict SHA-256 checks."""
        if self.source_kind != "v1_checkout":
            return {}
        found: dict[str, ModelEntry] = {}
        for spec in KNOWN_V2_MODELS:
            ckpt_path = self.asset_root / spec["rel_dir"] / "best_model.npz"
            if not ckpt_path.is_file():
                continue
            ckpt_sha = sha256_file(ckpt_path)
            expected_ckpt = spec["expected_checkpoint_sha256"]
            if expected_ckpt and ckpt_sha != expected_ckpt:
                raise AssetValidationError(
                    f"v2 checkpoint sha256 mismatch for {spec['model_id']}: expected {expected_ckpt}, got {ckpt_sha}"
                )
            found[spec["model_id"]] = ModelEntry(
                model_id=spec["model_id"],
                profile=spec["profile"],
                checkpoint_path=ckpt_path,
                checkpoint_sha256=ckpt_sha,
                preregistered_config_path=None,
                preregistered_config_sha256=None,
                selection_path=None,
                selection_sha256=None,
                default_for_profile=spec["default_for_profile"],
                description=spec["description"],
            )
        return found

    @property
    def all_models(self) -> dict[str, ModelEntry]:
        """Combined catalog of frozen V1 baseline models and registered V2 models."""
        return {**self.models, **self.v2_models}

    def resolve_multijoint_model(self) -> ModelEntry | None:
        """Return the selected multi-joint model and verify its registered hash."""
        model = self._bundled_multijoint_model
        if self.source_kind == "v1_checkout":
            for model_id, digest in KNOWN_MULTIJOINT_CHECKPOINTS.items():
                checkpoint = self.asset_root / V2_MODELS_RELPATH / model_id / "best_model.npz"
                if checkpoint.is_file():
                    model = ModelEntry(model_id, "adjacent_static_parent_v1", checkpoint, digest)
                    break
        if model is None:
            return None
        actual = sha256_file(model.checkpoint_path)
        if actual != model.checkpoint_sha256:
            raise AssetValidationError(
                f"Multi-joint checkpoint sha256 mismatch for {model.model_id}: "
                f"expected {model.checkpoint_sha256}, got {actual}"
            )
        return model

    def resolve_multijoint_checkpoint(self) -> Path | None:
        """Return the selected multi-joint checkpoint path when available."""
        model = self.resolve_multijoint_model()
        return model.checkpoint_path if model is not None else None

    def resolve_model(self, model_id: str | None, *, profile: str) -> ModelEntry:
        require_profile(profile)
        if model_id is None:
            return self.default_model_for_profile(profile)
        available = self.all_models
        if model_id not in available:
            raise AssetValidationError(
                f"Unknown model_id {model_id!r}; available: {sorted(available.keys())}"
            )
        return available[model_id]

    def three_hash_record(self, profile: str, *, materialized_xml_sha256: str | None = None) -> dict[str, Any]:
        """Explicitly separate upstream raw hashes, profile rule hash, and materialized XML hash."""
        spec = profile_transform_spec(profile)
        return {
            "raw_xml_sha256": self.raw_xml_sha256,
            "mesh_tree_sha256": self.mesh_tree_sha256,
            "teacher_sha256": self.teacher_sha256,
            "profile_name": profile,
            "profile_rule_sha256": spec["profile_rule_sha256"],
            "materialized_xml_sha256": materialized_xml_sha256,
            "asset_manifest_sha256": self.manifest_sha256,
        }


def _find_repo_checkout(start: Path | None = None) -> Path | None:
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if parent in seen:
                continue
            seen.add(parent)
            if (parent / V1_XML_RELPATH).is_file() and (parent / V1_TEACHER_RELPATH).is_file():
                return parent
    return None


def default_runtime_asset_root() -> Path:
    """Directory managed by the runtime bundle installer."""
    configured = Path(os.environ.get("XDG_DATA_HOME", "")).expanduser()
    data_home = configured if configured.is_absolute() else Path.home() / ".local/share"
    return data_home / "sensory-translator-v2/bundle"


def resolve_asset_root(asset_root: str | Path | None = None) -> Path:
    """Resolve an explicit root, environment override, checkout, or installed bundle."""
    if asset_root is not None:
        return Path(asset_root).expanduser().resolve()
    env_val = os.environ.get(ENV_ASSET_ROOT)
    if env_val:
        return Path(env_val).expanduser().resolve()
    checkout = _find_repo_checkout()
    if checkout is not None:
        return checkout
    installed = default_runtime_asset_root()
    if installed.is_dir():
        return installed.resolve()
    raise AssetValidationError(
        f"No runtime bundle found. Run `sensory-engine-v2 install-assets <release.zip>` "
        f"or pass --asset-root (or set {ENV_ASSET_ROOT})."
    )


def resolve_runtime_root(runtime_root: str | Path | None = None, *, asset_root: Path | None = None) -> Path:
    """Resolve runtime output/cache root outside source trees."""
    if runtime_root is not None:
        return Path(runtime_root).expanduser().resolve()
    env_val = os.environ.get(ENV_RUNTIME_ROOT)
    if env_val:
        return Path(env_val).expanduser().resolve()
    if asset_root is not None and (asset_root / "results").is_dir():
        return (asset_root / "results/v2_runtime").resolve()
    return (Path.home() / ".cache/sensory_translator_v2").resolve()


def _preserve_symlink_abspath(path: str | Path) -> Path:
    """Resolve user/relative segments without dereferencing Python venv symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def resolve_physics_python(
    physics_python: str | Path | None = None,
    *,
    asset_root: Path | None = None,
) -> Path:
    """Resolve Python executable holding MuJoCo + dm_control without breaking venv symlinks."""
    if physics_python is not None:
        return _preserve_symlink_abspath(physics_python)
    env_val = os.environ.get(ENV_PHYSICS_PYTHON)
    if env_val:
        return _preserve_symlink_abspath(env_val)
    search_bases: list[Path] = []
    if asset_root is not None:
        search_bases.append(asset_root)
    checkout = _find_repo_checkout()
    if checkout is not None:
        search_bases.append(checkout)
    for base in search_bases:
        venv_py = base / ".venv-flymimic/bin/python"
        if venv_py.is_file():
            return _preserve_symlink_abspath(venv_py)
    return _preserve_symlink_abspath(sys.executable)


def resolve_scripts_root(
    scripts_root: str | Path | None = None,
    *,
    asset_root: Path | None = None,
) -> Path | None:
    """Resolve v1 frozen scripts directory when available (for v1_subprocess backend only)."""
    if scripts_root is not None:
        cand = Path(scripts_root).expanduser().resolve()
        return cand if cand.is_dir() else None
    env_val = os.environ.get(ENV_SCRIPTS_ROOT)
    if env_val:
        cand = Path(env_val).expanduser().resolve()
        return cand if cand.is_dir() else None
    for base in ([asset_root] if asset_root is not None else []) + [Path.cwd().resolve()]:
        cand = base / "scripts"
        if (cand / "differentiate_neuromuscle_plant.py").is_file():
            return cand.resolve()
    checkout = _find_repo_checkout()
    if checkout is not None and (checkout / "scripts/differentiate_neuromuscle_plant.py").is_file():
        return (checkout / "scripts").resolve()
    return None


def _compute_mesh_tree_sha256(mesh_hashes: dict[str, str]) -> str:
    return canonical_json_sha256(sorted(mesh_hashes.items()))


def _load_v2_bundle(
    root: Path,
    *,
    runtime_root: Path,
    physics_python: Path,
    scripts_root: Path | None,
    verify_hashes: bool,
) -> AssetCatalog:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise AssetValidationError(f"v2 asset bundle missing manifest.json: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssetValidationError(f"Invalid asset bundle JSON {manifest_path}: {exc}") from None
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise AssetValidationError(
            f"Unsupported bundle schema {manifest.get('schema_version')!r}; expected {BUNDLE_SCHEMA!r}"
        )

    def _obj_path(sha: str, label: str) -> Path:
        if not isinstance(sha, str) or len(sha) != 64:
            raise AssetValidationError(f"Malformed sha256 for {label}: {sha!r}")
        obj = root / "objects" / sha[:2] / sha
        if not obj.is_file():
            raise AssetValidationError(f"Missing content-addressed asset {label} at {obj}")
        if verify_hashes:
            actual = sha256_file(obj)
            if actual != sha:
                raise AssetValidationError(
                    f"Hash mismatch for {label} ({obj}): expected {sha}, got {actual}"
                )
        return obj

    fly_xml = manifest.get("raw_xml", {})
    raw_xml_sha = str(fly_xml.get("sha256", ""))
    if raw_xml_sha != EXPECTED_XML_SHA256:
        raise AssetValidationError(
            f"Bundle XML hash {raw_xml_sha} != expected frozen FlyMimic XML {EXPECTED_XML_SHA256}"
        )
    xml_path = _obj_path(raw_xml_sha, "raw_xml")

    stl_map = manifest.get("stl_meshes", {})
    if len(stl_map) != 71:
        raise AssetValidationError(f"Expected 71 STL meshes in bundle, found {len(stl_map)}")
    mesh_paths: dict[str, Path] = {}
    mesh_hashes: dict[str, str] = {}
    for rel_key, digest in sorted(stl_map.items()):
        mesh_paths[rel_key] = _obj_path(str(digest), rel_key)
        mesh_hashes[rel_key] = str(digest)
    mesh_tree_sha = _compute_mesh_tree_sha256(mesh_hashes)
    if manifest.get("mesh_tree_sha256") != mesh_tree_sha:
        raise AssetValidationError("Bundle mesh_tree_sha256 mismatch")

    teacher_info = manifest.get("teacher", {})
    teacher_sha = str(teacher_info.get("sha256", ""))
    if teacher_sha != EXPECTED_TEACHER_SHA256:
        raise AssetValidationError(
            f"Bundle teacher hash {teacher_sha} != expected {EXPECTED_TEACHER_SHA256}"
        )
    teacher_path = _obj_path(teacher_sha, "teacher")

    v2_checkpoint_hashes = {spec["model_id"]: spec["expected_checkpoint_sha256"] for spec in KNOWN_V2_MODELS}
    models: dict[str, ModelEntry] = {}
    for model_id, entry in sorted(manifest.get("models", {}).items()):
        ckpt_sha = str(entry["checkpoint_sha256"])
        expected_ckpt = v2_checkpoint_hashes.get(model_id)
        if verify_hashes and expected_ckpt and ckpt_sha != expected_ckpt:
            raise AssetValidationError(
                f"v2 checkpoint sha256 mismatch for {model_id}: expected {expected_ckpt}, got {ckpt_sha}"
            )
        ckpt_path = _obj_path(ckpt_sha, f"model:{model_id}")
        prereg_sha = entry.get("preregistered_config_sha256")
        sel_sha = entry.get("selection_sha256")
        models[model_id] = ModelEntry(
            model_id=model_id,
            profile=require_profile(entry["profile"]),
            checkpoint_path=ckpt_path,
            checkpoint_sha256=ckpt_sha,
            preregistered_config_path=_obj_path(prereg_sha, f"{model_id}:prereg") if prereg_sha else None,
            preregistered_config_sha256=prereg_sha,
            selection_path=_obj_path(sel_sha, f"{model_id}:selection") if sel_sha else None,
            selection_sha256=sel_sha,
            default_for_profile=entry.get("default_for_profile"),
            description=str(entry.get("description", "")),
        )

    multijoint_model = None
    multijoint_info = manifest.get("multijoint_checkpoint")
    if multijoint_info is not None:
        if not isinstance(multijoint_info, dict) or not isinstance(multijoint_info.get("model_id"), str):
            raise AssetValidationError("Invalid registered multi-joint checkpoint metadata in bundle")
        model_id = multijoint_info["model_id"]
        digest = KNOWN_MULTIJOINT_CHECKPOINTS.get(model_id)
        if digest is None or multijoint_info.get("sha256") != digest:
            raise AssetValidationError("Invalid registered multi-joint checkpoint metadata in bundle")
        checkpoint = _obj_path(digest, "multi-joint checkpoint")
        multijoint_model = ModelEntry(model_id, "adjacent_static_parent_v1", checkpoint, digest)

    logical = {
        "schema_version": "sensory_translator_v2_asset_catalog_v1",
        "raw_xml": {"sha256": raw_xml_sha},
        "mesh_tree_sha256": mesh_tree_sha,
        "teacher": {"sha256": teacher_sha},
        "models": {
            mid: {
                "profile": m.profile,
                "checkpoint_sha256": m.checkpoint_sha256,
                "preregistered_config_sha256": m.preregistered_config_sha256,
                "selection_sha256": m.selection_sha256,
                "default_for_profile": m.default_for_profile,
            }
            for mid, m in sorted(models.items())
        },
    }
    if multijoint_model is not None:
        logical["multijoint_checkpoint"] = {
            "model_id": multijoint_model.model_id,
            "sha256": multijoint_model.checkpoint_sha256,
        }
    return AssetCatalog(
        source_kind="v2_bundle",
        asset_root=root,
        runtime_root=runtime_root,
        physics_python=physics_python,
        scripts_root=scripts_root,
        xml_path=xml_path,
        raw_xml_sha256=raw_xml_sha,
        mesh_paths=mesh_paths,
        mesh_hashes=mesh_hashes,
        mesh_tree_sha256=mesh_tree_sha,
        teacher_path=teacher_path,
        teacher_sha256=teacher_sha,
        models=models,
        manifest_sha256=canonical_json_sha256(logical),
        provenance={"bundle_manifest": str(manifest_path)},
        _bundled_multijoint_model=multijoint_model,
    )


def _load_v1_checkout(
    root: Path,
    *,
    runtime_root: Path,
    physics_python: Path,
    scripts_root: Path | None,
    verify_hashes: bool,
) -> AssetCatalog:
    xml_path = root / V1_XML_RELPATH
    mesh_dir = root / V1_MESH_DIR_RELPATH
    teacher_path = root / V1_TEACHER_RELPATH
    if not xml_path.is_file():
        raise AssetValidationError(f"FlyMimic XML not found under {root}: {xml_path}")
    if not mesh_dir.is_dir():
        raise AssetValidationError(f"FlyMimic STL mesh dir not found under {root}: {mesh_dir}")
    if not teacher_path.is_file():
        raise AssetValidationError(f"Frozen teacher rollout not found under {root}: {teacher_path}")

    raw_xml_sha = sha256_file(xml_path)
    if raw_xml_sha != EXPECTED_XML_SHA256:
        raise AssetValidationError(
            f"Raw XML sha256 mismatch: expected {EXPECTED_XML_SHA256}, got {raw_xml_sha}"
        )
    teacher_sha = sha256_file(teacher_path)
    if teacher_sha != EXPECTED_TEACHER_SHA256:
        raise AssetValidationError(
            f"Teacher sha256 mismatch: expected {EXPECTED_TEACHER_SHA256}, got {teacher_sha}"
        )

    stl_files = sorted(mesh_dir.glob("*.stl"))
    if len(stl_files) != 71:
        raise AssetValidationError(f"Expected 71 STL meshes in {mesh_dir}, found {len(stl_files)}")
    mesh_paths: dict[str, Path] = {}
    mesh_hashes: dict[str, str] = {}
    for stl in stl_files:
        rel_key = f"meshes/stl/{stl.name}"
        mesh_paths[rel_key] = stl
        mesh_hashes[rel_key] = sha256_file(stl)
    mesh_tree_sha = _compute_mesh_tree_sha256(mesh_hashes)

    models: dict[str, ModelEntry] = {}
    for spec in KNOWN_V1_MODELS:
        model_dir = root / spec["rel_dir"]
        ckpt_path = model_dir / "best_model.npz"
        if not ckpt_path.is_file():
            continue
        ckpt_sha = sha256_file(ckpt_path)
        expected_ckpt = spec["expected_checkpoint_sha256"]
        if verify_hashes and expected_ckpt and ckpt_sha != expected_ckpt:
            raise AssetValidationError(
                f"v1 checkpoint sha256 mismatch for {spec['model_id']}: expected {expected_ckpt}, got {ckpt_sha}"
            )
        prereg = model_dir / "preregistered_config.json"
        sel = model_dir / "selection_before_physics.json"
        models[spec["model_id"]] = ModelEntry(
            model_id=spec["model_id"],
            profile=spec["profile"],
            checkpoint_path=ckpt_path,
            checkpoint_sha256=ckpt_sha,
            preregistered_config_path=prereg if prereg.is_file() else None,
            preregistered_config_sha256=sha256_file(prereg) if prereg.is_file() else None,
            selection_path=sel if sel.is_file() else None,
            selection_sha256=sha256_file(sel) if sel.is_file() else None,
            default_for_profile=spec["default_for_profile"],
            description=spec["description"],
        )
    if not models:
        raise AssetValidationError(f"No valid rate checkpoints discovered under {root}")

    logical = {
        "schema_version": "sensory_translator_v2_asset_catalog_v1",
        "raw_xml": {"sha256": raw_xml_sha},
        "mesh_tree_sha256": mesh_tree_sha,
        "teacher": {"sha256": teacher_sha},
        "models": {
            mid: {
                "profile": m.profile,
                "checkpoint_sha256": m.checkpoint_sha256,
                "preregistered_config_sha256": m.preregistered_config_sha256,
                "selection_sha256": m.selection_sha256,
                "default_for_profile": m.default_for_profile,
            }
            for mid, m in sorted(models.items())
        },
    }
    return AssetCatalog(
        source_kind="v1_checkout",
        asset_root=root,
        runtime_root=runtime_root,
        physics_python=physics_python,
        scripts_root=scripts_root,
        xml_path=xml_path,
        raw_xml_sha256=raw_xml_sha,
        mesh_paths=mesh_paths,
        mesh_hashes=mesh_hashes,
        mesh_tree_sha256=mesh_tree_sha,
        teacher_path=teacher_path,
        teacher_sha256=teacher_sha,
        models=models,
        manifest_sha256=canonical_json_sha256(logical),
        provenance={"v1_root": str(root)},
    )


def load_asset_catalog(
    *,
    asset_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    physics_python: str | Path | None = None,
    scripts_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> AssetCatalog:
    """Load and verify assets from either a content-addressed v2 bundle or a v1 checkout."""
    resolved_asset_root = resolve_asset_root(asset_root)
    resolved_runtime_root = resolve_runtime_root(runtime_root, asset_root=resolved_asset_root)
    resolved_physics_py = resolve_physics_python(physics_python, asset_root=resolved_asset_root)

    if (resolved_asset_root / "manifest.json").is_file() and (resolved_asset_root / "objects").is_dir():
        explicit_scripts = (
            Path(scripts_root).expanduser().resolve()
            if scripts_root is not None
            else (
                Path(os.environ[ENV_SCRIPTS_ROOT]).expanduser().resolve()
                if os.environ.get(ENV_SCRIPTS_ROOT)
                else None
            )
        )
        return _load_v2_bundle(
            resolved_asset_root,
            runtime_root=resolved_runtime_root,
            physics_python=resolved_physics_py,
            scripts_root=explicit_scripts if (explicit_scripts and explicit_scripts.is_dir()) else None,
            verify_hashes=verify_hashes,
        )
    resolved_scripts = resolve_scripts_root(scripts_root, asset_root=resolved_asset_root)
    return _load_v1_checkout(
        resolved_asset_root,
        runtime_root=resolved_runtime_root,
        physics_python=resolved_physics_py,
        scripts_root=resolved_scripts,
        verify_hashes=verify_hashes,
    )


def verify_asset_catalog(catalog: AssetCatalog) -> dict[str, Any]:
    """Re-verify every asset hash on disk in the loaded catalog."""
    raw_xml = sha256_file(catalog.xml_path)
    if raw_xml != catalog.raw_xml_sha256 or raw_xml != EXPECTED_XML_SHA256:
        raise AssetValidationError(f"XML hash mismatch during verify: {raw_xml}")
    teacher_sha = sha256_file(catalog.teacher_path)
    if teacher_sha != catalog.teacher_sha256 or teacher_sha != EXPECTED_TEACHER_SHA256:
        raise AssetValidationError(f"Teacher hash mismatch during verify: {teacher_sha}")
    checked = 2
    for rel_key, mesh_path in catalog.mesh_paths.items():
        actual = sha256_file(mesh_path)
        expected = catalog.mesh_hashes[rel_key]
        if actual != expected:
            raise AssetValidationError(f"STL hash mismatch for {rel_key}: {actual} != {expected}")
        checked += 1
    for mid, entry in catalog.models.items():
        actual = sha256_file(entry.checkpoint_path)
        if actual != entry.checkpoint_sha256:
            raise AssetValidationError(f"Checkpoint hash mismatch for {mid}")
        checked += 1
    report: dict[str, Any] = {
        "verified": True,
        "source_kind": catalog.source_kind,
        "asset_root": str(catalog.asset_root),
        "checked_files": checked,
        "raw_xml_sha256": raw_xml,
        "mesh_tree_sha256": catalog.mesh_tree_sha256,
        "teacher_sha256": teacher_sha,
        "manifest_sha256": catalog.manifest_sha256,
        "models": {mid: entry.to_dict() for mid, entry in sorted(catalog.models.items())},
    }
    multijoint_model = catalog.resolve_multijoint_model() if catalog.source_kind == "v2_bundle" else None
    if multijoint_model is not None:
        report["checked_files"] += 1
        report["multijoint_checkpoint"] = {
            "model_id": multijoint_model.model_id,
            "sha256": multijoint_model.checkpoint_sha256,
            "checkpoint_path": str(multijoint_model.checkpoint_path),
        }
    return report


def export_asset_bundle(
    destination: str | Path,
    *,
    asset_root: str | Path | None = None,
    overwrite: bool = False,
    include_v2_models: bool = False,
) -> dict[str, Any]:
    """Export verified assets, with optional inclusion of registered V2 checkpoints."""
    catalog = load_asset_catalog(asset_root=asset_root, verify_hashes=True)
    models = catalog.all_models if include_v2_models else catalog.models
    multijoint_model = (
        catalog.resolve_multijoint_model()
        if include_v2_models or catalog.source_kind == "v2_bundle" else None
    )
    if include_v2_models and multijoint_model is None and not any(
        spec["model_id"] in models for spec in KNOWN_V2_MODELS
    ):
        raise AssetValidationError(f"No registered V2 checkpoints found under {catalog.asset_root}")
    dest = Path(destination).expanduser().resolve()
    if dest.exists():
        if not overwrite and any(dest.iterdir()):
            raise AssetValidationError(f"Bundle destination is non-empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    objects_dir = dest / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    def _store(src: Path, digest: str) -> None:
        sub = objects_dir / digest[:2]
        sub.mkdir(parents=True, exist_ok=True)
        target = sub / digest
        if target.is_file() and sha256_file(target) == digest:
            return
        fd, tmp_name = tempfile.mkstemp(dir=sub, prefix=f".{digest[:12]}.", suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            shutil.copy2(src, tmp_path)
            tmp_path.replace(target)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    _store(catalog.xml_path, catalog.raw_xml_sha256)
    _store(catalog.teacher_path, catalog.teacher_sha256)
    for rel_key, mesh_path in catalog.mesh_paths.items():
        _store(mesh_path, catalog.mesh_hashes[rel_key])

    model_manifest: dict[str, Any] = {}
    for mid, model in sorted(models.items()):
        _store(model.checkpoint_path, model.checkpoint_sha256)
        if model.preregistered_config_path and model.preregistered_config_sha256:
            _store(model.preregistered_config_path, model.preregistered_config_sha256)
        if model.selection_path and model.selection_sha256:
            _store(model.selection_path, model.selection_sha256)
        model_manifest[mid] = {
            "profile": model.profile,
            "checkpoint_sha256": model.checkpoint_sha256,
            "preregistered_config_sha256": model.preregistered_config_sha256,
            "selection_sha256": model.selection_sha256,
            "default_for_profile": model.default_for_profile,
            "description": model.description,
        }

    manifest_payload = {
        "schema_version": BUNDLE_SCHEMA,
        "raw_xml": {
            "relative_path": "best_combined_arm_cvt3.xml",
            "sha256": catalog.raw_xml_sha256,
        },
        "stl_meshes": dict(sorted(catalog.mesh_hashes.items())),
        "mesh_tree_sha256": catalog.mesh_tree_sha256,
        "teacher": {
            "relative_path": "teacher_rollout.npz",
            "sha256": catalog.teacher_sha256,
        },
        "models": model_manifest,
    }
    if multijoint_model is not None:
        _store(multijoint_model.checkpoint_path, multijoint_model.checkpoint_sha256)
        manifest_payload["multijoint_checkpoint"] = {
            "model_id": multijoint_model.model_id,
            "sha256": multijoint_model.checkpoint_sha256,
        }
    manifest_tmp = dest / ".manifest.json.tmp"
    manifest_tmp.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(dest / "manifest.json")
    reloaded = load_asset_catalog(asset_root=dest, verify_hashes=True)
    return verify_asset_catalog(reloaded)


def materialize_profile_model(
    catalog: AssetCatalog,
    profile: str,
    *,
    cache_base: Path | None = None,
) -> dict[str, Any]:
    """Ensure a filesystem XML + sibling meshes/stl directory exists for MuJoCo loading.

    Using a deterministic directory keyed by `(raw_xml_sha256, mesh_tree_sha256, profile_rule_sha256)`
    keeps materialized relative mesh references intact without hardcoding checkout paths into XML.
    """
    spec = profile_transform_spec(profile)
    base = (cache_base or (catalog.runtime_root / "materialized")).resolve()
    key = sha256_bytes(
        f"{catalog.raw_xml_sha256}:{catalog.mesh_tree_sha256}:{spec['profile_rule_sha256']}".encode("utf-8")
    )[:24]
    model_dir = base / f"{profile}_{key}"
    mesh_out_dir = model_dir / "meshes/stl"
    xml_out_path = model_dir / "model.xml"
    done_marker = model_dir / ".ready.json"

    raw_xml_text = catalog.xml_path.read_text(encoding="utf-8")
    materialized_xml_text = transform_xml_for_profile(raw_xml_text, profile)
    materialized_xml_sha = sha256_bytes(materialized_xml_text.encode("utf-8"))

    if not (done_marker.is_file() and xml_out_path.is_file() and sha256_file(xml_out_path) == materialized_xml_sha):
        mesh_out_dir.mkdir(parents=True, exist_ok=True)
        for rel_key, src_mesh in catalog.mesh_paths.items():
            dst_mesh = model_dir / rel_key
            if not dst_mesh.is_file() or sha256_file(dst_mesh) != catalog.mesh_hashes[rel_key]:
                fd, tmp_name = tempfile.mkstemp(dir=dst_mesh.parent, prefix=".stl.", suffix=".tmp")
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    shutil.copy2(src_mesh, tmp_path)
                    tmp_path.replace(dst_mesh)
                except Exception:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    raise
        xml_tmp = xml_out_path.with_suffix(".xml.tmp")
        xml_tmp.write_text(materialized_xml_text, encoding="utf-8")
        xml_tmp.replace(xml_out_path)
        marker_tmp = done_marker.with_suffix(".json.tmp")
        marker_tmp.write_text(
            json.dumps(
                catalog.three_hash_record(profile, materialized_xml_sha256=materialized_xml_sha),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        marker_tmp.replace(done_marker)

    return {
        "xml_path": xml_out_path,
        "xml_text": materialized_xml_text,
        "model_dir": model_dir,
        "three_hashes": catalog.three_hash_record(
            profile,
            materialized_xml_sha256=materialized_xml_sha,
        ),
    }
