"""Core protocols, physical contracts, and strict validators for v2."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

PROFILES = ("original_v1", "multiccd_v1", "adjacent_static_parent_v1")

CONTROL_STEPS = 210
STATE_STEPS = 211
SUBSTEPS = 20
PHYSICS_TIMESTEP_S = 0.0001
CONTROL_TIMESTEP_S = 0.002
DURATION_S = 0.42
TIME_GRID_TOLERANCE_S = 1e-12

NQ = 14
NV = 14
NU = 15
NA = 15
NSTATE = 43

TARGET_ACTUATOR_COLUMNS = (13, 14)
TARGET_ACTUATOR_NAMES = ("LFTibia_flex_93434", "LFTibia_extensor_93932")
TARGET_JOINT = "joint_LFTibia_pitch"
TARGET_QPOS_ADDRESS = 6

LF_JOINT_NAMES: tuple[str, ...] = (
    "joint_LFCoxa_yaw",
    "joint_LFCoxa_pitch",
    "joint_LFCoxa_roll",
    "joint_LFTrochanter_yaw",
    "joint_LFTrochanter_pitch",
    "joint_LFTrochanter_roll",
    "joint_LFTibia_pitch",
)
LF_QPOS_ADDRESSES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
LF_ACTUATOR_COLUMNS: tuple[int, ...] = tuple(range(NU))
LF_ACTUATOR_NAMES: tuple[str, ...] = (
    "LFC_tergopleural_promotor_a",
    "LFC_tergopleural_promotor_b",
    "LFC_pleural_remotor_and_abductor",
    "LFC_pleural_promotor",
    "LFC_sternal_anterior_rotator",
    "LFC_sternal_posterior_rotator",
    "LFC_sternal_adductor",
    "LFF_trochanter_flexor_b",
    "LFF_sterno-tergo-trochanter_extensor_a",
    "LFF_sterno-tergo-trochanter_extensor_b",
    "LFF_accesory_trochanter_flexor",
    "LFF_trochanter_extensor",
    "LFF_trochanter_flexor_a",
    "LFTibia_flex_93434",
    "LFTibia_extensor_93932",
)
LF_JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "coxa": (0, 1, 2),
    "femur_trochanter": (3, 4, 5),
    "tibia": (6,),
}

DEFAULT_DIRECTION_SEED = 331
DEFAULT_JACOBIAN_EPS = 1e-7
DEFAULT_DIRECTION_EPS = 1e-6

# Flexor (gain=68.1213) to extensor (gain=303.877) static actuator force balance ratio from FlyMimic XML
EXT_TO_FLEX_RATIO = 68.1213 / 303.877
DEFAULT_FEEDBACK_KP = 20.0
DEFAULT_FEEDBACK_KD = 0.3

STATE_KEYS = (
    "qpos",
    "qvel",
    "actuator_activation",
    "state_actuator_ctrl",
    "time_s",
)
TRACE_SHAPES: dict[str, tuple[int, ...]] = {
    "qpos": (STATE_STEPS, NQ),
    "qvel": (STATE_STEPS, NV),
    "actuator_activation": (STATE_STEPS, NA),
    "state_actuator_ctrl": (STATE_STEPS, NU),
    "time_s": (STATE_STEPS,),
    "interval_applied_ctrl": (CONTROL_STEPS, NU),
}
TRACE_KEYS = tuple(TRACE_SHAPES.keys())

EXPECTED_XML_SHA256 = "59d7db31eb756c61661065c16cfbbb1e3400da1a9df8b1f02fe79dc87bd48724"
EXPECTED_TEACHER_SHA256 = "285926477784ac9ea1657f2d422642696bc3b4cd5866c7e2235e7fde964a5ebd"


class V2Error(RuntimeError):
    """Base error for sensory_translator_v2."""


class AssetValidationError(V2Error, ValueError):
    """Raised when runtime assets, checkpoints, or hashes fail validation."""


class BackendUnavailableError(V2Error, RuntimeError):
    """Raised when a configured physics backend or executable is unavailable."""


class BackendSessionError(V2Error, RuntimeError):
    """Raised when a backend session violates ordering, lifecycle, or protocol rules."""


class BackendTimeoutError(BackendSessionError, TimeoutError):
    """Raised when a persistent or subprocess backend call times out."""


class RateBoundaryError(V2Error, ValueError):
    """Raised when rate, excitation, goal, or boundary values fail strict checks."""


class RequestValidationError(V2Error, ValueError):
    """Raised on malformed SDK/CLI/HTTP inference requests."""


def require_real_float(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate and coerce a finite scalar real number, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise RequestValidationError(f"{name} must be a real number, not {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RequestValidationError(f"{name} must be finite")
    if positive and numeric <= 0.0:
        raise RequestValidationError(f"{name} must be positive")
    if nonnegative and numeric < 0.0:
        raise RequestValidationError(f"{name} must be nonnegative")
    if minimum is not None and numeric < minimum:
        raise RequestValidationError(f"{name} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise RequestValidationError(f"{name} must be <= {maximum}")
    return numeric


def require_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate an integer scalar, rejecting booleans and floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise RequestValidationError(f"{name} must be an integer, not {type(value).__name__}")
    numeric = int(value)
    if minimum is not None and numeric < minimum:
        raise RequestValidationError(f"{name} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise RequestValidationError(f"{name} must be <= {maximum}")
    return numeric


def require_profile(profile: Any) -> str:
    """Validate a physics profile identifier."""
    if type(profile) is not str or profile not in PROFILES:
        raise RequestValidationError(
            f"profile must be one of {PROFILES!r}, got {profile!r}"
        )
    return profile


def validate_vector(values: Any, name: str, *, length: int = STATE_STEPS) -> np.ndarray:
    """Validate a finite 1D float64 vector of exact length, rejecting booleans."""
    arr = np.asarray(values)
    if arr.dtype == np.bool_ or np.iscomplexobj(arr):
        raise RequestValidationError(f"{name} must contain real floating values")
    try:
        arr_f = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{name} could not convert to float64: {exc}") from None
    if arr_f.shape != (length,) or not bool(np.isfinite(arr_f).all()):
        raise RequestValidationError(f"{name} must be finite shape ({length},), got {arr_f.shape}")
    return arr_f


def validate_time_grid(time_s: Any, name: str = "time_s") -> np.ndarray:
    """Verify exact 2 ms boundaries from 0.0 to 0.420 seconds."""
    arr = validate_vector(time_s, name, length=STATE_STEPS)
    expected = np.arange(STATE_STEPS, dtype=np.float64) * CONTROL_TIMESTEP_S
    if not bool(np.all(np.abs(arr - expected) <= TIME_GRID_TOLERANCE_S)):
        raise RequestValidationError(
            f"{name} must cover 0..0.420s at exact 2ms intervals"
        )
    return arr


def validate_index_tuple(indices: Any, name: str, *, upper_bound: int) -> tuple[int, ...]:
    """Validate non-empty tuple of strictly increasing unique integer indices."""
    if isinstance(indices, (str, bytes)) or not hasattr(indices, "__iter__"):
        raise RequestValidationError(f"{name} must be an iterable of integer indices")
    coerced = tuple(require_int(idx, f"{name} item", minimum=0, maximum=upper_bound - 1) for idx in indices)
    if not coerced:
        raise RequestValidationError(f"{name} must not be empty")
    if len(set(coerced)) != len(coerced):
        raise RequestValidationError(f"{name} must not contain duplicate indices")
    return coerced


def validate_matrix(
    values: Any,
    name: str,
    *,
    rows: int,
    cols: int,
    bounds: tuple[float, float] | None = None,
    error_cls: type[V2Error] = RequestValidationError,
) -> np.ndarray:
    """Validate a finite 2D float64 array with exact shape and optional bounds."""
    arr = np.asarray(values)
    if arr.dtype == np.bool_ or np.iscomplexobj(arr):
        raise error_cls(f"{name} must be real floating array")
    try:
        arr_f = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise error_cls(f"{name} could not convert to float64: {exc}") from None
    if arr_f.shape != (rows, cols) or not bool(np.isfinite(arr_f).all()):
        raise error_cls(
            f"{name} must be finite float64 with shape ({rows}, {cols}), got {arr_f.shape}"
        )
    if bounds is not None:
        lo, hi = bounds
        if bool((arr_f < lo).any()) or bool((arr_f > hi).any()):
            raise error_cls(f"{name} must lie inside [{lo}, {hi}]")
    return arr_f.copy()


def validate_controls(
    controls: Any,
    name: str = "controls",
    *,
    bounds: tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:
    """Validate finite excitation controls shaped [210, 2]."""
    return validate_matrix(
        controls,
        name,
        rows=CONTROL_STEPS,
        cols=2,
        bounds=bounds,
        error_cls=RateBoundaryError,
    )


def validate_traces(traces: Any) -> dict[str, np.ndarray]:
    """Validate all 6 canonical physical state and control trace arrays."""
    if not isinstance(traces, dict):
        raise BackendSessionError("traces must be a mapping")
    missing = set(TRACE_KEYS) - set(traces.keys())
    if missing:
        raise BackendSessionError(f"traces missing keys: {sorted(missing)}")
    validated: dict[str, np.ndarray] = {}
    for key in TRACE_KEYS:
        arr = np.asarray(traces[key], dtype=np.float64)
        expected_shape = TRACE_SHAPES[key]
        if arr.shape != expected_shape or not bool(np.isfinite(arr).all()):
            raise BackendSessionError(
                f"trace {key!r} must be finite float64 {expected_shape}, got {arr.shape}"
            )
        validated[key] = arr.copy()
    return validated


@dataclass(frozen=True)
class BackendIdentity:
    """Identity and certification metadata reported by a backend."""

    name: str
    profile: str
    backend_version: str
    is_certified_frozen_physics: bool
    cache_mode: str = "fresh"
    asset_hashes: dict[str, Any] = field(default_factory=dict)
    runtime_versions: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.is_certified_frozen_physics) is not bool:
            raise TypeError("is_certified_frozen_physics must be a strict bool")
        require_profile(self.profile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile,
            "backend_version": self.backend_version,
            "is_certified_frozen_physics": self.is_certified_frozen_physics,
            "cache_mode": self.cache_mode,
            "asset_hashes": dict(self.asset_hashes),
            "runtime_versions": dict(self.runtime_versions),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SessionSpec:
    """Configuration defining an isolated backend evaluation session."""

    profile: str
    asset_manifest_sha256: str = ""
    teacher_sha256: str = ""
    raw_xml_sha256: str = ""
    profile_rule_sha256: str = ""
    direction_seed: int = DEFAULT_DIRECTION_SEED
    jacobian_eps: float = DEFAULT_JACOBIAN_EPS
    direction_eps: float = DEFAULT_DIRECTION_EPS
    cache_mode: str = "fresh"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_profile(self.profile)
        require_int(self.direction_seed, "direction_seed", minimum=0)
        require_real_float(self.jacobian_eps, "jacobian_eps", positive=True)
        require_real_float(self.direction_eps, "direction_eps", positive=True)


@dataclass(frozen=True)
class InitialState:
    """Stateless per-reset physical initial state offset."""

    initial_tibia_delta_rad: float = 0.0

    def __post_init__(self) -> None:
        coerced = require_real_float(self.initial_tibia_delta_rad, "initial_tibia_delta_rad")
        object.__setattr__(self, "initial_tibia_delta_rad", coerced)


@dataclass(frozen=True)
class EvaluationRequest:
    """Single-rollout request for forward physics and optional VJP."""

    controls: np.ndarray
    need_jacobian: bool = False
    cotangent_qpos: np.ndarray | None = None
    mse_target_rad: np.ndarray | None = None
    check_directions: bool = False
    direction_seed: int = DEFAULT_DIRECTION_SEED
    clip_boundary_gradient: str = "native"
    request_id: str = ""

    def __post_init__(self) -> None:
        if type(self.need_jacobian) is not bool:
            raise RequestValidationError("need_jacobian must be a strict bool")
        if type(self.check_directions) is not bool:
            raise RequestValidationError("check_directions must be a strict bool")
        if self.clip_boundary_gradient not in ("native", "one_sided"):
            raise RequestValidationError(
                "clip_boundary_gradient must be 'native' or 'one_sided'"
            )
        object.__setattr__(self, "controls", validate_controls(self.controls, "controls"))
        if self.cotangent_qpos is not None and self.mse_target_rad is not None:
            raise RequestValidationError("Specify at most one of cotangent_qpos or mse_target_rad")
        if self.cotangent_qpos is not None:
            object.__setattr__(
                self,
                "cotangent_qpos",
                validate_vector(self.cotangent_qpos, "cotangent_qpos", length=STATE_STEPS),
            )
        if self.mse_target_rad is not None:
            object.__setattr__(
                self,
                "mse_target_rad",
                validate_vector(self.mse_target_rad, "mse_target_rad", length=STATE_STEPS),
            )
        if self.need_jacobian and self.cotangent_qpos is None and self.mse_target_rad is None:
            raise RequestValidationError(
                "cotangent_qpos or mse_target_rad is required when need_jacobian=True"
            )
        if self.check_directions and not self.need_jacobian:
            raise RequestValidationError("check_directions requires need_jacobian=True")
        require_int(self.direction_seed, "direction_seed", minimum=0)


@dataclass
class Evaluation:
    """Verified rollout output from a backend evaluation."""

    request_id: str
    session_id: str
    profile: str
    is_certified_frozen_physics: bool
    traces: dict[str, np.ndarray]
    joint_rad: np.ndarray
    control_vjp: np.ndarray | None = None
    gradient_valid: bool | None = None
    direction_checks: dict[str, Any] | None = None
    independent_teacher_gate: dict[str, Any] | None = None
    timing_s: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.is_certified_frozen_physics) is not bool:
            raise TypeError("is_certified_frozen_physics must be a strict bool")
        require_profile(self.profile)
        self.traces = validate_traces(self.traces)
        self.joint_rad = validate_vector(self.joint_rad, "joint_rad", length=STATE_STEPS)
        if self.control_vjp is not None:
            cols = self.metadata.get("control_columns") if isinstance(self.metadata, dict) else None
            expected_cols = len(cols) if isinstance(cols, (list, tuple)) and len(cols) > 0 else 2
            self.control_vjp = validate_matrix(
                self.control_vjp,
                "control_vjp",
                rows=CONTROL_STEPS,
                cols=expected_cols,
                bounds=None,
                error_cls=RateBoundaryError,
            )


@dataclass
class Session:
    """Mutable handle tracking open backend session state."""

    session_id: str
    backend_name: str
    spec: SessionSpec
    identity: BackendIdentity
    created_utc: str
    closed: bool = False
    active_initial: InitialState | None = None
    active_seed: int | None = None
    evaluation_count: int = 0


@dataclass(frozen=True)
class RenderResult:
    """Metadata for rendered top-view simulation artifacts."""

    mp4_path: Path | None
    webm_path: Path | None
    preview_png_path: Path | None
    frame_count: int
    visible: bool
    report: dict[str, Any]


@runtime_checkable
class Backend(Protocol):
    """Canonical stateless-session physics backend protocol for Engine v2."""

    def describe(self) -> BackendIdentity: ...

    def open_session(self, spec: SessionSpec) -> Session: ...

    def reset(self, session: Session, initial: InitialState, seed: int) -> None: ...

    def evaluate(self, session: Session, request: EvaluationRequest) -> Evaluation: ...

    def close_session(self, session: Session) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ClosedLoopRequest:
    """Validated causal proprioceptive closed-loop rollout request."""

    feedforward_controls: np.ndarray
    target_rad: np.ndarray
    kp: float = DEFAULT_FEEDBACK_KP
    kd: float = DEFAULT_FEEDBACK_KD
    ext_to_flex_ratio: float = EXT_TO_FLEX_RATIO
    request_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feedforward_controls",
            validate_controls(self.feedforward_controls, "feedforward_controls"),
        )
        object.__setattr__(
            self,
            "target_rad",
            validate_vector(self.target_rad, "target_rad", length=STATE_STEPS),
        )
        object.__setattr__(self, "kp", require_real_float(self.kp, "kp", nonnegative=True))
        object.__setattr__(self, "kd", require_real_float(self.kd, "kd", nonnegative=True))
        object.__setattr__(
            self,
            "ext_to_flex_ratio",
            require_real_float(self.ext_to_flex_ratio, "ext_to_flex_ratio", nonnegative=True),
        )


@runtime_checkable
class RenderCapableBackend(Backend, Protocol):
    """Optional capability extension for top-view FlyMimic rendering."""

    def render_rollout(
        self,
        session: Session,
        controls: np.ndarray,
        output_dir: Path,
        *,
        stem: str = "simulation_top_view",
        initial: InitialState | None = None,
    ) -> RenderResult: ...


@runtime_checkable
class ClosedLoopCapableBackend(Backend, Protocol):
    """Optional capability extension for causal online proprioceptive feedback."""

    def evaluate_closed_loop(
        self,
        session: Session,
        request: ClosedLoopRequest,
    ) -> Evaluation: ...


def _coerce_joint_vector(
    value: Any,
    name: str,
    length: int,
    *,
    default_scalar: float | None = None,
    nonnegative: bool = False,
    positive: bool = False,
) -> np.ndarray:
    if value is None:
        if default_scalar is None:
            raise RequestValidationError(f"{name} is required")
        return np.full((length,), float(default_scalar), dtype=np.float64)
    if isinstance(value, (bool, np.bool_)):
        raise RequestValidationError(f"{name} must not be boolean")
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = require_real_float(arr.item(), name, nonnegative=nonnegative, positive=positive)
        return np.full((length,), scalar, dtype=np.float64)
    vec = validate_vector(arr, name, length=length)
    if nonnegative and bool((vec < 0.0).any()):
        raise RequestValidationError(f"{name} entries must be nonnegative")
    if positive and bool((vec <= 0.0).any()):
        raise RequestValidationError(f"{name} entries must be positive")
    return vec


@dataclass(frozen=True)
class MultiJointEvaluationRequest:
    """Zero-teacher-assist / multi-joint rollout request for arbitrary LF actuator/DOF subsets."""

    controls: np.ndarray
    control_columns: tuple[int, ...] = LF_ACTUATOR_COLUMNS
    target_qpos_indices: tuple[int, ...] = LF_QPOS_ADDRESSES
    initial_qpos_delta_rad: np.ndarray | None = None
    need_jacobian: bool = False
    cotangent_qpos: np.ndarray | None = None
    mse_target_rad: np.ndarray | None = None
    joint_weights: np.ndarray | None = None
    check_directions: bool = False
    direction_seed: int = DEFAULT_DIRECTION_SEED
    clip_boundary_gradient: str = "one_sided"
    request_id: str = ""

    def __post_init__(self) -> None:
        if type(self.need_jacobian) is not bool:
            raise RequestValidationError("need_jacobian must be a strict bool")
        if type(self.check_directions) is not bool:
            raise RequestValidationError("check_directions must be a strict bool")
        if self.clip_boundary_gradient not in ("native", "one_sided"):
            raise RequestValidationError(
                "clip_boundary_gradient must be 'native' or 'one_sided'"
            )
        cols = validate_index_tuple(self.control_columns, "control_columns", upper_bound=NU)
        qidxs = validate_index_tuple(self.target_qpos_indices, "target_qpos_indices", upper_bound=NQ)
        object.__setattr__(self, "control_columns", cols)
        object.__setattr__(self, "target_qpos_indices", qidxs)
        object.__setattr__(
            self,
            "controls",
            validate_matrix(
                self.controls,
                "controls",
                rows=CONTROL_STEPS,
                cols=len(cols),
                bounds=(0.0, 1.0),
                error_cls=RateBoundaryError,
            ),
        )
        if self.initial_qpos_delta_rad is not None:
            object.__setattr__(
                self,
                "initial_qpos_delta_rad",
                validate_vector(self.initial_qpos_delta_rad, "initial_qpos_delta_rad", length=len(qidxs)),
            )
        if self.cotangent_qpos is not None and self.mse_target_rad is not None:
            raise RequestValidationError("Specify at most one of cotangent_qpos or mse_target_rad")
        if self.cotangent_qpos is not None:
            object.__setattr__(
                self,
                "cotangent_qpos",
                validate_matrix(self.cotangent_qpos, "cotangent_qpos", rows=STATE_STEPS, cols=len(qidxs)),
            )
        if self.mse_target_rad is not None:
            object.__setattr__(
                self,
                "mse_target_rad",
                validate_matrix(self.mse_target_rad, "mse_target_rad", rows=STATE_STEPS, cols=len(qidxs)),
            )
        if self.joint_weights is not None:
            object.__setattr__(
                self,
                "joint_weights",
                _coerce_joint_vector(self.joint_weights, "joint_weights", len(qidxs), positive=True),
            )
        if self.need_jacobian and self.cotangent_qpos is None and self.mse_target_rad is None:
            raise RequestValidationError(
                "cotangent_qpos or mse_target_rad is required when need_jacobian=True"
            )
        if self.check_directions and not self.need_jacobian:
            raise RequestValidationError("check_directions requires need_jacobian=True")
        require_int(self.direction_seed, "direction_seed", minimum=0)


@dataclass(frozen=True)
class MultiJointClosedLoopRequest:
    """Causal multi-joint proprioceptive reflex request across all controlled muscles."""

    feedforward_controls: np.ndarray
    target_rad: np.ndarray
    control_columns: tuple[int, ...] = LF_ACTUATOR_COLUMNS
    target_qpos_indices: tuple[int, ...] = LF_QPOS_ADDRESSES
    initial_qpos_delta_rad: np.ndarray | None = None
    kp: np.ndarray | float = 8.0
    kd: np.ndarray | float = 0.04
    jacobian_refresh_steps: int = 15
    damping: float = 0.10
    feedback_clip: float = 0.8
    request_id: str = ""

    def __post_init__(self) -> None:
        cols = validate_index_tuple(self.control_columns, "control_columns", upper_bound=NU)
        qidxs = validate_index_tuple(self.target_qpos_indices, "target_qpos_indices", upper_bound=NQ)
        object.__setattr__(self, "control_columns", cols)
        object.__setattr__(self, "target_qpos_indices", qidxs)
        object.__setattr__(
            self,
            "feedforward_controls",
            validate_matrix(
                self.feedforward_controls,
                "feedforward_controls",
                rows=CONTROL_STEPS,
                cols=len(cols),
                bounds=(0.0, 1.0),
                error_cls=RateBoundaryError,
            ),
        )
        object.__setattr__(
            self,
            "target_rad",
            validate_matrix(self.target_rad, "target_rad", rows=STATE_STEPS, cols=len(qidxs)),
        )
        if self.initial_qpos_delta_rad is not None:
            object.__setattr__(
                self,
                "initial_qpos_delta_rad",
                validate_vector(self.initial_qpos_delta_rad, "initial_qpos_delta_rad", length=len(qidxs)),
            )
        object.__setattr__(
            self,
            "kp",
            _coerce_joint_vector(self.kp, "kp", len(qidxs), nonnegative=True),
        )
        object.__setattr__(
            self,
            "kd",
            _coerce_joint_vector(self.kd, "kd", len(qidxs), nonnegative=True),
        )
        object.__setattr__(
            self,
            "jacobian_refresh_steps",
            require_int(self.jacobian_refresh_steps, "jacobian_refresh_steps", minimum=1, maximum=CONTROL_STEPS),
        )
        object.__setattr__(self, "damping", require_real_float(self.damping, "damping", positive=True))
        object.__setattr__(self, "feedback_clip", require_real_float(self.feedback_clip, "feedback_clip", positive=True))


@runtime_checkable
class MultiJointCapableBackend(Backend, Protocol):
    """Optional capability extension for multi-joint / all-muscle physics evaluation and PD control."""

    def evaluate_multijoint(
        self,
        session: Session,
        request: MultiJointEvaluationRequest,
    ) -> Evaluation: ...

    def evaluate_multijoint_closed_loop(
        self,
        session: Session,
        request: MultiJointClosedLoopRequest,
    ) -> Evaluation: ...
