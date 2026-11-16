"""Standalone goal-conditioned continuous motor rate FFN and NPZ serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from ..contracts import AssetValidationError, RateBoundaryError, require_int, require_real_float
from ..goals import GOAL_CONDITION_SAMPLES

DEFAULT_MAX_RATE_HZ = 200.0


@runtime_checkable
class RatePolicyProtocol(Protocol):
    """Frozen tensor contract any retrainable internal rate policy must satisfy."""

    condition_dim: int
    n_pools: int
    knots_per_pool: int
    max_rate_hz: float

    def forward(self, conditions: torch.Tensor) -> torch.Tensor: ...


class ConditionalRateFFN(nn.Module):
    """SiLU MLP mapping [B, condition_dim] goal/state conditions to [B, n_pools, K] rate knots."""

    architecture_name = "GoalStateConditionedContinuousRateFFN"

    def __init__(
        self,
        condition_dim: int = 19,
        hidden_dim: int = 64,
        knots_per_pool: int = GOAL_CONDITION_SAMPLES,
        max_rate_hz: float = DEFAULT_MAX_RATE_HZ,
        *,
        n_pools: int = 2,
        num_layers: int = 1,
        base_knots_hz: torch.Tensor | np.ndarray | None = None,
        condition_mean: torch.Tensor | np.ndarray | None = None,
        condition_std: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.condition_dim = require_int(condition_dim, "condition_dim", minimum=1)
        self.hidden_dim = require_int(hidden_dim, "hidden_dim", minimum=1)
        self.knots_per_pool = require_int(knots_per_pool, "knots_per_pool", minimum=2)
        self.n_pools = require_int(n_pools, "n_pools", minimum=1)
        self.num_layers = require_int(num_layers, "num_layers", minimum=1)
        self.max_rate_hz = require_real_float(max_rate_hz, "max_rate_hz", positive=True)

        self.hidden = nn.Linear(self.condition_dim, self.hidden_dim, dtype=torch.float64)
        self.extra_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.float64)
                for _ in range(self.num_layers - 1)
            ]
        )
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(self.hidden_dim, self.n_pools * self.knots_per_pool, dtype=torch.float64)

        if base_knots_hz is None:
            base_tensor = torch.zeros((self.n_pools, self.knots_per_pool), dtype=torch.float64)
        else:
            base_tensor = torch.as_tensor(base_knots_hz, dtype=torch.float64).clone()
        if base_tensor.shape != (self.n_pools, self.knots_per_pool):
            raise RateBoundaryError(f"base_knots_hz must have shape ({self.n_pools}, {self.knots_per_pool})")

        if condition_mean is None:
            mean_tensor = torch.zeros((self.condition_dim,), dtype=torch.float64)
        else:
            mean_tensor = torch.as_tensor(condition_mean, dtype=torch.float64).clone()
        if mean_tensor.shape != (self.condition_dim,):
            raise RateBoundaryError(f"condition_mean must have shape ({self.condition_dim},)")

        if condition_std is None:
            std_tensor = torch.ones((self.condition_dim,), dtype=torch.float64)
        else:
            std_tensor = torch.as_tensor(condition_std, dtype=torch.float64).clone()
        if std_tensor.shape != (self.condition_dim,):
            raise RateBoundaryError(f"condition_std must have shape ({self.condition_dim},)")

        self.register_buffer("base_knots_hz", base_tensor)
        self.register_buffer("condition_mean", mean_tensor)
        self.register_buffer("condition_std", std_tensor)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(conditions, torch.Tensor)
            or conditions.ndim != 2
            or conditions.shape[0] < 1
            or conditions.shape[1] != self.condition_dim
        ):
            raise RateBoundaryError(
                f"conditions must have shape [B>=1, {self.condition_dim}]"
            )
        cond_f64 = conditions.to(dtype=torch.float64)
        if not bool(torch.isfinite(cond_f64).all()):
            raise RateBoundaryError("conditions must be finite")
        normalized = (cond_f64 - self.condition_mean) / self.condition_std
        hidden = self.activation(self.hidden(normalized))
        for layer in self.extra_layers:
            hidden = hidden + self.activation(layer(hidden))
        residual = self.output_layer(hidden).reshape(-1, self.n_pools, self.knots_per_pool)
        raw_knots = self.base_knots_hz.unsqueeze(0) + self.max_rate_hz * residual
        return raw_knots.clamp(0.0, self.max_rate_hz)


class ResidualNormRateMLP(nn.Module):
    """Retrainable LayerNorm + Pre-Act Residual SiLU MLP satisfying RatePolicyProtocol."""

    architecture_name = "ResidualNormRateMLP"

    def __init__(
        self,
        condition_dim: int = 19,
        hidden_dim: int = 64,
        knots_per_pool: int = GOAL_CONDITION_SAMPLES,
        max_rate_hz: float = DEFAULT_MAX_RATE_HZ,
        *,
        n_pools: int = 2,
        num_layers: int = 2,
        base_knots_hz: torch.Tensor | np.ndarray | None = None,
        condition_mean: torch.Tensor | np.ndarray | None = None,
        condition_std: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.condition_dim = require_int(condition_dim, "condition_dim", minimum=1)
        self.hidden_dim = require_int(hidden_dim, "hidden_dim", minimum=1)
        self.knots_per_pool = require_int(knots_per_pool, "knots_per_pool", minimum=2)
        self.n_pools = require_int(n_pools, "n_pools", minimum=1)
        self.num_layers = require_int(num_layers, "num_layers", minimum=1)
        self.max_rate_hz = require_real_float(max_rate_hz, "max_rate_hz", positive=True)

        self.in_proj = nn.Linear(self.condition_dim, self.hidden_dim, dtype=torch.float64)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim, dtype=torch.float64) for _ in range(self.num_layers)]
        )
        self.blocks = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.float64) for _ in range(self.num_layers)]
        )
        self.activation = nn.SiLU()
        self.out_norm = nn.LayerNorm(self.hidden_dim, dtype=torch.float64)
        self.output_layer = nn.Linear(self.hidden_dim, self.n_pools * self.knots_per_pool, dtype=torch.float64)
        with torch.no_grad():
            self.output_layer.weight.mul_(0.05)
            self.output_layer.bias.zero_()

        base_t = (
            torch.zeros((self.n_pools, self.knots_per_pool), dtype=torch.float64)
            if base_knots_hz is None
            else torch.as_tensor(base_knots_hz, dtype=torch.float64).clone()
        )
        mean_t = (
            torch.zeros((self.condition_dim,), dtype=torch.float64)
            if condition_mean is None
            else torch.as_tensor(condition_mean, dtype=torch.float64).clone()
        )
        std_t = (
            torch.ones((self.condition_dim,), dtype=torch.float64)
            if condition_std is None
            else torch.as_tensor(condition_std, dtype=torch.float64).clone()
        )
        if base_t.shape != (self.n_pools, self.knots_per_pool):
            raise RateBoundaryError(f"base_knots_hz must have shape ({self.n_pools}, {self.knots_per_pool})")
        if mean_t.shape != (self.condition_dim,) or std_t.shape != (self.condition_dim,):
            raise RateBoundaryError(f"condition_mean/std must have shape ({self.condition_dim},)")
        self.register_buffer("base_knots_hz", base_t)
        self.register_buffer("condition_mean", mean_t)
        self.register_buffer("condition_std", std_t)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(conditions, torch.Tensor)
            or conditions.ndim != 2
            or conditions.shape[0] < 1
            or conditions.shape[1] != self.condition_dim
        ):
            raise RateBoundaryError(f"conditions must have shape [B>=1, {self.condition_dim}]")
        cond_f64 = conditions.to(dtype=torch.float64)
        if not bool(torch.isfinite(cond_f64).all()):
            raise RateBoundaryError("conditions must be finite")
        normalized = (cond_f64 - self.condition_mean) / self.condition_std
        h = self.activation(self.in_proj(normalized))
        for norm, linear in zip(self.norms, self.blocks):
            h = h + self.activation(linear(norm(h)))
        residual = self.output_layer(self.out_norm(h)).reshape(-1, self.n_pools, self.knots_per_pool)
        raw_knots = self.base_knots_hz.unsqueeze(0) + self.max_rate_hz * residual
        return raw_knots.clamp(0.0, self.max_rate_hz)


RatePolicyBuilder = Callable[[dict[str, Any]], nn.Module]


def _build_conditional_ffn(config: dict[str, Any]) -> ConditionalRateFFN:
    return ConditionalRateFFN(
        condition_dim=int(config.get("condition_dim", 19)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        knots_per_pool=int(config.get("knots_per_pool", GOAL_CONDITION_SAMPLES)),
        max_rate_hz=float(config.get("max_rate_hz", DEFAULT_MAX_RATE_HZ)),
        n_pools=int(config.get("n_pools", 2)),
        num_layers=int(config.get("num_layers", 1)),
    )


def _build_residual_norm_mlp(config: dict[str, Any]) -> ResidualNormRateMLP:
    return ResidualNormRateMLP(
        condition_dim=int(config.get("condition_dim", 19)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        knots_per_pool=int(config.get("knots_per_pool", GOAL_CONDITION_SAMPLES)),
        max_rate_hz=float(config.get("max_rate_hz", DEFAULT_MAX_RATE_HZ)),
        n_pools=int(config.get("n_pools", 2)),
        num_layers=int(config.get("num_layers", 2)),
    )


RATE_ARCHITECTURES: dict[str, RatePolicyBuilder] = {
    "GoalStateConditionedContinuousRateFFN": _build_conditional_ffn,
    "ConditionalRateFFN": _build_conditional_ffn,
    "ResidualNormRateMLP": _build_residual_norm_mlp,
}


def register_rate_architecture(name: str, builder: RatePolicyBuilder) -> None:
    """Register a custom RatePolicyProtocol internal architecture for plug-and-play retraining."""
    if not name or not isinstance(name, str):
        raise ValueError("Architecture name must be a non-empty string")
    RATE_ARCHITECTURES[name] = builder


def build_rate_policy_from_config(config: dict[str, Any]) -> nn.Module:
    """Instantiate a registered internal rate policy strictly from checkpoint/training config."""
    arch = str(
        config.get("architecture")
        or config.get("model_type")
        or "GoalStateConditionedContinuousRateFFN"
    )
    builder = RATE_ARCHITECTURES.get(arch)
    if builder is None:
        raise AssetValidationError(
            f"Unknown rate policy architecture {arch!r}; registered={sorted(RATE_ARCHITECTURES)}"
        )
    return builder(config)


def _scalar_str(arr: np.ndarray, name: str) -> str:
    if arr.ndim != 0 or arr.dtype.kind not in ("U", "S"):
        raise AssetValidationError(f"Checkpoint field {name!r} must be a scalar string")
    val = arr.item()
    return val.decode("utf-8") if isinstance(val, bytes) else str(val)


def load_rate_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Load a strict non-pickle v1/v2 RatePolicyProtocol NPZ checkpoint."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise AssetValidationError(f"Rate checkpoint does not exist: {ckpt_path}")
    actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise AssetValidationError(
            f"Checkpoint sha256 mismatch for {ckpt_path}: expected {expected_sha256}, got {actual_sha}"
        )
    try:
        with np.load(ckpt_path, allow_pickle=False) as archive:
            files = set(archive.files)
            required_meta = {"format_version", "configuration_json", "metadata_json"}
            if not required_meta.issubset(files):
                missing = sorted(required_meta - files)
                raise AssetValidationError(f"Checkpoint missing metadata fields: {missing}")
            ver_arr = archive["format_version"]
            if ver_arr.ndim != 0 or int(ver_arr.item()) != 1:
                raise AssetValidationError(f"Unsupported checkpoint format_version: {ver_arr}")
            config = json.loads(_scalar_str(archive["configuration_json"], "configuration_json"))
            metadata = json.loads(_scalar_str(archive["metadata_json"], "metadata_json"))
            state_arrays: dict[str, torch.Tensor] = {}
            for key in sorted(files):
                if key.startswith("state__"):
                    tensor_name = key[len("state__"):]
                    arr = np.asarray(archive[key], dtype=np.float64)
                    if not bool(np.isfinite(arr).all()):
                        raise AssetValidationError(f"Non-finite values in checkpoint key {key!r}")
                    state_arrays[tensor_name] = torch.from_numpy(arr.copy())
    except AssetValidationError:
        raise
    except Exception as exc:
        raise AssetValidationError(f"Failed to read non-pickle checkpoint {ckpt_path}: {exc}") from None

    model = build_rate_policy_from_config(config)
    expected_keys = set(model.state_dict().keys())
    if set(state_arrays.keys()) != expected_keys:
        raise AssetValidationError(
            f"Checkpoint state keys mismatch: expected {sorted(expected_keys)}, got {sorted(state_arrays.keys())}"
        )
    model.load_state_dict(state_arrays, strict=True)
    model.eval()
    return model, config, metadata


def save_rate_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a RatePolicyProtocol checkpoint in strict non-pickle NPZ v1 format."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    arch_name = str(getattr(model, "architecture_name", type(model).__name__))
    cfg = dict(
        config
        or {
            "activation": "SiLU",
            "architecture": arch_name,
            "buffer_dtype": "float64",
            "condition_dim": int(getattr(model, "condition_dim")),
            "hidden_dim": int(getattr(model, "hidden_dim", 0)),
            "knots_per_pool": int(getattr(model, "knots_per_pool")),
            "n_pools": int(getattr(model, "n_pools")),
            "num_layers": int(getattr(model, "num_layers", 1)),
            "max_rate_hz": float(getattr(model, "max_rate_hz")),
            "output_parameterization": "native_clamp(base_hz+max_rate_hz*residual,0,max_rate_hz)",
            "parameter_dtype": "float64",
            "rate_unit": "Hz commanded continuous intensity, not realized spikes or current",
        }
    )
    if "architecture" not in cfg:
        cfg["architecture"] = arch_name
    meta = dict(metadata or {})
    arrays: dict[str, Any] = {
        "format_version": np.array(1, dtype=np.int64),
        "configuration_json": np.array(json.dumps(cfg, sort_keys=True, separators=(",", ":"))),
        "metadata_json": np.array(json.dumps(meta, sort_keys=True, separators=(",", ":"))),
    }
    for name, tensor in model.state_dict().items():
        arrays[f"state__{name}"] = tensor.detach().cpu().to(dtype=torch.float64).numpy()
    tmp_path = ckpt_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(ckpt_path)
    return ckpt_path
