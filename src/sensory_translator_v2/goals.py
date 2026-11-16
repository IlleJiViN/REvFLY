"""Goal construction, trajectory validation, and condition encoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import (
    CONTROL_TIMESTEP_S,
    DURATION_S,
    STATE_STEPS,
    require_real_float,
    validate_matrix,
    validate_time_grid,
    validate_vector,
)

ANGLE_SCALE_RAD = math.pi
VELOCITY_SCALE_RAD_S = 10.0
REFERENCE_DURATION_S = DURATION_S
GOAL_CONDITION_SAMPLES = 16


def validate_joint_goal(
    target_rad: Any,
    time_s: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate target tibia angle trajectory [211] and 2ms time boundaries [211]."""
    target_arr = validate_vector(target_rad, "target_rad", length=STATE_STEPS)
    if time_s is None:
        time_arr = np.arange(STATE_STEPS, dtype=np.float64) * CONTROL_TIMESTEP_S
    else:
        time_arr = validate_time_grid(time_s, "time_s")
    return target_arr, time_arr


def goal_from_scale(
    scale: float,
    teacher_joint_rad: Any,
    *,
    initial_tibia_delta_deg: float = 0.0,
) -> dict[str, Any]:
    """Build a scaled tibia trajectory relative to the nominal starting joint pose."""
    scale_val = require_real_float(scale, "scale", positive=True)
    delta_deg = require_real_float(initial_tibia_delta_deg, "initial_tibia_delta_deg")
    teacher_arr = validate_vector(teacher_joint_rad, "teacher_joint_rad", length=STATE_STEPS)
    nominal_q0 = float(teacher_arr[0])
    target_arr = nominal_q0 + scale_val * (teacher_arr - nominal_q0)
    time_arr = np.arange(STATE_STEPS, dtype=np.float64) * CONTROL_TIMESTEP_S
    return {
        "goal_scale": scale_val,
        "initial_tibia_deg": float(np.degrees(nominal_q0 + math.radians(delta_deg))),
        "target_deg": np.degrees(target_arr).tolist(),
        "target_rad": target_arr.tolist(),
        "time_s": time_arr.tolist(),
    }


def encode_joint_condition(
    target_rad: torch.Tensor,
    initial_q_rad: torch.Tensor,
    initial_qvel_rad_s: torch.Tensor,
    duration_s: float = REFERENCE_DURATION_S,
    *,
    goal_samples: int = GOAL_CONDITION_SAMPLES,
) -> torch.Tensor:
    """Encode target joint waveform and initial pose/velocity into [B, 19] features."""
    if not isinstance(target_rad, torch.Tensor) or target_rad.ndim != 2 or target_rad.shape[0] < 1:
        raise ValueError("target_rad must be a 2D tensor with shape [B>=1, T>=2]")
    batch, steps = target_rad.shape
    if steps < 2:
        raise ValueError("target_rad requires at least 2 time samples")
    if initial_q_rad.shape != (batch,) or initial_qvel_rad_s.shape != (batch,):
        raise ValueError("initial_q_rad and initial_qvel_rad_s must have shape [B]")
    duration_val = require_real_float(duration_s, "duration_s", positive=True)
    if goal_samples < 2:
        raise ValueError("goal_samples must be >= 2")

    target_f64 = target_rad.to(dtype=torch.float64)
    q0_f64 = initial_q_rad.to(device=target_f64.device, dtype=torch.float64)
    v0_f64 = initial_qvel_rad_s.to(device=target_f64.device, dtype=torch.float64)
    if not (
        bool(torch.isfinite(target_f64).all())
        and bool(torch.isfinite(q0_f64).all())
        and bool(torch.isfinite(v0_f64).all())
    ):
        raise ValueError("encode_joint_condition inputs must be finite")

    sampled = F.interpolate(
        target_f64.unsqueeze(1),
        size=goal_samples,
        mode="linear",
        align_corners=True,
    ).squeeze(1)
    relative = (sampled - q0_f64.unsqueeze(1)) / ANGLE_SCALE_RAD
    initial_q_feat = (q0_f64 / ANGLE_SCALE_RAD).unsqueeze(1)
    initial_v_feat = (v0_f64 / VELOCITY_SCALE_RAD_S).unsqueeze(1)
    duration_feat = torch.full(
        (batch, 1),
        duration_val / REFERENCE_DURATION_S,
        device=target_f64.device,
        dtype=torch.float64,
    )
    return torch.cat([relative, initial_q_feat, initial_v_feat, duration_feat], dim=1)


def validate_multijoint_goal(
    target_rad: Any,
    *,
    n_joints: int = 7,
    time_s: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate multi-joint target trajectory [211, J] and 2ms time grid [211]."""
    target_arr = validate_matrix(target_rad, "target_rad", rows=STATE_STEPS, cols=n_joints)
    if time_s is None:
        time_arr = np.arange(STATE_STEPS, dtype=np.float64) * CONTROL_TIMESTEP_S
    else:
        time_arr = validate_time_grid(time_s, "time_s")
    return target_arr, time_arr


def encode_multijoint_condition(
    target_rad: torch.Tensor,
    initial_q_rad: torch.Tensor,
    initial_qvel_rad_s: torch.Tensor,
    duration_s: float = REFERENCE_DURATION_S,
    *,
    goal_samples: int = GOAL_CONDITION_SAMPLES,
) -> torch.Tensor:
    """Encode [B, T, J] joint trajectories + [B, J] initial state into [B, J*(K+2)+1] features."""
    if not isinstance(target_rad, torch.Tensor) or target_rad.ndim != 3 or target_rad.shape[0] < 1:
        raise ValueError("target_rad must be a 3D tensor with shape [B>=1, T>=2, J>=1]")
    batch, steps, n_joints = target_rad.shape
    if steps < 2 or n_joints < 1:
        raise ValueError("target_rad requires T>=2 and J>=1")
    if initial_q_rad.shape != (batch, n_joints) or initial_qvel_rad_s.shape != (batch, n_joints):
        raise ValueError("initial_q_rad and initial_qvel_rad_s must have shape [B, J]")
    duration_val = require_real_float(duration_s, "duration_s", positive=True)
    if goal_samples < 2:
        raise ValueError("goal_samples must be >= 2")

    target_f64 = target_rad.to(dtype=torch.float64)
    q0_f64 = initial_q_rad.to(device=target_f64.device, dtype=torch.float64)
    v0_f64 = initial_qvel_rad_s.to(device=target_f64.device, dtype=torch.float64)
    if not (
        bool(torch.isfinite(target_f64).all())
        and bool(torch.isfinite(q0_f64).all())
        and bool(torch.isfinite(v0_f64).all())
    ):
        raise ValueError("encode_multijoint_condition inputs must be finite")

    # Interpolate each joint channel: [B, J, T] -> [B, J, goal_samples]
    sampled = F.interpolate(
        target_f64.permute(0, 2, 1),
        size=goal_samples,
        mode="linear",
        align_corners=True,
    )
    relative = ((sampled - q0_f64.unsqueeze(-1)) / ANGLE_SCALE_RAD).reshape(batch, n_joints * goal_samples)
    initial_q_feat = q0_f64 / ANGLE_SCALE_RAD
    initial_v_feat = v0_f64 / VELOCITY_SCALE_RAD_S
    duration_feat = torch.full(
        (batch, 1),
        duration_val / REFERENCE_DURATION_S,
        device=target_f64.device,
        dtype=torch.float64,
    )
    return torch.cat([relative, initial_q_feat, initial_v_feat, duration_feat], dim=1)
