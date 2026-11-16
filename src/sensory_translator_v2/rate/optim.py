"""Shared optimization helpers for motor-rate refinement and training."""

from __future__ import annotations

import numpy as np


def lbfgs_direction(
    grad: np.ndarray,
    s_hist: list[np.ndarray],
    y_hist: list[np.ndarray],
    *,
    init_step_rms_hz: float = 0.05,
    max_step_rms_hz: float = 15.0,
) -> np.ndarray:
    """Compute safeguarded two-loop L-BFGS descent direction with RMS step normalization."""
    g_vec = np.asarray(grad, dtype=np.float64).reshape(-1)
    g_rms = float(np.sqrt(np.mean(g_vec ** 2)))
    if g_rms <= 1e-18:
        return np.zeros_like(grad, dtype=np.float64)

    def _steepest() -> np.ndarray:
        return ((-init_step_rms_hz / g_rms) * g_vec).reshape(grad.shape)

    m = len(s_hist)
    if m == 0:
        return _steepest()

    q = g_vec.copy()
    alphas = np.zeros(m, dtype=np.float64)
    rhos = np.zeros(m, dtype=np.float64)
    for idx in range(m - 1, -1, -1):
        denom = float(np.dot(y_hist[idx], s_hist[idx]))
        if denom <= 1e-20:
            return _steepest()
        rhos[idx] = 1.0 / denom
        alphas[idx] = rhos[idx] * float(np.dot(s_hist[idx], q))
        q = q - alphas[idx] * y_hist[idx]

    last_yy = float(np.dot(y_hist[-1], y_hist[-1]))
    if last_yy <= 1e-24:
        return _steepest()
    gamma = float(np.dot(s_hist[-1], y_hist[-1])) / last_yy
    if gamma <= 0.0:
        return _steepest()

    r = gamma * q
    for idx in range(m):
        beta = rhos[idx] * float(np.dot(y_hist[idx], r))
        r = r + s_hist[idx] * (alphas[idx] - beta)
    d_vec = -r
    if float(np.dot(d_vec, g_vec)) >= 0.0:
        return _steepest()
    d_rms = float(np.sqrt(np.mean(d_vec ** 2)))
    if d_rms > max_step_rms_hz:
        d_vec = d_vec * (max_step_rms_hz / d_rms)
    return d_vec.reshape(grad.shape)
