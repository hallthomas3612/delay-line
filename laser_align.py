# laser_align.py
from __future__ import annotations
from typing import Callable, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AlignConfig:
    tol_beta: float = 1e-8
    tol_step: float = 1e-10
    max_iters: int = 50
    step_cap: Optional[float] = None
    per_axis_caps: Optional[np.ndarray] = None
    damping: float = 0.0
    line_search: bool = True
    ls_factor: float = 0.5
    ls_max_shrinks: int = 8
    verbose: bool = False

@dataclass
class AlignResult:
    alpha: np.ndarray
    iters: int
    converged: bool
    reason: str
    history: Dict[str, list] = field(default_factory=dict)

def _apply_caps(delta: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    d = delta.copy()
    if cfg.step_cap is not None:
        n = np.linalg.norm(d)
        if n > cfg.step_cap and n > 0:
            d *= cfg.step_cap / n
    if cfg.per_axis_caps is not None:
        d = np.clip(d, -np.abs(cfg.per_axis_caps), np.abs(cfg.per_axis_caps))
    return d

def pseudo_inverse_step(J: np.ndarray, B: np.ndarray) -> np.ndarray:
    pinv = np.linalg.pinv(J)
    return - pinv @ B

def damped_least_squares_step(J: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    JTJ = J.T @ J
    n = JTJ.shape[0]
    A = JTJ + (lam ** 2) * np.eye(n)
    rhs = J.T @ B
    delta = - np.linalg.solve(A, rhs)
    return delta

def nullspace(J: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    n = Vt.shape[1]
    if s.size == 0:
        return np.empty((n, 0))
    tol = rtol * s[0]
    rank = int((s > tol).sum())
    k = n - rank
    if k <= 0:
        return np.empty((n, 0))
    Vnull = Vt[-k:, :].T
    return Vnull

def align_mirrors(
    J_fn: Callable[[np.ndarray], np.ndarray],
    B_fn: Callable[[np.ndarray], np.ndarray],
    alpha0: np.ndarray,
    cfg: Optional[AlignConfig] = None,
) -> AlignResult:
    if cfg is None:
        cfg = AlignConfig()
    alpha = np.array(alpha0, dtype=float).reshape(-1)
    hist = {"||B||": [], "||delta||": [], "alpha": []} if cfg.verbose else {}

    B = B_fn(alpha)
    if cfg.verbose:
        hist["||B||"].append(float(np.linalg.norm(B)))
        hist["alpha"].append(alpha.copy())

    for k in range(1, cfg.max_iters + 1):
        J = J_fn(alpha)
        if cfg.damping == 0.0:
            delta = pseudo_inverse_step(J, B)
        else:
            delta = damped_least_squares_step(J, B, cfg.damping)

        delta = _apply_caps(delta, cfg)

        if np.linalg.norm(delta) <= cfg.tol_step:
            return AlignResult(alpha=alpha, iters=k-1, converged=False, reason="Stalled (tiny step)", history=hist)

        step_alpha = alpha + delta
        B_new = B_fn(step_alpha)
        if cfg.line_search:
            shrinks = 0
            while np.linalg.norm(B_new) >= np.linalg.norm(B) and shrinks < cfg.ls_max_shrinks:
                delta *= cfg.ls_factor
                step_alpha = alpha + delta
                B_new = B_fn(step_alpha)
                shrinks += 1

        alpha = step_alpha
        B = B_new

        if cfg.verbose:
            hist["||B||"].append(float(np.linalg.norm(B)))
            hist["||delta||"].append(float(np.linalg.norm(delta)))
            hist["alpha"].append(alpha.copy())

        if np.linalg.norm(B) <= cfg.tol_beta:
            return AlignResult(alpha=alpha, iters=k, converged=True, reason="||B|| <= tol_beta", history=hist)

    return AlignResult(alpha=alpha, iters=cfg.max_iters, converged=False, reason="Max iterations reached", history=hist)
