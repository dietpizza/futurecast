"""Labels subpackage initialization."""

from futurecast.labels.residual_alpha import (
    compute_residual_alpha_target,
    compute_forward_returns,
)

__all__ = [
    "compute_residual_alpha_target",
    "compute_forward_returns",
]
