"""CoveragePrior: per-tool learnable Beta-distributed coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass
class CoveragePrior:
    """Per-tool learnable Beta-distributed coverage prior.

    Tracks P(tool returns an answer | category) as independent
    Beta(alpha[c], beta[c]) distributions. Compose with any Tool
    implementation — no inheritance required.
    """

    alpha: NDArray[np.float64]  # shape (num_categories,)
    beta: NDArray[np.float64]   # shape (num_categories,)

    @classmethod
    def from_priors(
        cls,
        categories: tuple[str, ...],
        priors: dict[str, float],
        strength: float = 10.0,
    ) -> Self:
        """Create from a {category: prior_probability} dict."""
        alpha = np.array(
            [priors.get(c, 0.5) * strength for c in categories],
            dtype=np.float64,
        )
        beta = np.array(
            [(1.0 - priors.get(c, 0.5)) * strength for c in categories],
            dtype=np.float64,
        )
        return cls(alpha=alpha, beta=beta)

    def coverage(self) -> NDArray[np.float64]:
        """E[Beta(alpha, beta)] per category, clipped to [0.1, 1.0]."""
        return np.clip(self.alpha / (self.alpha + self.beta), 0.1, 1.0)

    def update(self, category_posterior: NDArray[np.float64], was_useful: bool) -> None:
        """Soft Beta update: alpha += P(c) on success, beta += P(c) on failure."""
        if was_useful:
            self.alpha += category_posterior
        else:
            self.beta += category_posterior

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Deserialise from a dict."""
        return cls(
            alpha=np.array(d["alpha"], dtype=np.float64),
            beta=np.array(d["beta"], dtype=np.float64),
        )
