"""Category definitions and inference — re-exports from credence."""

from credence.environment.categories import (
    CATEGORIES,
    NUM_CATEGORIES,
    make_keyword_category_infer_fn,
)

__all__ = ["CATEGORIES", "NUM_CATEGORIES", "make_keyword_category_infer_fn"]
