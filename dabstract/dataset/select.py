import numpy as np
from typing import (
    TypeVar,
)

tvSubsampleFunc = TypeVar("subsample_fct")


def random_subsample(ratio: int = 1, **kwargs) -> tvSubsampleFunc:
    """Subsampling fct: random"""

    def func(data):
        indexes = np.arange(len(data))
        if ratio < 1:
            if isinstance(indexes, np.ndarray):
                indexes = np.random.choice(
                    indexes, int(np.ceil(len(indexes) * ratio)), replace=False
                )
            elif isinstance(indexes, list):
                for k in range(len(indexes)):
                    indexes[k] = np.random.choice(
                        indexes[k], int(np.ceil(len(indexes[k]) * ratio)), replace=False
                    )
        return indexes

    return func


def subsample_by_str(key: str = None, keep: str = None, **kwargs) -> tvSubsampleFunc:
    """Subsampling fct: by string"""

    def func(data):
        if isinstance(key, list):
            for tmp in key:
                data = data[tmp]
        elif isinstance(key, str):
            data = data[key]
        assert keep is not None
        return [k for k in np.arange(len(data)) if data[k] == keep]

    return func
