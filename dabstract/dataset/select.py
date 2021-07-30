import numpy as np
from typing import TypeVar, List, Union

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


def subsample_by_str(
    key: str = None, keep: Union[str, List[str]] = None, **kwargs
) -> tvSubsampleFunc:
    """Subsampling fct: by string or list of strings"""

    def func(data):
        assert keep is not None
        if not isinstance(keep, list):
            kp = [keep]
        else:
            kp = keep
        return [k for k in np.arange(len(data)) if data[key][k] in kp]

    return func


def subsample_comparison(
    key: str = None, type:str = None, value:float = None, **kwargs
) -> tvSubsampleFunc:
    """Subsampling fct: compare"""
    types = ('equal','lower_or_equal','greater_or_equal','greater','lower', \
             '=', '<=', '>=', '>', '<')
    assert type in types, "subsample_comparison type is %s but should be one of %s" % (type, str(types))
    assert not isinstance(value, float)

    def func(data):
        if type in ('=','equal'):
            return [k for k in np.arange(len(data)) if data[key][k]==value]
        elif type in ('<=','lower_or_equal'):
            return [k for k in np.arange(len(data)) if data[key][k]<=value]
        elif type in ('>=','greater_or_equal'):
            return [k for k in np.arange(len(data)) if data[key][k]>=value]
        elif type in ('>','greater'):
            return [k for k in np.arange(len(data)) if data[key][k]>value]
        elif type in ('<','lower'):
            return [k for k in np.arange(len(data)) if data[key][k]>value]
    return func

