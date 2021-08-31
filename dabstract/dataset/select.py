import numpy as np
import datetime as dt
from typing import TypeVar, List, Union, Tuple

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
        key: str = None,
        type:str = None,
        value: Union[float, dt.datetime, Tuple[Union[float, dt.datetime]]] = None,
        convert_str_to_datetime: bool = False,
        datetime_str: str = '%Y-%m-%d %H:%M:%S',
        **kwargs
) -> tvSubsampleFunc:
    """Subsampling fct: compare"""
    types = ('equal','lower_or_equal','greater_or_equal','greater','lower', 'within',\
             '=', '<=', '>=', '>', '<','<x<')
    assert type in types, "subsample_comparison type is %s but should be one of %s" % (type, str(types))
    assert not isinstance(value, float)
    if type in ('<x<', 'within'):
        assert len(value) == 2, "value should have a length of two, i.e. (start_value, stop_value)"

    if convert_str_to_datetime:
        value = dt.datetime.strptime(value, datetime_str)

    def func(data):
        if isinstance(data[key], np.ndarray):
            if type in ('=','equal'):
                return np.where(data[key]==value)[0]
            elif type in ('<=','lower_or_equal'):
                return np.where(data[key]<=value)[0]
            elif type in ('>=','greater_or_equal'):
                return np.where(data[key]>=value)[0]
            elif type in ('>','greater'):
                return np.where(data[key]>value)[0]
            elif type in ('<','lower'):
                return np.where(data[key]<value)[0]
            elif type in ('<x<','within'):
                return np.where((data[key]>value[0]) & (data[key]<value[1]))[0]
        else:
            if type in ('=','equal'):
                return [k for k in np.arange(len(data)) if data[key][k]==value]
            elif type in ('<=','lower_or_equal'):
                return [k for k in np.arange(len(data)) if data[key][k]<=value]
            elif type in ('>=','greater_or_equal'):
                return [k for k in np.arange(len(data)) if data[key][k]>=value]
            elif type in ('>','greater'):
                return [k for k in np.arange(len(data)) if data[key][k]>value]
            elif type in ('<','lower'):
                return [k for k in np.arange(len(data)) if data[key][k]<value]
            elif type in ('<x<', 'within'):
                return [k for k in np.arange(len(data)) if ((data[key][k]>value[0]) & (data[key][k]<value[1]))]
    return func

