import numbers
import copy
import numpy as np
from tqdm import tqdm
import inspect
import os

import warnings

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import (
    Union,
    Any,
    List,
    TypeVar,
    Callable,
    Dict,
    Iterable,
    Generator,
    Tuple,
)

tvDictSeqAbstract = TypeVar("DictSeqAbstract")
tvSeqAbstract = TypeVar("SeqAbstract")

from dabstract.utils import list_intersection, list_difference
from dabstract.dataprocessor import ProcessingChain


class Abstract:
    pass


class UnpackAbstract(Abstract):
    """Unpack a dictionary into a list"""

    def __init__(self, data: dict or tvDictSeqAbstract, keys: List[str]):
        self._data = data
        self._keys = keys
        self._abstract = True if isinstance(data, Abstract) else False

    def get(self, index: int, return_info: bool = False) -> List[Any]:
        if isinstance(index, numbers.Integral):
            out = list()
            if len(self._keys) == 1:
                out = self._data[self._keys[0]][index]
            else:
                for key in self._keys:
                    out.append(self._data[key][index])
            if return_info:
                return out, dict()
            else:
                return out
        else:
            return self._data[index]

    def __getitem__(self, index) -> Union[Dict[str, Any], Any]:
        return self.get(index)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return self._data.__repr__() + "\n Unpack of keys: " + str(self._keys)


def parallel_op(
    data: Iterable,
    type: str = "threadpool",
    *args: list,
    workers: int = 0,
    buffer_len: int = 3,
    return_info: bool = False,
    **kwargs
) -> Generator:
    """Apply a multiproc generator to the input sequence"""
    # define function to evaluate
    if isinstance(data, Abstract):

        def func(index):
            return data.get(index, *args, return_info=return_info, **kwargs)

    else:

        def func(index):
            return data[index]

    # get parallel util
    if type == "threadpool":
        parr = ThreadPoolExecutor
    elif type == "processpool":
        parr = ProcessPoolExecutor

    # create generator
    if workers > 0:
        Q = Queue()
        with parr(workers) as E:
            for k in range(len(data)):
                if Q.qsize() >= buffer_len:
                    yield Q.get().result()
                Q.put(E.submit(func, k))
            while not Q.empty():
                yield Q.get().result()
    else:
        for k in range(len(data)):
            yield func(k)


class DataAbstract(Abstract):
    """Allow for multi-indexing and multi-processing on a sequence or dictseq"""

    def __init__(
        self,
        data: Iterable,
        workers: int = 0,
        buffer_len: int = 3,
        load_memory: bool = False,
        **kwargs: Dict
    ):
        self._data = data
        self._abstract = True if isinstance(data, Abstract) else False
        self._workers = workers
        self._buffer_len = buffer_len
        self._load_memory = load_memory
        self._kwargs = kwargs

    def __iter__(self) -> Any:
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __iter__(self) -> Generator:
        return parallel_op(
            self._data,
            workers=self._workers,
            buffer_len=self._buffer_len,
            return_info=False,
            **self._kwargs
        )

    def __call__(self):
        return self.__iter__()

    def get(
        self,
        index: Iterable,
        *args: list,
        return_info: bool = False,
        workers: int = 0,
        buffer_len: int = 3,
        return_generator: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Any:
        if isinstance(index, numbers.Integral):
            if self._abstract:
                data, info = self._data.get(
                    index, return_info=True, *args, **kwargs, **self._kwargs
                )
            else:
                data, info = self._data[index], {}
            return (data, info) if return_info else data
        elif isinstance(index, (tuple, list, np.ndarray, slice)):
            # generator
            _data = SelectAbstract(self._data, index)
            gen = parallel_op(
                _data,
                *args,
                workers=workers,
                buffer_len=buffer_len,
                return_info=return_info,
                **kwargs,
                **self._kwargs
            )
            # return
            if return_generator:
                return gen
            else:
                for k, tmp in enumerate(tqdm(gen, disable=not verbose)):
                    if return_info:
                        tmp_data, tmp_info = tmp[0], tmp[1]
                    else:
                        tmp_data = tmp
                    if len(_data) == 1:
                        return (tmp_data, tmp_info) if return_info else tmp_data
                    else:
                        if k == 0:
                            if return_info:
                                info_out = [dict()] * len(self._data)
                            if isinstance(tmp_data, (np.ndarray)):
                                data_out = np.zeros((len(_data),) + tmp_data.shape)
                            elif isinstance(
                                tmp_data, (np.int, np.int64, int, np.float64)
                            ):
                                data_out = np.zeros((len(_data), 1))
                            else:
                                data_out = [None] * len(_data)
                        data_out[k] = tmp_data
                        if return_info:
                            info_out[k] = tmp_info
                return (data_out, info_out) if return_info else data_out
        elif isinstance(index, str):
            return DataAbstract(KeyAbstract(self, index))
        else:
            raise TypeError(
                "Index should be a number. Note that a str works too as it does not provide any error but it will only a \n \
                            value which is not None in case a it actually contains a key. \n \
                            This is because a SeqAbstract may contain a DictSeqAbstract with a single active key \n \
                            and other data including no keys."
            )

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> List[str]:
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self) -> str:
        return (
            class_str(self._data)
            + "\n data abstract: multi_processing "
            + str((True if self._workers > 0 else False))
        )


class MapAbstract(Abstract):
    """Add a mapping on input data"""

    def __init__(
        self,
        data: Iterable,
        map_fct: Callable,
        info: List[Dict] = None,
        *arg: list,
        **kwargs: dict
    ):
        assert callable(map_fct), map_fct
        self._map_fct = map_fct
        self._data = data
        self._chain = True if isinstance(map_fct, ProcessingChain) else False
        self._abstract = True if isinstance(data, Abstract) else False
        self._kwargs = kwargs
        self._info = info
        self._args = arg

    def __iter__(self) -> Generator:
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __setitem__(self, k, v):
        raise NotImplementedError("MapAbstract does not support item assignment. \
                First excecute the mapping and then asign, or asign and add mapping afterwards.")

    def get(
        self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            if index < 0:
                index = index % len(self)
            if self._abstract:
                data, info = self._data.get(index, return_info=True, *arg, **kwargs)
            else:
                data, info = self._data[index], kwargs
            if self._info is not None:
                info = dict(info, **self._info[index])
            if self._chain:
                data, info = self._map_fct(
                    data, *self._args, **dict(self._kwargs, **info), return_info=True
                )
            else:
                data = self._map_fct(data)
            return (data, info) if return_info else data
        elif isinstance(index, str):
            warnings.warn(
                "Ignoring a mapping. Mapping works on __getitem__, so if you have a nested DictSeqAbstract with active key, then you will access the active key without mapping and the meta information"
            )
            return self._data[index]
        else:
            raise TypeError(
                "Index should be a number. Note that a str works too as it does not provide any error but it will only a \
                            value which is not None in case a it actually contains a key. \
                            This is because a SeqAbstract may contain a DictSeqAbstract with a single active key \
                            and other data including no keys."
            )
            # ToDo(gert) add a way to raise a error in case data does not contain any key.

    def shape(self) -> Union[Tuple, int]:
        data = self[0]
        if isinstance(data, np.ndarray):
            return data.shape
        elif hasattr(data, "__len__"):
            return len(data)
        else:
            return []

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> List[str]:
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self) -> str:
        return class_str(self._data) + "\n map: " + str(self._map_fct)


def Map(
    data,
    map_fct: Callable,
    info: List[Dict] = None,
    lazy: bool = True,
    workers: int = 1,
    buffer_len: int = 3,
    *arg: list,
    **kwargs: Dict
) -> Union[MapAbstract, DataAbstract, np.ndarray, list]:
    """Factory function to allow for choice between lazy and direct mapping"""
    _abstract = True if isinstance(data, Abstract) else False
    if lazy:
        return MapAbstract(data, map_fct, info=info, *arg, **kwargs)
    else:
        return DataAbstract(
            MapAbstract(data, map_fct, info=info, *arg, **kwargs),
            workers=workers,
            buffer_len=buffer_len,
        )[:]


# ToDo(gert)
# def Replicate(data, factor, type = 'on_sample', lazy=True, workers=1, buffer_len=3, *arg, **kwargs):
#     """Factory function to allow for choice between lazy and direct replication
#     """
#     _abstract = (True if isinstance(data, abstract) else False)
#     if lazy:
#         return ReplicateAbstract(data, factor, type = 'on_sample', **kwargs)
#     else:
#         #ToDo: replace by a list and np equivalent
#         return DataAbstract(ReplicateAbstract(data, factor, type = 'on_sample', *arg, **kwargs),
#                             workers=workers,
#                             buffer_len=buffer_len)[:]
# ToDo(gert)
# class ReplicateAbstract(abstract):
#     """Replicate data a particular factor
#     """
#     def __init__(self, data, factor, type = 'on_sample', **kwargs):
#         self._data = data
#         self._type = type
#         self._factor = factor
#         self._abstract = (True if isinstance(data, abstract) else False)
#         if self._type == 'on_sample':
#             self.rep_function = (lambda x: int(np.floor(x / self._factor)))
#         elif self._type == 'full':
#             self.rep_function = (lambda x: int(np.floor(x / len(self._data))))
#         else:
#             raise NotImplemented
#
#     def __iter__(self):
#         for k in range(len(self)):
#             yield self[k]
#
#     def __getitem__(self, index):
#         return self.get(index)
#
#     def get(self, index, return_info=False, *arg, **kwargs):
#         if isinstance(index, numbers.Integral):
#             if index < 0:
#                 index = index % len(self)
#             assert index < len(self)
#             k = self.rep_function(index)
#             if self._abstract:
#                 data, info = self._data.get(k, return_info=True, *arg, **kwargs, **self._kwargs)
#             else:
#                 data, info = self._data[k], {}
#             return ((data, info) if return_info else data)
#         elif isinstance(index, str):
#             return KeyAbstract(self, index)
#         else:
#             raise TypeError('Index should be a str our number')
#
#     def __len__(self):
#         return len(self._data) * self._factor
#
#     def keys(self):
#         if hasattr(self._data, 'keys'):
#             return self._data.keys()
#         else:
#             return self._data._data.keys()
#
#     def __repr__(self):
#         return self._data.__repr__() + "\n replicate: " + str(
#             self._factor) + ' ' + self._type


class SampleReplicateAbstract(Abstract):
    """Replicate data on sample-by-sample basis"""

    def __init__(self, data: Iterable, factor: int, **kwargs: Dict):
        self._data = data
        self._factor = factor
        if isinstance(self._factor, numbers.Integral):
            self._factor = self._factor * np.ones(len(data))
        self._abstract = True if isinstance(data, Abstract) else False

    def __iter__(self) -> Generator:
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def get(
        self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            assert index < len(self), "Index should be lower than len(dataset)"
            if index < 0:
                index = index % len(self)
            for k, factor in enumerate(self._factor):
                if factor <= index:
                    index -= factor
                else:
                    # get
                    if self._abstract:
                        data, info = self._data.get(k, return_info=True, **kwargs)
                    else:
                        data, info = self._data[k], dict()
                    # return
                    return (data, info) if return_info else data
        elif isinstance(index, str):
            return KeyAbstract(self, index)
        else:
            raise TypeError("Index should be a str our number")

    def __len__(self) -> int:
        return int(np.sum(self._factor))

    def keys(self) -> List[str]:
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self) -> str:
        return (
            self._data.__repr__()
            + "\n replicate: "
            + str(self._factor.min())
            + " - "
            + str(self._factor.max())
        )


def SampleReplicate(
    data: Iterable,
    factor: int,
    lazy: bool = True,
    workers: int = 1,
    buffer_len: int = 3,
    *arg: List,
    **kwargs: Dict
) -> Union[SampleReplicateAbstract, DataAbstract, np.ndarray, list]:
    """Factory function to allow for choice between lazy and direct sample replication"""
    _abstract = True if isinstance(data, Abstract) else False
    if lazy:
        return SampleReplicateAbstract(data, factor, **kwargs)
    else:
        # ToDo: replace by a list and np equivalent
        return DataAbstract(
            SampleReplicateAbstract(data, factor, *arg, **kwargs),
            workers=workers,
            buffer_len=buffer_len,
        )[:]


class SplitAbstract(Abstract):
    """Split the datastream"""

    def __init__(
        self,
        data: Iterable,
        split_size: int = None,
        constraint: str = None,
        sample_len: int = None,
        sample_period: int = None,
        type: str = "seconds",
        **kwargs: Dict
    ):
        self._data = data
        assert split_size is not None, "Please provide a split in " + type
        self._type = type
        self._split_size = split_size
        self._constraint = constraint
        self._sample_len = sample_len
        if isinstance(self._sample_len, numbers.Integral):
            self._sample_len = self._sample_len * np.ones(len(data))
        self._sample_period = sample_period
        self._abstract = True if isinstance(data, Abstract) else False
        self._kwargs = kwargs
        self._init_split()

    def _init_split(self):
        # init window_size
        if self._type == "seconds":
            self._window_size = int(self._split_size / self._sample_period)
        elif self._type == "samples":
            self._window_size = int(self._split_size)
        if self._constraint == "power2":
            self._window_size = int(2 ** np.ceil(np.log2(self._window_size)))
        assert self._window_size > 0
        # prepare splits
        self._split_range, self._split_len = [None] * len(self._data), np.zeros(
            len(self._data), dtype=int
        )
        for j in range(len(self._data)):
            num_frames = max(
                1,
                int(
                    np.floor(
                        (
                            (self._sample_len[j] - (self._window_size - 1) - 1)
                            / self._window_size
                        )
                        + 1
                    )
                ),
            )
            self._split_range[j] = np.tile(
                np.array([0, self._window_size]), (num_frames, 1)
            ) + np.tile(
                np.transpose(np.array([np.arange(num_frames) * self._window_size])),
                (1, 2),
            )
            self._split_len[j] = num_frames

    def __iter__(self) -> Generator:
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def get(
        self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            assert index < len(self)
            if index < 0:
                index = index % len(self)
            for k, split_len in enumerate(self._split_len):
                if split_len <= index:
                    index -= split_len
                else:
                    read_range = self._split_range[k][int(index)]
                    # get data
                    if self._abstract:
                        data, info = self._data.get(
                            k,
                            return_info=True,
                            *arg,
                            **kwargs,
                            **self._kwargs,
                            read_range=read_range
                        )
                    else:
                        data, info = self._data[k][read_range[0] : read_range[1]], {}
                    return (data, info) if return_info else data
        elif isinstance(index, str):
            return KeyAbstract(self, index)
        else:
            raise TypeError("Index should be a str our number")

    def __len__(self) -> int:
        return int(np.sum(self._split_len))

    def keys(self) -> List[str]:
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self):
        return (
            self._data.__repr__()
            + "\n split: "
            + str(self._window_size * self._sample_period)
            + " "
            + self._type
        )


def Split(
    data: Iterable,
    split_size: int = None,
    constraint: str = None,
    sample_len: int = None,
    sample_period: int = None,
    type: str = "seconds",
    lazy: bool = True,
    workers: bool = 1,
    buffer_len: int = 3,
    *arg: List,
    **kwargs: Dict
) -> Union[SplitAbstract, DataAbstract, np.ndarray, list]:
    """Factory function to allow for choice between lazy and direct example splittin"""
    _abstract = True if isinstance(data, Abstract) else False
    if lazy:
        return SplitAbstract(
            data,
            split_size=split_size,
            constraint=constraint,
            sample_len=sample_len,
            sample_period=sample_period,
            type=type,
            **kwargs
        )
    else:
        # ToDo: replace by a list and np equivalent
        return DataAbstract(
            SplitAbstract(
                data,
                split_size=split_size,
                constraint=constraint,
                sample_len=sample_len,
                sample_period=sample_period,
                type=type,
                *arg,
                **kwargs
            ),
            workers=workers,
            buffer_len=buffer_len,
        )[:]


class SelectAbstract(Abstract):
    """Select a subset of your input sequence. Selection is performed directly, this means that it should be a
    variable which is readily available from memory.
    """

    def __init__(
        self,
        data: Iterable,
        selector: Union[List[int], Callable, numbers.Integral],
        eval_data: Any = None,
        **kwargs: Dict
    ):
        self._data = data
        self._eval_data = data if eval_data is None else eval_data
        self._selector = selector
        if callable(selector):
            if len(inspect.getargspec(selector)[0]) == 1:
                self._indices = selector(self._eval_data)
            else:
                self._indices = np.where(
                    [selector(self._eval_data, k) for k in range(len(self._eval_data))]
                )[0]
        elif isinstance(selector, slice):
            self._indices = np.arange(
                (0 if selector.start is None else selector.start),
                (len(self._eval_data) if selector.stop is None else selector.stop),
                (1 if selector.step is None else selector.step),
            )
        elif isinstance(selector, (tuple, list, np.ndarray)):
            self._indices = selector
        elif isinstance(selector, numbers.Integral):
            self._indices = [selector]
        self._abstract = True if isinstance(data, Abstract) else False
        self._kwargs = kwargs
        if self._abstract:
            if hasattr(self._data, "_lazy"):
                if not self._data._lazy:
                    return DataAbstract(self)[:]

    def __iter__(self) -> Generator:
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def get(
        self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            assert index < len(self)
            index = self._indices[index]
            if self._abstract:
                data, info = self._data.get(
                    index, return_info=True, *arg, **kwargs, **self._kwargs
                )
            else:
                data, info = self._data[index], {}
            return (data, info) if return_info else data
        elif isinstance(index, str):
            return SelectAbstract(self._data[index], self._indices)
            # return KeyAbstract(self, index)
        else:
            raise TypeError("Index should be a str our number")

    def __len__(self) -> int:
        return len(self._indices)

    def keys(self) -> List[str]:
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self) -> str:
        return self._data.__repr__() + "\n select: " + str(type(self._selector))


def Select(
    data,
    selector: Union[List[int], Callable, numbers.Integral],
    eval_data: Any = None,
    lazy: bool = True,
    workers: int = 1,
    buffer_len: int = 3,
    *arg,
    **kwargs
) -> Union[SelectAbstract, DataAbstract, np.ndarray, list]:
    """Factory function to allow for choice between lazy and direct example selection"""
    _abstract = True if isinstance(data, Abstract) else False
    if lazy:
        return SelectAbstract(data, selector, eval_data=eval_data, **kwargs)
    else:
        # ToDo: replace by a list and np equivalent
        return DataAbstract(
            SelectAbstract(data, selector, eval_data=eval_data, *arg, **kwargs),
            workers=workers,
            buffer_len=buffer_len,
        )[:]


class FilterAbstract(Abstract):
    """Filter on the fly. Interesting when the variable to filter on takes long to compute."""

    def __init__(self, data: Iterable, filter_fct: Callable, **kwargs):
        assert callable(filter_fct), filter_fct
        self.filter_fct = filter_fct
        self.abstract = data
        self._kwargs = kwargs

    def __iter__(self) -> Generator:
        for index in range(len(self)):
            data = self.get(index)
            if self.filter_fct(data):
                yield data

    def __getitem__(self, index: int) -> Any:
        data = self.get(index)
        if self.filter_function(data):
            return data
        raise IndexError("Not available.")

    def get(
        self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            assert index < len(self)
            if self._abstract:
                data, info = self._data.get(
                    index, return_info=True, *arg, **kwargs, **self._kwargs
                )
            else:
                data, info = self._data[index], {}
            return (data, info) if return_info else data
        elif isinstance(index, str):
            return KeyAbstract(self, index)
        else:
            raise TypeError("Index should be a str our number")

    def __len__(self) -> int:
        raise Exception("Length not available as filter is evaluated on the fly")

    def keys(self) -> List[str]:
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self) -> str:
        return self._data.__repr__() + "\n filter: " + str(type(self.filter_function))


def Filter(
    data: Iterable,
    filter_fct: Callable,
    lazy: bool = True,
    workers: int = 1,
    buffer_len: int = 3,
    *arg: List,
    **kwargs: Dict
) -> Union[FilterAbstract, DataAbstract, np.ndarray, list]:
    """Factory function to allow for choice between lazy and direct example selection"""
    _abstract = True if isinstance(data, Abstract) else False
    if lazy:
        return FilterAbstract(data, filter_fct, **kwargs)
    else:
        # ToDo: replace by a list and np equivalent
        return DataAbstract(
            FilterAbstract(data, filter_fct, *arg, **kwargs),
            workers=workers,
            buffer_len=buffer_len,
        )[:]


class KeyAbstract(Abstract):
    """Error handling wrapper for a concatenated sequence where one might have a dictseq and the other doesnt.
    This will allow for key/index indexing even if the particular index does not have a key.
    """

    def __init__(self, data: Iterable, key: str, **kwargs: Dict):
        assert isinstance(data, Abstract)
        self._data = data
        self._key = key
        self._kwargs = kwargs

    def __iter__(self) -> Generator:
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def get(
        self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            assert index < len(self)
            try:
                data, info = self._data.get(
                    key=self._key,
                    index=index,
                    return_info=True,
                    *arg,
                    **kwargs,
                    **self._kwargs
                )
            except:
                data, info = None, {}
            return (data, info) if return_info else data
        else:
            return KeyAbstract(self, index)

        # ToDo(gert)
        # if isinstance(index, str):
        #     assert key is None
        #     return self._data[index]
        # elif isinstance(index,numbers.Integral):
        #     if key is None:
        #         data, info = dict(), dict()
        #         for k, key in enumerate(self._active_keys):
        #             data[key], info[key] = self._data[key].get(index=index, return_info=True,**kwargs)
        #         if len(self._active_keys)==1:
        #             data, info = data[key], info[key]
        #     else:
        #         assert isinstance(key,str)
        #         data, info = self._data[key].get(index=index, return_info=True, **kwargs)
        #     return ((data, info) if return_info else data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> List[str]:
        if hasattr(self._data, "keys"):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self) -> str:
        return "key_abstract of key " + self._key + " on " + str(self._data)


class DictSeqAbstract(Abstract):
    """DictSeq base class"""

    def __init__(self, name: str = ""):
        self._nr_keys = 0
        self._name = name
        self._data = dict()
        self._active_keys = []
        self._lazy = dict()
        self._abstract = dict()
        self._adjust_mode = False

    def add(
        self,
        key: str,
        data: Iterable,
        lazy: bool = True,
        info: List[Dict] = None,
        **kwargs: Dict
    ) -> None:
        assert hasattr(
            data, "__getitem__"
        ), "provided data instance must have __getitem__ method."
        assert (
            key != "all"
        ), "The name 'all' is reserved for referring to all keys when applying a transform."
        if not self._adjust_mode:
            if self._nr_keys > 0:
                assert len(data) == len(self), "len(self) is not the same as len(data)"
        new_key = False if key in self.keys() else True
        if (not lazy) and isinstance(data, Abstract):
            data = DataAbstract(data)[:]
        elif info is not None:
            data = SeqAbstract().concat(data, info=info)
        self._data.update({key: data})
        self._lazy.update({key: lazy})
        self._abstract.update({key: isinstance(data, Abstract)})
        if new_key:
            self._reset_active_keys()
            self._nr_keys += 1
        return self

    def add_dict(self, dct: Dict, lazy: bool = True) -> None:
        for key in dct:
            self.add(key, dct[key], lazy=lazy)
        return self

    def concat(
        self, data: Iterable, intersect: bool = False, adjust_base: bool = True
    ) -> None:
        from dabstract.dataset.helpers import FolderDictSeqAbstract

        if isinstance(data, list):
            for d in data:
                self.concat(d, intersect=intersect)
        else:
            self2 = self if adjust_base else copy.deepcopy(self)
            self2._adjust_mode = True
            data = copy.deepcopy(data)
            assert isinstance(data, DictSeqAbstract)
            if self2._nr_keys != 0:
                if not intersect:
                    assert (
                        data.keys() == self2.keys()
                    ), "keys do not match. Set intersect=True for keeping common keys."
                    keys = data.keys()
                else:
                    # get diff
                    keys = list_intersection(data.keys(), self2.keys())
                    rem_keys = list_difference(data.keys(), self2.keys())
                    # remove ones which are not identical
                    for rem_key in rem_keys:
                        self2.remove(rem_key)
                for key in keys:
                    if self2._lazy[key]:
                        # make sure that data format is as desired by the base dict
                        if not isinstance(
                            self2[key],
                            (SeqAbstract, DictSeqAbstract, FolderDictSeqAbstract),
                        ):
                            self2[key] = SeqAbstract().concat(self2[key])
                        # concatenate SeqAbstract
                        if isinstance(
                            data[key], SeqAbstract
                        ):  # if already a SeqAbstract, concat cleaner to avoid overhead
                            for _data in data[key]._data:
                                self2[key].concat(_data)
                        else:  # if not just concat at once
                            self2[key].concat(data[key])
                    else:
                        assert (
                            self2[key].__class__ == data[key].__class__
                        ), "When using lazy=False, datatypes should be same in case of concatenation."
                        if isinstance(self2[key], list):
                            self2[key] = self2[key] + data[key]
                        elif isinstance(self2[key], np.ndarray):
                            self2[key] = np.concatenate((self2[key], data[key]))
                self2._adjust_mode = False
            else:
                self2.__dict__.update(data.__dict__)

            return self2

    def remove(self, key: str) -> None:
        del self._data[key]
        self.reset_active_keys()
        self._nr_keys -= 1
        return self

    def add_map(self, key: str, map_fct: Callable, *arg: List, **kwargs: Dict) -> None:
        self[key] = Map(self[key], map_fct, lazy=self._lazy[key], *arg, **kwargs)

    def add_alias(self, key: str, new_key: str) -> None:
        assert new_key not in self.keys(), "alias key already in existing keys."
        self.add(new_key, self[key])

    def set_active_keys(self, keys: Union[List[str], str]) -> None:
        self._set_active_keys(keys)

    def reset_active_key(self) -> None:
        warnings.warn(
            "reset_active_key() in DictSeqAbstract is deprecated. Please use reset_active_keys()"
        )
        self._reset_active_keys()

    def reset_active_keys(self) -> None:
        self._reset_active_keys()

    def _set_active_keys(self, keys: Union[List[str], str]) -> None:
        if isinstance(keys, list):
            for key in keys:
                assert key in self.keys(), "key " + key + " does not exists."
            self._active_keys = keys
        else:
            assert keys in self.keys(), "key " + keys + " does not exists."
            self._active_keys = [keys]

    def _reset_active_keys(self) -> None:
        self._active_keys = self.keys()

    def get_active_keys(self) -> List[str]:
        return self._active_keys

    def __len__(self) -> int:
        nr_examples = [len(self._data[key]) for key in self._data]
        assert all([nr_example == nr_examples[0] for nr_example in nr_examples])
        return nr_examples[0] if len(nr_examples) > 0 else 0

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __add__(self, other: Iterable) -> None:
        assert isinstance(other, DictSeqAbstract)
        return self.concat(other, adjust_base=False)

    def __setitem__(self, k: str, v: Any) -> None:
        assert isinstance(k, str), "Assignment only possible by key (str)."
        new_key = False if k in self.keys() else True
        lazy = True if new_key else self._lazy[k]  # make sure that lazy is kept
        self.add(k, v, lazy=lazy)

    def get(
        self,
        index: int,
        key: str = None,
        return_info: bool = False,
        *arg: List,
        **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, str):
            assert key is None
            return self._data[index]
        elif isinstance(index, numbers.Integral):
            if key is None:
                data, info = dict(), dict()
                for k, key in enumerate(self._active_keys):
                    if self._abstract[key]:
                        data[key], info[key] = self._data[key].get(
                            index=index, return_info=True, **kwargs
                        )
                    else:
                        data[key], info[key] = self._data[key][index], dict()
                if len(self._active_keys) == 1:
                    data, info = data[key], info[key]
            else:
                assert isinstance(key, str)
                data, info = self._data[key].get(
                    index=index, return_info=True, **kwargs
                )
            return (data, info) if return_info else data
        else:
            raise IndexError("index should be a number or str")

    def unpack(self, keys: List[str]) -> UnpackAbstract:
        return UnpackAbstract(self._data, keys)

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def shape(self) -> Union[Tuple[int], int]:
        if len(self._active_keys) == 1:
            data = self[0]
            if isinstance(data, np.ndarray):
                return data.shape
            elif hasattr(data, "__len__"):
                return len(data)
            else:
                return []
        raise NotImplementedError(
            "Shape only available if a single active key is set and if that item has a .shape() method."
        )

    def summary(self) -> Dict:
        summary = dict()
        for name, data in zip(self.keys(), self._data):
            summary[name] = data.summary()
        return summary

    def __repr__(self) -> str:
        return "dict_seq containing: " + str(self.keys())


class SeqAbstract(Abstract):
    """Seq base class"""

    def __init__(self, data: Iterable = None, name: str = "seq"):
        self._nr_sources = 0
        self._data = []
        self._info = []
        self._kwargs = []
        self._name = name
        if data is not None:
            if isinstance(data, list):
                for _data in data:
                    self.concat(_data)
            else:
                raise AssertionError("Input data should be a list")

    def concat(self, data: Iterable, info: List[Dict] = None, **kwargs: Dict) -> None:
        # Check
        assert hasattr(
            data, "__getitem__"
        ), "provided data instance must have __getitem__ method."
        if isinstance(data, DictSeqAbstract):
            assert (
                len(data._active_keys) == 1
            ), "You can only add a dict_abstract in case there is only one active key."
        # Add
        data = copy.deepcopy(data)
        if isinstance(data, SeqAbstract):
            for _data in data._data:
                self.concat(_data)
        else:
            self._data.append(data)
        self._nr_sources += 1
        # Information to propagate to transforms or use for split
        if info is not None:
            assert isinstance(info, list), "info should be a list"
            assert isinstance(
                info[0], dict
            ), "The items in info should contain a dict()"
            assert len(info) == len(
                data
            ), "info should be a list with len(info)==len(data)"
        self._info.append(info)
        self._kwargs.append(kwargs)
        return self

    def __len__(self) -> int:
        return np.sum([len(data) for data in self._data])

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __setitem__(self, index: int, value: Iterable):
        if isinstance(index, numbers.Integral):
            if index < 0:
                index = index % len(self)
            for k, data in enumerate(self._data):
                if len(data) <= index:
                    index -= len(data)
                else:
                    data[index] = value
                return None
            raise IndexError("Index should be lower than len(dataset)")
        elif isinstance(index, str):
            return KeyAbstract(self, index)
        else:
            raise IndexError(
                "index should be a number (or key in case of a nested dict_seq)."
            )

    def __add__(self, other: Union[tvSeqAbstract, Iterable]):
        # assert isinstance(other)
        return self.concat(other)

    def get(
        self,
        index: int,
        key: str = None,
        return_info: bool = False,
        *arg: List,
        **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            if index < 0:
                index = index % len(self)
            for k, data in enumerate(self._data):
                if len(data) <= index:
                    index -= len(data)
                else:
                    info = dict() if self._info[k] is None else self._info[k][index]
                    # get
                    if isinstance(self._data[k], Abstract):
                        data, info = data.get(
                            index, key=key, return_info=True, **info, **kwargs
                        )
                    else:
                        assert key is None
                        data, info = data[index], dict(**info, **kwargs)
                    # return
                    return (data, info) if return_info else data
            raise IndexError("Index should be lower than len(dataset)")
        elif isinstance(index, str):
            return KeyAbstract(self, index)
        else:
            raise IndexError(
                "index should be a number (or key in case of a nested dict_seq)."
            )

    def summary(self) -> Dict:
        return {"nr_examples": self.nr_examples, "name": self._name}

    def shape(self) -> Union[Tuple[int], int]:
        data = self[0]
        if isinstance(data, np.ndarray):
            return data.shape
        elif hasattr(data, "__len__"):
            return len(data)
        else:
            return []

    def __repr__(self):
        r = "seq containing:"
        for data in self._data:
            if not isinstance(data, (Abstract)):
                r += "\n[ \t" + str(type(data)) + "\t]"
            else:
                r += "\n[ \t" + repr(data) + "\t]"
            # r += '\n'
        return r


def class_str(data: Callable):
    if isinstance(data, Abstract):
        return repr(data)
    else:
        return str(data.__class__)
    return
