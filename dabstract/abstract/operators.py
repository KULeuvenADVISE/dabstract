import numbers
import copy
import numpy as np
from tqdm import tqdm
import inspect
import os
import itertools
import datetime

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

from dabstract.abstract import base as base
from dabstract.dataprocessor import ProcessingChain

class ShuffleAbstract(base.Abstract):
    """
    The class is an abstract wrapper that shuffles an iterable

    Parameters
    ----------
    data : iterable
    reshuffle : bool
        Reshuffle each call for an iteration or not

    Returns
    ----------
    BatchAbstract class
    """

    def __init__(self, data: Iterable, reshuffle: bool = True):
        super().__init__(data)
        self._data = data
        self._idx = np.arange(len(self))
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._idx)

    def _call_on_iter(self):
        self.shuffle()
        super()._call_on_iter()

    def get(self, index: int, *args, return_info: bool = False, **kwargs) -> List[Any]:
        """
        Parameters
        ----------
        index : int
            index to retrieve data from
        return_info : bool
            return tuple (data, info) if True else data (default = False)
            info contains the information that has been propagated through the chain of operations
        Returns
        ----------
        List of Any
        """
        if isinstance(index, numbers.Integral):
            if self._abstract:
                data, info = self._data.get(
                    self._idx[index], return_info=True, *args, **kwargs
                )
            else:
                data, info = self._data[self._idx[index]], {}
            return (data, info) if return_info else data
        else:
            raise NotImplementedError("You should provide a numbers.Integral when indexing a ShuffleAbstract.")

    def __repr__(self) -> str:
        return "%s\n Shuffle" % self._data.__repr__()

class BatchAbstract(base.Abstract):
    """
    The class is an abstract wrapper around an iterable to batch the dataset.
    Parameters
    ----------
    data : iterable
    batch_size : int
        size of each of the batches
    unzip : bool
        In case each sample contains a list that needs to be unzipped.
        This is similar to the zip(...) operation.
    unzip : bool
        In case each sample contains a list that needs to be unzipped.
        This is similar to the unzip(*...) operation.
    drop_last : bool
        Remove the last batch if the size does not match the batch_size.
    Returns
    ----------
    BatchAbstract class
    """

    def __init__(self, data: Iterable,
                 batch_size: int,
                 drop_last: bool = False,
                 unzip: bool = False,
                 zip: bool = False):
        super().__init__(data)
        self._data_abstract = DataAbstract(data, unzip=unzip, zip=zip)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def get(self, index: int, return_info: bool = False) -> List[Any]:
        """
        Parameters
        ----------
        index : int
            index to retrieve data from
        return_info : bool
            return tuple (data, info) if True else data (default = False)
            info contains the information that has been propagated through the chain of operations
        Returns
        ----------
        List of Any
        """
        # assert return_info is False, "BatchAbstract breaks the information flow. It is meant as a final step to batch data as input to a learner."
        if isinstance(index, numbers.Integral):
            input_index = index * self.batch_size
            tmp = self._data_abstract[input_index:min(input_index + self.batch_size,len(self._data))]
            return (tmp, {}) if return_info else tmp
        else:
            raise NotImplementedError("You should provide a numbers.Integral when indexing a BatchAbstract.")

    def __len__(self) -> int:
        length = len(self._data) / self.batch_size
        if self.drop_last:
            return int(length)
        return int(np.ceil(length))

    def __repr__(self) -> str:
        return "%s\n Batch of size %d" % (self._data.__repr__(), self.batch_size)


class UnpackAbstract(base.Abstract):
    """
    The class is an abstract wrapper around a dictionary or DictSeqAbstract to unpack this dictionary in a lazy manner.
    Unpacking refers to copying the content of the dictionary into a list.

    Unpacking is based on the parameter "keys". For example, consider a Dict or DictSeqAbstract with the
    following content::

        $   data = {'data': [10,5,8],
        $           'label': [1,1,2],
        $           'other': ['some','other','information']

    To index this such that it returns a tuple containing the indexed item of keys 'data' and 'label',
    one can do::

        $   data_up = UnpackAbstract(data,keys=['data','label'])
        $   print(data_up[0])
        [10, 1]

    To index through the data one could directly use default indexing, i.e. [idx] or use the .get() method.

    The UnpackAbstract contains the following methods::

        .get - return entry form UnpackAbstract

    The full explanation for each method is provided as a docstring at each method.

    Parameters
    ----------
    data : dict or tvDictSeqAbstract
        dictionary or DictSeqAbstract to be unpacked
    keys : List[str]
        list containing the strings that are used as keys

    Returns
    ----------
    UnpackAbstract class
    """

    def __init__(self, data: dict or tvDictSeqAbstract, keys: List[str]):
        super().__init__(data)
        self._keys = keys

    def get(self, index: int, return_info: bool = False) -> List[Any]:
        """
        Parameters
        ----------
        index : int
            index to retrieve data from
        Returns
        ----------
        List of Any
        """
        if isinstance(index, numbers.Integral):
            out = list()
            if len(self._keys) == 1:
                out = self._data[self._keys[0]][index]
            else:
                for key in self._keys:
                    out.append(self._data[key][index])
            return out, {} if return_info else out
        else:
            raise NotImplementedError("You should index with integers.")

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return self._data.__repr__() + "\n Unpack of keys: " + str(self._keys)


def parallel_op(
        data: Iterable,
        type: str = "threadpool",
        workers: int = 0,
        buffer_len: int = 3,
        return_info: bool = False,
        *args: list,
        **kwargs: Dict
) -> Generator:
    """
    Apply parallelisation to an iterable. This works for any iterable including dabstract functions.

    Consider the following pseudo code as an example::

        $ class IterableToParallize()
        $   def __init__(data, process_function)
        $       self.data = data
        $       self.process_function = process_function
        $   def __getitem__(k)
        $       return self.process_function(self.data[k])
        $
        $   iterable = IterableToParallize(data, process_function)

    which could also be created using the abstract.MapAbstract as::

        $ iterable = MapAbstract(data, process_function)

    To get the data one could simply loop over the data like::

        $ for example in iterable:
        $   do something

    However, if this is costly, one would use this function to speed that up::

        $ par_iterable = parallel_op(iterable, workers = 5)
        $ for example in par_iterable:
        $   do something

    Parameters
    ----------
    data : Iterable
        Iterable object to be parralelise
    type : str ['threadpool','processpool']
        String to select either 'threadpool' or 'processpool'
    workers : int
        Amount of parallel workers
    buffer_len : int
        The length of the buffer in case of a generator::

            for data in dataset:
                do_something(data)

        This will cue up buffer_len instances of data while do_something() is busy.
    return_info : bool
        Return information that has been propagated through a chain of processors and abstract's.
        For example, if one has used WavDataReader from dabstract.dataprocessor this will retrieve you the sampling
        frequency ('fs')
    args : list
        additional param to provide to iterable
    kwargs : dict
        additional param to provide to iterable

    Returns
    -------
    data : Generator
        The generator will return Union[Generator, Tuple[Generator, Dict]]
        When return_info is True, it returns a tuple of the exanoke and a Dictionary containing propagated information
        When return_info is False, it returns the example
    """
    # check
    assert hasattr(data, "__len__"), "Can only use parallel_op it object has __len__"

    # define function to evaluate
    if isinstance(data, base.Abstract):

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


class DataAbstract(base.Abstract):
    """
    DataAbstract combines the functionality offered by the function parallel_op to allow parallel processing with arbitrary
    access of the data. In parallel_op you're only given a Generator, however, you might be interested in more flexibility such
    as indexing your data in a particular range and still having this parallized. Additionally, as all classes in abstract
    follow the convention that they can only process one example at a time (i.e., data[0] is possible but not data[0:5]) this
    function is also used to simply add multi-indexing to any iterable and provide automatic stacking of that data either into a list or np.ndarray if possible.

    This function is reused multiple times throughout the Dataset class and is key to the lazy/eager processing flow in this framework.

    Consider the following case where you have created a ProcessingChain and you use MapAbstract to have a lazy processor of your data::

        $   processor = ProcessingChain().add(some_function).add(another_function)
        $   lazy_processed_data = MapAbstract(data, processor)

    In this situation you can index lazy_processed_data only one by one. Using DataAbstract, multi-indexing is provided::

        $   lazy_processed_data = DataAbstract(lazy_processed_data)
        $   lazy_processed_data_subset = lazy_processed_data[0:5]

    By default no multi processing is active. You can keep your same workflow as before and add an argument such as the amount of
    workers (default=0) and so on. i.e, ::

        $   lazy_multiprocessed_data = DataAbstract(lazy_processed_data, workers=5)
        $   lazy_multiprocessed_data_subset = lazy_multiprocessed_data[0:5]

    Similarly, you can use it as a Generator::

        $   for example in lazy_multiprocessed_data:
        $       do_something

    This class has like other abstract classes a .get() method, which enables you to provide additional args and kwargs to your
    abstract function and whether or not to return propagated information. More information on that can be read in the docstring
    of that specific method.

    Parameters
    ----------
    data : Iterable
        Iterable object to be parralelise and multi-index
    output_data_type : str ['auto','numpy','list']
        When multi-indexing (e.g., data[0:5]) it could be handy to automatically try to stack these examples into a np.ndarray or a list.
        In case of 'auto' it always tries to stack it in a np.ndarray. If not feasible due to different sizes it will provide a list
        In case of 'np.ndarray' or 'list' it obviously only tries to go for the former or the latter.
    type : str ['threadpool','processpool']
        String to select either 'threadpool' or 'processpool'
    workers : int
        Amount of parallel workers
    buffer_len : int
        The length of the buffer in case of a generator::

            for data in dataset:
                do_something(data)

        This will cue up buffer_len instances of data while do_something() is busy.

    Returns
    -------
    data : Generator
        The generator will return Union[Generator, Tuple[Generator, Dict]]
        When return_info is True, it returns a tuple of the exanoke and a Dictionary containing propagated information
        When return_info is False, it returns the example
    """

    def __init__(
            self,
            data: Iterable,
            output_datatype: str = "auto",
            workers: int = 0,
            buffer_len: int = 3,
            load_memory: bool = False,
            unzip: bool = False,
            zip: bool = False
    ):
        super().__init__(data)
        self._output_datatype = output_datatype
        self._workers = workers
        self._buffer_len = buffer_len
        self._load_memory = load_memory
        self._unzip = unzip
        self._zip = zip

    def __iter__(self) -> Generator:
        super().__iter__()
        return parallel_op(
            self._data,
            workers=self._workers,
            buffer_len=self._buffer_len,
            return_info=False,
        )

    def get(
            self,
            index: Iterable = None,
            return_info: bool = False,
            #workers: int = 0,
            #buffer_len: int = 3,
            return_generator: bool = False,
            verbose: bool = False,
            *args: list,
            **kwargs: Dict
    ) -> Any:
        """
        Parameters
        ----------
        index : Iterable
            Indices to retrieve data from
        return_info : bool
            Return information that has been propagated through a chain of processors and abstract's.
            For example, if one has used WavDataReader from dabstract.dataprocessor this will retrieve you the sampling
            frequency ('fs')
        workers : int
            Amount of workers used for loading the data (default = 1)
        buffer_len : int
            Buffer_len of the pool (default = 3)
        return_generator : bool
            Return generator object with the data if True or return tuple (data, info) if return_info is True
            else return data (default = False)
        verbose : bool
            If True show progress (default = False)
        args : List
            additional param to provide to the function if needed
        kwargs : Dict
            additional param to provide to the function if needed

        Returns
        -------
        data : Any
            When iterating if will return a Generator
            For each sample generator will return Union[Generator, Tuple[Generator, Dict]]
                When return_info is True, it returns a tuple of the data and a Dictionary containing propagated information
                When return_info is False, it returns a Generator
            When indexing the dataset it will return:
                When return_info is True, it returns a tuple of a List or np.ndarray and a Dictionary containing propagated information
                When return_info is False, it returns a List or np.ndarray
        """
        if isinstance(index, numbers.Integral):
            if self._abstract:
                data, info = self._data.get(
                    index, return_info=True, *args, **kwargs
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
                workers=self._workers,
                buffer_len=self._buffer_len,
                return_info=return_info,
                **kwargs,
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
                        # init data_out
                        if k == 0:
                            if return_info:
                                info_out = [dict()] * len(self._data)

                            if self._unzip:
                                assert not self._zip, "Zip and unzip can't both be active."
                                data_out = [None] * len(tmp_data)
                                for j in range(len(tmp_data)):
                                    data_out[j] = self._check_data_type(_data, tmp_data[j])
                            else:
                                data_out = self._check_data_type(_data,tmp_data)

                        # failsafe in not all outputs are numpy's, revert back to list
                        #ToDo: put within the method, as this temporarily doubles mem usage
                        elif self._output_datatype == "auto":
                            if self._unzip:
                                for j in range(len(tmp_data)):
                                    data_out[j] = self._check_numpy_array(data_out[j], tmp_data[j], k)
                            else:
                                data_out = self._check_numpy_array(data_out, tmp_data, k)

                        # save data
                        if self._zip:
                            raise NotImplementedError("zipping not supported yet.")
                        elif self._unzip:
                            for j in range(len(tmp_data)):
                                data_out[j][k] = tmp_data[j]
                        else:
                            data_out[k] = tmp_data
                        if return_info:
                            info_out[k] = tmp_info
                return (data_out, info_out) if return_info else data_out
        elif index is None:
            if return_generator:
                gen = parallel_op(
                    self._data,
                    *args,
                    workers=self._workers,
                    buffer_len=self._buffer_len,
                    return_info=return_info,
                    **kwargs,
                )
                return gen
            else:
                raise NotImplementedError
        elif isinstance(index, str):
            return DataAbstract(KeyAbstract(self, index))
        else:
            raise TypeError(
                "Index should be a number. Note that a str works too as it does not provide any error but it will only return a \n \
                            value which is not None in case it actually contains a key. \n \
                            This is because a SeqAbstract may contain a DictSeqAbstract with a single active key \n \
                            and other data including no keys."
            )

    def _check_numpy_array(self, data_out, tmp_data, k):
        if isinstance(data_out, np.ndarray):
            if (
                    np.squeeze(data_out[0]).shape
                    != np.squeeze(tmp_data).shape
            ):
                tmp_data_out = data_out
                data_out = [None] * len(data_out)
                for j in range(0, k):
                    data_out[j] = tmp_data_out[j]
        return data_out

    def _check_data_type(self, _data, tmp_data):
        if isinstance(
                tmp_data, (np.ndarray)
        ) and self._output_datatype in ("numpy", "auto"):
            #assert not self._unzip and not self._zip, "No (un)zip available for numpy datatype."
            data_out = np.zeros((len(_data),) + tmp_data.shape, tmp_data.dtype)
        elif isinstance(tmp_data, np.datetime64) and \
                self._output_datatype in ("numpy", "auto"):
            #assert not self._unzip and not self._zip, "No (un)zip available for numpy datatype."
            data_out = np.zeros((len(_data), 1), dtype=tmp_data.dtype)
        elif isinstance(tmp_data, (np.int, np.int64, int)) and \
                self._output_datatype in ("numpy", "auto"):
            #assert not self._unzip and not self._zip, "No (un)zip available for numpy datatype."
            data_out = np.zeros((len(_data), 1), dtype=tmp_data.dtype)
        elif self._output_datatype in ("list", "auto"):
            data_out = [None] * len(_data)
        else:
            raise NotImplementedError("datatype of %s and target datatype %s not supported by DataAbstract" % \
                                      (str(tmp_data.__class__),str(self._output_datatype)))
        return data_out

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (
                class_str(self._data)
                + "\n data abstract: multi_processing "
                + str((True if self._workers > 0 else False))
        )


class MapAbstract(base.Abstract):
    """
    The class applies a mapping to data in a lazy manner.

    For example, consider the following function::

        $   def some_function(input, multiplier, logarithm=False)
        $       output = input * multiplier
        $       if logarithm:
        $           output = np.log10(output)
        $       return output

    You can apply this function with multiplier=5 and logarithm=True as follows::

        $   data = [1,2,3]
        $   data_map = MapAbstract(data,map_fct=some_function, 5, logarithm=True)
        $   print(data_map[0])
        0.6989

    Similarly, one could use a lambda function::

        $   data = [1,2,3]
        $   data_map = MapAbstract(data, lambda x: np.log10(x*5))
        $   print(data_map[0])
        0.6989

    Another example is to use the ProcessingChain. This would allow propagation of information.
    For example, assume the following ProcessingChain::

        $   class custom_processor(Processor):
        $       def process(self, data, **kwargs):
        $           return data + 1, {'multiplier': 3}
        $   class custom_processor2(Processor):
        $       def process(self, data, **kwargs):
        $           return data * kwargs['multiplier'], {}
        $   dp = ProcessingChain()
        $   dp.add(custom_processor)
        $   dp.add(custom_processor2)

    And add this to some data with a MapAbstract::

        $   data = [1,2,3]
        $   data_map = MapAbstract(data,map_fct=dp)
        $   print(data_map[0])
        6

    When using a ProcessingChain one can utilise the fact that it propagates the so-called 'info' through lazy operations.
    To obtain the information that has been progated, one can use the .get() method::

        $   print(data_map.get(0, return_info=True)
        (6, {'multiplier': 3, 'output_shape': ()})

    For more information on how to use a ProcessingChain, please check dabstract.dataprocessor.ProcessingChain.

    There are cases when one would like to use a function that has not been defined as a dabstract Processor, but
    where it still is desired to for example propagate information, e.g. sampling frequency.
    One can encapsulate information in a mapping function such as::

        $   data = [1,2,3]
        $   data_map = MapAbstract(data, (lambda x): x, info=({'fs': 16000}, {'fs': 16000}, {'fs': 16000}))
        $   print(data_map[0])
        (1, {'fs': 16000})

    To index through the data one could directly use default indexing, i.e. [idx] or use the .get() method.

    The MapAbstract contains the following methods::

        .get - return entry from MapAbstract
        .keys - return attribute keys of data

    The full explanation for each method is provided as a docstring at each method.

    Parameters
    ----------
    data : Iterable
        Iterable object to be mapped
    map_fct : Callable
        Callable object that defines the mapping
    info : List[Dict]
        List of Dictionary containing information that will be propagated through the chain of operations.
        Useful when the mapping function is not a ProcessingChain
        (default = None)
    arg : list
        additional param to provide to the function if needed
    kwargs : dict
        additional param to provide to the function if needed

    Returns
    ----------
    MapAbstract class
    """

    def __init__(
            self,
            data: Iterable,
            map_fct: Callable,
            info: List[Dict] = None,
            *args: list,
            **kwargs: Dict
    ):
        super().__init__(data)
        assert callable(map_fct), map_fct
        self._map_fct = map_fct
        self._chain = True if isinstance(map_fct, ProcessingChain) else False
        self._info = info
        self._args = args
        self._kwargs = kwargs

    def get(
            self, index: int, return_info: bool = False, *args: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        """

        Parameters
        ----------
        index : int
            index to retrieve data from
        return_info : bool
            return tuple (data, info) if True else data (default = False)
            info contains the information that has been propagated through the chain of operations
        arg : List
            additional param to provide to the function if needed
        kwargs : Dict
            additional param to provide to the function if needed

        Returns
        -------
        List OR np.ndarray OR Any
        """
        if isinstance(index, numbers.Integral):
            if index < 0:
                index = index % len(self)
            if self._abstract:
                data, info = self._data.get(index, *args, return_info=True, **kwargs)
            else:
                data, info = self._data[index], kwargs
            if self._chain:
                data, info = self._map_fct(
                    data, *self._args, **dict(self._kwargs, **info), return_info=True
                )
            else:
                data = self._map_fct(data, *self._args, **dict(self._kwargs, **info))
            if self._info is not None:
                info = dict(info, **self._info[index])
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

    def __len__(self) -> int:
        return len(self._data)

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
    """
    Factory function to allow for choice between lazy and direct mapping.

    For both an instance of MapAbstract is created. Different from lazy mapping, is that with direct mapping all examples
    are immediately evaluated.

    To have more information on mapping, please read the docstring of MapAbstract().

    Parameters
    -------
    data :
        The data that needs to be mapped
    map_fct : Callable
        Callable object that defines the mapping
    info : List[Dict]
        List of Dictionary containing information that has been propagated through the chain of operations
        (default = None)
    lazy : bool
        apply lazily or not (default = True)
    workers : int
        amount of workers used for loading the data (default = 1)
    buffer_len : int
        buffer_len of the pool (default = 3)
    arg : list
        additional param to provide to the function if needed
    kwargs : Dict
        additional param to provide to the function if needed

    Returns
    -------
    MapAbstract OR DataAbstract OR np.ndarray OR list
    """

    if lazy:
        return MapAbstract(data, map_fct, *arg, info=info, **kwargs)
    else:
        return DataAbstract(
            MapAbstract(data, map_fct, *arg, info=info, **kwargs),
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
##
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
#             raise TypeError('Index should be a str or number')
#
#     def __len__(self):
#         return len(self._data) * self._factor
#
#     def __repr__(self):
#         return self._data.__repr__() + "\n replicate: " + str(
#             self._factor) + ' ' + self._type


class SampleReplicateAbstract(base.Abstract):
    """
    Replicate data on sample-by-sample basis.

    Sample replication is based on the parameter 'factor'. This parameter is used to control to replication ratio.
    For example::

        $ data = [1, 2, 3]
        $ data_rep = SampleReplicateAbstract([1, 2, 3], factor = 3)
        $ print([tmp for tmp in data_rep])
        [1, 1, 1, 2, 2, 2, 3, 3, 3]

    The SampleReplicateAbstract contains the following methods::

    .get - return entry form SampleReplicateAbstract
    .keys - return the list of keys

    The full explanation for each method is provided as a docstring at each method.

    Parameters
    -------
    data : Iterable
        input data to replicate on a sample-by-sample basis
    factor : int
        integer used to compute an index for element in data used as sample
    kwargs : Dict
        additional param to provide to the function if needed

    Returns
    -------
    SampleReplicateAbstract class
    """

    def __init__(self, data: Iterable, factor: int, **kwargs: Dict):
        super().__init__(data)
        self._factor = factor
        if isinstance(self._factor, numbers.Integral):
            self._factor = self._factor * np.ones(len(data))

    def get(
            self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        """
        Parameters
        ----------
        index : int
            index to sample from data
        return_info : bool
            return tuple (data, info) if True else data (default = False)
        arg : List
            additional param to provide to the function if needed
        kwargs : Dict
            additional param to provide to the function if needed

        Returns
        -------
        List OR np.ndarray OR Any
        """
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
            raise TypeError("Index should be a str or number")

    def __len__(self) -> int:
        return int(np.sum(self._factor))

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
    """
    Factory function to allow for choice between lazy and direct sample replication.

    For both an instance of SampleReplicateAbstract is created. Different from sample replication, is that with direct
    sample replication all examples are immediately evaluated.

    To have more information on sample replication, please read the docstring of SampleReplicateAbstract().

    Parameters
    -------
    data : Iterable
        input data to perform sample replication on
    factor : int
        integer used to compute an index for element in data used as sample
    lazy : bool
        apply lazily or not (default = True)
    workers : int
        amount of workers used for loading the data (default = 1)
    buffer_len : int
        buffer_len of the pool (default = 3)
    arg : List
        additional param to provide to the function if needed
    kwargs : Dict
        additional param to provide to the function if needed

    Returns
    -------
    SampleReplicateAbstract OR DataAbstract OR np.ndarray OR list
    """
    if lazy:
        return SampleReplicateAbstract(data, factor, *arg, **kwargs)
    else:
        # ToDo: replace by a list and np equivalent
        if isinstance(data, np.ndarray):
            # faster implementation suited for np.ndarrays
            return np.repeat(data, factor)
        elif isinstance(data, list):
            # faster implementation suited for lists
            return list(
                itertools.chain(*[[tmp_data for k in range(tmp_fact)] for tmp_fact, tmp_data in zip(factor, data)]))
        else:
            # slowest implementation for other iterable datatypes
            return DataAbstract(
                SampleReplicateAbstract(data, factor, *arg, **kwargs),
                workers=workers,
                buffer_len=buffer_len,
            )[:]


class SplitAbstract(base.Abstract):
    """
    The class is an abstract wrapper around an iterable to split this iterable in a lazy manner. Splitting refers
    to dividing the a particular example in multiple chunks, i.e. 60s examples are divided into 1s segments.

    Splitting is based on the parameters split_size, constraint, sample_len, sample_period and type.

    If type is set to 'samples' one has to define 'sample_len' and 'split_size'. In that case 'sample_len' refers to
    the amount of samples in one example, and split_size the size of one segment. 'sample_len' can be set as an integer
    if all examples are of the same size OR a list of integers if these are different between examples.

    If type is set to 'seconds' one has to define 'sample_len', 'split_size' and 'sample_period'. In this case each of
    these variables are not samples but defined in terms of seconds. 'sample_period' additionally specifies the sample period
    of these samples in order to properly split.

    The SplitAbstract contains the following methods::

        .get - return entry from SplitAbstract
        .keys - return attribute keys of data

    The full explanation for each method is provided as a docstring at each method.

    Parameters
    ----------
    data : Iterable
        Iterable object to be splitted
    split_size : int
        split size in seconds/samples depending on 'metric'
    constraint : str
        option 'power2' creates sizes with a order of 2 (used for autoencoders)
    sample_len : int or List[int]
        sample length (default = None)
    sample_period : int
        sample period (default = None)
    type : str
        split_size type ('seconds','samples') (default = 'seconds')

    Returns
    -------
    SplitAbstract class
    """

    def __init__(
            self,
            data: Iterable,
            split_len: int = None,
            sample_len: Union[int, List[int]] = None,
    ):
        super().__init__(data)
        self._split_len = split_len
        self._sample_len = sample_len
        if isinstance(self._split_len, numbers.Integral):
            self._split_len = self._split_len * np.ones(len(data))
        if isinstance(self._sample_len, numbers.Integral):
            self._sample_len = self._sample_len * np.ones(len(data))
        self._init_split()

    def get_param(self):
        return {'split_len': self._split_len,
                'sample_len': self._sample_len,
                'splits': self._splits}

    def _init_split(self):
        assert all(self._split_len > 0)
        # prepare splits
        self._split_range, self._splits = [None] * len(self._data), np.zeros(
            len(self._data), dtype=int
        )
        for j in range(len(self._data)):
            num_frames = max(
                1,
                int(
                    np.floor(
                        (
                                (self._sample_len[j] - (self._split_len[j] - 1) - 1)
                                / self._split_len[j]
                        )
                        + 1
                    )
                ),
            )
            self._split_range[j] = np.tile(
                np.array([0, self._split_len[j]]), (num_frames, 1)
            ) + np.tile(
                np.transpose(np.array([np.arange(num_frames) * self._split_len[j]])),
                (1, 2),
            )
            self._splits[j] = num_frames
        lol = 0

    def get(
            self, index: int, return_info: bool = False, *args: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        """
        Parameters
        ----------
        index : int
            index to retrieve data from
        return_info : bool
            return tuple (data, info) if True else data (default = False)
            info contains the information that has been propagated through the chain of operations
        arg : List
            additional param to provide to the function if needed
        kwargs : Dict
            additional param to provide to the function if needed
        Returns
        -------
        List OR np.ndarray OR Any
        """
        if isinstance(index, numbers.Integral):
            assert index < len(self)
            if index < 0:
                index = index % len(self)
            for k, split_len in enumerate(self._splits):
                if split_len <= index:
                    index -= split_len
                else:
                    read_range = self._split_range[k][int(index)]
                    # get data
                    if self._abstract:
                        splittable = self._data.is_splittable()
                        data, info = self._data.get(
                            k,
                            *args,
                            return_info=True,
                            **kwargs,
                            **({'read_range': read_range} if splittable else {})
                        )
                        if not splittable:
                            try:
                                data = data[int(read_range[0]): int(read_range[1]) + 1]
                            except:
                                raise ValueError("data is not splittable.")
                    else:
                        data, info = self._data[k][int(read_range[0]): int(read_range[1]) + 1], {}
                    return (data, info) if return_info else data
        elif isinstance(index, str):
            return KeyAbstract(self, index)
        else:
            raise TypeError("Index should be a str or number")

    def __len__(self) -> int:
        return int(np.sum(self._splits))

    def __repr__(self):
        split_ratios = self._split_len / self._sample_len
        return "%s\n\tsplit: %.2f (mean relative length), %d to %d" % (self._data.__repr__(),
                                                                       split_ratios.mean(),
                                                                       len(self._data),
                                                                       len(self))


def Split(
        data: Iterable,
        split_len: float = None,
        sample_len: float = None,
        lazy: bool = True,
        workers: bool = 1,
        buffer_len: int = 3
) -> Union[SplitAbstract, DataAbstract, np.ndarray, list]:
    """
    Factory function to allow for choice between lazy and direct example splitting.

    For both an instance of SplitAbstract is created. Different from lazy splitting, is that with direct splitting
    all examples are immediately evaluated.

    To have more information on splitting, please read the docstring of SplitAbstract().

    Parameters
    ----------
    data : Iterable
        Iterable object to be splitted
    split_size : int
        split size in seconds/samples depending on 'metric'
    constraint : str
        option 'power2' creates sizes with a order of 2 (used for autoencoders)
    sample_len : int
        sample length (default = None)
    sample_period : int
        sample period (default = None)
    type : str
        split_size type ('seconds','samples') (default = 'seconds')
    lazy : bool
        apply lazily or not (default = True)
    workers : int
        amount of workers used for loading the data (default = 1)
    buffer_len : int
        buffer_len of the pool (default = 3)
    arg : List
        additional param to provide to the function if needed
    kwargs : Dict
        additional param to provide to the function if needed

    Returns
    -------
    SplitAbstract OR DataAbstract OR np.ndarray OR list
    """
    if lazy:
        return SplitAbstract(
            data,
            split_len=split_len,
            sample_len=sample_len,
        )
    else:
        # ToDo: replace by a list and np equivalent
        return DataAbstract(
            SplitAbstract(
                data,
                split_len=split_len,
                sample_len=sample_len,
            ),
            workers=workers,
            buffer_len=buffer_len,
        )[:]


class SelectAbstract(base.Abstract):
    """
    Select a subset of your input sequence.

    Selection is based on a so called 'selector' which may have the form of a Callable or a list/np.ndarray of integers.
    Important for these Callables is that they accept two arguments: (1) data to base selection on and (2) index of the
    variable to be evaluated.

    Regarding the selector one can use  set of build-in selectors in dabstract.dataset.select, lambda function, an own custom function
    or indices. For example:

    1) random subsampling with::

        $  SelectAbstract(data, dabstract.dataset.select.random_subsample('ratio': 0.5))

    2) select based on a key and a particular value::

        $  SelectAbstract(data, dabstract.dataset.select.subsample_by_str('ratio': 0.5))

    3) use the lambda function such as::

        $  SelectAbstract(data, (lambda x,k: x['data']['subdb'][k]))

    4) directly use indices::

        $  indices = np.array[0,1,2,3,4])
        $  SelectAbstract(data, indices)

    If no 'eval_data' is used, the evaluation is performed on data available in 'data'. If 'eval_data' is available
    the evaluation is performed on 'eval_data'

    The SelectAbstract contains the following methods::

        .get - return entry from SelectAbstract
        .keys - return the list of keys

    The full explanation for each method is provided as a docstring at each method.

    Parameters
    ----------
    data : Iterable
        input data to perform selection on, if eval_data is None
    selector : List[int] OR Callable OR numbers.Integral
        selection criterium
    eval_data : Any
        if eval_data not None, then selection will be performed on eval_data, else data (default = None)
    kwargs : Dict
        additional param to provide to the function if needed

    Returns
    -------
    SelectAbstract class
    """

    def __init__(
            self,
            data: Iterable,
            selector: Union[List[int], Callable, numbers.Integral],
            eval_data: Any = None,
            *args,
            **kwargs: Dict
    ):
        super().__init__(data)
        assert hasattr(self._data, "__len__"), (
                "Can only use %s it object has __len__" % self.__class__.__name__
        )
        self._eval_data = data if eval_data is None else eval_data
        self._selector = selector
        self.set_indices(selector, *args, **kwargs)

    def set_indices(self, selector, *args, **kwargs):
        if callable(selector):
            if len(inspect.getfullargspec(selector).args) == 1:
                self._indices = selector(self._eval_data, *args, **kwargs)
            elif len(inspect.getfullargspec(selector).args) == 2:
                self._indices = np.where(
                    [
                        selector(self._eval_data, k, *args, **kwargs)
                        for k in range(len(self._eval_data))
                    ]
                )[0]
            else:
                raise NotImplementedError(
                    "Selector not supported. Please consult the docstring for options."
                )
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

    def get_indices(self):
        return self._indices

    def get(
            self, index: int, return_info: bool = False, *args: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        """
        Parameters
        ----------
        index : int
            index to retrieve data from
        return_info : bool
            return tuple (data, info) if True else data (default = False)
        arg : List
            additional param to provide to the function if needed
        kwargs : Dict
            additional param to provide to the function if needed

        Returns
        -------
        List OR np.ndarray OR Any
        """
        if isinstance(index, numbers.Integral):
            assert index < len(self)
            index = self._indices[index]
            if self._abstract:
                data, info = self._data.get(index, *args, return_info=True, **kwargs)
            else:
                data, info = self._data[index], {}
            return (data, info) if return_info else data
        elif isinstance(index, str):
            return SelectAbstract(self._data[index], self._indices)
            # return KeyAbstract(self, index)
        else:
            raise TypeError("Index should be a str or number")

    def __len__(self) -> int:
        return len(self._indices)

    def __repr__(self) -> str:
        return self._data.__repr__() + "\n select: " + str(type(self._selector))


def Select(
        data,
        selector: Union[List[int], Callable, numbers.Integral],
        eval_data: Any = None,
        lazy: bool = True,
        workers: int = 1,
        buffer_len: int = 3,
        *args: List,
        **kwargs: Dict
) -> Union[SelectAbstract, DataAbstract, np.ndarray, list]:
    """
    Factory function to allow for choice between lazy and direct example selection.

    For both an instance of SelectAbstract is created. Different from lazy selecting, is that with direct selecting
    all examples are immediately evaluated.

    For more information on the functionality of Select please check the docstring of SelectAbstract().

    Parameters
    ----------
    data : Iterable
        input data to perform selection on, if eval_data is None
    selector : List[int] OR Callable OR numbers.Integral
        selection criterium
    eval_data : Any
        if eval_data not None, then selection will be performed on eval_data, else data (default = None)
    lazy : bool
        apply lazily or not (default = True)
    workers : int
        amount of workers used for loading the data (default = 1)
    buffer_len : int
        buffer_len of the pool (default = 3)
    arg/kwargs:
        additional param to provide to the function if needed

    Returns
    -------
    SelectAbstract OR DataAbstract OR np.ndarray OR list
    """
    if lazy:
        return SelectAbstract(data, selector, *args, eval_data=eval_data, **kwargs)
    else:
        # ToDo: replace by a list and np equivalent
        return DataAbstract(
            SelectAbstract(data, selector, *args, eval_data=eval_data, **kwargs),
            workers=workers,
            buffer_len=buffer_len,
        )[:]


class FilterAbstract(base.Abstract):
    """
    Filter on the fly. Interesting when the variable to filter on takes long to compute.

    When the FilterAbstract wrapper is applied, the length of your data is undefined as filtering is based on a net yet
    excecuted function 'filter_fct'.

    The FilterAbstract class contain the following methods
    ::
    .get - return entry from FilterAbstract
    .keys - show the set of keys

    The full explanation for each method is provided as a docstring at each method.

    Parameters
    ----------
    data : Iterable
        Iterable object to be filtered
    filter_fct : Callable
        Callable function that needs to be applied
    return_none : bool
        If True, return None if filter_fct is False
        If False, raises IndexError
    kwargs:
        additional param to provide to the function if needed

    Returns
    -------
    FilterAbstract class
    """

    def __init__(
            self,
            data: Iterable,
            filter_fct: Callable,
            return_none: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(data)
        assert callable(filter_fct), filter_fct
        self._filter_fct = filter_fct
        self._return_none = return_none
        self._args = args
        self._kwargs = kwargs

    def __iter__(self) -> Generator:
        for data in self._data:
            if self._filter_fct(data, *self._args, **self._kwargs):
                yield data

    def get(
            self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        """
        Parameters
        ----------
        index : int
            index to retrieve data from
        return_info : bool
            return tuple (data, info) if True else data (default = False)
        arg : List
            additional param to provide to the function if needed
        kwargs : Dict
            additional param to provide to the function if needed

        Returns
        -------
        List OR np.ndarray OR Any
        """
        if isinstance(index, numbers.Integral):
            assert index < len(self._data)
            if self._abstract:
                data, info = self._data.get(
                    index, return_info=True, *self._args, **self._kwargs
                )
            else:
                data, info = self._data[index], {}

            if self._filter_fct(data):
                return (data, info) if return_info else data
            elif not self._return_none:
                raise IndexError("Not available.")
            return None, info

        elif isinstance(index, str):
            return KeyAbstract(self, index)
        else:
            raise TypeError("Index should be a str or number")

    @property
    def len_defined(self):
        return self._return_none

    def __len__(self) -> int:
        if self.len_defined:
            return len(self._data)
        else:
            raise NotImplementedError("Length undefined when return_none is False")

    def __repr__(self) -> str:
        return self._data.__repr__() + "\n filter: " + str(type(self._filter_fct))


def Filter(
        data: Iterable,
        filter_fct: Callable,
        return_none: bool = True,
        lazy: bool = True,
        workers: int = 1,
        buffer_len: int = 3,
        *arg: List,
        **kwargs: Dict
) -> Union[FilterAbstract, DataAbstract, np.ndarray, List]:
    """
    Factory function to allow for choice between lazy and direct example selection.

    For both an instance of FilterAbstract is created. Different from lazy filtering, is that with direct filtering
    all examples are immediately evaluated.

    For more information on the functionality of Filter please check the docstring of FilterAbstract().

    Parameters
    ----------
    data : Iterable
        Iterable object to be filtered
    filter_fct : Callable
        Callable function that needs to be applied.
    return_none : bool
        If True, return None if filter_fct is False
        If False, raises IndexError
    lazy : bool
        apply lazily or not (default = True)
    workers : int
        amount of workers used for loading the data (default = 1)
    buffer_len : int
        buffer_len of the pool (default = 3)
    arg : List
        additional param to provide to the function if needed
    kwargs : Dict
        additional param to provide to the function if needed

    Returns
    -------
    FilterAbstract OR DataAbstract OR np.ndarray OR list
    """
    if lazy:
        return FilterAbstract(data, filter_fct, *arg, return_none=return_none, **kwargs)
    else:
        # ToDo: replace by a list and np equivalent
        tmp = DataAbstract(
            FilterAbstract(data, filter_fct, *arg, return_none=True, **kwargs),
            output_datatype="list",
            workers=workers,
            buffer_len=buffer_len,
        )[:]
        if return_none:
            return tmp
        else:
            return DataAbstract(
                SelectAbstract(tmp, lambda x, k: x[k] is not None),
                workers=workers,
                buffer_len=buffer_len,
            )[:]


class KeyAbstract(base.Abstract):
    """Error handling wrapper for a concatenated sequence where one might have a dictseq and the other doesnt.
    This will allow for key/index indexing even if the particular index does not have a key.
    """

    def __init__(self, data: Iterable, key: str):
        super().__init__(data)
        self._key = key

    def get(
            self, index: int, return_info: bool = False, *arg: List, **kwargs: Dict
    ) -> Union[List, np.ndarray, Any]:
        if isinstance(index, numbers.Integral):
            assert index < len(self)
            try:
                data, info = self._data.get(
                    key=self._key,
                    index=index,
                    *arg,
                    return_info=True,
                    **kwargs,
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

    def __repr__(self) -> str:
        return "key_abstract of key " + self._key + " on " + str(self._data)


def class_str(data: Callable):
    if isinstance(data, base.Abstract):
        return repr(data)
    else:
        return str(data.__class__)
    return
