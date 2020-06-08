import numbers
import copy
import numpy as np
from tqdm import tqdm
import inspect

from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from dabstract.utils import intersection
from dabstract.dataprocessor import processing_chain

class abstract():
    pass

def class_str(data):
    if isinstance(data,abstract):
        return repr(data)
    else:
        return str(data.__class__)
    return

def GeneratorAbstract(data, *args, multi_processing=False, workers=2, buffer_len=3, return_info=False, **kwargs):
    # define function to evaluate
    if isinstance(data, abstract):
        def func(index):
            return data.get(index, *args, return_info=return_info, **kwargs)
    else:
        def func(index):
            return data[index]
    # create generator
    if multi_processing:
        Q = Queue()
        with ThreadPoolExecutor(workers) as E:
            for k in range(len(data)):
                if Q.qsize() >= buffer_len:
                    yield Q.get().result()
                Q.put(E.submit(func, k))
            while not Q.empty():
                yield Q.get().result()
    else:
        for k in range(len(data)):
            yield func(k)

class DataAbstract(abstract):
    def __init__(self, data, multi_processing=False, workers=2, buffer_len=3, load_memory=False, **kwargs):
        self._data = data
        self._abstract = (True if isinstance(data, abstract) else False)
        self._multi_processing = multi_processing
        self._workers = workers
        self._buffer_len = buffer_len
        self._load_memory = load_memory
        self._kwargs = kwargs

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index):
        return self.get(index)

    def __iter__(self):
        return GeneratorAbstract(self._data, multi_processing=self._multi_processing, \
                                 workers=self._workers, buffer_len=self._buffer_len, return_info=False, **self._kwargs)

    def get(self, index, *args, return_info=False, multi_processing=False, workers=2, buffer_len=3, return_generator=False, verbose=False, **kwargs):
        if isinstance(index, numbers.Integral):
            if self._abstract:
                data, info = self._data.get(index, return_info=True, *args,**kwargs,**self._kwargs)
            else:
                data, info = self._data[index], {}
            return ((data, info) if return_info else data)
        elif isinstance(index, (tuple, list, np.ndarray, slice)):
            # generator
            _data = SelectAbstract(self._data,index)
            gen = GeneratorAbstract(_data, \
                                    *args, multi_processing=multi_processing, workers=workers, buffer_len=buffer_len, \
                                    return_info=return_info, **kwargs, **self._kwargs)
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
                        return ((tmp_data, tmp_info) if return_info else tmp_data)
                    else:
                        if k == 0:
                            if return_info: info_out = [dict()] * len(self._data)
                            if isinstance(tmp_data, (np.ndarray)):
                                data_out = np.zeros((len(_data),) + tmp_data.shape)
                            elif isinstance(tmp_data, (np.int, np.int64, int)):
                                data_out = np.zeros((len(_data), 1))
                            else:
                                data_out = [None] * len(_data)
                        data_out[k] = tmp_data
                        if return_info: info_out[k] = tmp_info
                return ((data_out, info_out) if return_info else data_out)
        elif isinstance(index,str):
            return DataAbstract(KeyAbstract(self,index))
        else:
            raise TypeError('Index should be a number. Note that a str works too as it does not provide any error but it will only a \n \
                            value which is not None in case a it actually contains a key. \n \
                            This is because a SeqAbstract may contain a DictSeqAbstract with a single active key \n \
                            and other data including no keys.')
    def __len__(self):
        return len(self._data)

    def keys(self):
        if hasattr(self._data,'keys'):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self):
        return class_str(self._data) + "\n data abstract: multi_processing " + str(self._multi_processing)

class MapAbstract(abstract):
    def __init__(self, data, map_fct, *arg, **kwargs):
        assert callable(map_fct), map_fct
        self._map_fct = map_fct
        self._data = data
        self._chain = (True if isinstance(map_fct, processing_chain) else False)
        self._abstract = (True if isinstance(data, abstract) else False)
        self._kwargs = kwargs
        self._args = arg

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, return_info=False, *arg, **kwargs):
        if isinstance(index, numbers.Integral):
            if self._abstract:
                data, info = self._data.get(index, return_info=True,*arg,**kwargs)
            else:
                data, info = self._data[index], kwargs
            if self._chain:
                data, info = self._map_fct(data, *self._args, **dict(self._kwargs, **info), return_info=True)
            else:
                data = self._map_fct(data)
            return ((data, info) if return_info else data)
        elif isinstance(index,str):
            return KeyAbstract(self,index)
        else:
            raise TypeError('Index should be a number. Note that a str works too as it does not provide any error but it will only a \
                            value which is not None in case a it actually contains a key. \
                            This is because a SeqAbstract may contain a DictSeqAbstract with a single active key \
                            and other data including no keys.')
            #ToDo(gert) add a way to raise a error in case data does not contain any key.

    def __len__(self):
        return len(self._data)

    def keys(self):
        if hasattr(self._data,'keys'):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self):
        return class_str(self._data) + "\n map: " + str(self._map_fct)

class SplitAbstract(abstract):
    def __init__(self, data, split_size=None, constraint=None,
                 sample_len=None, sample_period=None, type='seconds', **kwargs):
        self._data = data
        assert split_size is not None, "Please provide a split in " + type
        self._type = type
        self._split_size = split_size
        self._constraint = constraint
        self._sample_len = sample_len
        if isinstance(self._sample_len,numbers.Integral):
            self._sample_len = self._sample_len * np.ones(len(data))
        self._sample_period = sample_period
        self._abstract = (True if isinstance(data, abstract) else False)
        self._kwargs = kwargs
        self._init_split()

    def _init_split(self):
        # init window_size
        if self._type == 'seconds':
            self._window_size = int(self._split_size / self._sample_period)
        elif self._type == 'samples':
            self._window_size = int(self._split_size)
        if self._constraint == 'power2':
            self._window_size = int(2 ** np.ceil(np.log2(self._window_size)))
        # prepare splits
        self._split_range, self._split_len = [None] * len(self._data), np.zeros(len(self._data),dtype=int)
        for j in range(len(self._data)):
            num_frames = max(1, int(np.floor(((self._sample_len[j] - (self._window_size - 1) - 1) / self._window_size) + 1)))
            self._split_range[j] =  np.tile(np.array([0, self._window_size]), (num_frames, 1)) + \
                                    np.tile(np.transpose(np.array([np.arange(num_frames) * self._window_size])), (1, 2))
            self._split_len[j] = num_frames

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, return_info=False, *arg, **kwargs):
        if isinstance(index, numbers.Integral):
            assert index<len(self)
            if index < 0:
                index = index % len(self)
            for k,split_len in enumerate(self._split_len):
                if split_len <= index:
                    index -= split_len
                else:
                    range = self._split_range[k][int(index)]
                    # get data
                    if self._abstract:
                        data,info = self._data.get(k,return_info=True,*arg,**kwargs,**self._kwargs, range=range)
                    else:
                        data,info = self._data[k], {}
                    # apply range if not already done in
                    #ToDo(gert): add check for WavDatareader and NumpyDatareader, maybe check for len()?
                    data = data[range[0]:range[1]]

            return ((data, info) if return_info else data)
        elif isinstance(index,str):
            return KeyAbstract(self,index)
        else:
            raise TypeError('Index should be a str our number')

    def __len__(self):
        return int(np.sum(self._split_len))

    def keys(self):
        if hasattr(self._data,'keys'):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self):
        return self._data.__repr__() + "\n split: " + str(self._window_size*self._sample_period) + ' ' + self._type

class SelectAbstract(abstract):
    def __init__(self, data, selector, **kwargs):
        self._data = data
        self._selector = selector
        if callable(selector):
            if len(inspect.getargspec(selector)[0])==1:
                self._indices = selector(data)
            else:
                self._indices = np.where([selector(self._data,k) for k in range(len(self._data))])[0]
        elif isinstance(selector,slice):
            self._indices = np.arange(  (0 if selector.start is None else selector.start), \
                                        (len(self._data) if selector.stop is None else selector.stop), \
                                        (1 if selector.step is None else selector.step))
        elif isinstance(selector,(tuple,list,np.ndarray)):
            self._indices = selector
        elif isinstance(selector,numbers.Integral):
            self._indices = [selector]
        self._abstract = (True if isinstance(data, abstract) else False)
        self._kwargs = kwargs

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, return_info=False, *arg, **kwargs):
        if isinstance(index, numbers.Integral):
            assert index<len(self)
            index = self._indices[index]
            if self._abstract:
                data,info = self._data.get(index,return_info=True,*arg,**kwargs,**self._kwargs)
            else:
                data,info = self._data[index], {}
            return ((data, info) if return_info else data)
        elif isinstance(index,str):
            return KeyAbstract(self,index)
        else:
            raise TypeError('Index should be a str our number')

    def __len__(self):
        return len(self._indices)

    def keys(self):
        if hasattr(self._data,'keys'):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self):
        return self._data.__repr__() + "\n select: " + str(type(self._selector))

class FilterAbstract(abstract):
    def __init__(self, abstract, filter_fct, **kwargs):
        assert callable(filter_fct), filter_fct
        self.filter_fct = filter_fct
        self.abstract = abstract
        self._kwargs = kwargs

    def __iter__(self):
        for index in range(len(self)):
            data = self.get(index)
            if self.filter_fct(data):
                yield data

    def __getitem__(self, index):
        data = self.get(index)
        if self.filter_function(data):
            return data
        raise IndexError('Not available.')

    def get(self, index, return_info=False, *arg, **kwargs):
        if isinstance(index, numbers.Integral):
            assert index<len(self)
            if self._abstract:
                data,info = self._data.get(index,return_info=True,*arg,**kwargs,**self._kwargs)
            else:
                data,info = self._data[index], {}
            return ((data, info) if return_info else data)
        elif isinstance(index,str):
            return KeyAbstract(self,index)
        else:
            raise TypeError('Index should be a str our number')

    def __len__(self):
        raise Exception('Length not available as filter is evaluated on the fly')

    def keys(self):
        if hasattr(self._data,'keys'):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self):
        return self._data.__repr__() + "\n filter: " + str(type(self.filter_function))

class KeyAbstract(abstract):
    def __init__(self, data, key, **kwargs):
        assert isinstance(data,abstract)
        self._data = data
        self._key = key
        self._kwargs = kwargs

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, return_info=False, *arg, **kwargs):
        assert index<len(self)
        try:
            data, info = self._data.get(key=self._key, index=index, return_info=True, *arg, **kwargs, **self._kwargs)
        except:
            data, info = None, {}
        return ((data, info) if return_info else data)

    def __len__(self):
        return len(self._data)

    def keys(self):
        if hasattr(self._data,'keys'):
            return self._data.keys()
        else:
            return self._data._data.keys()

    def __repr__(self):
        return 'key_abstract of key ' + self._key + ' on ' + str(self._data)

class DictSeqAbstract(abstract):
    def __init__(self, name=''):
        self._nr_keys = 0
        self._name = name
        self._data = dict()
        self._active_keys = None

    def add(self, key, data,  **kwargs):
        assert hasattr(data, '__getitem__'), "provided data instance must have __getitem__ method."
        assert key != 'all', "The name 'all' is reserved for referring to all keys when applying a transform."
        if self._nr_keys>0: assert len(data)==len(self), "len(self) is not the same as len(data)"
        if not isinstance(data,abstract):
            data = SeqAbstract().concat(data)
        self._data.update({key: data})
        self._nr_keys += 1
        self.__len__()
        self.reset_active_key()
        return self

    def add_dict(self,dct):
        for key in dct:
            self.add(key,dct[key])
        return self

    def concat(self, data, intersect=False):
        if isinstance(data,list):
            for d in data:
                self.concat(d,intersect=intersect)
        else:
            data = copy.deepcopy(data)
            assert isinstance(data, DictSeqAbstract)
            if self._nr_keys != 0:
                if not intersect:
                    assert data.keys() == self.keys(), "keys do not match. Set intersect=True for keeping common keys."
                    keys = enumerate(data.keys())
                else:
                    keys = intersection(data.keys(), self.keys())
                for k, key in keys:
                    if not isinstance(self._data[key], SeqAbstract):
                        self._data[key] = SeqAbstract().concat(self._data[key])
                    if not isinstance(data[key], SeqAbstract):
                        self._data[key].concat(data[key])
                    else:
                        for _data in data[key]._data:
                            self._data[key].concat(_data)
            else:
                self.__dict__.update(data.__dict__)
            return self

    def remove(self,key):
        del self._data[key]
        self._nr_keys -= 1
        return self

    def add_map(self,key, map_fct, *arg, **kwargs):
        assert key in self.keys()
        self[key] = MapAbstract(self[key], map_fct, *arg, **kwargs)

    def add_select(self,map_fct, *arg, **kwargs):
        for key in self.keys():
            self[key] = SeqAbstract().add(SelectAbstract(self[key], map_fct, *arg, **kwargs))

    def set_active_keys(self,keys):
        if isinstance(keys,list):
            for key in keys:
                assert key in self.keys(), "key " + key + " does not exists."
            self._active_keys = keys
        else:
            assert keys in self.keys(), "key " + keys + " does not exists."
            self._active_keys = [keys]

    def reset_active_key(self):
        self._active_keys = self.keys()

    def __len__(self):
        nr_examples = [len(self._data[key]) for key in self._data]
        assert all([nr_example == nr_examples[0] for nr_example in nr_examples])
        return nr_examples[0]

    def __getitem__(self, index):
        return self.get(index)

    def __add__(self, other):
        assert isinstance(other, DictSeqAbstract)
        return self.concat(other)

    def __setitem__(self, k, v):
        assert k in self.keys()
        self._data[k] = v

    def get(self, index, key=None, return_info=False, verbose=False, **kwargs):
        if isinstance(index, str):
            assert key is None
            return self._data[index]
        elif isinstance(index,numbers.Integral):
            if key is None:
                data, info = dict(), dict()
                for k, key in enumerate(self._active_keys):
                    data[key], info[key] = self._data[key].get(index=index, return_info=True,**kwargs)
                if len(self._active_keys)==1:
                    data, info = data[key], info[key]
            else:
                assert isinstance(key,str)
                data, info = self._data[key].get(index=index, return_info=True, **kwargs)
            return ((data, info) if return_info else data)
        else:
            raise IndexError('index should be a number or str')

    def keys(self):
        return list(self._data.keys())

    def summary(self):
        summary = dict()
        for name, data in zip(self.keys(), self._data):
            summary[name] = data.summary()
        return summary

    def __repr__(self):
        return 'dict_seq containing: ' + str(self.keys())

class SeqAbstract(abstract):
    def __init__(self, name='seq'):
        self._nr_sources = 0
        self._data = []
        self._info = []
        self._kwargs = []
        self._name = name

    def concat(self, data, info=None, **kwargs):
        # Check
        assert hasattr(data, '__getitem__'), "provided data instance must have __getitem__ method."
        if isinstance(data,DictSeqAbstract):
            assert len(data._active_keys)==1, "You can only add a dict_abstract in case there is only one active key."
        # Add
        data = copy.deepcopy(data)
        if isinstance(data,SeqAbstract):
            for _data in data._data:
                self.concat(_data)
        else:
            self._data.append(data)
        self._nr_sources += 1
        # Information to propagate to transforms or use for split
        if info is not None:
            assert isinstance(info,list), "info should be a list"
            assert isinstance(info[0],dict()), "The items in info should contain a dict()"
            assert len(info)==self.nr_examples[-1], "info should be a list with len(info)==len(data)"
        self._info.append(info)
        self._kwargs.append(kwargs)
        return self

    def __len__(self):
        return np.sum([len(data) for data in self._data])

    def __getitem__(self, index):
        return self.get(index)

    def __add__(self, other):
        assert isinstance(other)
        return self.concat(other)

    def get(self, index, key=None, return_info=False, **kwargs):
        if isinstance(index,numbers.Integral):
            if index < 0:
                index = index % len(self)
            for k,data in enumerate(self._data):
                if len(data) <= index:
                    index -= len(data)
                else:
                    info = (dict() if self._info[k] is None else self._info[k][index])
                    # get
                    if isinstance(self._data[k],abstract):
                        data, info = data.get(index, key=key, return_info=True, **info, **kwargs)
                    else:
                        assert key is None
                        data = data[index]
                    # return
                    return ((data, info) if return_info else data)
            raise IndexError('Index should be lower than len(dataset)')
        elif isinstance(index,str):
            return KeyAbstract(self, index)
        else:
            raise IndexError('index should be a number (or key in case of a nested dict_seq).')

    def summary(self):
        return {'nr_examples': self.nr_examples, 'name': self._name}

    def __repr__(self):
        r = 'seq containing:'
        for data in self._data:
            if not isinstance(data,(abstract)):
                r += '\n[ \t' + str(type(data)) + '\t]'
            else:
                r += '\n[ \t' + repr(data) + '\t]'
            #r += '\n'
        return r