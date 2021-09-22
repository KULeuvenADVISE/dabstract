import numbers
import copy
import numpy as np
import os
import warnings

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
)

tvDictSeqAbstract = TypeVar("DictSeqAbstract")
tvSeqAbstract = TypeVar("SeqAbstract")

import abc

from dabstract.abstract import base as base
from dabstract.abstract import operators as ops
from dabstract.utils import list_intersection, list_difference, safe_len


class ContainerAbstract(base.Abstract):
    def __init__(self, allow_dive: bool = False, allow_nested: bool = False):
        self._allow_dive = allow_dive
        self._allow_nested = allow_nested
        self._name = self.__class__.__name__
        self._group = None

    @abc.abstractmethod
    def concat(self):
        raise NotImplementedError

    @property
    def allow_dive(self):
        return self._allow_dive

    @allow_dive.setter
    def allow_dive(self, value: bool):
        self._allow_dive = value

    @property
    def allow_nested(self):
        return self._allow_nested

    @allow_nested.setter
    def allow_nested(self, value: bool):
        self._allow_nested = value

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group: str):
        self._group = group
        for key in self.keys():
            if self.is_abstract()[key]:
                self[key].group = group

    @property
    def adjust_mode(self):
        return self._adjust_mode

    @adjust_mode.setter
    def adjust_mode(self, value: bool):
        self._adjust_mode = value
        for key in self.keys():
            if isinstance(self[key], ContainerAbstract):
                self[key].adjust_mode = value

    @abc.abstractmethod
    def is_abstract(self, key):
        raise NotImplementedError


class DictSeqAbstract(ContainerAbstract):
    """DictSeq base class"""

    def __init__(self, allow_dive: bool = False, allow_nested: bool = False):
        super().__init__(allow_dive=allow_dive, allow_nested=allow_nested)
        self._nr_keys = 0
        self._data = dict()
        self._active_keys = []
        self._lazy = dict()
        self._abstract = dict()
        self._adjust_mode = False
        self._allowed_keys = None
        self._restricted_keys = None
        self.set_data()

    def set_data(self):
        pass

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
        assert (
                "." not in key
        ), "A dot (.) is reserved for to support a dotted path. Please select a name that does not contain a dot."
        assert hasattr(data, "__len__"), (
                "Can only use %s it object has __len__" % self.__class__.__name__
        )
        if not self.adjust_mode:
            if self._allowed_keys is not None:
                assert key in self._allowed_keys, (
                        "Key %s is not in the list of allowed keys: %s" % (key, str(self._allowed_keys))
                )
            if self._restricted_keys is not None:
                assert key not in self._restricted_keys, (
                        "Key %s is in the list of restricted keys: %s" % (key, str(self._restricted_keys))
                )
            if self._nr_keys > 0 and len(self) > 0:
                assert len(data) == len(self), "len(self) is not the same as len(data)"
        new_key = False if key in self.keys() else True
        if (not lazy) and isinstance(data, base.Abstract):
            data = ops.DataAbstract(data)[:]
        elif info is not None:
            data = SeqAbstract().concat(data, info=info)
        self._data.update({key: data})
        self._lazy.update({key: lazy})
        self._abstract.update({key: isinstance(data, base.Abstract)})
        if new_key:
            self._reset_active_keys()
            self._nr_keys += 1
        return self

    def set(
            self,
            key: str,
            data: Iterable,
            lazy: bool = True,
            info: List[Dict] = None,
            **kwargs: Dict
    ) -> None:
        self.add(key=key, data=data, lazy=lazy, info=info, **kwargs)

    def add_dict(self, dct: Dict, lazy: bool = True) -> None:
        for key in dct:
            self.add(key, dct[key], lazy=lazy)
        return self

    def concat(
            self, data: Iterable,
            intersect: bool = False,
            adjust_base: bool = True,
            allow_dive: bool = False
    ) -> None:
        if isinstance(data, list):
            for d in data:
                self.concat(d, intersect=intersect)
        else:
            self2 = self if adjust_base else copy.deepcopy(self)
            self2.adjust_mode = True
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
                        if rem_key in data.keys():
                            data.remove(rem_key)
                        else:
                            self2.remove(rem_key)
                for key in keys:
                    if self2._lazy[key]:
                        # make sure that data format is as desired by the base dict
                        if not isinstance(
                                self2[key],
                                SeqAbstract,
                        ):
                            self2[key] = SeqAbstract(allow_dive=allow_dive).concat(self2[key])
                        self2[key].concat(data[key])
                    else:
                        try:
                            assert (
                                    self2[key].__class__ == data[key].__class__
                            ), "When using lazy=False, datatypes should be same in case of concatenation."
                            if isinstance(self2[key], list):
                                self2[key] = self2[key] + data[key]
                            elif isinstance(self2[key], np.ndarray):
                                # print(key)
                                self2[key] = np.concatenate((self2[key], data[key]))
                        except:
                            lol = 0
                self2.adjust_mode = False
            else:
                self2.__dict__.update(data.__dict__)

            return self2

    def remove(self, key: str) -> None:
        del self._data[key]
        self.reset_active_keys()
        self._nr_keys -= 1
        return self

    def add_map(self, key: str, map_fct: Callable, *arg: List, **kwargs: Dict) -> None:
        self[key] = ops.Map(self[key], map_fct, lazy=self._lazy[key], *arg, **kwargs)

    def add_select(self, selector, *arg, eval_data=None, **kwargs):
        def iterative_select(data, indices, *arg, lazy=True, **kwargs):
            if isinstance(data, DictSeqAbstract):
                data.adjust_mode = True
                for key in data.keys():
                    if isinstance(data[key], DictSeqAbstract):
                        if data[key].allow_dive:
                            data[key] = iterative_select(
                                data[key], indices, *arg, lazy=data._lazy[key], **kwargs
                            )
                            continue
                    data[key] = ops.Select(
                        data[key], indices, *arg, lazy=data._lazy[key], **kwargs
                    )
                data.adjust_mode = False
            else:
                data = ops.Select(data, indices, *arg, lazy=lazy, **kwargs)
            return data

        # get indices for all to ensure no discrepancy between items
        indices = ops.Select(
            self,
            selector,
            *arg,
            eval_data=(self if eval_data is None else eval_data),
            **kwargs,
        ).get_indices()
        # Add selection
        iterative_select(self, indices, *arg, **kwargs)

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

    def _call_on_iter(self):
        for key in self._abstract:
            if self._abstract[key]:
                self._data[key]._call_on_iter()

    def __len__(self) -> int:
        nr_examples = [len(self._data[key]) for key in self._data]
        assert all([nr_example == nr_examples[0] for nr_example in nr_examples])
        return nr_examples[0] if len(nr_examples) > 0 else 0

    def __add__(self, other: Iterable) -> None:
        assert isinstance(other, DictSeqAbstract)
        return self.concat(other, adjust_base=False)

    def __setitem__(self, k: str, v: Any) -> None:
        # ToDo: add checks on length. In this case these are avoided.
        # ToDo: functionality of "self['something'] = " and self.set("something", value) differs
        # ToDo: what is the distinction between a set and an add? should be made more clear
        assert isinstance(k, str), "Assignment only possible by (dotted) key (str)."
        fields, tmp = k.split('.'), self
        k = fields.pop()
        for f in fields:
            tmp = tmp[f]
        new_key = False if k in self.keys() else True
        lazy = True if new_key else self._lazy[k]  # make sure that lazy is kept
        if isinstance(tmp, DictSeqAbstract):
            tmp.set(k, v, lazy=lazy)
        else:
            tmp.set(k, v)

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
            fields, tmp = index.split('.'), self._data
            for f in fields:
                tmp = tmp[f]
            return tmp
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
                if self._abstract[key]:
                    data, info = self._data[key].get(
                        index=index, return_info=True, **kwargs
                    )
                else:
                    data, info = self._data[key][index], dict()
            return (data, info) if return_info else data
        else:
            raise IndexError("index should be a number or str")

    def unpack(self, keys: List[str]) -> ops.UnpackAbstract:
        return ops.UnpackAbstract(self._data, keys)

    def keys(self, dive: bool = False) -> List[str]:
        # ToDo: test iterative dive
        def iterative_dive(data, prefix=''):
            keys = []
            for key in data.keys():
                keypath = key if prefix == '' else prefix + '.%s' % key
                if isinstance(data[key], DictSeqAbstract):
                    if data[key].allow_dive:
                        keys += iterative_dive(data[key], prefix=keypath)
                        continue
                keys.append(keypath)
            return keys

        if dive:
            return iterative_dive(self._data)
        else:
            return list(self._data.keys())

    def is_lazy(self, dive: bool = False) -> List[str]:
        def iterative_dive(data, prefix=''):
            lazys = {}
            for key in data.keys():
                keypath = key if prefix == '' else prefix + '.%s' % key
                if isinstance(data[key], DictSeqAbstract):
                    if data[key].allow_dive:
                        lazys.update(iterative_dive(data[key], prefix=keypath))
                        continue
                lazys.update({keypath: data._lazy[key]})
            return lazys

        if dive:
            return iterative_dive(self)
        else:
            return self._lazy

    def summary(self) -> Dict:
        summary = dict()
        for name, data in zip(self.keys(), self._data):
            summary[name] = data.summary()
        return summary

    def __repr__(self) -> str:
        """string print representation of function"""
        return "%s containing: %s" % (self.__class__.__name__, str(self.keys()))

    def pop(self, key) -> Any:
        if self._abstract[key]:
            self[key] = self[key].pop()
        else:
            raise NotImplementedError("Can't pop a data object that is not of type Abstract")

    def is_abstract(self, dive: bool = False) -> List[str]:
        def iterative_dive(data, prefix=''):
            abstracts = {}
            for key in data.keys():
                keypath = key if prefix == '' else prefix + '.%s' % key
                if isinstance(data[key], DictSeqAbstract):
                    if data[key].allow_dive:
                        abstracts.update(iterative_dive(data[key], prefix=keypath))
                        continue
                abstracts.update({keypath: data._abstract[key]})
            return abstracts

        if dive:
            return iterative_dive(self)
        else:
            return self._abstract

    @property
    def allowed_keys(self):
        return self._allowed_keys

    @allowed_keys.setter
    def allowed_keys(self, value: Union[List[str], None]):
        assert isinstance(value, List) or None, "list of allowed keys should be a List or None"
        if isinstance(value, Iterable):
            assert all([isinstance(key, str) for key in value]), "The iterable of allowed keys should contain all strings."
        self._allowed_keys = value

    @property
    def restricted_keys(self):
        return self._allowed_keys

    @restricted_keys.setter
    def restricted_keys(self, value: Union[Iterable[str], None]):
        assert isinstance(value, Iterable) or None, "The iterable of restricted_keys should be a List or None"
        if isinstance(value, Iterable):
            assert all([isinstance(key, str) for key in value]), "The iterable of restricted keys should contain all strings."
        self._restricted_keys = value


class SeqAbstract(ContainerAbstract):
    """Seq base class"""

    def __init__(self, data: Iterable = None, allow_dive: bool = False, allow_nested: bool = False):
        super().__init__(allow_dive=allow_dive, allow_nested=allow_nested)
        self._nr_sources = 0
        self._data = []
        self._abstract = []
        self._info = []
        self._kwargs = []
        if data is not None:
            if isinstance(data, list):
                for _data in data:
                    self.concat(_data)
            else:
                raise AssertionError("Input data should be a list")
        self.set_data()

    def set_data(self):
        pass

    def concat(self, data: Iterable, info: List[Dict] = None, **kwargs: Dict) -> None:
        # Add data
        if isinstance(data, SeqAbstract):
            for _data in data._data:
                self._concat(_data)
        else:
            self._concat(data)
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
        # kwargs
        self._kwargs.append(kwargs)
        return self

    def _concat(self, data: Iterable, info: List[Dict] = None, **kwargs: Dict) -> None:
        data = copy.deepcopy(data)
        # Check
        assert hasattr(
            data, "__getitem__"
        ), "provided data instance must have __getitem__ method."
        if isinstance(data, DictSeqAbstract):
            assert (
                    len(data._active_keys) == 1
            ), "You can only add a dict_abstract in case there is only one active key."
        assert hasattr(self._data, "__len__"), (
                "Can only use %s it object has __len__" % self.__class__.__name__
        )
        self._data.append(data)
        self._nr_sources += 1
        self._abstract.append(isinstance(data, base.Abstract))
        return self

    def set(self, index: int, value: Iterable, **kwargs):
        if isinstance(index, numbers.Integral):
            if index < 0:
                index = index % len(self)
            for k, data in enumerate(self._data):
                if len(data) <= index:
                    index -= len(data)
                else:
                    assert safe_len(data[index]) == safe_len(value), \
                        "When you change an entry in a SeqAbstract container the length should be equal."
                    data[index] = value
                return None
            raise IndexError("Index should be lower than len(dataset)")
        elif isinstance(index, str):
            if index[0] == '[' and index[-1] == ']':
                assert safe_len(self._data[int(index[1:-1])]) == safe_len(value), \
                    "When you change an entry in a SeqAbstract container the length should be equal."
                self._data[int(index[1:-1])] = value
            else:
                return ops.KeyAbstract(self, index)
        else:
            raise IndexError(
                "index should be a number (or key in case of a nested dict_seq)."
            )

    def __len__(self) -> int:
        return np.sum([len(data) for data in self._data])

    def __setitem__(self, index: int, value: Iterable):
        self.set(index, value)

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
        # ToDo: check what this key thing is still doing here
        if isinstance(index, numbers.Integral):
            if index < 0:
                index = index % len(self)
            for k, data in enumerate(self._data):
                if len(data) <= index:
                    index -= len(data)
                else:
                    info = dict() if self._info[k] is None else self._info[k][index]
                    # get
                    if self._abstract[k]:
                        data, info = data.get(
                            index,
                            *arg,
                            return_info=True,
                            **(info if key is None else dict(info, key=key)),
                            **kwargs,
                        )
                    else:
                        assert key is None
                        data, info = data[index], dict(**info, **kwargs)
                    # return
                    return (data, info) if return_info else data
            raise IndexError("Index should be lower than len(dataset)")
        elif isinstance(index, str):
            if index[0] == '[' and index[-1] == ']':
                return self._data[int(index[1:-1])]
            else:
                return ops.KeyAbstract(self, index)
        else:
            raise IndexError(
                "index should be a number (or key in case of a nested dict_seq)."
            )

    def _call_on_iter(self):
        for k, _abstract in enumerate(self._abstract):
            if _abstract:
                self._data[k]._call_on_iter()

    def summary(self) -> Dict:
        return {"nr_examples": self.nr_examples, "name": self._name}

    def __repr__(self):
        r = "%s containing:" % str(self.name)
        for data in self._data:
            if not isinstance(data, (base.Abstract)):
                r += "\n[ \t" + str(type(data)) + "\t]"
            else:
                r += "\n[ \t" + repr(data) + "\t]"
            # r += '\n'
        return r

    def pop(self) -> Any:
        raise NotImplementedError

    def keys(self, dive: bool = False):
        if self.allow_dive and dive:
            return ["[%d]" % k for k in range(len(self._data))]
        else:
            return []
