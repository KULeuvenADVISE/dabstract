from abc import ABC, abstractmethod
from typing import Any


class Abstract(ABC):
    def __init__(self, data: Any, allow_dive: bool = False):
        self._data = data
        self._abstract = True if isinstance(data, Abstract) else False
        self._allow_dive = allow_dive
        self._name = self.__class__.__name__
        self._group = None
        if self._abstract:
            assert self._data.len_defined, (
                    "Can only use %s it data has __len__" % self.__class__.__name__
            )
        else:
            assert hasattr(self._data, "__len__"), (
                    "Can only use %s it data has __len__" % self.__class__.__name__
            )

    def __iter__(self) -> Any:
        self._call_on_iter()
        for k in range(len(self)):
            yield self[k]

    def _call_on_iter(self):
        if self._abstract:
            self._data._call_on_iter()

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def get(
            self,
            index: Any,
            return_info: bool = False,
            **kwargs
    ) -> Any:
        # get data
        if self._abstract:
            data, info = self._data.get(index, return_info=True)
        else:
            data, info = self._data[index], {}
        # return
        return (data, info) if return_info else data

    def __setitem__(self, k, v):
        raise NotImplementedError(
            "%s does not support item assignment." % self.__class__.__name__
        )

    def __call__(self, *args, **kwargs) -> Any:
        return self.get(*args, **kwargs)

    def __len__(self):
        return len(self._data)

    def pop(self) -> Any:
        return self._data

    def len_defined(self):
        return True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group: str):
        self._group = group
        if self.is_abstract:
            self._data.group = group

    @property
    def is_abstract(self):
        return self._abstract

    def _abstract_handler(self,
                          base_class: object,
                          **kwargs: object) -> object:
        return None
