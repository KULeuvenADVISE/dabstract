from abc import ABC, abstractmethod
from typing import Any

class Abstract(ABC):
    def __init__(self, data: Any):
        self._data = data
        self._abstract = True if isinstance(data, Abstract) else False
        if self._abstract:
            assert self._data.len_defined, (
                "Can only use %s it data has __len__" % self.__class__.__name__
            )
        else:
            assert hasattr(self._data, "__len__"), (
                "Can only use %s it data has __len__" % self.__class__.__name__
            )

    def __iter__(self) -> Any:
        for k in range(len(self)):
            yield self[k]

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

    @property
    def len_defined(self):
        return True

    @property
    def is_splittable(self):
        if self._abstract:
            return self._data.is_splittable()
        else:
            False