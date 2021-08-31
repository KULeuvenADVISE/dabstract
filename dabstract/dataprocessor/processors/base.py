from abc import ABC
from typing import Union, List, Optional, TypeVar, Callable, Dict, Iterable

tvProcessingChain = TypeVar("ProcessingChain")


class Processor:
    """base class for processor"""

    def __init__(self):
        pass

    def process(self, data: Iterable, **kwargs) -> (Iterable, Dict):
        return data, {}

    def inv_process(self, data: Iterable, **kwargs) -> Iterable:
        return data

    # def fit(self, data: Iterable, **kwargs) -> None:
    #     raise NotImplementedError

    def __call__(self, data: Iterable, return_info: bool = False, **kwargs) -> Iterable:
        tmp = self.process(data, **kwargs)
        return tmp if return_info else tmp[0]


class ExternalProcessor(Processor, ABC):
    """base class for an external function"""

    def __init__(self, fct: Callable):
        self.fct = fct
        self.__class__.__name__ = fct.__name__

    def process(self, data, **kwargs) -> (Iterable, Dict):
        return self.fct(data), {}


class Dummy(Processor, ABC):
    pass
