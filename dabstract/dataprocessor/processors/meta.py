import numpy as np
import numbers
import sklearn.preprocessing as pp

from dabstract.dataprocessor.processors import base

from typing import Dict, Any, List, Union, Callable, Iterable

class MultiLabelFilter(base.Processor):
    def __init__(self, filter_type: str = 'majority'):
        self.filter_type = filter_type

    def process(self, data: Iterable, **kwargs) -> np.ndarray:
        # filter
        if self.filter_type in 'majority':
            values, counts = np.unique(data, return_counts=True)
            return values[np.argmax(counts)], {}

        elif self.filter_type == 'random_tie_majority':
            values, counts = np.unique(data, return_counts=True)
            return values[np.argmax(np.random.random(len(counts)) * (counts == counts.max()))], {}

        else:
            raise NotImplementedError

class OneHotEncoder(base.Processor):
    def __init__(self, classes: Union[List, np.ndarray] = None):
        self.classes = classes

    def process(self,
                data: Iterable,
                **kwargs) -> np.ndarray:
        assert self.is_fitted(), \
            "Encoder is not fitted. Please call fit(.)"
        return self._encoder.transform(data.reshape(-1,1))

    def fit(self, data: Iterable, **kwargs) -> None:
        self._encoder = pp.OneHotEncoder(categories = 'auto' if self.classes is None else self.classes)
        self._encoder.fit(data)

    def is_fitted(self):
        return hasattr(self, '_encoder')

class MultiLabelBinarizer(base.Processor):
    def __init__(self, classes: Union[List, np.ndarray] = None):
        self.classes = classes

    def process(self,
                data: Iterable,
                **kwargs) -> np.ndarray:
        assert self.is_fitted(), \
            "Encoder is not fitted. Please call fit(.)"
        return self._encoder.transform(data.reshape(-1,1))

    def fit(self, data: Iterable, **kwargs) -> None:
        self._encoder = pp.MultiLabelBinarizer(classes = self.classes)
        self._encoder.fit(data)

    def is_fitted(self):
        return hasattr(self, '_encoder')