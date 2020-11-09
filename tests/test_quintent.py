import numpy as np
import os

from dabstract.dataset.dataset import Dataset
from dabstract.dataset.abstract import Abstract
from dabstract.dataset.helpers import *
from dabstract.dataprocessor.processors import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *


def test_sample_replicate():
    """Test SampleReplicate"""
    from dabstract.dataset.abstract import SampleReplicate
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3)}
    data.add_dict(data_dict)
    # lazy sample replicate
    data_lazy_sample = SampleReplicate(data, 3, lazy=False)
    # eager sample replicate
    data_eager_sample = SampleReplicate(data, 3)
    # checks
    assert data_lazy_sample.get(0) == {'test1': 1.0}
    assert data_lazy_sample.get(-1) == {'test1': 1.0}
    assert data_eager_sample.get(0) == {'test1': 1.0}
    assert data_eager_sample.get(-1) == {'test1': 1.0}


def test_sample_replicate_abstract():
    """Test SampleReplicateAbstract"""
    from dabstract.dataset.abstract import SampleReplicateAbstract
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3)}
    data.add_dict(data_dict)
    data_sample_replicate_abstract = SampleReplicateAbstract(data, 3)
    # checks
    assert data_sample_replicate_abstract.get(0) == {'test1': 1.0}
    assert data_sample_replicate_abstract.get(-1) == {'test1': 1.0}
    assert data_sample_replicate_abstract.keys() == {'test1', 'test2'}


def test_map():
    """Test Map"""
    from dabstract.dataset.abstract import Map
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3)}
    data.add_dict(data_dict)
    # eager mapping
    map_eager_data = Map(data, (lambda x: 2*x), lazy=False)
    # lazy mapping
    map_lazy_data = Map(data, (lambda x: 2*x))
    # checks
    assert map_eager_data[0] == {'test1': 2.0, 'test2': 0.0}
    assert map_eager_data[-1] == {'test1': 2.0, 'test2': 0.0}
    assert map_lazy_data.get(0) == {'test1': 2.0, 'test2': 0.0}
    assert map_lazy_data.get(-1) == {'test1': 2.0, 'test2': 0.0}


def test_map_abstract():
    """Test MapAbstract"""
    from dabstract.dataset.abstract import SeqAbstract, MapAbstract
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3)}
    data.add_dict(data_dict)
    data_map_abstract = MapAbstract(data, (lambda x: x+1))
    # checks
    assert data_map_abstract.get(0) == 2.0
    assert data_map_abstract.shape() == 3
    assert data_map_abstract.keys() == {'test1', 'test2'}


def test_unpack_abstract_get():
    """Test UnpackAbstract.get"""
    from dabstract.dataset.abstract import UnpackAbstract
    # db init
    data = UnpackAbstract()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    data_unpack_first = data.get(0)
    data_unpack_second = data.get(1)
    data_unpack_last = data.get(2)
    assert data_unpack_first == [1.0, 0.0, '1']
    assert data_unpack_second == [1.0, 0.0, '2']
    assert data_unpack_last == [1.0, 0.0, '3']


if __name__ == "__main__":
    test_sample_replicate()
    test_sample_replicate_abstract()
    test_map()
    test_map_abstract()
    test_unpack_abstract_get()
