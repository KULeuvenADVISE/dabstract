import numpy as np
import os

from dabstract.dataset.abstract import Abstract
from dabstract.dataset.helpers import *
from dabstract.dataprocessor.processors import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *


def test_SampleReplicate():
    """Test SampleReplicate"""
    from dabstract.dataset.abstract import SampleReplicate
    # data init
    data = ['1', '2', '3', '4']
    # lazy sample replicate
    data_lazy_sample = SampleReplicate(data, factor=3, lazy=False)
    # eager sample replicate
    data_eager_sample = SampleReplicate(data, factor=3, lazy=True)
    # checks
    assert data_lazy_sample == ['1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4']
    assert data_lazy_sample.get(3) == ['2']
    assert data_lazy_sample.get(-1) == ['4']
    assert data_eager_sample == ['1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4']
    assert data_eager_sample.get(3) == ['2']
    assert data_eager_sample.get(-1) == ['4']


def test_SampleReplicateAbstract():
    """Test SampleReplicateAbstract"""
    from dabstract.dataset.abstract import SampleReplicateAbstract
    # data init
    data = {'test1': ['1', '2', '3'],
            'test2': np.zeros(3)}
    data_sample_replicate_abstract = SampleReplicateAbstract(data, factor=3)
    # checks
    assert data_sample_replicate_abstract.get(0) == {'test1': '1', 'test2': 0.0}
    assert data_sample_replicate_abstract.get(-1) == {'test1': '3', 'test2': 0.0}
    assert data_sample_replicate_abstract.keys() == {'test1', 'test2'}


def test_Map():
    """Test Map"""
    from dabstract.dataset.abstract import Map, MapAbstract
    # data init
    data = [1, 2, 3, 4]

    """Map using lambda function"""
    # eager mapping lambda function
    map_eager_data_lambda = Map(data, (lambda x: 2*x), lazy=False)
    # lazy mapping lambda function
    map_lazy_data_lambda = Map(data, (lambda x: 2*x), lazy=True)
    # checks
    assert map_eager_data_lambda[0] == 2
    assert map_eager_data_lambda[-1] == 8
    assert map_lazy_data_lambda.get(0) == 2
    assert map_lazy_data_lambda.get(-1) == 8

    """Map using user defined function"""
    #def some_function(input, multiplier, logarithm=False):
    #    output = input*multiplier
    #    if logarithm:
    #        output = np.log10(output)
    #    return output

    # eager mapping defined function
    #map_eager_data_def = Map(data, map_fct=some_function, 2, logarithm=True, lazy=False)
    # lazy mapping defined function
    #map_lazy_data_def = Map(data, map_fct=some_function, 2, logarithm=True, lazy=True)
    # checks
    #assert map_eager_data_def[0] == 0.3010
    #assert map_eager_data_def[-1] == 0.9031
    #assert map_lazy_data_def.get(0) == 0.3010
    #assert map_lazy_data_def.get(-1) == 0.9031

    """Map using processingchain"""
    class custom_processor(Processor):
        def process(self, data, **kwargs):
            return data + 1, {'multiplier': 3}
    class custom_processor2(Processor):
        def process(self, data, **kwargs):
            return data * kwargs['multiplier'], {}
    dp = ProcessingChain()
    dp.add(custom_processor)
    dp.add(custom_processor2)

    # eager mapping using processingchain
    map_eager_data_dp = Map(data, map_fct=dp, lazy=False)
    # lazy mapping using processingchain
    map_lazy_data_dp = Map(data, map_fct=dp, lazy=True)
    # checks
    assert map_eager_data_dp[0] == 6
    assert map_eager_data_dp[-1] == 15
    assert map_eager_data_dp.get(0, return_info=True) == (6, {'multiplier': 3, 'output_shape': ()})
    assert map_lazy_data_dp.get(0) == 6
    assert map_lazy_data_dp.get(-1) == 15
    assert map_lazy_data_dp.get(-1, return_info=True) == (15, {'multiplier': 3, 'output_shape': ()})

    """Map with lambda function and additional information"""
    # eager mapping using lambda function and information
    map_eager_data_lambda_info = Map(data, (lambda x: 2*x), info=({'test1': 'a', 'test2': 'b', 'test3': 3, 'test4': 4}), lazy=False)
    # lazy mapping using lambda function and information
    map_lazy_data_lambda_info = Map(data, (lambda x: 2*x), info=({'test1': 'a', 'test2': 'b', 'test3': 3, 'test4': 4}), lazy=True)
    # checks
    assert map_eager_data_lambda_info[0] == (2, {'test1': 'a'})
    assert map_eager_data_lambda_info[-1] == (8, {'test4': 4})
    assert map_lazy_data_lambda_info.get(0) == (2, {'test1': 'a'})
    assert map_lazy_data_lambda_info.get(-1) == (8, {'test4': 4})


def test_MapAbstract():
    """Test MapAbstract"""
    from dabstract.dataset.abstract import MapAbstract
    # data init
    data = {'test1': np.ones(3),
            'test2': np.zeros(3)}
    data_map_abstract = MapAbstract(data, (lambda x: x+1))
    # checks
    assert data_map_abstract.get(0) == {'test1': 2, 'test2': 1}
    assert data_map_abstract.keys() == {'test1', 'test2'}


def test_UnpackAbstract():
    """Test UnpackAbstract.get"""
    from dabstract.dataset.abstract import UnpackAbstract, DictSeqAbstract
    # data of type dictionary init
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data_dict_unpack = UnpackAbstract(data_dict, keys=['test1', 'test3'])
    # checks dictionary
    assert data_dict_unpack.get(0) == [1.0, '1']
    assert data_dict_unpack.get(1) == [1.0, '2']
    assert data_dict_unpack.get(2) == [1.0, '3']

    # data of type DictSeqAbstract init
    data_DSA = DictSeqAbstract()
    data_DSA.add(data_dict)
    data_DSA_unpack = UnpackAbstract(data_DSA, keys=['test1', 'test3'])
    # check DictSeqAbastract
    assert data_DSA_unpack.get(0) == [1.0, '1']
    assert data_DSA_unpack.get(1) == [1.0, '2']
    assert data_DSA_unpack.get(2) == [1.0, '3']


if __name__ == "__main__":
    test_SampleReplicate()
    test_SampleReplicateAbstract()
    test_Map()
    test_MapAbstract()
    test_UnpackAbstract()
