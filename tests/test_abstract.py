from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *

def test_SampleReplicate():
    """Test SampleReplicate"""
    from dabstract.abstract import SampleReplicate, DictSeqAbstract
    ## test with list
    # data init
    data = ['1', '2', '3', '4']
    # lazy sample replicate of a list
    data_eager_sample = SampleReplicate(data, factor=3, lazy=False)
    # eager sample replicate of a list
    data_lazy_sample = SampleReplicate(data, factor=3, lazy=True)
    # checks
    assert data_eager_sample == ['1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4']
    assert data_eager_sample[3] == '2'
    assert data_eager_sample[-1] == '4'
    assert [k for k in data_lazy_sample] == ['1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4']
    assert data_lazy_sample[3] == '2'
    assert data_lazy_sample[-1] == '4'

    ## Test with DictSeqAbstract
    # data
    dsa = DictSeqAbstract().add_dict({'test1': ['1', '2', '3'], 'test2': np.zeros(3)})
    # eager sample replicate of a DictAbstract
    dsa_eager_sample = SampleReplicate(dsa, factor=3, lazy=False)
    # lazy sample replicate of a DictAbstract
    dsa_lazy_sample = SampleReplicate(dsa, factor=3, lazy=True)
    # checks
    assert dsa_eager_sample[0] == {'test1': '1', 'test2': 0.0}
    assert dsa_eager_sample[-1] == {'test1': '3', 'test2': 0.0}
    assert dsa_eager_sample[0].keys() == {'test1', 'test2'}
    assert dsa_lazy_sample[0] == {'test1': '1', 'test2': 0.0}
    assert dsa_lazy_sample[-1] == {'test1': '3', 'test2': 0.0}
    assert dsa_lazy_sample.keys() == ['test1', 'test2']

def test_Map():
    """Test Map"""
    from dabstract.abstract import Map
    # data init
    data = [1, 2, 3, 4]

    ## Map using lambda function
    # eager mapping lambda function
    map_eager_data_lambda = Map(data, (lambda x: 2*x), lazy=False)
    # lazy mapping lambda function
    map_lazy_data_lambda = Map(data, (lambda x: 2*x), lazy=True)
    # checks
    assert map_eager_data_lambda[0] == 2
    assert map_eager_data_lambda[-1] == 8
    assert map_lazy_data_lambda[0] == 2
    assert map_lazy_data_lambda[-1] == 8

     ## Map using defined function
    def some_function(input, multiplier, logarithm=False):
        output = input*multiplier
        if logarithm:
            output = np.log10(output)
        return output

    # eager mapping defined function
    map_eager_data_def = Map(data, some_function, multiplier=2, logarithm=True, lazy=False)
    # lazy mapping defined function
    map_lazy_data_def = Map(data, some_function, multiplier=2, logarithm=True, lazy=True)
    # checks
    assert map_eager_data_def[0] == 0.3010299956639812
    assert map_eager_data_def[-1] == 0.9030899869919435
    assert map_lazy_data_def[0] == 0.3010299956639812
    assert map_lazy_data_def[-1] == 0.9030899869919435

    ## Map using ProcessingChain
    class custom_processor(Processor):
        def process(self, data, **kwargs):
            print(kwargs)
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
    assert map_lazy_data_dp[0] == 6
    assert map_lazy_data_dp[-1] == 15
    assert map_lazy_data_dp.get(-1, return_info=True) == (15, {'multiplier': 3, 'output_shape': ()})

    ## Map using lambda function with additional information
    # eager mapping using lambda function and information
    map_eager_data_lambda_info = Map(data, (lambda x: 2*x), info=({'test': 1}, {'test': 2}, {'test': 'a'}, {'test': 'b'}), lazy=False)
    # lazy mapping using lambda function and information
    map_lazy_data_lambda_info = Map(data, (lambda x: 2*x), info=({'test': 1}, {'test': 2}, {'test': 'a'}, {'test': 'b'}), lazy=True)
    # checks
    assert map_eager_data_lambda_info[0] == 2
    assert map_eager_data_lambda_info[-1] == 8
    assert map_lazy_data_lambda_info.get(0, return_info=True) == (2, {'test': 1})
    assert map_lazy_data_lambda_info.get(-1, return_info=True) == (8, {'test': 'b'})

def test_UnpackAbstract():
    """Test UnpackAbstract.get"""
    from dabstract.abstract import UnpackAbstract, DictSeqAbstract
    # data of type dictionary init
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data_dict_unpack = UnpackAbstract(data_dict, keys=['test1', 'test3'])
    # checks dictionary
    assert data_dict_unpack[0] == [1.0, '1']
    assert data_dict_unpack[1] == [1.0, '2']
    assert data_dict_unpack[2] == [1.0, '3']

    # data of type DictSeqAbstract init
    dsa = DictSeqAbstract()
    dsa.add_dict(data_dict)
    dsa_unpack = UnpackAbstract(dsa, keys=['test1', 'test3'])
    # check DictSeqAbastract
    assert dsa_unpack[0] == [1.0, '1']
    assert dsa_unpack[1] == [1.0, '2']
    assert dsa_unpack[2] == [1.0, '3']


if __name__ == "__main__":
    test_SampleReplicate()
    test_Map()
    test_UnpackAbstract()
