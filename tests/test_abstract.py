from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *

def test_Split():
    from dabstract.abstract import Split, DictSeqAbstract
    """Test Split"""
    ## Checks on list
    # data
    data = np.ones((1, 100))
    # split
    data_split_lazy = Split(data=data, split_size=50, sample_len=data.shape[1], sample_period=1, type='seconds', lazy=True)
    data_split_direct = Split(data=data, split_size=50, sample_len=data.shape[1], sample_period=1, type='seconds', lazy=False)
    # check
    np.testing.assert_array_equal(data_split_direct, np.ones((2, 50)))
    np.testing.assert_array_equal(data_split_lazy, np.ones((2, 50)))

    ## Checks on a DictSeqAbstract
    # data
    data = {'test1': np.ones((1, 100))}
    data = DictSeqAbstract().add_dict(data)
    # split
    data_split_lazy = Split(data=data, split_size=50, sample_len=len(data.get(0)), sample_period=1, type='seconds', lazy=True)
    data_split_direct = Split(data=data, split_size=50, sample_len=len(data.get(0)), sample_period=1, type='seconds', lazy=False)
    # check
    np.testing.assert_array_equal(data_split_direct, np.ones((2, 50)))
    np.testing.assert_array_equal(data_split_lazy, np.ones((2, 50)))

    ## Checks on samples
    # data
    data = np.ones((1, 100))
    # split
    data_split_lazy_samples = Split(data=data, split_size=50, sample_len=data.shape[1], sample_period=1, type='samples', lazy=True)
    data_split_direct_samples = Split(data=data, split_size=50, sample_len=data.shape[1], sample_period=1, type='samples', lazy=False)
    # check
    np.testing.assert_array_equal(data_split_direct_samples, np.ones((2, 50)))
    np.testing.assert_array_equal(data_split_lazy_samples, np.ones((2, 50)))

    ## Checks on constraint='power2'
    data = np.ones((1, 100))
    # split
    data_split_lazy_power2_size4 = Split(data=data, split_size=4, sample_len=data.shape[1], sample_period=1, constraint='power2', lazy=True)
    data_split_direct_power2_size4 = Split(data=data, split_size=4, sample_len=data.shape[1], sample_period=1, constraint='power2', lazy=False)
    data_split_lazy_power2_size5 = Split(data=data, split_size=5, sample_len=data.shape[1], sample_period=1, constraint='power2', lazy=True)
    data_split_direct_power2_size5 = Split(data=data, split_size=5, sample_len=data.shape[1], sample_period=1, constraint='power2', lazy=False)
    # check
    np.testing.assert_array_equal(data_split_direct_power2_size4, np.ones((25, 4)))
    np.testing.assert_array_equal(data_split_lazy_power2_size4, np.ones((25, 4)))
    np.testing.assert_array_equal(data_split_direct_power2_size5, np.ones((12, 8)))
    np.testing.assert_array_equal(data_split_lazy_power2_size5, np.ones((12, 8)))


def test_Select():
    # TODO: eval_data
    from dabstract.abstract import Select, DictSeqAbstract
    """Test Select"""
    ## Checks on a DictSeqAbstract
    # data
    data = {"test1": np.ones(3), "test2": np.zeros(3), "test3": [1, 2, 3]}
    data = DictSeqAbstract().add_dict(data)
    indices = np.array([0, 1])
    # select
    data_select_indices_lazy = Select(data, indices, lazy=True)
    data_select_indices_direct = Select(data, indices, lazy=False)
    # check
    assert data_select_indices_lazy[0] == {"test1": 1, "test2": 0, "test3": 1}
    assert data_select_indices_direct[0] == {"test1": 1, "test2": 0, "test3": 1}
    assert data_select_indices_lazy[-1] == {"test1": 1, "test2": 0, "test3": 2}
    assert data_select_indices_direct[-1] == {"test1": 1, "test2": 0, "test3": 2}

    ## Checks on a list
    # data
    data = [1, 2, 3, 4]
    indices = np.array([0, 1])
    # select
    data_select_indices_lazy = Select(data, indices, lazy=True)
    data_select_indices_direct = Select(data, indices, lazy=False)
    # check
    assert data_select_indices_lazy[0] == 1
    assert data_select_indices_lazy[-1] == 2
    assert data_select_indices_direct[0] == 1
    assert data_select_indices_direct[-1] == 2

    ## Lambda function
    # Data
    data = {"test1": np.ones(3), "test2": np.zeros(3), "test3": [1, 2, 3]}
    data = DictSeqAbstract().add_dict(data)
    test3_criterium = np.array([1, 2])
    # select
    data_select_lambda_lazy = Select(data=data, selector=(lambda x, k: x["test3"][k] in test3_criterium), lazy=True)
    data_select_lambda_direct = Select(data=data, selector=(lambda x, k: x["test3"][k] in test3_criterium), lazy=False)
    # check
    assert data_select_lambda_lazy[0] == {"test1": 1, "test2": 0, "test3": 1}
    assert data_select_lambda_direct[0] == {"test1": 1, "test2": 0, "test3": 1}
    assert data_select_lambda_lazy[-1] == {"test1": 1, "test2": 0, "test3": 2}
    assert data_select_lambda_direct[-1] == {"test1": 1, "test2": 0, "test3": 2}

    ## eval_data not None
    # data
    # data = [1, 2, 3, 4]
    # eval_data = [5, 6, 7]
    # test3_criterium = [5, 6]
    data = {"test1": np.ones(3), "test2": np.zeros(3), "test3": [1, 2, 3]}
    eval_data = {"test1": np.ones(3), "test2": np.zeros(3), "test3": [5, 6, 7]}
    data = DictSeqAbstract().add_dict(data)
    eval_data = DictSeqAbstract().add_dict(eval_data)
    test3_criterium = np.array([5, 6])
    # select
    data_select_indices_lazy = Select(data, selector=(lambda x: x in test3_criterium), eval_data=eval_data, lazy=True)
    data_select_indices_direct = Select(data, selector=(lambda x: x in test3_criterium), eval_data=eval_data, lazy=False)
    # # check
    # # TODO


def test_Filter():
    from dabstract.abstract import Filter, DictSeqAbstract
    """Test Filter"""
    ## List
    # data
    data = [1, 2, 3, 4]
    test3_criterium = [1, 2]
    # filter
    data_filter_lazy = Filter(data, filter_fct=(lambda x: x in test3_criterium), return_none=False, lazy=True)
    data_filter_direct = Filter(data, filter_fct=(lambda x: x in test3_criterium), return_none=False, lazy=False)
    data_filter_lazy_none = Filter(data, filter_fct=(lambda x: x in test3_criterium), return_none=True, lazy=True)
    data_filter_direct_none = Filter(data, filter_fct=(lambda x: x in test3_criterium), return_none=True, lazy=False)
    # check
    assert data_filter_lazy[0] == 1 and data_filter_lazy[1] == 2
    assert data_filter_direct[0] == 1 and data_filter_direct[1] == 2
    assert data_filter_lazy_none[0] == 1 and data_filter_lazy_none[1] == 2
    assert data_filter_direct_none == [1, 2, None, None]

    ## DictSeqAbstract
    # data
    data = {"test1": np.ones(3), "test2": np.zeros(3), "test3": [1, 2, 3]}
    data = DictSeqAbstract().add_dict(data)
    test3_criterium = [1, 2]
    # filter
    data_filter_lazy = Filter(data, filter_fct=(lambda x: x['test3'] in test3_criterium), return_none=False, lazy=True)
    data_filter_direct = Filter(data, filter_fct=(lambda x: x['test3'] in test3_criterium), return_none=False, lazy=False)
    data_filter_lazy_none = Filter(data, filter_fct=(lambda x: x['test3'] in test3_criterium), return_none=True, lazy=True)
    data_filter_direct_none = Filter(data, filter_fct=(lambda x: x['test3'] in test3_criterium), return_none=True, lazy=False)
    # check
    assert data_filter_lazy[0] == {"test1": 1, "test2": 0, "test3": 1} and data_filter_lazy[1] == {"test1": 1, "test2": 0, "test3": 2}
    assert data_filter_direct[0] == {"test1": 1, "test2": 0, "test3": 1} and data_filter_direct[1] == {"test1": 1, "test2": 0, "test3": 2}
    assert data_filter_lazy_none[0] == {"test1": 1, "test2": 0, "test3": 1} and data_filter_lazy_none[1] == {"test1": 1, "test2": 0, "test3": 2}
    assert data_filter_direct_none == [{'test1': 1.0, 'test2': 0.0, 'test3': 1}, {'test1': 1.0, 'test2': 0.0, 'test3': 2}, None]


if __name__ == "__main__":
    test_Split()
    test_Select()
    test_Filter()