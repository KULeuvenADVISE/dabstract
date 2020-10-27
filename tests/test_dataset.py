import numpy as np
import os

from dabstract.dataset.dataset import Dataset
from dabstract.dataset.helpers import *
from dabstract.dataprocessor.processors import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *


def get_dataset():
    """Get dataset class"""

    class EXAMPLE(Dataset):
        def __init__(self,
                     paths=None,
                     test_only=0,
                     **kwargs):
            # init dict abstract
            super().__init__(name=self.__class__.__name__,
                             paths=paths,
                             test_only=test_only)

        # Data: get data
        def set_data(self, paths):
            # audio
            chain = ProcessingChain().add(WavDatareader())
            from dabstract.dataset.helpers import FolderDictSeqAbstract
            tmp = FolderDictSeqAbstract(paths['data'], map_fct=chain, file_info_save_path=paths['data'])
            self.add('data', tmp)
            # add labels
            self.add('binary_anomaly', self._get_binary_anomaly(paths), lazy=False)
            self.add('group', self['data']['subdb'], lazy=False)
            return self

        def prepare(self, paths):
            if not os.path.isdir(paths['data']):
                from scipy.io.wavfile import write
                # normal class
                files = 20
                duration = 60
                sampling_rate = 16000
                subdb = 'normal'
                for k in range(files):
                    os.makedirs(os.path.join(paths['data'], subdb), exist_ok=True)
                    write(os.path.join(paths['data'], subdb, str(k) + '.wav'), sampling_rate,
                          0.1 * np.random.rand(duration * 16000))
                labels = np.zeros(files)
                np.save(os.path.join(paths['data'], subdb + '_labels.npy'), labels)

                # abnormal class
                files = 20
                duration = 60
                sampling_rate = 16000
                subdb = 'abnormal'
                for k in range(files):
                    os.makedirs(os.path.join(paths['data'], subdb), exist_ok=True)
                    write(os.path.join(paths['data'], subdb, str(k) + '.wav'), sampling_rate,
                          np.random.rand(duration * 16000))
                labels = np.ones(files)
                np.save(os.path.join(paths['data'], subdb + '_labels.npy'), labels)

        def _get_binary_anomaly(self, paths):
            subdbs = np.unique(self['data']['subdb'])
            labels = [None] * len(subdbs)
            for k, subdb in enumerate(subdbs):
                subdb_id = np.where([s == subdb for s in self['data']['subdb']])[0]
                reorder = np.array([int(os.path.splitext(filename)[0]) \
                                    for filename in \
                                    [self['data']['filename'][k] for k in subdb_id]])
                labels[k] = np.load(os.path.join(paths['meta'], subdb + '_labels.npy'))[reorder]
            return listnp_combine(labels)

    return EXAMPLE


def test_EXAMPLE_dataset():
    """Test dataset loading"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data')})

    # checks
    assert len(db) == 40
    assert isinstance(db['data'], FolderDictSeqAbstract)
    assert isinstance(db['binary_anomaly'], np.ndarray)
    assert isinstance(db['group'], List)
    assert isinstance(db['group'][0], str)
    assert isinstance(db['test_only'], np.ndarray)
    assert isinstance(db['dataset_id'], np.ndarray)


def test__getitem__():
    """Test __get_item__"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data')})

    # checks
    assert isinstance(db[0], Dict)
    assert all([key in ('data', 'binary_anomaly', 'group', 'test_only', 'dataset_id') for key in db[0]])
    assert np.all(db[0]['data'] == db['data'][0])
    assert db[0]['binary_anomaly'] == db['binary_anomaly'][0]
    assert db[0]['group'] == db['group'][0]
    assert db[0]['test_only'] == db['test_only'][0]
    assert db[0]['dataset_id'] == db['dataset_id'][0]


def test__setitem__():
    """Test __get_item__"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data')})
    EXAMPLE = get_dataset()
    db2 = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                         'meta': os.path.join('data', 'data')})

    # checks
    db['data'] = db2['data']
    assert db['data'] == db2['data']

    db['binary_anomaly'][0] = db['binary_anomaly'][1]
    assert db['binary_anomaly'][0] == db['binary_anomaly'][1]

    db['group'][0] = db['group'][1]
    assert db['group'][0] == db['group'][1]

    db['test_only'][0] = 1
    assert db['test_only'][0] == 1

    db['dataset_id'][0] = 1
    assert db[0]['dataset_id'] == 1


def test__add__():
    """Test __get_item__"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data')})
    EXAMPLE = get_dataset()
    db2 = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                         'meta': os.path.join('data', 'data')})

    # checks
    db3 = db + db2
    assert [db3[key][0] == db[key][0] for key in db.keys()]
    assert [db3[key][-1] == db2[key][-1] for key in db.keys()]
    db4 = db + db + db + db + db
    assert len(db4) == 5 * len(db)


def test_add():
    """Test __get_item__"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data')})
    EXAMPLE = get_dataset()
    db2 = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                         'meta': os.path.join('data', 'data')})

    # checks
    db3 = db + db2
    assert [db3[key][0] == db[key][0] for key in db.keys()]
    assert [db3[key][-1] == db2[key][-1] for key in db.keys()]
    db4 = db + db + db + db + db
    assert len(db4) == 5 * len(db)


def test_add():
    """Test add"""
    data = Dataset()
    data.add('test1', np.ones(3))
    data.add('test2', np.zeros(3))
    data.add('test3', ['1', '2', '3'])
    # checks
    assert data[0] == {'test1': 1.0, 'test_only': 0.0, 'dataset_id': 0, 'test2': 0.0, 'test3': '1'}


def test_add_dict():
    """Test add_dict"""
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    assert data[0] == {'test1': 1.0, 'test_only': 0.0, 'dataset_id': 0, 'test2': 0.0, 'test3': '1'}


def test_concat():
    """Test concat"""
    # set dataset 1
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # set dataset 2
    data1 = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data1.add_dict(data_dict)
    # set dataset 2
    data2 = Dataset()
    data_dict = {'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data2.add_dict(data_dict)

    # checks
    data3 = data.concat(data, adjust_base=False)
    assert len(data3) == 6
    assert data[0] == {'test1': 1.0, 'test_only': 0.0, 'dataset_id': 0, 'test2': 0.0, 'test3': '1'}
    assert data[-1] == {'test1': 1.0, 'test_only': 0.0, 'dataset_id': 0, 'test2': 0.0, 'test3': '3'}

    data3 = data.concat(data2, intersect=True, adjust_base=False)
    assert len(data3) == 6
    assert data3[0] == {'test2': 0.0, 'test3': '1', 'test_only': 0.0, 'dataset_id': 0}
    assert data3[-1] == {'test2': 0.0, 'test3': '3', 'test_only': 0.0, 'dataset_id': 1}

    data.concat(data2, intersect=True, adjust_base=True).concat(data2, intersect=True, adjust_base=True)
    assert len(data) == 9
    assert data[0] == {'test2': 0.0, 'test3': '1', 'test_only': 0.0, 'dataset_id': 0}
    assert data[-1] == {'test2': 0.0, 'test3': '3', 'test_only': 0.0, 'dataset_id': 1}


def test_remove():
    """Test remove"""
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    data.remove('test1')
    assert all([key in ('test2', 'test3', 'test_only', 'dataset_id') for key in data[0].keys()])
    assert all([key in ('test2', 'test3', 'test_only', 'dataset_id') for key in data.keys()])


def test_add_map():
    """Test add_map"""
    # define dataset
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)

    # define processing chain
    class custom_processor(Processor):
        def process(self, data, **kwargs):
            return data + 1, {'multiplier': 3}

    class custom_processor2(Processor):
        def process(self, data, **kwargs):
            return data * kwargs['multiplier'], {}

    dp = ProcessingChain()
    dp.add(custom_processor)
    dp.add(custom_processor2)
    # checks
    data.add_map('test1', lambda x: x + 1)
    assert data[0] == {'test1': 2.0, 'test2': 0.0, 'test3': '1', 'test_only': 0.0, 'dataset_id': 0}
    data.add_map('test1', dp)
    assert data[0] == {'test1': 9.0, 'test2': 0.0, 'test3': '1', 'test_only': 0.0, 'dataset_id': 0}


def test_add_split():
    """Test add_split"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data')})
    db.add('example_id', np.arange(len(db)), lazy=False)
    # checks
    db_split = copy.deepcopy(db)
    db_split.add_split(5)
    assert len(db) * 60 / 5 == len(db_split)
    assert all(db_split['data'][0] == db['data'][0][0:16000 * 5])
    db_split['example_id'][0] == db_split['example_id'][int(60 / 5 - 1)]

    db_split = copy.deepcopy(db)
    db_split.add_split(reference_key='data', split_size=16000, type='samples')
    assert len(db) * 60 == len(db_split)
    assert all(db_split['data'][0] == db['data'][0][0:16000])
    db_split['example_id'][0] == db_split['example_id'][59]

    db_split = copy.deepcopy(db)
    db_split.add_split(reference_key='data', split_size=16000, type='samples', constraint='power2')
    assert all(db_split['data'][0] == db['data'][0][0:2 ** 14])


def test_add_select():
    """Test add_select"""
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    data2 = copy.deepcopy(data)
    data2.add_select([1, 2])
    assert data2[0] == data[1]

    from dabstract.dataset.select import subsample_by_str
    data2 = copy.deepcopy(data)
    data2.add_select(subsample_by_str('test3', ['2', '3']))
    assert data2[0] == data[1]

    data2 = copy.deepcopy(data)
    data2.add_select((lambda x, k: x['test3'][k] in ('2', '3')))
    assert data2[0] == data[1]


def test_add_alias():
    """Test add_alias"""
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    data.add_alias('test1', 'test1_alias')
    assert data['test1'][0] == data['test1_alias'][0]


def test_keys():
    """Test add_alias"""
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    assert data.keys() == ['test1', 'test2', 'test3', 'test_only', 'dataset_id']


def test_active_keys():
    """Test active_keys"""
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    data.set_active_keys(['test1', 'test3'])
    assert data[0] == {'test1': 1.0, 'test3': '1'}

    data.set_active_keys(['test1'])
    assert data[0] == 1.0

    data.reset_active_keys()
    assert data[0] == {'test1': 1.0, 'test2': 0.0, 'test3': '1', 'test_only': 0.0, 'dataset_id': 0}


def test_unpack():
    """Test unpack"""
    # db init
    data = Dataset()
    data_dict = {'test1': np.ones(3),
                 'test2': np.zeros(3),
                 'test3': ['1', '2', '3']}
    data.add_dict(data_dict)
    # checks
    data_unpack = data.unpack(['test1', 'test2', 'test3'])
    assert data_unpack[0] == [1.0, 0.0, '1']


def test_load_memory():
    """Test load_memory"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data')})
    # check
    db2 = copy.deepcopy(db)
    db2.load_memory('data', verbose=False)
    assert all([np.all(db[0][key] == db2[0][key]) for key in db.keys()])

    db2 = copy.deepcopy(db)
    db2.load_memory('data', keep_structure=True, verbose=False)
    assert all([np.all(db[0][key] == db2[0][key]) for key in db.keys()])


def test_prepare_feat():
    """Test prepare_feat"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data'),
                        'feat': os.path.join('data', 'feat')})
    # define chain
    dp = ProcessingChain()
    dp.add(Framing(windowsize=10, stepsize=10, axis=0))
    dp.add(FFT(axis=1))
    dp.add(Aggregation(methods=['mean', 'std'], axis=0, combine='concatenate'))
    # check
    db.prepare_feat('data', 'avgFFT', dp, 'feat', verbose=False)
    assert np.all(db['feat'][0] == dp(db['data'][0], fs=16000))


def test_xval():
    """Test prepare_feat"""
    # db init
    EXAMPLE = get_dataset()
    db = EXAMPLE(paths={'data': os.path.join('data', 'data'),
                        'meta': os.path.join('data', 'data'),
                        'feat': os.path.join('data', 'feat')})
    db.add('set', ['test'] * len(db))
    # add xval
    from dabstract.dataset.xval import xval_from_item
    db2 = copy.deepcopy(db)
    db2.add('set', ['test'] * len(db))
    db3 = copy.deepcopy(db)
    db3.add('set', ['train'] * len(db))
    db4 = db2 + db3
    db4.set_xval(xval_from_item(key='set'))
    # check
    test_set = db4.get_xval_set(fold=0, set='test')
    assert len(test_set) == len(db2)

    train_set = db4.get_xval_set(fold=0, set='train')
    assert len(train_set) == len(db3)

    assert all([np.all(test_set[0][key] == db2[0][key]) for key in ['data', 'test_only', 'binary_anomaly', 'group', 'set']])
    assert all([np.all(train_set[0][key] == db3[0][key]) for key in ['data', 'test_only', 'binary_anomaly', 'group', 'set']])


if __name__ == "__main__":
    test_EXAMPLE_dataset()
    test__getitem__()
    test__setitem__()
    test__add__()
    test_add()
    test_add_dict()
    test_concat()
    test_remove()
    test_add_map()
    test_add_split()
    test_add_select()
    test_add_alias()
    test_keys()
    test_active_keys()
    test_unpack()
    test_load_memory()
    test_prepare_feat()
    test_xval()
