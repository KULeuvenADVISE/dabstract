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
    assert all([key in ('test2','test3','test_only','dataset_id') for key in data[0].keys()])
    assert all([key in ('test2','test3','test_only','dataset_id') for key in data.keys()])

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
            return data+1, {'multiplier': 3}
    class custom_processor2(Processor):
        def process(self, data, **kwargs):
            return data*kwargs['multiplier'], {}
    dp = ProcessingChain()
    dp.add(custom_processor)
    dp.add(custom_processor2)
    # checks
    data.add_map('test1',lambda x: x+1)
    assert data[0] == {'test1': 2.0, 'test2': 0.0, 'test3': '1', 'test_only': 0.0, 'dataset_id': 0}
    data.add_map('test1',dp)
    assert data[0] == {'test1': 9.0, 'test2': 0.0, 'test3': '1', 'test_only': 0.0, 'dataset_id': 0}

def test_add_split():
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