import numpy as np
import os

from dabstract.dataset.dataset import Dataset
from dabstract.dataset.helpers import *
from dabstract.dataprocessor.processors import *
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
    assert len(db)==40
    assert isinstance(db['data'],FolderDictSeqAbstract)
    assert isinstance(db['binary_anomaly'],np.ndarray)
    assert isinstance(db['group'], List)
    assert isinstance(db['group'][0], str)
    assert isinstance(db['test_only'], List)
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
    db3 = db+db2
    assert [db3[key][0] == db[key][0] for key in db.keys()]
    assert [db3[key][-1] == db2[key][-1] for key in db.keys()]
    db4 = db+db+db+db+db
    assert len(db4)==5*len(db)

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


if __name__ == "__main__":
    test_EXAMPLE_dataset()
    test__getitem__()
    test__setitem__()
    test__add__()
    test_add()