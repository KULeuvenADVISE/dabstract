import numpy as np
import os

from dabstract.dataset.dataset import Dataset
from dabstract.dataset.helpers import *
from dabstract.dataprocessor.processors import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import *

from dabstract.dataset.containers import MetaContainer, WavFolderContainer

class DetectionDataset(Dataset):
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
        tmp = WavFolderContainer(paths['data'],
                               map_fct=chain,
                               file_info_save_path=paths['data'])
        self.add('data', tmp)

        # add labels
        self.add('binary_anomaly',
                 MetaContainer(self._get_binary_anomaly(paths),
                               meta_type='single_label'),
                 lazy=True)
        self.add('time_anomaly',
                 MetaContainer(self._get_time_anomaly(paths),
                               meta_type='multi_time_label',
                               duration = self['data'].get_duration(),
                               time_step = self['data'].get_time_step()),
                 lazy=True)
        self.add_split(20)
        print(self['time_anomaly'][0])
        print(self['time_anomaly']._data[0])

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
        return listnp_combine(labels).reshape(-1,1)

    def _get_time_anomaly(self, paths):
        labels = self._get_binary_anomaly(paths)
        probability = 0.7
        time_label = []
        for k in range(len(labels)):
            total_samples = self['data'].get_samples(k)
            sample_idx = np.sort(np.random.randint(0,total_samples, 5))
            sample_idx = sample_idx + (total_samples - sample_idx.max())
            end_time = sample_idx / self['data'].get_fs(k)
            start_time = np.concatenate((np.array([0]),end_time[0:-1]))
            sample_label = np.random.binomial(1, probability * labels[k] + (1-probability) * (1 - labels[k]), 5)
            time_label.append(np.stack((sample_label, start_time, end_time),axis=1))
        return time_label