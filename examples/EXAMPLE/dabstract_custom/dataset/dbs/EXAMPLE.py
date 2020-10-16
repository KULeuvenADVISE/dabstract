from pylab import plot, imshow, show

from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataset.dataset import dataset
from dabstract.dataset.helpers import FolderDictSeqAbstract
from dabstract.dataprocessor.processors import *
from dabstract.utils import listnp_combine

class EXAMPLE(dataset):
    def __init__(self,
                 paths=None,
                 split=None,
                 select=None,
                 test_only=0,
                 **kwargs):
        # init dict abstract
        super().__init__(name=self.__class__.__name__,
                         paths=paths,
                         split=split,
                         select=select,
                         test_only=test_only)

    # Data: get data
    def set_data(self, paths):
        # audio
        chain = processing_chain().add(WavDatareader())
        self.add('data', FolderDictSeqAbstract(paths['data'], file_info_save_path=paths['feat'], map_fct=chain))
        # add labels
        self.add('binary_anomaly',self._get_binary_anomaly(paths))
        self.add('group', self['data']['subdb'])
        return self

    def prepare(self,paths):
        if not os.path.isdir(paths['data']):
            from scipy.io.wavfile import write
            # abnormal class
            files = 100
            duration = 60
            sampling_rate = 16000
            subdb = 'abnormal'
            filt = FIR_filter(type='highpass', f=4000, taps=100, fs=sampling_rate, axis=0)
            for k in range(files):
                os.makedirs(os.path.join(paths['data'],subdb), exist_ok=True)
                data = filt.process(np.random.normal(size=duration * 16000))[0]
                write(os.path.join(paths['data'],subdb, str(k) + '.wav'), sampling_rate, data/data.max())
            labels = np.ones(files)
            np.save(os.path.join(paths['data'],subdb + '_labels.npy'), labels)

            # normal class
            files = 100
            duration = 60
            sampling_rate = 16000
            subdb = 'normal'
            filt = FIR_filter(type='lowpass', f=4000, taps=100, fs=sampling_rate, axis=0)
            for k in range(files):
                os.makedirs(os.path.join(paths['data'],subdb), exist_ok=True)
                data = filt.process(np.random.normal(size=duration * 16000))[0]
                write(os.path.join(paths['data'],subdb, str(k) + '.wav'), sampling_rate, data/data.max())
            labels = np.zeros(files)
            np.save(os.path.join(paths['data'],subdb + '_labels.npy'), labels)

    def _get_binary_anomaly(self, paths):
        subdbs = np.unique(self['data']['subdb'])
        labels = [None] * len(subdbs)
        for k,subdb in enumerate(subdbs):
            subdb_id = np.where([s == subdb for s in self['data']['subdb']])[0]
            reorder = np.array([int(os.path.splitext(filename)[0]) \
                                for filename in \
                                [self['data']['filename'][k] for k in subdb_id]])
            labels[k] = np.load(os.path.join(paths['meta'], subdb + '_labels.npy'))[reorder]
        return listnp_combine(labels)