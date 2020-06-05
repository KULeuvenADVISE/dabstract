from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataset.dataset import dict_dataset
from dabstract.dataprocessor.processors import *
from dabstract.utils import listnp_combine

class EXAMPLE(dict_dataset):
    def __init__(self,
                 paths=None,
                 split=None,
                 filter=None,
                 xval=None,
                 test_only=0,
                 **kwargs):
        # init dict abstract
        super().__init__(name=self.__class__.__name__,
                         paths=paths,
                         split=split,
                         filter=filter,
                         xval=xval,
                         test_only=test_only)

    # Data: get data
    def set_data(self, paths):
        # audio
        chain = processing_chain().add(WavDatareader())
        self.add('data',self.dict_from_folder(paths['data'],transform=chain.process_single))
        # add labels
        if not hasattr(self, 'binary_anomaly'):
            subdbs = np.unique(self['data']['subdb'])
            labels = [None] * len(subdbs)
            for k,subdb in enumerate(subdbs):
                subdb_id = np.where([s == subdb for s in self['data']['subdb'][:]])[0]
                reorder = np.array([int(os.path.splitext(filename)[0]) \
                                    for filename in \
                                    [self['data']['filename'][k] for k in subdb_id]])
                labels[k] = np.load(os.path.join(paths['meta'], subdb + '_labels.npy'))[reorder]

            self.binary_anomaly = listnp_combine(labels)
            self.binary_anomaly = self.binary_anomaly.reshape((len(self.binary_anomaly), 1))
        self.add('binary_anomaly',self.binary_anomaly)
        self.add('group', self['data']['subdb'])
        return self

    def prepare(self,paths):
        if not os.path.isdir(paths['data']):
            from scipy.io.wavfile import write
            # normal class
            files = 20
            duration = 60
            sampling_rate = 16000
            subdb = 'normal'
            for k in range(files):
                os.makedirs(os.path.join(paths['data'],subdb), exist_ok=True)
                write(os.path.join(paths['data'],subdb, str(k) + '.wav'), sampling_rate, 0.1 * np.random.rand(duration * 16000))
            labels = np.zeros(files)
            np.save(os.path.join(paths['data'],subdb + '_labels.npy'), labels)

            # abnormal class
            files = 20
            duration = 60
            sampling_rate = 16000
            subdb = 'abnormal'
            for k in range(files):
                os.makedirs(os.path.join(paths['data'],subdb), exist_ok=True)
                write(os.path.join(paths['data'],subdb, str(k) + '.wav'), sampling_rate, np.random.rand(duration * 16000))
            labels = np.ones(files)
            np.save(os.path.join(paths['data'],subdb + '_labels.npy'), labels)