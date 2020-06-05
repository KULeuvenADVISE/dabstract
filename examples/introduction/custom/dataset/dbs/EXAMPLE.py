from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataset.dataset import dataset
from dabstract.dataprocessor.processors import *
from dabstract.utils import listnp_combine

class EXAMPLE(dataset):
    def __init__(self,
                 paths=None,
                 split=None,
                 select=None,
                 xval=None,
                 test_only=0,
                 xval_func=None,
                 xval_save=False,
                 xval_dir='',
                 xval_overwrite=False,
                 **kwargs):
        # init dict abstract
        super().__init__(name=self.__class__.__name__,
                         paths=paths,
                         split=split,
                         select=select,
                         xval=xval,
                         test_only=test_only,
                         xval_func=xval_func,
                         xval_save=xval_save,
                         xval_dir=xval_dir,
                         xval_overwrite=xval_overwrite)

    # Data: get data
    def set_data(self, paths):
        # audio
        chain = processing_chain().add(WavDatareader())
        self.add('data',self.dict_from_folder(paths['data'],map_fct=chain, save_path=paths['tmp']))
        # add labels
        self.add('binary_anomaly',self._get_binary_anomaly(paths))
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