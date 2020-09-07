import dcase_util
import pandas

from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataset.dataset import dataset
from dabstract.dataprocessor.processors import *
from dabstract.utils import stringlist2ind

class DCASE2020Task1A(dataset):
    def __init__(self,
                 paths=None,
                 split=None,
                 filter=None,
                 test_only=0,
                 **kwargs):
        # init dict abstract
        super().__init__(name=self.__class__.__name__,
                         paths=paths,
                         split=split,
                         filter=filter,
                         test_only=test_only)

    # Data: get data
    def set_data(self, paths):
        # audio
        chain = processing_chain().add(WavDatareader(select_channel=0))
        self.add('audio',self.dict_from_folder(paths['data'],map_fct=chain,save_path=os.path.join(paths['feat'],self.__class__.__name__, 'audio', 'raw')))
        # get meta
        labels = pandas.read_csv(os.path.join(paths['meta'], 'meta.csv'), delimiter='\t')
        # make sure audio and meta is aligned
        filenames = labels['filename'].to_list()
        resort = np.array([filenames.index(os.path.join('audio',filename)) for filename in self['audio']['example']])
        labels = labels.reindex(resort)
        # add labels
        self.add('identifier', labels['identifier'].to_list())
        self.add('source', labels['source_label'].to_list())
        self.add('scene', labels['scene_label'].to_list())
        self.add('scene_id', stringlist2ind(labels['scene_label'].to_list()))
        self.add('group', stringlist2ind(labels['identifier'].to_list()))
        return self

    def prepare(self,paths):
        dcase_util.datasets.dataset_factory(
            dataset_class_name='TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet',
            data_path=os.path.split(os.path.split(paths['data'])[0])[0],
        ).initialize()