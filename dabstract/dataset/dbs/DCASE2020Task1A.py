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
        chain = processing_chain().add(WavDatareader())
        self.add('audio',self.dict_from_folder(paths['data'],map_fct=chain,save_path=paths['feat']))
        # add labels
        labels = pandas.read_csv(os.path.join(paths['meta'],'meta.csv'), delimiter='\t')
        self.add('identifier', labels['identifier'].to_list())
        self.add('source', labels['source_label'].to_list())
        self.add('scene', labels['source_label'].to_list())
        self.add('group', stringlist2ind(labels['identifier'].to_list()))
        return self

    def prepare(self,paths):
        dcase_util.datasets.dataset_factory(
            dataset_class_name='TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet',
            data_path=os.path.split(os.path.split(paths['data'])[0])[0],
        ).initialize()