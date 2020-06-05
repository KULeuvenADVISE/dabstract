import dcase_util
import pandas

from dabstract.dataprocessor.processing_chain import processing_chain
from dabstract.dataset.dataset import dict_dataset
from dabstract.dataprocessor.processors import *
from dabstract.utils import listnp_combine, group_to_ind

class DCASE2020Task1B(dict_dataset):
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
        chain = processing_chain().add(WavDatareader(select_channel=0))
        self.add('audio',self.dict_from_folder(paths['data'],transform=chain.process_single,save_info=True,save_path=paths['tmp']))
        # add labels
        labels = pandas.read_csv(os.path.join(paths['meta'],'meta.csv'), delimiter='\t')
        self.add('identifier', labels['identifier'].to_list())
        self.add('source', labels['source_label'].to_list())
        self.add('scene', labels['source_label'].to_list())
        self.add('group', group_to_ind(labels['identifier'].to_list()))
        return self

    #naar 1 abstractiecode gaan
    # actieve set gebruiken!
    #tranformenfilter zijn klasses!
    #immutableobjects

    def prepare(self,paths):
        # dcase_util.datasets.dataset_factory(
        # dataset_class_name='TAUUrbanAcousticScenes_2020_3Class_DevelopmentSet',
        # data_path=os.path.split(os.path.split(paths['data'])[0])[0],
        # ).initialize()
        pass

