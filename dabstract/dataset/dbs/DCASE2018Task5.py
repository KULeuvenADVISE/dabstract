import dcase_util
import pandas

from dabstract.dataprocessor.processing_chain import ProcessingChain
from dabstract.dataset.dataset import Dataset
from dabstract.dataprocessor.processors import *
from dabstract.utils import stringlist2ind


class DCASE2018Task5(Dataset):
    """DCASE2020Task1A dataset

    This class downloads the datasets and prepares it in the dabstract format.

    Parameters
    ----------
    paths : dict or str:
        Path configuration in the form of a dictionary.
        For example::
            $   paths={ 'data': path_to_data,
            $           'meta': path_to_meta,
            $           'feat': path_to_feat}
    test_only : bool
        To specify if this dataset should be used for testing or both testing and train.
        This is only relevant if multiple datasets are combined and set_xval() is used.
        For example::
            test_only = 0 -> use for both train and test
            test_only = 1 -> use only for test

    check dabstract.dataset.dataset.Dataset for more info

    Returns
    -------
        DCASE2020Task1B dataset class
    """

    def __init__(self, paths=None, test_only=0, **kwargs):
        # init dict abstract
        super().__init__(name=self.__class__.__name__, paths=paths, test_only=test_only)

    # Data: get data
    def set_data(self, paths):
        """Set the data"""

        # audio
        chain = ProcessingChain().add(WavDatareader(select_channel=0))
        from dabstract.dataset.helpers import FolderDictSeqAbstract

        self.add(
            "audio",
            FolderDictSeqAbstract(
                paths["data"],
                map_fct=chain,
                file_info_save_path=os.path.join(
                    paths["feat"], self.__class__.__name__, "audio", "raw"
                ),
            ),
        )
        # get meta
        if os.path.exists(os.path.join(paths["meta"], "meta_dabstract.txt")):
            labels = pandas.read_csv(
                os.path.join(paths["meta"], "meta_dabstract.txt"), delimiter="\t", header=None
            )
        else:
            labels = pandas.read_csv(
                os.path.join(paths["meta"], "meta.txt"), delimiter="\t", header=None
            )
            # make sure audio and meta is aligned
            filenames = labels[0].to_list()
            resort = np.array(
                [
                    filenames.index("audio/" + filename)
                    for filename in self["audio"]["example"]
                ]
            )
            labels = labels.reindex(resort)
            labels.to_csv(os.path.join(paths["meta"], "meta_dabstract.txt"), sep="\t", header = False, index=False)

        # add labels
        self.add("identifier", labels[2].to_list(), lazy=False)
        #self.add("source", [filename for filename in filenames], lazy=False)
        self.add("scene", labels[1].to_list(), lazy=False)
        self.add(
            "scene_id", stringlist2ind(self['scene']), lazy=False
        )
        self.add("group", stringlist2ind(self['identifier']), lazy=False)
        return self

    def prepare(self, paths):
        pass
        # """Prepare the data"""
        # dcase_util.datasets.dataset_factory(
        #     dataset_class_name="DCASE2018_Task5_DevelopmentSet",
        #     data_path=os.path.split(os.path.split(paths["data"])[0])[0],
        # ).initialize()