import dcase_util
import pandas

from dabstract.dataprocessor.processing_chain import ProcessingChain
from dabstract.dataset.dataset import Dataset
from dabstract.dataprocessor.processors import *
from dabstract.utils import stringlist2ind


class DCASE2020Task1A(Dataset):
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
        labels = pandas.read_csv(
            os.path.join(paths["meta"], "meta.csv"), delimiter="\t"
        )
        # make sure audio and meta is aligned
        filenames = labels["filename"].to_list()
        resort = np.array(
            [
                filenames.index("audio/" + filename)
                for filename in self["audio"]["example"]
            ]
        )
        labels = labels.reindex(resort)
        # add labels
        self.add("identifier", labels["identifier"].to_list(), lazy=False)
        self.add("source", labels["source_label"].to_list(), lazy=False)
        self.add("scene", labels["scene_label"].to_list(), lazy=False)
        self.add(
            "scene_id", stringlist2ind(labels["scene_label"].to_list()), lazy=False
        )
        self.add("group", stringlist2ind(labels["identifier"].to_list()), lazy=False)
        return self

    def prepare(self, paths):
        """Prepare the data"""

        dcase_util.datasets.dataset_factory(
            dataset_class_name="TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet",
            data_path=os.path.split(os.path.split(paths["data"])[0])[0],
        ).initialize()
