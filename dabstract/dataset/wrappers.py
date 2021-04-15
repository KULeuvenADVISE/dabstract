import numpy as np
import numbers

from dabstract.abstract.abstract import DictSeqAbstract, SeqAbstract, MapAbstract, Abstract
from dabstract.dataset.helpers import get_dir_info

from typing import Any, List, Optional, TypeVar, Callable, Dict, Iterable, Union
tvDataset = TypeVar("Dataset")

class MetaWrapper(Abstract):
    time_input_types = ('multi_time_label','multi_endtime_label')

    # ToDo: add summaries related to meta
    def __init__(self,
                 data: Iterable = None,
                 input_type: str = 'auto',
                 output_type: str = 'as_is',
                 sample_length: int = None,
        ):
        # init
        super().__init__()
        self._data
        self.input_type = input_type
        self.output_type = output_type
        self.sample_length = sample_length
        if self.sample_length is not None:
            assert input_type in self.time_input_types, \
                "You should not specify a duration when input type does not have time dependency"
            if isinstance(self.sample_length, numbers.Integral):
                self.sample_length = self.sample_length * np.ones(len(data))
            else:
                assert len(self.sample_length) == len(self)

        # checks
        if self.input_type == 'auto':
            raise NotImplementedError("Please manually select an input type. Don't be lazy.")

        elif self.input_type == 'single_label':
            assert self.output_type == 'as_is', "Only output_type as_is is supported with single_label"
            assert isinstance(data, Iterable)

        elif self.input_type == 'multi_label':
            assert self.output_type in ('as_is','multi_label','majority','random_tie_majority'), \
                "Only output_type as_is, multi_label and majority is supported with input_type multi_label"
            assert isinstance(data, Iterable), \
                "data should be a nested np.arrray or list in case its input_type is multi_label"
            assert isinstance(data[0], Iterable), \
                "data should be a nested np.arrray or list in case its input_type is multi_label"

        elif self.input_type == 'multi_endtime_label':
            assert isinstance(data, Iterable), \
                "data should be a np.arrray or list in case its input_type is multi_endtime_label"
            assert isinstance(data[0], np.ndarray), \
                "data[k] should be a np.ndarray in case its input_type is multi_endtime_label"
            assert len(data[0].shape), \
                "data[k] should be a np.ndarray matrix in case its input_type is multi_endtime_label"

        elif self.input_type == 'multi_time_label':
            assert isinstance(data, Iterable), \
                "data should be a np.arrray or list in case its input_type is multi_time_label"
            assert isinstance(data[0], np.ndarray), \
                "data[k] should be a np.ndarray in case its input_type is multi_time_label"
            assert len(data[0].shape), \
                "data[k] should be a np.ndarray matrix in case its input_type is multi_time_label"
        else:
            raise NotImplementedError("%s is not a valid input type." % self.input_type)

    def get(
        self,
        index: int,
        return_info: bool = False,
        **kwargs
    ) -> Union[List, np.ndarray, Any]:
        # get data
        data = self._data[index]

        if return_info:
            data, info = data
        else:
            info = {}

        # reformat meta
        if self.input_type == 'auto':
            raise NotImplementedError("Please manually select an input type. Don't be lazy.")

        elif self.input_type == 'single_label':
            pass

        elif self.input_type == 'multi_label':

            if self.output_type == 'majority':
                values, counts = np.unique(data, return_counts=True)
                return values[np.argmax(counts)]

            elif self.output_type == 'random_tie_majority':
                values, counts = np.unique(data, return_counts=True)
                return values[np.argmax(np.random.random(len(counts)) * (counts == counts.max()))]

        elif self.input_type == 'multi_endtime_label':
            # update data if different range is required
            if 'read_range' in kwargs:
                pass

            if self.output_type == 'multi_label':
                pass

            elif self.output_type == 'majority':
                values, counts = np.unique(data[:,1], return_counts=True)
                return values[np.argmax(counts)]

            elif self.output_type == 'random_tie_majority':
                values, counts = np.unique(data[:,1], return_counts=True)
                return values[np.argmax(np.random.random(len(counts)) * (counts == counts.max()))]

            elif self.output_type == 'first':
                return data[:, 1][0]

            elif self.output_type == 'center':
                mid_idx = data[:, 0].max()/2
                return data[:, 1][np.where(data[:, 0]>mid_idx)[0][0]]

            elif self.output_type == 'last':
                return data[:, 1][-1]


        else:
            raise NotImplementedError("%s is not a valid input type." % self.input_type)

        return (data, info) if return_info else data

    def get_fs(self, index: int = None):
        return 1 if self.input_type in self.time_input_types else None

    def get_output_shape(self, index: int = None):
        return None

    def get_samples(self, index: int = None):
        return self.sample_length if self.input_type in self.time_input_types else None

    def get_duration(self, index: int = None):
        return self.sample_length if self.input_type in self.time_input_types else None

    def get_time_step(self, index: int = None):
        return 1

class FolderWrapper(DictSeqAbstract):
    """Get meta information of the files in a directory and place them in a DictSeq

    This function gets meta information (e.g. sampling frequency, length) of files in your provided directory.
    It return a FolderDictSeq with the filenames/information/subfolders.
    A FolderDictSeq is inherited from DictSeq and has similar functionality. However,
    for a FolderDictSeq the active_keys are fixed to 'data'. In essence FolderDictSeq is a
    data container showing information of a walk through a folder.
    Additionally, this format keeps track of relevant information to either wav or numpy files.
    prepare_feat and add_split only work on data fields that have this structure.

    Parameters
    ----------
    path : str
        path to the directory to check
    extension : str
        only evaluate files with that extension
    map_fct : Callable
        add a mapping function y = f(x) to the 'data'
    filepath : str
        in case you already have the files you want to obtain information from,
        the dir tree search is not done and this is used instead
    file_info_save_path: : str
        save the information to this location
        this function can be costly, so saving is useful
    overwrite_file_info : bool
        overwrite file info file

    Returns
    -------
    DictSeqAbstract : DictSeqAbstract
        dictseq containing file information as a list,
        formatted as::
            output['filepath'] = list of paths to files
            output['example'] = example string (i.e. filename without extension)
            output['filename'] = filename
            output['subdb'] = relative subdirectory (starting from 'path') to file
            output['info'][file_id] = { 'output_shape': .., #output shape of the wav file
                                        'fs': .., # sampling frequency
                                        'time_step' ..: # sample period
                                        }
    """

    def __init__(
        self,
        path: str,
        extension: str = ".wav",
        map_fct: Callable = None,
        file_info_save_path: bool = None,
        filepath: str = None,
        overwrite_file_info: bool = False,
        info: List[Dict] = None,
        **kwargs
    ):
        super().__init__()
        if "save_path" in kwargs:
            file_info_save_path = kwargs["save_path"]
            warnings.warn(
                "save_path is deprecated in dataset.py/dict_from_folder(). Change to 'file_info_save_path'",
                DeprecationWarning,
            )
        # get info
        fileinfo = get_dir_info(
            path,
            extension=extension,
            file_info_save_path=file_info_save_path,
            filepath=filepath,
            overwrite_file_info=overwrite_file_info,
        )
        # overwrite file info
        if info is not None:
            fileinfo["info"] = info
        # add data
        self.add("data", fileinfo["filepath"], info=fileinfo["info"])
        self.add_dict(fileinfo, lazy=False)
        # add map
        if map_fct is not None:
            self["data"] = MapAbstract(self["data"], map_fct=map_fct)
        # set active key
        self._set_active_keys("data")

    def set_active_keys(self, keys: List[str]) -> None:
        """Disables set of active keys"""
        raise Exception(
            "A FolderDictSeqAbstract should always have data as the only active key. Setting not possible. Please use DictSeqAbstract if other functionality is needed."
        )

    def reset_active_keys(self) -> None:
        """Disables reset of active keys"""
        raise Exception(
            "A FolderDictSeqAbstract should always have data as the only active key. Resetting not possible. Please use DictSeqAbstract if other functionality is needed."
        )

    def __setitem__(self, k: int, v: Any) -> None:
        if isinstance(k, str):
            self._data[k] = v
        elif isinstance(k, numbers.Integral):
            self._data["data"][k] = v
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        """string print representation of function"""
        return "FolderWrapper containing: " + str(self.keys())

class WavFolderWrapper(FolderWrapper):
    # ToDo: add summaries related to the wav folder
    def __init__(
        self,
        path: str,
        map_fct: Callable = None,
        file_info_save_path: bool = None,
        filepath: str = None,
        overwrite_file_info: bool = False,
        info: List[Dict] = None,
        **kwargs
    ):
        super().__init__(path=path,
                         extension='.wav',
                         map_fct = map_fct,
                         file_info_save_path=file_info_save_path,
                         filepath=filepath,
                         overwrite_file_info=overwrite_file_info,
                         info=info,
                         **kwargs)

    def _get_info(self, key: str, index: int = None):
        assert key in self['info'][0], '%s not available in %s instance' % (key, self.__class__.__name__)
        if index is None:
            return np.array([tmp[key] for tmp in self['info']])
        else:
            return self['info'][index][key]
        return

    def get_info(self, index: int = None):
        if index is None:
            self['info']
        else:
            return self['info'][index]

    def get_fs(self, index: int = None):
        return self._get_info(key='fs', index=index)

    def get_output_shape(self, index: int = None):
        return self._get_info(key='output_shape', index=index)

    def get_samples(self, index: int = None):
        tmp = self._get_info(key='output_shape', index=index)
        return tmp[:,0] if index is None else tmp[0]

    def get_duration(self, index: int = None):
        return self.get_samples(index=index)/self.get_fs(index=index)

    def get_time_step(self, index: int = None):
        return self._get_info(key='time_step', index=index)

class FeatureFolderWrapper(FolderWrapper):
    # ToDo: add summaries related to the feature folder
    def __init__(
        self,
        path: str,
        map_fct: Callable = None,
        file_info_save_path: bool = None,
        filepath: str = None,
        overwrite_file_info: bool = False,
        info: List[Dict] = None,
        **kwargs
    ):
        super().__init__(path=path,
                         extension='.npy',
                         map_fct = map_fct,
                         file_info_save_path=file_info_save_path,
                         filepath=filepath,
                         overwrite_file_info=overwrite_file_info,
                         info=info,
                         **kwargs)

    def _get_info(self, key: str, index: int = None):
        assert key in self['info'][0], '%s not available in %s instance' % (key, self.__class__.__name__)
        if index is None:
            return np.array([tmp[key] for tmp in self['info']])
        else:
            return self['info'][index][key]
        return

    def get_info(self, index: int = None):
        if index is None:
            self['info']
        else:
            return self['info'][index]

    def get_fs(self, index: int = None):
        return self._get_info(key='fs', index=index)

    def get_output_shape(self, index: int = None):
        return self._get_info(key='output_shape', index=index)

    def get_samples(self, index: int = None):
        tmp = self._get_info(key='output_shape', index=index)
        return tmp[:,0] if index is None else tmp[0]

    def get_duration(self, index: int = None):
        return self.get_samples(index=index)/self.get_fs(index=index)

    def get_time_step(self, index: int = None):
        return self._get_info(key='time_step', index=index)