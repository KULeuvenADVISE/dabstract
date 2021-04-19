import numpy as np
import numbers
import warnings
import copy

from dabstract.abstract import  DictSeqAbstract, SeqAbstract, MapAbstract, Abstract
from dabstract.dataset.helpers import get_dir_info

from typing import Any, List, Optional, TypeVar, Callable, Dict, Iterable, Union
tvDataset = TypeVar("Dataset")

class Container():
    pass

class MetaContainer(Container, Abstract):
    time_meta_types = ('multi_time_label')

    def __init__(self,
                 data: Iterable = None,
                 meta_type: str = 'auto',
                 duration: Union[int, List[int]] = None,
                 time_step: Union[int, List[int]] = None,
        ):
        # init
        super().__init__(data=data)
        self.meta_type = meta_type
        self.duration = duration
        self.time_step = time_step
        if (self.duration is not None) or (self.time_step is not None):
            assert meta_type in self.time_meta_types, \
                "You should not specify a duration when input type does not have time dependency"
        if meta_type in self.time_meta_types:
            assert (self.duration is not None) or (self.time_step is not None), \
                "You should specify a duration when input type does has time dependency"
            if isinstance(self.duration, numbers.Integral):
                self.duration = self.duration * np.ones(len(data))
            else:
                assert len(self.duration) == len(self)
            if isinstance(self.time_step, numbers.Integral):
                self.time_step = self.time_step * np.ones(len(data))
            else:
                assert len(self.time_step) == len(self)

        # checks
        if self.meta_type == 'auto':
            raise NotImplementedError("Please manually select an input type. Don't be lazy. ;)")

        elif self.meta_type == 'single_label':
            assert isinstance(data, Iterable)

        elif self.meta_type == 'multi_label':
            assert isinstance(data, Iterable), \
                "data should be a nested np.arrray or list in case its input_type is multi_label"
            assert isinstance(data[0], Iterable), \
                "data should be a nested np.arrray or list in case its input_type is multi_label"

        elif self.meta_type == 'multi_time_label':
            assert isinstance(data, Iterable), \
                "data should be a np.arrray or list in case its input_type is multi_time_label"
            assert isinstance(data[0], np.ndarray), \
                "data[k] should be a np.ndarray in case its input_type is multi_time_label"
            assert len(data[0].shape), \
                "data[k] should be a np.ndarray matrix in case its input_type is multi_time_label"

        else:
            raise NotImplementedError("%s is not a valid input type." % self.meta_type)

    def get(
        self,
        index: int,
        return_info: bool = False,
        read_range = None,
        **kwargs
    ) -> Union[List, np.ndarray, Any]:
        # get data
        if self._abstract:
            data, info = self._data.get(index, return_info=True)
        else:
            data, info = self._data[index], {}
        data = copy.deepcopy(data)

        # reformat meta
        if read_range is not None:
            if self.meta_type == 'multi_time_label':
                start_idx = np.where(data[:,2]>read_range[0])[0]
                stop_idx = np.where(data[:,1]>read_range[1])[0]
                #if len(start_idx) > 0 and len(stop_idx) > 0:
                data = data[np.max([start_idx[0],0]):np.min([stop_idx[0],data.shape[0]])]
                data[0,1] = np.max([read_range[0],data[0,1]])
                data[-1,2] = np.min([read_range[1],data[-1,2]])
                # elif len(start_idx) == 0:
                #     data = data[None,0,:]
                #     data[0,1] = read_range[0]
                # elif len(stop_idx) == 0:
                #     data = data[None,0,:]
                #     data[0,2] = read_range[1]
                # else:
                #     pass
            else:
                raise NotImplementedError("%s is not a valid input type to cope but read_range input. Something is going wrong. Please check." % self.input_type)

        return (data, info) if return_info else data

    def get_duration(self, index: int = None):
        if index is None:
            return self.duration if self.meta_type in self.time_meta_types else None
        else:
            return self.duration[index] if self.meta_type in self.time_meta_types else None

    def get_time_step(self, index: int = None):
        if index is None:
            return self.time_step if self.meta_type in self.time_meta_types else None
        else:
            return self.time_step[index] if self.meta_type in self.time_meta_types else None

    def get_split_len(self, index: int = None):
        return self.get_duration(index=index)

    def is_splittable(self):
        return self.meta_type in self.time_meta_types


class FolderContainer(Container, DictSeqAbstract):
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
        return "%s containing: %s" % (self.__class__.__name__, str(self.keys()))

    def _get_info(self, key: str, index: int = None):
        assert key in self['info'][0], '%s not available in %s instance' % (key, self.__class__.__name__)
        if index is None:
            return np.array([tmp[key] for tmp in self['info']])
        else:
            return self['info'][index][key]
        return

    def get_info(self, index: int = None):
        if index is None:
            return self['info']
        else:
            return self['info'][index]

    def is_splittable(self):
        # assumes that this is the case for ALL info
        # should we do a check on everything or assume this is always the case?
        return 'time_step' in self['info'][0] and 'output_shape' in self['info'][0]

    def get_split_len(self, index: int = None):
        return self.get_samples(index=index)


class WavFolderContainer(FolderContainer):
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

class FeatureFolderContainer(FolderContainer):
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