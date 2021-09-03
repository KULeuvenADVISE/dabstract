from abc import ABC

import numpy as np
import numbers
import warnings
import copy
import cv2 as cv

import datetime as dt

from dabstract.abstract import DictSeqAbstract, SeqAbstract, MapAbstract, Abstract, SelectAbstract, SplitAbstract, \
    SampleReplicate, Select
from dabstract.dataprocessor import ProcessingChain, Processor
from dabstract.dataset.helpers import get_dir_info
from dabstract.utils import listdictnp_combine

import abc
from typing import Any, List, Optional, TypeVar, Callable, Dict, Iterable, Union

tvDataset = TypeVar("Dataset")


class MethWrap(ABC):
    pass


class Container(MethWrap, Abstract, ABC):
    pass


class InfoMethWrap(MethWrap):
    @property
    def required_info(self):
        return []

    def _info_checks(self, info: Union[Iterable, None]):
        if info is None:
            return [{}] * len(self)
        else:
            return info
        assert all([all([key in self.required_info for key in tmp.keys()]) for tmp in info]), \
            "keys in the info argument do not all contain the requirements (%s)" % str(self.required_info)

    def _get_info(self, key: str, index: int = None):
        assert key in self['info'][0], '%s not available in %s instance' % (key, self.__class__.__name__)
        if index is None:
            return np.array([tmp[key] for tmp in self['info']])
        else:
            return self['info'][index][key]

    def get_info(self, index: int = None):
        if index is None:
            return self['info']
        else:
            return self['info'][index]

    def concat(self, *arg: Any, **kwargs: dict):
        raise NotImplementedError("Concatenation not supported. \
        Containers should always have a fixed length. \
        If you want to concat other containers to it, please use a SeqAbstract.")


class InfoContainer(InfoMethWrap, DictSeqAbstract, Container):
    def __init__(self, data: Union[dict, Iterable] = None, info: Iterable = None):
        super().__init__()
        if isinstance(data, dict):
            self.add_dict(data)
        elif data is not None:
            self.add('data', data)
        if info in self.keys():
            assert info is None, "info should be None when data contains a dict with info as key."
        self.add('info', self._info_checks(info))


class TimeMethWrap(InfoMethWrap):
    @property
    def required_info(self):
        return ['time_axis', 'output_shape', 'time_step']

    @property
    def is_splittable(self):
        # assumes that this is the case for ALL info
        # should we do a check on everything or assume this is always the case?
        return ((self['info'][0]['time_axis'] is not None) if 'time_axis' in self['info'][
            0] else False) and 'output_shape' in self['info'][0]

    def get_output_shape(self, index: int = None):
        return self._get_info(key='output_shape', index=index)

    def get_samples(self, index: int = None):
        tmp = self._get_info(key='output_shape', index=index)
        return tmp[:, 0] if index is None else tmp[0]

    def get_duration(self, index: int = None):
        return self.get_samples(index=index) * self.get_time_step(index=index)

    def get_time_step(self, index: int = None):
        return self._get_info(key='time_step', index=index)

    def get_split_len(self, index: int = None):
        return self.get_samples(index=index)

    @staticmethod
    def get_augmented_wrapper(base_class):
        if base_class == SplitAbstract:
            return SplitTimeContainer
        elif base_class == SelectAbstract:
            return SelectTimeContainer
        else:
            return None


class TimeContainer(TimeMethWrap, DictSeqAbstract, Container):
    pass


class AudioMethWrap(TimeMethWrap):
    def get_fs(self, index: int = None):
        return self._get_info(key='fs', index=index)


class AudioContainer(AudioMethWrap, DictSeqAbstract, Container):
    pass


class FolderContainer(InfoMethWrap, DictSeqAbstract, Container):
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
            type=None,
            extension: str = '.ts',
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
            type=type,
            extension=extension,
            file_info_save_path=file_info_save_path,
            filepath=filepath,
            overwrite_file_info=overwrite_file_info,
        )
        # overwrite file info
        if info is not None:
            fileinfo["info"] = info
        fileinfo["info"] = self._info_checks(fileinfo["info"])
        # init class
        # ToDo check why info is needed to propagate (legacy prepare_feat?)
        self.add("data", fileinfo["filepath"], info=fileinfo["info"])
        self.add_dict(fileinfo, lazy=False)
        # add map
        if map_fct is not None:
            self["data"] = MapAbstract(self["data"], map_fct=map_fct)
        # set active key
        self._set_active_keys("data")
        # check info
        if info in self.keys():
            assert info is None, "info should be None when data contains a dict with info as key."

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


class TimeFolderContainer(TimeMethWrap, FolderContainer):
    pass


class SplitTimeContainer(TimeMethWrap, SplitAbstract, Container):
    pass
    # #ToDo: Could be a thing to swith to Abstract methods?
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     a = self['info']
    #     a[0]
    #     self._data_rep = {}
    #     for key in self._data.keys():
    #         if key not in self._data.get_active_keys():
    #             self._data_rep[key] = SampleReplicate(self._data[key], factor=self._splits, lazy=self._data._lazy[key])

    # def get(
    #         self, index: int, return_info: bool = False, *args: List, **kwargs: Dict
    # ) -> Union[List, np.ndarray, Any]:
    #     if isinstance(index, numbers.Integral):
    #         return super().get(index, return_info=return_info, *args, **kwargs)
    #     elif isinstance(index, str):
    #         return self._data_rep[index]


class SelectTimeContainer(TimeMethWrap, SelectAbstract, Container):
    pass


class SelectAudioContainer(AudioMethWrap, SelectAbstract, Container):
    pass


class AudioFolderContainer(AudioMethWrap, FolderContainer):
    # ToDo: add summaries related to the wav folder
    def __init__(
            self,
            path: str,
            extension: str = '.wav',
            map_fct: Callable = None,
            file_info_save_path: bool = None,
            filepath: str = None,
            overwrite_file_info: bool = False,
            info: List[Dict] = None,
            **kwargs
    ):
        super().__init__(path=path,
                         type='audio',
                         extension=extension,
                         map_fct=map_fct,
                         file_info_save_path=file_info_save_path,
                         filepath=filepath,
                         overwrite_file_info=overwrite_file_info,
                         info=info,
                         **kwargs)

    @staticmethod
    def get_feature_container():
        return AudioFeatureFolderContainer


class AudioFeatureFolderContainer(AudioMethWrap, FolderContainer):
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
                         map_fct=map_fct,
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
        return tmp[:, 0] if index is None else tmp[0]

    def get_duration(self, index: int = None):
        return self.get_samples(index=index) / self.get_fs(index=index)

    def get_time_step(self, index: int = None):
        return self._get_info(key='time_step', index=index)


# ToDo: change structure of inheritance such that is modular such as the FolderContainer and so on
class MetaContainer(Container):
    time_meta_types = ('multi_label_time_range', 'binary_time_point')

    def __init__(self,
                 data: Iterable = None,
                 meta_type: str = 'auto',
                 output_meta_type: str = 'as_is',
                 duration: Union[int, List[int]] = None,
                 time_step: Union[int, List[int]] = None,
                 ):
        # init
        super().__init__(data=data)
        self.meta_type = meta_type
        self.output_meta_type = output_meta_type
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

        elif self.meta_type == 'multi_label_time_range':
            assert isinstance(data, Iterable), \
                "data should be a np.arrray or list in case its input_type is multi_label_time_range"
            assert isinstance(data[0], np.ndarray), \
                "data[k] should be a np.ndarray in case its input_type is multi_label_time_range"
            assert len(data[0].shape), \
                "data[k] should be a np.ndarray matrix in case its input_type is multi_label_time_range"

        elif self.meta_type == 'binary':
            raise NotImplementedError

        elif self.meta_type == 'binary_time_range':
            raise NotImplementedError

        elif self.meta_type == 'binary_time_point':
            assert isinstance(data, Iterable), \
                "data should be a np.arrray or list in case its input_type is binary_time_point"

        else:
            raise NotImplementedError("%s is not a valid input type." % self.meta_type)

        if self.output_meta_type == 'as_is':
            pass
        elif self.output_meta_type == 'binary':
            assert self.meta_type in ('binary', 'binary_time_point')
        elif self.output_meta_type == 'multi_label':
            assert self.meta_type in ('multi_label', 'multi_label_time_range')
        else:
            raise NotImplementedError("%s is not a valid output type." % self.output_meta_type)

    def get(
            self,
            index: int,
            return_info: bool = False,
            read_range=None,
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
            if self.meta_type == 'multi_label_time_range':
                start_idx = np.where(data[:, 2] > read_range[0])[0]
                stop_idx = np.where(data[:, 1] > read_range[1])[0]
                data = data[np.max([start_idx[0], 0]):np.min([stop_idx[0], data.shape[0]])]
                data[:, [1, 2]] -= read_range[0]
                data[0, 1] = np.max([0, data[0, 1]])
                data[-1, 2] = np.min([read_range[1], data[-1, 2]])
            elif self.meta_type == 'binary_time_point':
                if data is not None:
                    data = np.any((data > read_range[0]) & (data < read_range[1]))
                else:
                    data = False
            else:
                raise NotImplementedError(
                    "%s is not a valid input type to cope but read_range input. Something is going wrong. Please check." % self.input_type)

        # reformat output
        if self.output_meta_type == 'multi_label':
            if self.meta_type == 'multi_time_label':
                data = data[None, :, 0]
            elif self.meta_type == 'multi_label':
                pass
            else:
                raise NotImplementedError("multi_label only supported for meta_type multi_time_label and multi_label")
        elif self.output_meta_type == 'binary':
            if self.meta_type == 'binary_time_point':
                if not (type(data) == bool):
                    if data is None:
                        data = 0
                    elif not isinstance(data, (bool, np.bool_)):
                        data = 1
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

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

    @property
    def is_splittable(self):
        return self.meta_type in self.time_meta_types

# class CameraFolderContainer(FolderContainer):
#     # ToDo: add summaries related to the camera folder
#     def __init__(
#             self,
#             path: str,
#             extension: str = None,
#             map_fct: Callable = None,
#             file_info_save_path: bool = None,
#             filepath: str = None,
#             overwrite_file_info: bool = False,
#             info: List[Dict] = None,
#             **kwargs
#     ):
#         super().__init__(path=path,
#                          type='camera',
#                          extension=extension,
#                          map_fct=map_fct,
#                          file_info_save_path=file_info_save_path,
#                          filepath=filepath,
#                          overwrite_file_info=overwrite_file_info,
#                          info=info,
#                          **kwargs)
#
#     def get_fs(self, index: int = None):
#         return self._get_info(key='fs', index=index)
#
#     def get_output_shape(self, index: int = None):
#         return self._get_info(key='output_shape', index=index)
#
#     def get_samples(self, index: int = None):
#         return self._get_info(key='length', index=index)
#
#     def get_duration(self, index: int = None):
#         return self.get_samples(index=index) / self.get_fs(index=index)
#
#     def get_time_step(self, index: int = None):
#         return 1 / self._get_info(key='fs', index=index)
#
#
# class AsyncCameraFolderContainer(CameraFolderContainer):
#     # ToDo: add summaries related to the camera folder
#     def __init__(
#             self,
#             path: str,
#             duration: Any,
#             timestamp_ref_input: str = 'external',
#             timestamp_new_input: str = 'external',
#             timestamps_ref: Union[List[Any], str] = None,
#             timestamps_new: Union[List[Any], str] = None,
#             error_margin: float = 0,
#             extension: str = None,
#             map_fct: Callable = None,
#             file_info_save_path: bool = None,
#             filepath: str = None,
#             overwrite_file_info: bool = False,
#             info: List[Dict] = None,
#             **kwargs
#     ):
#         # get default shizzle
#         super().__init__(path=path,
#                          extension=extension,
#                          map_fct=map_fct,
#                          file_info_save_path=file_info_save_path,
#                          filepath=filepath,
#                          overwrite_file_info=overwrite_file_info,
#                          info=info,
#                          **kwargs)
#
#         # get ref timestamps
#         if timestamp_ref_input == 'external':
#             assert timestamps_ref is not None
#             assert isinstance(timestamps_ref, list)
#         elif timestamp_ref_input == 'strftime':
#             assert isinstance(timestamps_ref, str)
#             timestamps_ref = np.array(
#                 [dt.datetime.strptime(tmp, timestamps_ref).timestamp() for tmp in self['filename']])
#         else:
#             raise NotImplementedError("timestamp_ref_input should be either external or strftime.")
#
#         # get new timestamps
#         if timestamp_new_input == 'external':
#             assert timestamps_new is not None
#             assert isinstance(timestamps_new, list)
#             assert np.all(timestamps_new <= timestamps_ref.min())
#         elif timestamp_new_input == 'minmax':
#             assert timestamps_new is None
#             timestamps_new = np.arange(timestamps_ref.min(),
#                                        timestamps_ref.max() - duration,
#                                        duration)
#         else:
#             raise NotImplementedError("timestamp_new_input should be either external or minmax.")
#
#         # update self._data
#         self._prep_sync(timestamps_ref=timestamps_ref,
#                         timestamps_new=timestamps_new,
#                         duration=duration,
#                         error_margin=error_margin)
#         # remove methods which should not be available after init
#         del self._prep_sync
#         del self._new_info_handler
#         del self._new_data_handler
#
#     def _prep_sync(self,
#                    timestamps_ref: List[Any],
#                    timestamps_new: List[Any],
#                    duration: float,
#                    error_margin: float,
#                    fixed_output_shape: bool = True):
#         # init
#         new_data = dict()
#         for key in self.keys():
#             if key != 'data':
#                 new_data[key] = [None] * len(timestamps_new)
#         # update meta info
#         for k, timestamp_new in enumerate(timestamps_new):
#             frameidx, fileidx = self.get_read_info(timestamps_ref,
#                                                    timestamp_new,
#                                                    duration,
#                                                    error_margin,
#                                                    fixed_output_shape)
#
#             # prep meta
#             for key in self.keys():
#                 if key == 'info':
#                     new_data['info'][k] = self._new_info_handler(frameidx, fileidx)
#                 elif key == 'data':
#                     pass
#                 else:
#                     new_data[key][k] = [self[key][file_id] for file_id in np.unique(fileidx)]
#         # apply a data wrapper to the reader
#         new_data['data'] = MapAbstract(self['timestamp'],
#                                        ProcessingChain().add(self.CCTVreader(**self._acc_info,
#                                                                              observation_window=self.observation_window,
#                                                                              error_margin=5,
#                                                                              id=id)))
#         pass
#
#     def get_read_info(self,
#                       timestamps_ref,
#                       timestamp_new,
#                       duration,
#                       error_margin,
#                       fixed_output_shape):
#         file_idx_geq = np.where((timestamps_ref >= timestamp_new))[0]
#         file_idx_leq = np.where((timestamps_ref <= timestamp_new + duration + error_margin))[0]
#         file_idx = np.intersect1d(file_idx_geq - 1,
#                                   file_idx_leq)
#
#         # prep range to read from which file
#         files_fs = np.array([self['info'][file_id]['fs'] for file_id in file_idx])
#         files_lengths = np.array([self['info'][file_id]['length'] for file_id in file_idx])
#         files_timestamps = np.linspace(timestamps_ref[file_idx[0]],
#                                        timestamps_ref[file_idx[-1] + 1],
#                                        files_lengths.sum() + 1)[:-1]
#         # ToDo Fix issue with length
#         if fixed_output_shape:
#             start = np.where((files_timestamps >= timestamp_new))[0][0]
#             assert np.all(files_fs[0] == files_fs)
#             idx = np.arange(start, start + duration * files_fs[0], dtype=int)
#         else:
#             idx = np.where((files_timestamps >= timestamp_new) & (files_timestamps < (timestamp_new + duration)))[0]
#
#         files_frameidx = np.concatenate([np.arange(file_length) for file_length in files_lengths])
#         files_fileidx = np.concatenate(
#             [file_id * np.ones(file_length, dtype=int) for file_id, file_length in
#              zip(file_idx, files_lengths)])
#
#         return files_frameidx[idx], files_fileidx[idx]
#
#     class CCTVreader(Processor):
#         def __init__(self,
#                      filepaths: List[str],
#                      timestamps: List[float],
#                      observation_window: int,
#                      error_margin: int,
#                      fixed_output_shape=0):
#             assert id in [1, 2], "camera_id should be 1 or 2"
#             self.id = id
#             self.error_margin = error_margin
#             self.observation_window = observation_window
#             self.filepaths = filepaths
#             self.timestamps = timestamps
#             self.fixed_output_shape = fixed_output_shape
#
#         def process(self, timestamp):
#             frameidx, fileidx = self.get_read_info(self.timestamps,
#                                                    timestamp,
#                                                    self.duration,
#                                                    self.error_margin,
#                                                    self.fixed_output_shape)
#
#             # get readers and info
#             vid, info = [None] * len(fileidx), [None] * len(fileidx)
#             for k, file_id in enumerate(fileidx):
#                 vid[k] = cv.VideoCapture(self.filepaths[file_id])
#                 info[k] = {'width': int(vid[k].get(cv.CAP_PROP_FRAME_WIDTH)),
#                            'height': int(vid[k].get(cv.CAP_PROP_FRAME_HEIGHT)),
#                            'fs': vid[k].get(cv.CAP_PROP_FPS),
#                            'length': int(vid[k].get(cv.CAP_PROP_FRAME_COUNT))}
#                 info[k].update({'duration': 1 / info[k]['fs'] * info[k]['length']})
#
#             # get feed
#             frames = [None] * len(frameidx)
#             for k, frameid in enumerate(frameidx):
#                 vid[files_fileidx[frameid]].set(cv.CAP_PROP_POS_FRAMES, files_frameidx[frameid])
#                 ret, frames[k] = vid[files_fileidx[frameid]].read()
#
#             # return
#             return np.array(frames), {'width': 640,
#                                       'height': 400,
#                                       'fs': 25.0}
#
#     def _new_info_handler(self,
#                           frameidx,
#                           fileidx):
#         info = [self['info'][k] for k in np.unique(fileidx)]
#         info = listdictnp_combine(info, method='stack')
#         new_info = dict()
#         for key in info:
#             if key == 'output_shape':
#                 assert np.all(info[key][:, 1])
#                 assert np.all(info[key][:, 2])
#                 new_info['output_shape'] = np.array([len(frameidx), info[key][0][1], info[key][0][2]])
#             elif key == 'duration':
#                 new_info['duration'] = len(frameidx) / info['fs']
#             elif key == 'length':
#                 new_info['length'] = len(frameidx)
#             else:
#                 assert np.all(info[key])
#                 new_info[key] = info[key][0]
#         return new_info
#
#     @property
#     def is_splittable(self):
#         # assumes that this is the case for ALL info
#         # should we do a check on everything or assume this is always the case?
#         return ((self['info'][0]['time_axis'] is not None) if 'time_axis' in self['info'][
#             0] else False) and 'output_shape' in self['info'][0]
