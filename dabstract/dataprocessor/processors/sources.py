import numpy as np
from soundfile import read as read_wav
import cv2 as cv

from dabstract.dataprocessor.processors import base

from typing import Dict, Any, List, Union, Callable

class CameraDatareader(base.Processor):
    """
    ToDo
    """

    def __init__(
        self,
        **kwargs
    ):
        pass

    def process(self, file: str, read_range: List[int] = None, **kwargs) -> (np.ndarray, Dict):
        """
        ToDo
        """
        # Prepare
        vid = cv.VideoCapture(file)
        info = {'width': int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
                'height': int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)),
                'fs': vid.get(cv.CAP_PROP_FPS),
                'length': int(vid.get(cv.CAP_PROP_FRAME_COUNT))}

        if read_range is None:
            frame_idx = np.arange(0, info['length'])
        else:
            frame_idx = np.arange(read_range[0], read_range[1])

        # get feed
        frames = [None] * len(frame_idx)
        for k, frame_id in enumerate(frame_idx):
            vid.set(cv.CAP_PROP_POS_FRAMES, frame_id)
            ret, frames[k] = vid.read()

        return np.array(frames), info


class WavDatareader(base.Processor):
    """
    This processor class reads waveform data file and adds optional resampling if desired.
    For reading it uses the read_wav function from the soundfile package.

    For example consider the following usage with ProcessingChain class::

        $ dp = ProcessingChain()
        $ dp.add(WavDatareader())
        $ wavdata = dp("path/to/test.wav")
        $ print(wavdata)
        $ [-0.0859375 -0.078125  -0.078125  ...  0.         0.         0.       ]

    which reads the file located at path/to/test.wav. If the file contains multiple channels one can only specify
    which channel(s) to read as by adding the select_channel argument to WavDatareader()::

        WavDatareader(select_channel=0) or WavDatareader(select_channel=[0,1])

    Similarly one can read a specific range of the file::

        WavDatareader(read_range=[start_sample,end_sample])

    If a sampling frequency is specified (fs: float) then this is used to compare it with the sampling frequency of the
    actual wav file that has been read. In case it differs it either gives an error OR resamples the data based on the
    resample flat (resample: bool). In case resample is True, one can specify the resample_axis (default = 0) and
    resample_window (default = 'hann').

    Once can also directly use this processor without using ProcessingChain as::

        $ wr = WavDatareader()
        $ wavdata = wr("path/to/test.wav")
        $ print(wavdata)
        $ [-0.0859375 -0.078125  -0.078125  ...  0.         0.         0.       ]

    Parameters
    ----------
    select_channel : int or None: None
        Select which channel(s) must be read from the waveform file. By default all channels are read (default = None)
    fs : float or None
        Defines the sampling frequency of the waveform file.
    read_range : (int, int) or None
        Defines the sample range that must be readed from the waveform file.
        (default = None -> reads the entire range).
    dtype : str = 'float64'
        Data type of the returned array, by default 'float64'.
        Floating point audio data is typically in the range from -1.0 to 1.0.
        Integer data is in the range from -2**15 to 2**15-1 for 'int16' and from -2**31 to 2**31-1 for 'int32'
    resample : bool = False
        Defines whether a resampling on the loaded waveform data must be performed or not.
        (default = False)
    resample_axis : int = 0
        The axis over which the resampling should be done.
        (default = 0)
    resample_window : str = 'hann
        The window type that is used to do the resampling. See scipy.signal.resample for a list of valid window types.
        (Default = "hann")

    Returns
    ----------
    WavDatareader instance

    Check .process() method for what it returns when using this instance.

    """

    def __init__(
        self,
        select_channel: int = None,
        fs: float = None,
        read_range: (int, int) = None,
        dtype: str = 'float64',
        resample: bool =  False,
        resample_axis: int = 0,
        resample_window: str = 'hann',
        **kwargs
    ):
        self.select_channel = select_channel
        self.fs = fs
        self.read_range = read_range
        self.dtype = dtype
        self.resample = resample
        if self.resample:
            assert fs is not None
            from dabstract.dataprocessor.processors import Resample
            self.resampler = Resample(target_fs = fs, axis=resample_axis, window=resample_window)

    def process(self, file: str, **kwargs) -> (np.ndarray, Dict):
        """
        Process method for WavDatareader. This is called by the ProcessingChain in a sequential fashion or when using the
        reader directly.

        Input
        ----------
        file: str
            location to wav file

        Returns
        ----------
        data: ndarray
            The loaded waveform data by means of a numpy ndaddray,
            The format is SAMPLESxCHANNELS
        info : dict{'fs': Float}
            dictionary representing information to be propagated,
            containing 'fs' with a float
        """

        args = dict()
        info = dict()

        # get read params
        if self.read_range is not None:
            args.update({"start": self.read_range[0], "stop": self.read_range[1]})
        if "read_range" in kwargs:
            #info.update({'read_ranged': True})
            args.update(
                {"start": int(kwargs["read_range"][0]), "stop": int(kwargs["read_range"][1])}
            )
        if hasattr(self, "dtype"):
            args.update({"dtype": self.dtype})

        # read
        data, fs = read_wav(file, **args)
        info.update({'fs': fs, 'time_axis': 0})

        # data selection
        if self.select_channel is not None:
            data = data[:, self.select_channel]
            info.update({'channels': 1})

        # resample
        if self.fs is not None:
            if self.resample:
                data = self.resampler.process(data, fs = fs)[0]
                info.update({'fs': self.fs})
            else:
                assert (
                    fs == self.fs
                ), "Input fs and provided fs different. Downsampling not supported currently."

        # updata self info
        return data, info

class NumpyDatareader(base.Processor):
    """
    This processor class simply reads a numpy ndarray with an optional memory mapping in case a specific
    range of samples is desired.

    For example consider the following usage with ProcessingChain class::

        $ dp = ProcessingChain()
        $ dp.add(NumpyDatareader())
        $ numpydata = dp("path/to/test.npy")
        $ print(numpydata)
        $ [-0.0859375 -0.078125  -0.078125  ...  0.         0.         0.       ]

    Similarly one can read a specific range of the file::

        NumpyDatareader(read_range=[start_sample,end_sample])

    Note that one can only memory map the first dimension of your ndarray, i.e. the rows.
    Once can also directly use this processor without using ProcessingChain as::

        $ nr = NumpyDatareader()
        $ numpydata = nr("path/to/test.npy")
        $ print(numpydata)
        $ [-0.0859375 -0.078125  -0.078125  ...  0.         0.         0.       ]

    Parameters
    ----------
    read_range : (int, int) or None
        Defines the sample range that must be readed from the waveform file.
        (default = None -> reads the entire range).

    Returns
    ----------
    NumpyDatareader instance

    Check .process() method for what it returns when using this instance.

    """

    def __init__(self, read_range: (int, int) = None, **kwargs):
        self.read_range = read_range

    def process(self, file: str, **kwargs) -> (np.ndarray, Dict):
        """
        Process method for NumpyDatareader.
        This is called by the ProcessingChain in a sequential fashion or when using the reader directly.

        Input
        ----------
        file: str
            location to wav file

        Returns
        ----------
        data: ndarray
            The loaded npy data
        info : dict{'fs': Float}
            dictionary representing information to be propagated,
            containing 'fs' with a float
        """

        args = dict()
        info = dict()

        # get read params
        if self.read_range is not None:
            args.update({"read_range": self.read_range})
        if "read_range" in kwargs:
            args.update({"read_range": kwargs["read_range"]})

        if "read_range" in args:
            data = np.load(file, mmap_mode="r")
            data = data[int(args["read_range"][0]) : int(args["read_range"][1]), :]
            info.update({'read_ranged': True})
        else:
            data = np.load(file)

        return data, info