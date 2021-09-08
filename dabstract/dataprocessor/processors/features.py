import numpy as np
import sklearn.preprocessing as pp
import sys
import os
import scipy
import scipy.signal as signal
import librosa
import numbers

from dabstract.utils import listnp_combine, flatten_nested_lst
from dabstract.dataprocessor.processors import base, decorators

from typing import Dict, Any, List, Union, Callable


class Normalizer(base.Processor):
    """
    This processor class applies a normalisation operation on an input array. It currently support minmax and standard
    normalisation. Once can also specify the axis to normalise over. For example, consider a set of matrices such that we have a
    tensor of EXAMPLESxDIM0xDIM1. If DIM0 is a time axis and DIM1 features (e.g. STFT) one may want to normalise over DIM1 only.
    In case DIM0 and DIM1 all represent features (e.g. in case of an image) one may want to normalise over each seperate feature at once.
    Similarly one can choose one scale over all features.

    For example, consider the following function::

        $ fit_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=float)
        $ example = np.array([1, 2, 3, 4, 5], dtype=float)
        $ dp = ProcessingChain()
        $ dp.add(Normalizer(type='minmax'))
        $ dp.fit(fit_data)
        $ output = dp(example)
        $ print(output)
        [0.  0.  0.  0.  0. ]

    that transforms the input data array such that the minimum value over the colums corresponds to 0 and the maximum value
    to 1. This .fit operation is required to fit the normalisation parameters (i.e. the range) given the data. By default
    the axis parameter is set top -1 (last axis) and therefor normalises over columns.

    Once can also directly use this processor without using ProcessingChain as::

        $ fit_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=float)
        $ example = np.array([1, 2, 3, 4, 5], dtype=float)
        $ nr = Normalizer(type='minmax')
        $ nr.fit(fit_data)
        $ output = nr(example)
        $ print(output)
        [0.  0.  0.  0.  0. ]

    It currently supports normalisation within a range (minmax) and based on standard distribution (standard).
    Both approaches use the sklearn toolbox to do the normalisation such that this is a wrapper to include the axis dependend
    normalisation.

    If one want to normalise over a different axis you can alter the axis argument such as::

        $ fit_data2 = np.array( [[[0,1,5],[0,1,2]], [[0,2,2],[0,2,2]], [[3,3,2],[0,3,2]]], dtype=float)
        $ example = fit_data2[2]
        $ nr = Normalizer(type='minmax', axis = 0)
        $ nr.fit(fit_data2)
        $ output = nr(example)
        $ print(output)
        [[0.6, 0.6, 0.4],[0., 1.,2/3]]

    Similarly one can choose to normalise over all samples by setting axis to 'all'.

    In order to fit the normaliser one should use the .fit() method. Similar to other .fit() methods the data you provide
    is of size SAMPLESxDIM0xDIM1x.. . When using process() one should provide a SINGLE sample of size DIM0xDIM1x.. .
    Additionally, normalisation is a reversible operation such that a inv_process() exists. This is useful if one would like
    to forward and backward transform for some reason, i.e. in a ML pipeline where the predicted output is transformed back
    to the original space.

    Parameters
    ----------
    type : str
        The normalisation data type. The two valid options are "minmax" and "standard". "minmax" scales the data such
        that the data range according to 'feature_range', "standard" scales the data such that the mean is set to 0 and
        the standard deviation to 1.
    feature_range : [int, int] = [0,1]
        min and max range. Only applicable for type == 'minmax'
    axis : Union(int,List[int], str) = -1
        axis to normalise over (default = -1)
        One can select multiple axis, e.g. [0, -1] or even select all axis 'all'

    Returns
    ----------
    Normalizer instance

    Check .process() method for what it returns when using this instance.

    """

    def __init__(self, type: str = None,
                 feature_range: (int, int) = [0, 1],
                 axis: Union[str, List[int], int] = -1,
                 subsample_type=None,
                 subsample_rate=None,
                 **kwargs):
        if type is None:
            AssertionError("Specify normalization type in processors.py/Normalizer")
        self.type = type
        self.feature_range = feature_range
        self.axis = axis
        self.subsample_type = subsample_type
        self.subsample_rate = subsample_rate

    @decorators.subsample_data
    @decorators.load_memory
    def fit(self, data: np.ndarray, **kwargs) -> None:
        # check axis
        if isinstance(self.axis, str):
            if self.axis == 'all':
                self.axis = np.arange(0, len(data.shape) - 1)
            elif self.axis == 'feature':
                self.axis = [0]
            else:
                raise NotImplementedError("axis should be a list of integers OR a string all/feature")
        elif isinstance(self.axis, numbers.Integral):
            self.axis = [self.axis]
        elif isinstance(self.axis, list):
            if all([isinstance(axis, numbers.Integral) for axis in self.axis]):
                pass
            else:
                raise NotImplementedError("axis should be a list of integers OR a string all/feature")
        else:
            raise NotImplementedError("axis should be a list of integers OR a string all/feature")
        # adjust -1
        for k, axis in enumerate(self.axis):
            if axis == -1:
                self.axis[k] = len(data.shape) - 2

        # set reshape parameters based on fit dataset
        self._base_ids = np.arange(0, len(data.shape))
        flatten_ids = np.setdiff1d(self._base_ids, np.array(self.axis) + 1)
        norm_ids = np.intersect1d(self._base_ids, np.array(self.axis) + 1)
        target_ids = np.concatenate([norm_ids, flatten_ids])
        self._transform_idx = self._base_ids[
            np.array([np.where(target_ids == base_id)[0][0] for base_id in self._base_ids])]
        self._inverse_transform_idx = target_ids
        self._inverse_reshape = np.array(data.shape)[self._inverse_transform_idx]
        self._reshape = np.concatenate([self._inverse_reshape[:len(self.axis)].prod(keepdims=True), np.array([-1])])
        self._inverse_reshape[
            len(self.axis)] = 1  # adjust # examples for usage during the process() and inv_process() methods

        # reorder axis and flatten over non-norm axis
        data = np.moveaxis(data, self._base_ids,
                           self._transform_idx)  # reorder such that norm_indices + flatten_indices
        data = data.reshape(self._reshape).T  # flatten and transpose to SAMPLES x FEATURES

        # fit
        if self.type == "minmax":
            self.scaler = pp.MinMaxScaler(feature_range=self.feature_range)
        elif self.type == "standard":
            self.scaler = pp.StandardScaler()
        self.scaler.fit(data)

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
        """
        Process method for Normalizer in a forward fashion.
        This is called by the ProcessingChain in a sequential fashion or when using the reader directly.

        Input
        ----------
        data: np.ndarray
            a single example

        Returns
        ----------
        data: ndarray
            The normalised data
        info : dict{'fs': Float}
            dictionary representing information to be propagated,
            containing 'fs' with a float
        """

        return self._scale_data(data, self.scaler.transform), {}

    def inv_process(self, data: np.ndarray, **kwargs):
        """
        Process method for Normalizer in a backward fashion.
        This is called by the ProcessingChain in a sequential fashion or when using the reader directly.

        Input
        ----------
        data: np.ndarray
            a single example

        Returns
        ----------
        data: ndarray
            The de-normalised data
        """
        return self._scale_data(data, self.scaler.inverse_transform)

    def _scale_data(self, data: np.ndarray, scaler_function: Callable):
        """
        Hidden method that does the effective scaling given the scaler function and data input
        """
        # reorder axis and flatten over non-norm axis
        orig_data = data
        data = data.reshape((1,) + tuple(data.shape))
        data = np.moveaxis(data, self._base_ids, self._transform_idx)
        data = data.reshape(self._reshape)

        if (self.type == "minmax") | (self.type == "standard"):
            # ToDo: avoid double transpose by incorporating in .fit()
            data = scaler_function(data.T).T
        else:
            raise NotImplementedError("Only MinMax and standard normalisation supported.")

        # unflatten and reorder
        data = data.reshape(self._inverse_reshape)
        data = np.moveaxis(data, self._base_ids, self._inverse_transform_idx)

        return data[0, :]

    def is_fitted(self):
        return hasattr(self, 'scaler')


class Scaler(base.Processor):
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
    resample flat (resample: bool).

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

    def __init__(self, **kwargs):
        self.type = kwargs["type"]

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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

        if self.type == "uint16":
            data = data / 2 ** 16
        elif self.type == "int16":
            data = data / (pow(2, 15) - 1)
        elif self.type == "wav_2_01":
            data = (data + 1) / 2
        else:
            print("Not supported.")
            sys.exit()

        return data, {}

    def inv_process(self, data: np.ndarray) -> np.ndarray:
        if self.type == "uint16_scaler":
            data = data * 2 ** 16
        elif self.type == "int16":
            data = data * (pow(2, 15) - 1)
        else:
            print("Not supported.")
            sys.exit()
        return data


class Framing(base.Processor):
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
    resample flat (resample: bool).

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
            windowsize: float = None,
            stepsize: float = None,
            window_func: str = "hamming",
            axis: int = -1,
            **kwargs
    ):
        # inits
        self.windowsize = windowsize
        self.stepsize = stepsize
        self.window_func = Windowing(window_func=window_func, axis=axis)
        self.axis = axis
        if "fs" in kwargs:
            self.fs = kwargs["fs"]

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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

        # inits
        if "time_step" in kwargs:
            if kwargs["time_step"] is not None:
                fs = 1 / kwargs["time_step"]
        elif "fs" in kwargs:
            fs = kwargs["fs"]
        elif hasattr(self, "fs"):
            fs = self.fs
        else:
            assert 0, "fs not provided in Framing"
        assert self.windowsize is not None
        frame_length = int(round(self.windowsize * fs))
        if self.stepsize is None:
            frame_step = 1
        else:
            frame_step = int(round(self.stepsize * fs))
        if self.axis == -1:
            axis = len(data.shape) - 1
        else:
            axis = self.axis
        signal_length = data.shape[axis]

        # segment
        num_frames = int(
            np.floor(((signal_length - (frame_length - 1) - 1) / frame_step) + 1)
        )
        assert num_frames > 0, "num of frames is 0 in Framing()"
        indices = (
                np.tile(np.arange(0, frame_length), (num_frames, 1))
                + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
        ).T
        )
        frames = np.take(data, indices, axis=axis)
        data_lo = np.take(
            data, np.setdiff1d(np.arange(signal_length), indices), axis=axis
        )

        # window fct
        self.window_func.axis = axis + 1
        frames = self.window_func.process(frames)[0]

        # update info
        info = {}
        if 'time_axis' in kwargs:
            if axis == kwargs['time_axis']:
                info = {"time_step": self.stepsize,
                        "length": num_frames}

        # return
        return frames, info


class Windowing(base.Processor):
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
    resample flat (resample: bool).

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

    def __init__(self, axis=-1, window_func="hamming", symmetry=True, **kwargs):
        self.axis = axis
        self.window_func = window_func
        self.symmetry = symmetry

    def process(self, data, **kwargs):
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

        # init
        if self.axis == -1:
            axis = len(data.shape) - 1
        else:
            axis = self.axis

        # get window
        if self.window_func == "none" or self.window_func == "None":
            return data, {}
        elif self.window_func is None:
            return data, {}
        else:
            hw = signal.get_window(
                self.window_func, np.shape(data)[axis], fftbins=self.symmetry
            )

        # window
        data *= np.reshape(
            hw, [(data.shape[k] if k == axis else 1) for k in range(len(data.shape))]
        )

        # return
        return data, dict()


class FFT(base.Processor):
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
    resample flat (resample: bool).

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
            type: str = "real",
            nfft: str = "nextpow2",
            format: str = "magnitude",
            dc_reset: bool = False,
            norm: str = None,
            axis: int = -1,
            **kwargs
    ):
        self.format = format
        self.dc_reset = dc_reset
        self.axis = axis
        self.nfft = nfft
        self.type = type
        self.norm = norm

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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

        # do fft
        if self.nfft == "nextpow2":
            nfft = 2 ** np.ceil(np.log2(np.shape(data)[self.axis]))
        elif self.nfft == "original":
            nfft = np.shape(data)[self.axis]

        if self.type == "real":
            data = np.fft.rfft(data, n=int(nfft), axis=self.axis, norm=self.norm)
        elif self.type == "full":
            data = np.fft.fft(data, n=int(nfft), axis=self.axis, norm=self.norm)

        # agg complex
        if self.format == "magnitude":
            data = np.absolute(data)
        elif self.format == "power":
            data = np.absolute(data) ** 2
        elif self.format == "split":
            data = np.concatenate((np.real(data), np.imag(data)), axis=self.axis)

        # remove dc
        if self.dc_reset:
            # do sel slicing
            if self.axis == -1:
                axis = len(np.shape(data)) - 1
            else:
                axis = self.axis
            sel_tuple = tuple(
                [
                    (slice(0, data.shape[k]) if k != axis else 0)
                    for k in range(len(data.shape))
                ]
            )
            data[sel_tuple] = 0

        return data, {"nfft": nfft, 'frequency_axis': self.axis}


class Filterbank(base.Processor):
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
    resample flat (resample: bool).

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
            n_bands: int = None,
            scale: str = "linear",
            nfft: int = None,
            fmin: int = 0,
            norm: str = None,
            fmax: int = np.Inf,
            axis: int = -1,
            **kwargs
    ):
        assert n_bands is not None, "The amount of n_bands should be provided."
        self.n_bands = n_bands
        self.scale = scale
        self.axis = axis
        self.fmin = fmin
        self.fmax = fmax
        self.norm = norm
        if "fs" in kwargs:
            self.fs = kwargs["fs"]
        self.nfft = nfft

    def process(self, data: np.ndarray, **kwargs):
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

        # inits
        if "fs" in kwargs:
            fs = kwargs["fs"]
        elif hasattr(self, "fs"):
            fs = self.fs
        else:
            print("No fs given in Filterbank()")
            sys.exit()

        if 'nfft' in kwargs:
            nfft = kwargs['nfft']
            if self.nfft is not None:
                assert nfft == self.nfft, "The nfft that was set mismatches with the one provided by a previous layer. Please check"
        elif self.nfft is not None:
            nfft = self.nfft
        else:
            raise NotImplementedError("No nfft provided in Filterbank()")

        if 'frequency_axis' in kwargs:
            assert self.axis == kwargs['frequency_axis']

        low_freq = self.fmin
        high_freq = np.min((fs / 2, self.fmax))

        # create filterbank
        if self.scale in ("mel", "linear"):
            if self.scale == "mel":
                # Define the Mel frequency of high_freq and low_freq
                low_freq_mel = 2595 * np.log10(1 + low_freq / 700)
                high_freq_mel = 2595 * np.log10(1 + high_freq / 700)
                # Define the start Mel frequencies, start frequencies and start bins
                start_freq_mel = low_freq_mel + np.arange(0, self.n_bands, 1) / (
                        self.n_bands + 1
                ) * (high_freq_mel - low_freq_mel)
                start_freq_hz = 700 * (10 ** (start_freq_mel / 2595) - 1)
                # Define the stop Mel frequencies, start frequencies and start bins
                stop_freq_mel = low_freq_mel + np.arange(2, self.n_bands + 2, 1) / (
                        self.n_bands + 1
                ) * (high_freq_mel - low_freq_mel)
                stop_freq_hz = 700 * (10 ** (stop_freq_mel / 2595) - 1)
            elif self.scale == "linear":
                # linear spacing
                hz_points = np.linspace(low_freq, high_freq, self.n_bands + 2)
                start_freq_hz = hz_points[0:-2]
                stop_freq_hz = hz_points[2:]

            # get bins
            start_bin = np.round(nfft / fs * start_freq_hz)
            stop_bin = np.round(nfft / fs * stop_freq_hz)
            # The middle bins of the filters are the start frequencies of the next filter.
            middle_bin = np.append(start_bin[1:], stop_bin[-2])
            # Compute the width of the filters
            tot_len = stop_bin - start_bin + 1
            low_len = middle_bin - start_bin + 1
            high_len = tot_len - low_len + 1
            # Allocate the empty filterbank
            fbank = np.zeros((self.n_bands, int(np.floor(nfft / 2 + 1))))
            # Compute the filter weights matrix
            for m in range(1, self.n_bands + 1):
                weights_low = np.arange(1, low_len[m - 1] + 1) / (low_len[m - 1])
                for k in range(0, int(low_len[m - 1])):
                    fbank[m - 1, int(start_bin[m - 1] + k)] = weights_low[k]
                weights_high = np.arange(high_len[m - 1], 0, -1) / (high_len[m - 1])
                for k in range(0, int(high_len[m - 1])):
                    fbank[m - 1, int(middle_bin[m - 1] + k)] = weights_high[k]

            # apply norm
            if self.norm == "slaney":
                enorm = 2.0 / (stop_freq_hz - start_freq_hz)
                fbank *= enorm[:, np.newaxis]
        elif self.scale in ("melLibrosa"):
            fbank = librosa.filters.mel(
                fs,
                nfft,
                n_mels=self.n_bands,
                fmin=low_freq,
                fmax=high_freq,
                norm=self.norm,
            )

        # Apply the mel/linear warping
        filter_banks = np.dot(data, fbank.T)
        filter_banks = np.where(
            filter_banks == 0, np.finfo(float).eps, filter_banks
        )  # Numerical Stability

        return filter_banks, {}


class Logarithm(base.Processor):
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
    resample flat (resample: bool).

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

    def __init__(self, type: str = "base10", **kwargs):
        self.type = type

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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

        if self.type == "base10":
            return 20 * np.log10(data), {}
        elif self.type == "natural":
            return np.log(data), {}

    def inv_process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
        if self.type == "base10":
            return (10 ** data) / 20, {}
        elif self.type == "natural":
            return np.exp(data), {}


class Aggregation(base.Processor):
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
    resample flat (resample: bool).

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
            methods: List[str] = ["mean", "std"],
            axis: int = 0,
            combine: str = None,
            combine_axis: int = None,
    ):
        self.methods = methods
        self.axis = axis
        self.combine = combine
        if combine_axis is None:
            self.combine_axis = axis
        else:
            self.combine_axis = combine_axis

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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

        # aggregate data
        tmp = [None] * len(self.methods)
        for k in range(len(self.methods)):
            if self.methods[k] == "mean":
                tmp[k] = np.mean(data, axis=self.axis)
            elif self.methods[k] == "std":
                tmp[k] = np.std(data, axis=self.axis)
            elif self.methods[k] == "kurtosis":
                tmp[k] = scipy.stats.kurtosis(data, axis=self.axis)
            else:
                print("Aggregation method not supported")
                sys.exit()
        output = (
            listnp_combine(tmp, method=self.combine, axis=self.combine_axis)
            if self.combine is not None
            else tmp
        )

        # update info
        info = {}
        if self.axis == kwargs['time_axis']:
            if 'duration' in kwargs:
                info.update({'time_step': kwargs['duration']})
            info.update({'length': 1,
                         'time_axis': None})

        return output, info


class FIRFilter(base.Processor):
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
    resample flat (resample: bool).

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
            type: str = type,
            f: float = None,
            taps: int = None,
            axis: int = 1,
            fs: float = None,
            window: str = "hamming",
    ):
        self.type = type
        self.f = f
        self.taps = taps
        self.taps |= 1  # make uneven
        self.axis = axis
        self.window = window
        self.fs = fs

    def get_filter(self, fs: int):
        if self.type == "bandstop":
            self.filter = signal.firwin(self.taps, self.f, window=self.window, fs=fs)
        elif self.type == "bandpass":
            self.filter = signal.firwin(
                self.taps, self.f, window=self.window, fs=fs, pass_zero=False
            )
        elif self.type == "highpass":
            self.filter = signal.firwin(
                self.taps, self.f, window=self.window, fs=fs, pass_zero=False
            )
        elif self.type == "lowpass":
            self.filter = signal.firwin(self.taps, self.f, window=self.window, fs=fs)
        else:
            raise NotImplementedError
        self.filter_fs = fs

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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

        if "fs" in kwargs:
            fs = kwargs["fs"]
        elif hasattr(self, "fs"):
            fs = self.fs
        else:
            raise Exception(
                "Sampling frequency should be provided to FIR_filter as init or passed on the process()"
            )
        if not hasattr(self, "filter"):
            self.get_filter(fs)
        else:
            if fs != self.filter_fs:
                self.get_filter(fs)

        return signal.lfilter(self.filter, 1.0, data, axis=self.axis), {}


class Resample(base.Processor):
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
    resample flat (resample: bool).

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

    def __init__(self,
                 target_fs: float = None,
                 fs: float = None,
                 axis: int = 0,
                 window: str = 'hann'):
        self.target_fs = target_fs
        self.fs = fs
        self.axis = axis
        self.window = window

    def process(self, data, **kwargs):
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

        if 'fs' in kwargs:
            fs = kwargs['fs']
        else:
            fs = self.fs
        data = scipy.signal.resample(data,
                                     int(np.round(self.target_fs / fs * data.shape[self.axis])),
                                     axis=self.axis,
                                     window=self.window)
        return data, {'fs': self.target_fs, 'time_step': 1 / self.target_fs}


class ExpandDims(base.Processor):
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
    resample flat (resample: bool).

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

    def __init__(self, axis: int = -1):
        self.axis = axis

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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

        data = np.expand_dims(data, axis=self.axis)
        return data, {}
