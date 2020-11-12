import numpy as np
import sklearn.preprocessing as pp
from soundfile import read as read_wav
import sys
import os
import scipy
import scipy.signal as signal
import librosa

from dabstract.utils import listnp_combine, flatten_nested_lst
from dabstract.dataprocessor import Processor

from typing import Dict


class WavDatareader(Processor):
    """Processor to read wav data"""

    def __init__(
        self,
        select_channel: int = None,
        fs: int = None,
        read_range: (int, int) = None,
        dtype=None,
        **kwargs
    ):
        self.select_channel = select_channel
        self.fs = fs
        self.read_range = read_range
        self.dtype = dtype

    def process(self, file: str, **kwargs) -> (np.ndarray, Dict):
        # get read params
        args = dict()
        if self.read_range is not None:
            args.update({"start": self.read_range[0], "stop": self.read_range[1]})
        if "read_range" in kwargs:
            args.update(
                {"start": kwargs["read_range"][0], "stop": kwargs["read_range"][1]}
            )
        if hasattr(self, "dtype"):
            args.update({"dtype": self.dtype})

        # read
        data, fs = read_wav(file, **args)
        if self.fs is not None:
            assert (
                fs == self.fs
            ), "Input fs and provided fs different. Downsampling not supported currently."

        # data selection
        if self.select_channel is not None:
            data = data[:, self.select_channel]

        # updata self info
        return data, {"fs": fs}


class NumpyDatareader(Processor):
    """Processor to read numpy data"""

    def __init__(self, read_range: (int, int) = None, **kwargs):
        self.read_range = read_range

    def process(self, file: str, **kwargs) -> (np.ndarray, Dict):
        # get read params
        args = dict()
        if self.read_range is not None:
            args.update({"read_range": self.read_range})
        if "range" in kwargs:
            args.update({"read_range": kwargs["read_range"]})

        if "range" in args:
            data = np.load(file, mmap_mode="r")
            data = data[args["read_range"][0] : args["read_range"][1], :]
        else:
            data = np.load(file)
        return data, {}


class Normalizer(Processor):
    """Processor to normalize data based on fitted parameters"""

    def __init__(self, type: str = None, feature_range: (int, int) = [0, 1], **kwargs):
        if type is None:
            AssertionError("Specify normalization type in processors.py/Normalizer")
        self.type = type
        self.feature_range = feature_range

    def fit(self, data: np.ndarray, **kwargs) -> None:
        if self.type == "minmax":
            self.scaler = pp.MinMaxScaler(feature_range=self.feature_range)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif (
                len(data.shape) == 3
            ):  # based on the assumption that we will at most read 3D data
                data = flatten_nested_lst(data)
            self.scaler.fit(data)
        elif self.type == "standard":
            self.scaler = pp.StandardScaler()
            if (
                len(data.shape) >= 3
            ):  # based on the assumption that we will at most read 3D data
                data = data.reshape(np.prod(data.shape[:-1]), data.shape[-1])
            self.scaler.fit(data)

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
        if (self.type == "minmax") | (self.type == "standard"):
            if len(data.shape) <= 1:
                data = data.reshape(1, -1)
                data = self.scaler.transform(data)
                return data.reshape(-1), {}
            elif len(data.shape) == 2:
                return self.scaler.transform(data), {}
            elif len(data.shape) == 3:
                for k in range(data.shape[0]):
                    data[k] = self.scaler.transform(data[k])
                return data, {}
            elif len(data.shape) == 4:
                for k in range(data.shape[0]):
                    for i in range(data[k].shape[2]):
                        data[k, :, :, i] = self.scaler.transform(data[k, :, :, i])
                return data, {}
        else:
            print("Not supported.")
            sys.exit()

    def inv_process(self, data: np.ndarray, **kwargs):
        if (self.type == "minmax") | (self.type == "standard"):
            if len(data.shape) <= 1:
                data = data.reshape(1, -1)
                data = self.scaler.inverse_transform(data)
                return data.reshape(-1)
            elif len(data.shape) == 2:
                return self.scaler.inverse_transform(data)
            elif len(data.shape) == 3:
                for k in range(data.shape[0]):
                    data[k] = self.scaler.inverse_transform(data[k])
            elif len(data.shape) == 4:
                for k in range(data.shape[0]):
                    for i in range(data[k].shape[2]):
                        data[k, :, :, i] = self.scaler.inverse_transform(
                            data[k, :, :, i]
                        )
            else:
                print("Not supported.")
                sys.exit()
        else:
            print("Not supported.")
            sys.exit()

        return data


class Scaler(Processor):
    """Processor to scale data"""

    def __init__(self, **kwargs):
        self.type = kwargs["type"]

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
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


class Framing(Processor):
    """Processor to frame data"""

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

        # return
        return frames, {"time_step": self.stepsize} if axis == 0 else dict()


class Windowing(Processor):
    def __init__(self, axis=-1, window_func="hamming", symmetry=True, **kwargs):
        self.axis = axis
        self.window_func = window_func
        self.symmetry = symmetry

    def process(self, data, **kwargs):
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


class FFT(Processor):
    """Processor to apply a FFT"""

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
        # do fft
        if self.nfft == "nextpow2":
            NFFT = 2 ** np.ceil(np.log2(np.shape(data)[self.axis]))
        elif self.nfft == "original":
            NFFT = np.shape(data)[self.axis]

        if self.type == "real":
            data = np.fft.rfft(data, n=int(NFFT), axis=self.axis, norm=self.norm)
        elif self.type == "full":
            data = np.fft.fft(data, n=int(NFFT), axis=self.axis, norm=self.norm)

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

        return data, {"nfft": NFFT}


class Filterbank(Processor):
    def __init__(
        self,
        n_bands=40,
        scale="linear",
        Nfft="nextpow2",
        fmin=0,
        norm=None,
        fmax=np.Inf,
        axis=-1,
        **kwargs
    ):
        self.n_bands = n_bands
        self.scale = scale
        self.axis = axis
        self.fmin = fmin
        self.fmax = fmax
        self.norm = norm
        if "fs" in kwargs:
            self.fs = kwargs["fs"]
        self.Nfft = Nfft

    def process(self, data, **kwargs):
        # inits
        if "fs" in kwargs:
            fs = kwargs["fs"]
            if self.fmax == "half_fs":
                self.fmax = fs / 2
        elif hasattr(self, "fs"):
            fs = self.fs
        else:
            print("No fs given in Filterbank()")
            sys.exit()

        if self.Nfft == "nextpow2":
            NFFT = np.shape(data)[self.axis] * 2 - 2
        elif self.Nfft == "original":
            NFFT = np.shape(data)[self.axis]

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
            start_bin = np.round(NFFT / fs * start_freq_hz)
            stop_bin = np.round(NFFT / fs * stop_freq_hz)
            # The middle bins of the filters are the start frequencies of the next filter.
            middle_bin = np.append(start_bin[1:], stop_bin[-2])
            # Compute the width of the filters
            tot_len = stop_bin - start_bin + 1
            low_len = middle_bin - start_bin + 1
            high_len = tot_len - low_len + 1
            # Allocate the empty filterbank
            fbank = np.zeros((self.n_bands, int(np.floor(NFFT / 2 + 1))))
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
                NFFT,
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


class Logarithm(Processor):
    """Processor to apply a logarithm"""

    def __init__(self, type: str = "base10", **kwargs):
        self.type = type

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
        if self.type == "base10":
            return 20 * np.log10(data), {}
        elif self.type == "natural":
            return np.log(data), {}

    def inv_process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
        if self.type == "base10":
            return (10 ** data) / 20, {}
        elif self.type == "natural":
            return np.exp(data), {}


class Aggregation(Processor):
    """Processor to aggregate data"""

    def __init__(
        self,
        methods: (str, str) = ["mean", "std"],
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

        return output, {"time_step": 0} if self.axis == 0 else dict()


class FIRFilter(Processor):
    """Processor to apply a FIR filter"""

    def __init__(
        self,
        type: str = type,
        f: int = None,
        taps: int = None,
        axis: int = 1,
        fs: int = None,
        window="hamming",
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

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
        if not hasattr(self, "filter"):
            if "fs" in kwargs:
                self.get_filter(kwargs["fs"])
            elif hasattr(self, "fs"):
                self.get_filter(self.fs)
            else:
                raise Exception(
                    "Sampling frequency should be provided to FIR_filter as init or passed on the process()"
                )
        return signal.lfilter(self.filter, 1.0, data, axis=self.axis), {}


class ExpandDims(Processor):
    """Processor to expand the dimensions"""

    def __init__(self, axis: int = -1):
        self.axis = axis

    def process(self, data: np.ndarray, **kwargs) -> (np.ndarray, Dict):
        data = np.expand_dims(data, axis=self.axis)
        return data, {}
