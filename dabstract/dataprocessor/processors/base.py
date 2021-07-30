from typing import Union, List, Optional, TypeVar, Callable, Dict, Iterable

tvProcessingChain = TypeVar("ProcessingChain")

class Processor:
    """base class for processor"""

    def __init__(self):
        pass

    def process(self, data: Iterable, **kwargs) -> (Iterable, Dict):
        return data, {}

    def inv_process(self, data: Iterable, **kwargs) -> Iterable:
        return data

    # def fit(self, data: Iterable, **kwargs) -> None:
    #     raise NotImplementedError

    def __call__(self, data: Iterable, return_info: bool = False, **kwargs) -> Iterable:
        tmp = self.process(data, **kwargs)
        return tmp if return_info else tmp[0]


class ExternalProcessor(Processor):
    """base class for an external function"""

    def __init__(self, fct: Callable):
        self.fct = fct
        self.__class__.__name__ = fct.__name__

    def process(self, data, **kwargs) -> (Iterable, Dict):
        return self.fct(data), {}


class Dummy(Processor):
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
    pass
