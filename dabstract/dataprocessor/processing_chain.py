import copy
import numpy as np
import os
import pickle
from pprint import pprint

from dabstract.utils import safe_import_module
from dabstract.dataprocessor.processors import base as base

from typing import Union, List, Optional, TypeVar, Callable, Dict, Iterable

tvProcessingChain = TypeVar("ProcessingChain")

class ProcessingChain:
    """Processing chain"""

    def __init__(self, chain: Optional[list] = list(), data: Optional[Iterable] = None):
        # init chain
        self._info = list()
        self._chain = list()
        self.add(chain)
        # fit
        if data is not None:
            self.fit(data=data)

    def add(
        self, name: Union[base.Processor, Dict, List, str], parameters: Dict = dict()
    ) -> tvProcessingChain:
        """Add to chain"""
        # Add new processor
        if isinstance(name, base.Processor):
            # if it's a dabstract processor, directly use
            self._info.append(
                {"name": name.__class__.__name__, "parameters": name.__dict__}
            )
            self._chain.append(name)
        elif isinstance(name, dict):
            # if a dictionary, check if it matches the expected format
            assert "chain" in name, "Specify a chain in your configuration."
            self.add(name["chain"])
        elif isinstance(name, list):
            # if a list, iterate over it
            for item in name:
                self.add(**item)
        elif isinstance(name, str):
            # if str, search for processor in dabstract processors
            if name not in ("none", "None"):
                module = safe_import_module("dabstract.dataprocessor.processors")
                if not hasattr(module, name):  # check customs
                    module = safe_import_module(
                        os.environ["dabstract_CUSTOM_DIR"] + ".processors"
                    )
                    assert hasattr(
                        module, name
                    ), "Processor is not supported in both dabstract.dataprocessor.processors and dabstract.custom.processors. Please check"
                self.add(getattr(module, name)(**parameters))
        elif callable(name):
            if isinstance(name, type):
                # if it is a class to be initialised
                self.add(name(**parameters))
            else:
                # if it is some function which does y = f(x), wrap it in a dabstract processor
                self.add(base.ExternalProcessor(name, **parameters))
        elif name is None:
            # add None
            pass
        else:
            raise NotImplementedError(
                "Input that you provided does not work for ProcessingChain()."
            )
        return self

    def process(self, data: Iterable, return_info: bool = False, **kwargs) -> Iterable:
        """process data"""
        kwargs = copy.deepcopy(kwargs)  # ensure immutability
        for chain in self._chain:
            # process
            data, info_out = chain.process(data, **kwargs)
            # update info dictionary
            kwargs.update(info_out)
        # add output shape info
        if len(self._chain) > 0:
            kwargs.update({"output_shape": np.shape(data)})
        return (data, kwargs) if return_info else data

    def __call__(self, data: Iterable, return_info: bool = False, **kwargs) -> Iterable:
        return self.process(data, return_info=return_info, **kwargs)

    def inv_process(self, data: Iterable = None) -> Iterable:
        """inverse process data"""
        for fid, chain in enumerate(reversed(self._chain)):
            assert hasattr(
                chain, "inv_process"
            ), "Not all processes in your chain contain inv_process methods."
            data = chain.inv_process(data)
        return data

    def fit(
        self,
        data: Iterable,
        load_memory: bool = True,
        workers: int = 1,
        buffer_len: int = 2,
        **kwargs
    ) -> tvProcessingChain:
        """fit parameters"""
        from dabstract.abstract import (
            SelectAbstract,
            MapAbstract,
            DataAbstract,
        )

        assert data is not None
        if len(self._chain) > 0:
            # init separate layers in the chain (+ causal recursive processing if init needs data)
            init_processor = ProcessingChain(chain=list())
            for k, chain in enumerate(self._chain):
                # fit if needed
                if hasattr(chain, "fit"):
                    # subsample based on init_subsample key
                    # ToDo: kind off hidden. Should we reallocate this or completely leave it up to the processor.fit()?
                    if "init_subsample" in self._info[k]["parameters"]:
                        if self._info[k]["parameters"]["init_subsample"] is not None:
                            sel_ind = np.random.choice(
                                np.arange(len(data)),
                                size=int(
                                    self._info[k]["parameters"]["init_subsample"]
                                    * len(data)
                                ),
                                replace=False,
                            )
                            data = SelectAbstract(data, (lambda x, k: k in sel_ind))
                    data_tmp = DataAbstract(
                        MapAbstract(data, init_processor),
                        workers=workers,
                        buffer_len=buffer_len,
                    )
                    # load into memory
                    if load_memory:
                        data_tmp, info_tmp = data_tmp.get(
                            slice(0, len(data)), return_info=True, **kwargs
                        )
                        chain.fit(data_tmp, info=info_tmp)
                    else:
                        chain.fit(data_tmp)
                # keep processor of previous stages (to be used for recursion if fit is needed)
                init_processor.add(chain)
        return self

    def save(self, filepath: str = None, ext = '.pickle'):
        """Save processor from filepath"""
        assert isinstance(filepath,str)
        with open(filepath + ext, 'wb') as f:
            pickle.dump(self.__dict__, f)  # save

    def load(self, filepath: str = None, ext = '.pickle'):
        """Load processor from filepath"""
        assert isinstance(filepath,str)
        with open(filepath + ext, "rb") as f:
            tmp = pickle.load(f)  # load
            self.__dict__.update(tmp)

    def exists(self, filepath: str = None, ext = '.pickle'):
        assert isinstance(filepath,str)
        return os.path.isfile(filepath + ext)

    def summary(self, verbose: bool = True) -> None:
        """Summary of processor"""
        if verbose:
            pprint(self._info)
