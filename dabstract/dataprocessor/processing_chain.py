import copy
import numpy as np
import os
from pprint import pprint
from pylab import plot, show, imshow

from dabstract.utils import safe_import_module

class processing_chain():
    def __init__(self, chain=list(), data=None):
        # init chain
        self._info = list()
        self._chain = list()
        self.add(chain)
        # fit
        if data is not None:
            self.fit(data=data)

    # add to chain
    def add(self, name, parameters=dict()):
        if isinstance(name,processor):
            self._info.append({  'name': name.__class__.__name__,
                                'parameters': name.__dict__})
            self._chain.append(name)
        elif isinstance(name,dict):
            assert 'chain' in name, 'Specify a chain in your configuration.'
            self.add(name['chain'])
        elif isinstance(name,list):
            for item in name:
                self.add(**item)
        else:
            if name is not 'none':
                # get processor class
                if isinstance(name,str):
                    module = safe_import_module('dabstract.dataprocessor.processors')
                    if not hasattr(module, name):  # check customs
                        module = safe_import_module(os.environ['dabstract_CUSTOM_DIR'] + '.processors')
                        assert hasattr(module, name), 'Processor is not supported in both dabstract.dataprocessor.processors and dabstract.custom.processors. Please check'
                    processor_class = getattr(module, name)
                else:
                    processor_class = name
                # initialize
                self._info.append({'name': name, 'parameters': parameters})
                self._chain.append(processor_class(**parameters))
        return self

    def process(self, data, return_info=False, **kwargs):
        kwargs = copy.deepcopy(kwargs) #ensure immutability
        for chain in self._chain:
            # process
            data, info_out = chain.process(data, return_info=True, **kwargs)
            # update info dictionary
            kwargs.update(info_out)
        # add output shape info
        if len(self._chain)>0:
            kwargs.update({'output_shape': np.shape(data)})
        return ((data, kwargs) if return_info else data)

    def __call__(self,data,return_info=False,**kwargs):
        return self.process(data, return_info=return_info, **kwargs)

    # inverse process
    def inv_process(self, data=None):
        for fid, chain in enumerate(reversed(self._chain)):
            assert hasattr(chain,'inv_process'), 'Not all processes in your chain contain inv_process methods.'
            data = chain.inv_process(data)
        return data

    # Initialize dataprocessing chain with data (including recursive data loading and processing if needed)
    def fit(self, data, **kwargs):
        from dabstract.dataset.abstract import abstract, SelectAbstract, MapAbstract, DataAbstract
        assert data is not None
        if len(self._chain) > 0:
            # init separate layers in the chain (+ causal recursive processing if init needs data)
            init_processor = processing_chain(chain=list())
            for k, chain in enumerate(self._chain):
                # fit if needed
                if hasattr(chain,'fit'):
                    if 'init_subsample' in self._info[k]['parameters']:
                        sel_ind = np.random.choice(np.arange(len(data)),size=int(self._info[k]['parameters']['init_subsample'] * len(data)), replace=False)
                        data = SelectAbstract(data, (lambda x,k: k in sel_ind))
                    data_tmp, info_tmp = DataAbstract(MapAbstract(data, init_processor)).get(slice(0,len(data)),return_info=True, **kwargs)
                    chain.fit(data_tmp, info_tmp)
                # keep processor of previous stages (to be used for recursion if fit is needed)
                init_processor.add(chain)
        return self

    def summary(self, verbose=True):
        if verbose: pprint(self._info)
        #return self._info

# base class for processor
class processor():
    def __init__(self):
        pass
    def process(self, data, **kwargs):
        return data, {}
    def inv_process(self, data, **kwargs):
        return data