import os
import types

from dabstract.utils import safe_import_module
from dabstract.dataset import dbs
from dabstract.dataset.abstract import *
from dabstract.dataset.dataset import dataset

def dataset_from_config(config, overwrite=False):
    assert isinstance(config,dict), "config should be a dictionary"
    assert 'datasets' in config, "config should have a datasets key"
    assert isinstance(config['datasets'],list), "config['dataset'] should be represented as a list where each item is a dictionary containing kwargs of your dataset."
    ddataset = dataset()
    # init datasets
    for k,db in enumerate(config['datasets']):
        ddataset += dataset_factory(name=db['name'], **db['parameters'])
    # add other functionality
    if 'xval' in config:
        ddataset.set_xval(**config['xval'], overwrite=overwrite)
    return ddataset

def dataset_factory(name=None,
                    paths=None,
                    xval=None,
                    filter=None,
                    test_only=0,
                    tmp_folder=None,
                    **kwargs):
    """Dataset factory
    Set up dataset from name and config
    """
    # get dataset
    if isinstance(name, str):
        # get db class
        module = dbs
        if not hasattr(module, name):  # check customs
            module = safe_import_module(os.environ['dabstract_CUSTOM_DIR'] + '.dataset.dbs')
            assert hasattr(module, name), 'Database class is not supported in both dabstract.dataset.dbs ' + os.environ['dabstract_CUSTOM_DIR'] + '.dataset.dbs. Please check'
        return getattr(module, name)(paths=paths,filter=filter, test_only=test_only, xval=xval, tmp_folder=tmp_folder, **kwargs)
    elif isinstance(name, DictSeqAbstract):
        pass
    elif isinstance(name,(type, types.ClassType)):
        return name(paths=paths,filter=filter, test_only=test_only, xval=xval, tmp_folder=tmp_folder, **kwargs)
