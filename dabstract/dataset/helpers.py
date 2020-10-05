import pathlib
import pickle
import types
import soundfile as sf

from dabstract.utils import safe_import_module
from dabstract.dataset.abstract import *
from dabstract.dataset import dbs

def dataset_from_config(config, overwrite_xval=False):
    """Create a dataset from configuration

    This function creates a dataset class from a dictionary definition.
    It is advised to define your configuration in yaml, read it using
    dabstract.utils.yaml_from_config and utilise it as such:

    Example:
        $  data = load_yaml_config(filename=path_to_dir, dir=path_to_yaml, walk=True/False,
        $  post_process=dataset_from_config, **kwargs)

        As a configuration one is advised to check the examples in dabstract/examples/introduction.
        e.g. introduction/configs/dbs/EXAMPLE_anomaly.yaml:

        A full format is defined as with placeholders:
        $ datasets:
        $  - name: dataset0
        $     parameters:
        $       paths: ..
        $       select: ..
        $       split: ..
        $       other: ..
        $       test_only: 1/0
        $   - name: dataset1
        $     parameters:
        $       paths: ..
        $       select: ..
        $       split: ..
        $       other: ..
        $       test_only: 1/0
        $ select:
        $ split:
        $ xval:

        For each dataset, name and parameters/path is mandatory.
        Select/split/test_only are default options to subsample, split or define
        that the dataset is for testing only (1/0) respectively.
        other refers to the fact that you can add custom keys/items to your own dataset

        select/split can be used for a dataset specifically or for the whole dataset (below)
        xval specifies the xval definition

        Select/split/xval are all defined in the following way:
        $   name:
        $   parameters:
        parameters is not mandatory
        name refers to either a string of a class that can be found in dataset.select/dataset.xval respectively or in
        the custom folder defined by os.environ["dabstract_CUSTOM_DIR"]

        More information on the possibilities of select, split and xval the reader is referred to:
        dataset.dataset.add_select()
        dataset.dataset.add_split()
        dataset.dataset.set_xval()

    Arguments:
        config (string): dictionary configuration
        overwrite_xval (bool): overwrite xval file

    Returns:
        dataset class
    """

    assert isinstance(config,dict), "config should be a dictionary"
    assert 'datasets' in config, "config should have a datasets key"
    assert isinstance(config['datasets'],list), "config['dataset'] should be represented as a list where each item is a dictionary containing kwargs of your dataset."
    from dabstract.dataset.dataset import dataset
    ddataset = dataset()
    # init datasets
    for k,db in enumerate(config['datasets']):
        ddataset.concat(dataset_factory(name=db['name'], **db['parameters']))
    # add other functionality
    if 'xval' in config:
        ddataset.set_xval(**config['xval'], overwrite=overwrite_xval)
    if 'split' in config:
        ddataset.add_split(config['split'])
    if 'select' in config:
        ddataset.add_select(config['select'])
    # if 'unpack' in config:
    #     ddataset = ddataset.unpack(config['unpack'])
    # if 'alias' in config:
    #     ddataset.add_alias(key,new_key)
    return ddataset

def dataset_factory(name=None,
                    paths=None,
                    xval=None,
                    select=None,
                    test_only=0,
                    tmp_folder=None,
                    **kwargs):
    """Dataset factory

    This function creates a dataset class from name and parameters.
    Specifically, this is used to search by name for that particular database class in
    - environment variable folder: os.environ["dabstract_CUSTOM_DIR"] = your_dir
    - dabstract.dataset.dbs folder

    if name is defined as a class object, than it uses this to init the dataset with the given kwargs.
    This function is mostly used by dataset_from_config(). One is advised to directly
    import the desired dataset class instead of using dataset_factory. This is only
    handy for configuration based experiments, which need a load from string.

    Example:
        $  data = dataset_factory(name='DCASE2020Task1B',
        $                         paths={'data': path_to_data,
        $                                'meta': path_to_meta,
        $                                'feat': path_to_feat},

        One is advised to check the examples in dabstract/examples/introduction on
        how to work with datasets

    Arguments:
        name (str/class): name of the class (or the class directly)
        kwargs: ToDo, not defined as this should be used only by load_from_config()
    Returns:
        dataset class
    """

    # get dataset
    if isinstance(name, str):
        # get db class
        module = dbs
        if not hasattr(module, name):  # check customs
            module = safe_import_module(os.environ['dabstract_CUSTOM_DIR'] + '.dataset.dbs')
            assert hasattr(module, name), 'Database class is not supported in both dabstract.dataset.dbs ' + os.environ['dabstract_CUSTOM_DIR'] + '.dataset.dbs. Please check'
        return getattr(module, name)(paths=paths,select=select, test_only=test_only, xval=xval, tmp_folder=tmp_folder, **kwargs)
    elif isinstance(name, DictSeqAbstract):
        pass
    elif isinstance(name,type):
        return name(paths=paths,select=select, test_only=test_only, xval=xval, tmp_folder=tmp_folder, **kwargs)

def dictseq_from_folder(path, extension='.wav', map_fct=None, file_info_save_path=None, filepath=None, overwrite_file_info=False, **kwargs):
    """** Deprecated **
    Get meta information of the files in a directory and place them in a DictSeq

    This function gets meta information (e.g. sampling frequency, length) of files in your provided directory.
    It return a DictSeq with the filenames/information/subfolders.
    This is mainly useful for obtaining apriori information of wav files such that they can be splitted in a lazy
    manner from disk.

    Example:
        $  dictseq = get_dir_info(path_to_directory, extension='.wav', file_info_save_path=None, filepath=None, overwrite_file_info=False)

    Arguments:
        path (str): path to the directory to check
        extension (string): only evaluate files with that extension
        map_fct: add a mapping function y = f(x) to the 'data'
        filepath: in case you already have the files you want to obtain information from, the dir tree search is not done
                    and this is used instead
        file_info_save_path: save the information to this location
                             this function can be costly, so saving is useful
        overwrite_file_info: overwrite file info file

    Returns:
        dictseq containing file information as a list, formatted as:
        output['filepath'] = list of paths to files
        output['example'] = example string (i.e. filename without extension)
        output['filename'] = filename
        output['subdb'] = relative subdirectory (starting from 'path') to file
        output['info'][file_id] = { 'output_shape': .., #output shape of the wav file
                                    'fs': .., # sampling frequency
                                    'time_step' ..: # sample period
                                    }
    """
    # if 'save_path' in kwargs:
    #     file_info_save_path = kwargs['save_path']
    #     warnings.warn("save_path is deprecated in dataset.py/dict_from_folder(). Change to 'file_info_save_path'", DeprecationWarning)
    #
    # data = DictSeqAbstract()
    # # get info
    # fileinfo = get_dir_info(path,extension=extension, file_info_save_path=file_info_save_path, filepath=filepath, overwrite_file_info=overwrite_file_info)
    # # add data
    # data.add('data', fileinfo['filepath'], active_key=True)
    # # add meta
    # for key in fileinfo:
    #     data.add(key,fileinfo[key])
    # # add map
    # data['data'] = MapAbstract(data['data'],map_fct=map_fct)
    # # set active key
    # data.set_active_keys('data')

    data = FolderDictSeqAbstract(path, extension=extension, map_fct=map_fct, file_info_save_path=file_info_save_path, filepath=filepath, overwrite_file_info=overwrite_file_info, **kwargs)
    warnings.warn("This function is deprecated. Please use the FolderDictSeqAbstract() class to initialise a data folder as DictSeqAbstract")
    return data

class FolderDictSeqAbstract(DictSeqAbstract):
    """Get meta information of the files in a directory and place them in a DictSeq

    This function gets meta information (e.g. sampling frequency, length) of files in your provided directory.
    It return a FolderDictSeq with the filenames/information/subfolders.
    A FolderDictSeq is inherited from DictSeq and has similar functionality. However,
    for a FolderDictSeq the active_keys are fixed to 'data'. In essence FolderDictSeq is a
    data container showing information of a walk through a folder.
    Additionally, this format keeps track of relevant information to either wav or numpy files.
    prepare_feat and add_split only work on data fields that have this structure.

    Example:
        $  folderdictseq = FolderDictSeqAbstract(path_to_directory, extension='.wav', file_info_save_path=None, filepath=None, overwrite_file_info=False)

    Arguments:
        path (str): path to the directory to check
        extension (string): only evaluate files with that extension
        map_fct: add a mapping function y = f(x) to the 'data'
        filepath: in case you already have the files you want to obtain information from, the dir tree search is not done
                    and this is used instead
        file_info_save_path: save the information to this location
                             this function can be costly, so saving is useful
        overwrite_file_info: overwrite file info file

    Returns:
        folderdictseq containing file information as a list, formatted as:
        output['filepath'] = list of paths to files
        output['example'] = example string (i.e. filename without extension)
        output['filename'] = filename
        output['subdb'] = relative subdirectory (starting from 'path') to file
        output['info'][file_id] = { 'output_shape': .., #output shape of the wav file
                                    'fs': .., # sampling frequency
                                    'time_step' ..: # sample period
                                    }
    """
    def __init__(self, path, extension='.wav', map_fct=None, file_info_save_path=None, filepath=None, overwrite_file_info=False, info=None, **kwargs):
        super().__init__()
        if 'save_path' in kwargs:
            file_info_save_path = kwargs['save_path']
            warnings.warn("save_path is deprecated in dataset.py/dict_from_folder(). Change to 'file_info_save_path'",
                          DeprecationWarning)
        # get info
        fileinfo = get_dir_info(path, extension=extension, file_info_save_path=file_info_save_path, filepath=filepath,
                                overwrite_file_info=overwrite_file_info)
        # overwrite file info
        if info is not None:
            fileinfo['info'] = info
        # add data
        self.add('data', fileinfo['filepath'], info = fileinfo['info'])
        # add meta
        # for key in fileinfo:
        #     self.add(key, fileinfo[key])
        self.add_dict(fileinfo, lazy=False)
        # add map
        if map_fct is not None:
            self['data'] = MapAbstract(self['data'], map_fct=map_fct)
        # set active key
        self._set_active_keys('data')

    def set_active_keys(self,keys):
        """ Disables set of active keys
        """
        raise Exception("A FolderDictSeqAbstract should always have data as the only active key. Setting not possible. Please use DictSeqAbstract if other functionality is needed.")

    def reset_active_keys(self):
        """ Disables reset of active keys
        """
        raise Exception("A FolderDictSeqAbstract should always have data as the only active key. Resetting not possible. Please use DictSeqAbstract if other functionality is needed.")

    def __repr__(self):
        """ string print representation of function
        """
        return 'folder_dict_seq containing: ' + str(self.keys())

def get_dir_info(path, extension='.wav', file_info_save_path=None, filepath=None, overwrite_file_info=False, **kwargs):
    """Get meta information of the files in a directory.

    This function gets meta information (e.g. sampling frequency, length) of files in your provided directory.
    It return a dictionary with the filenames/information/subfolders.
    This is mainly useful for obtaining apriori information of wav files such that they can be splitted in a lazy
    manner from disk.

    Example:
        $  dict = get_dir_info(path_to_directory, extension='.wav', file_info_save_path=None, filepath=None, overwrite_file_info=False),

    Arguments:
        path (str): path to the directory to check
        extension (string): only evaluate files with that extension
        filepath: in case you already have the files you want to obtain information from, the dir tree search is not done
                    and this is used instead
        file_info_save_path: save the information to this location
                             this function can be costly, so saving is useful
        overwrite_file_info: overwrite file info file
    Returns:
        dict containing file information as a list, formatted as:
        output['filepath'] = list of paths to files
        output['example'] = example string (i.e. filename without extension)
        output['filename'] = filename
        output['subdb'] = relative subdirectory (starting from 'path') to file
        output['info'][file_id] = { 'output_shape': .., #output shape of the wav file
                                    'fs': .., # sampling frequency
                                    'time_step' ..: # sample period
                                    }
    """

    def _get_dir_info(filepath, extension):
        info = [dict()] * len(filepath)
        if extension == '.wav':
            # import soundfile as sf
            for k in range(len(filepath)):
                f = sf.SoundFile(filepath[k])
                info[k]['output_shape'] = np.array([len(f), f.channels])
                info[k]['fs'] = f.samplerate
                info[k]['time_step'] = 1 / f.samplerate
        return info

    if 'save_path' in kwargs:
        file_info_save_path = kwargs['save_path']
        warnings.warn("'save_path' is deprecated in dataset.get_dir_info(). Change to 'file_info_save_path'", DeprecationWarning)

    # get dirs
    if not isinstance(filepath,list):
        filepath = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            tmp = [os.path.join(root, file) for file in files if extension in file]
            if len(tmp)>0:
                tmp.sort()
                filepath += tmp

    example = [os.path.relpath(file, path) for file in filepath if extension in file]
    filename = [os.path.split(file)[1] for file in example if extension in file]
    subdb = [os.path.split(file)[0] for file in example if extension in file]

    if file_info_save_path is not None:
        path = file_info_save_path

    # get additional info
    if not os.path.isfile(os.path.join(path, 'file_info.pickle')) or overwrite_file_info:
        info = _get_dir_info(filepath,extension)
        if (file_info_save_path is not None) and (info is not None):
            os.makedirs(path, exist_ok=True)
            with open(pathlib.Path(path, 'file_info.pickle'), "wb") as fp: pickle.dump((info, example),  fp)
    else:
        with open(os.path.join(path, 'file_info.pickle'), "rb") as fp:
            info, example_in = pickle.load(fp)
        info = [info[k] for k in range(len(example)) if example[k] in example_in]
        assert len(example_in) == len(filepath), "info file not of same size as directory"
    return {'filepath': filepath, 'example': example, 'filename': filename, 'subdb': subdb, 'info': info}