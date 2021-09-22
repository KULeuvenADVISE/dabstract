import pathlib
import pickle
import soundfile as sf
import cv2 as cv
from tqdm import tqdm

from dabstract.utils import safe_import_module
from dabstract.abstract import *
from dabstract.dataset import dbs

from typing import Any, List, Optional, TypeVar, Callable, Dict

tvDataset = TypeVar("Dataset")


def dataset_from_config(config: Dict, overwrite_xval: bool = False) -> tvDataset:
    """Create a dataset from configuration

    This function creates a dataset class from a dictionary definition.
    It is advised to define your configuration in yaml, read it using
    dabstract.utils.yaml_from_config and utilise it as such::

        $  data = load_yaml_config(filename=path_to_dir, path=path_to_yaml, walk=True/False,
        $  post_process=dataset_from_config, **kwargs)

    As a configuration one is advised to check the examples in dabstract/examples/introduction.
    e.g. introduction/configs/dbs/EXAMPLE_anomaly.yaml::

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
    Select/split/xval are all defined in the following way::
        $   name:
        $   parameters:

    `name` refers to either a string of a class that can be found in dataset.select/dataset.xval respectively or in
    the custom folder defined by os.environ["dabstract_CUSTOM_DIR"].
    `parameters` is not mandatory.

    More information on the possibilities of select, split and xval the reader is referred to:
    dataset.dataset.add_select()
    dataset.dataset.add_split()
    dataset.dataset.set_xval()

    Parameters
    ----------
    config : str
        dictionary configuration
    overwrite_xval : bool
        overwrite xval file
    """

    assert isinstance(config, dict), "config should be a dictionary"
    assert "datasets" in config, "config should have a datasets key"
    assert isinstance(
        config["datasets"], list
    ), "config['dataset'] should be represented as a list where each item is a dictionary containing kwargs of your dataset."

    # init datasets
    for k, db in enumerate(config["datasets"]):
        tmp_ddataset = dataset_factory(name=db["name"], **db["parameters"])
        if k == 0:
            ddataset = tmp_ddataset
        else:
            ddataset.concat(tmp_ddataset, intersect=True)
    # add other functionality
    if "select" in config:
        raise NotImplementedError("This is currently under dev. Please add select to each dataset branch.")
        if isinstance(config["select"], list):
            for _select in config["select"]:
                ddataset.add_select(**_select)
        elif isinstance(config["select"], dict):
            ddataset.add_select(**config["select"])
    if "split" in config:
        raise NotImplementedError("This is currently under dev. Please add select to each dataset branch.")
        if isinstance(config["split"], (int, float)):
            ddataset.add_split(config["split"])
        elif isinstance(config["split"], dict):
            ddataset.add_split(**config["split"])
        else:
            raise NotImplementedError
    if "xval" in config:
        ddataset.set_xval(**config["xval"], overwrite=overwrite_xval)
    return ddataset


def dataset_factory(
        name: (str, tvDataset, type) = None,
        paths: Dict[str, str] = None,
        xval: Optional[Dict[str, Union[str, int, Dict]]] = None,
        split: Optional[Dict[str, Union[str, int, Dict]]] = None,
        select: Optional[Dict[str, Union[str, int, Dict]]] = None,
        test_only: Optional[bool] = 0,
        **kwargs
) -> tvDataset:
    """Dataset factory

    This function creates a dataset class from name and parameters.
    Specifically, this is used to search by name for that particular database class in
    - environment variable folder: os.environ["dabstract_CUSTOM_DIR"] = your_dir
    - dabstract.dataset.dbs folder

    If name is defined as a class object, than it uses this to init the dataset with the given kwargs.
    This function is mostly used by dataset_from_config(). One is advised to directly
    import the desired dataset class instead of using dataset_factory. This is only
    handy for configuration based experiments, which need a load from string.
    For example::
        $  data = dataset_factory(name='DCASE2020Task1B',
        $                         paths={'data': path_to_data,
        $                                'meta': path_to_meta,
        $                                'feat': path_to_feat},

    One is advised to check the examples in dabstract/examples/introduction on
    how to work with datasets

    Parameters
    ----------
    select: Dict[str,Union[str,int, Dict]]
        selector configuration
    split: Dict[str,Union[str,int, Dict]]
        split configuration
    xval : Dict[str,Union[str,int, Dict]]
        xval configuration
    test_only : bool
        use the dataset for test (test_only=1) or both train and test (test_only=0)
    name : str/instance/object
        name of the class (or the class directly)
    paths : dict[str]
        dictionary containing paths to the data
    kwargs: ToDo, not defined as this should be used only by load_from_config()

    Returns
    -------
    dataset : Dataset class
    """
    from dabstract.dataset.dataset import Dataset

    # get dataset
    if isinstance(name, str):
        # get db class
        module = dbs
        if not hasattr(module, name):  # check customs
            module = safe_import_module(
                os.environ["dabstract_CUSTOM_DIR"] + ".dataset.dbs"
            )
            assert hasattr(module, name), (
                    "Database class is not supported in both dabstract.dataset.dbs "
                    + os.environ["dabstract_CUSTOM_DIR"]
                    + ".dataset.dbs. Please check"
            )
        db = getattr(module, name)(paths=paths, test_only=test_only, **kwargs)
    elif isinstance(name, Dataset):
        db = name
    elif isinstance(name, type):
        try:
            db = name(paths=paths, test_only=test_only, **kwargs)
        except:
            raise ValueError("Class is not a Dataset.")

    # add other functionality
    if select is not None:
        if isinstance(select, list):
            for _select in select:
                db.add_select(**_select)
        elif isinstance(select, dict):
            db.add_select(**select)
    if split is not None:
        if isinstance(split, (int, float)):
            db.add_split(split)
        elif isinstance(split, dict):
            db.add_split(**split)
        else:
            raise NotImplementedError
    if xval is not None:
        db.set_xval(**xval)

    return db


def get_dir_info(
        path: str,
        type: str = None,
        extension: str = None,
        file_info_save_path: bool = None,
        overwrite_file_info: bool = False,
        **kwargs
) -> Dict[str, List[Any]]:
    """Get meta information of the files in a directory.

    This function gets meta information (e.g. sampling frequency, length) of files in your provided directory.
    It return a dictionary with the filenames/information/subfolders.
    This is mainly useful for obtaining apriori information of wav files such that they can be splitted in a lazy
    manner from disk.

    Parameters
    ----------
    path : str
        path to the directory to check
    type : str ['audio', 'video'], default: None
        if extra meta information is desired then a type should be specified
    extension : str
        only evaluate files with that extension
    map_fct : Callable
        add a mapping function y = f(x) to the 'data'
    # filepath : str
    #     in case you already have the files you want to obtain information from,
    #     the dir tree search is not done and this is used instead
    file_info_save_path: : str
        save the information to this location
        this function can be costly, so saving is useful
    overwrite_file_info : bool
        overwrite file info file


    Returns
    -------
    dict : dict
        dict containing file information as a list,
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

    if "save_path" in kwargs:
        file_info_save_path = kwargs["save_path"]
        warnings.warn(
            "'save_path' is deprecated in dataset.get_dir_info(). Change to 'file_info_save_path'",
            DeprecationWarning,
        )

    # get dirs
    # if not isinstance(filepath, list):
    filepath = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        tmp = [os.path.join(root, file) for file in files if extension in file]
        if len(tmp) > 0:
            tmp.sort()
            filepath += tmp

    rfilepath = [os.path.relpath(file, path) for file in filepath if extension in file]
    filename = [os.path.split(file)[1] for file in rfilepath]
    rfolderstructure = [os.path.split(file)[0] for file in rfilepath]
    identifier = [os.path.splitext(file)[0] for file in filename]
    postfix = ['' for file in filename]

    if file_info_save_path is not None:
        path = file_info_save_path

    # get additional info
    if (
            not os.path.isfile(os.path.join(path, "file_info.pickle"))
            or overwrite_file_info
    ):
        info = _get_dir_info(filepath, type)
        if (file_info_save_path is not None) and (info is not None):
            os.makedirs(path, exist_ok=True)
            with open(pathlib.Path(path, "file_info.pickle"), "wb") as fp:
                pickle.dump((info, rfilepath), fp)
    else:
        with open(os.path.join(path, "file_info.pickle"), "rb") as fp:
            info, rfilepath_in = pickle.load(fp)
        if isinstance(info, list):
            print("This format of file_info.pickle is deprecated. Converting to the new format...")
            info = _get_dir_info(filepath, type)
            if (file_info_save_path is not None) and (info is not None):
                os.makedirs(path, exist_ok=True)
                with open(pathlib.Path(path, "file_info.pickle"), "wb") as fp:
                    pickle.dump((info, rfilepath), fp)
        if rfilepath != rfilepath_in:
            raise NotImplementedError("Amount of examples in info file does not match the "
                                      "amount of examples in the folder.")
    return {
        "filepath": filepath,
        "rfilepath": rfilepath,
        "filename": filename,
        "rfolder": rfolderstructure,
        "identifier": identifier,
        "postfix": postfix,
        **info,
    }


def _get_dir_info(filepath: str, type: str) -> List[Dict]:
    if type == "audio":
        print('Acquiring info from audio folder...')
        info = dict()
        for key in ['length', 'channels', 'time_axis']:
            info[key] = np.zeros(len(filepath), dtype='int')
        for key in ['fs', 'time_step', 'duration']:
            info[key] = np.zeros(len(filepath))
        info['output_shape'] = np.zeros((len(filepath),2),dtype=int)
        for k in tqdm(range(len(filepath))):
            f = sf.SoundFile(filepath[k])
            info['output_shape'][k] = np.array([len(f), f.channels])
            info['length'][k] = len(f)
            info['channels'][k] = f.channels
            info['time_axis'][k] = 0
            info['fs'][k] = f.samplerate
            info['time_step'][k] = 1 / f.samplerate
            info['duration'][k] = len(f) / f.samplerate
    elif type == "camera":
        raise NotImplementedError
        # print('Acquiring info from camera folder...')
        # for k in range(len(filepath)):
        #     info[k] = dict()
        #     vid = cv.VideoCapture(filepath[k])
        #     info[k] = {'width': int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
        #                'height': int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)),
        #                'fs': vid.get(cv.CAP_PROP_FPS),
        #                'length': int(vid.get(cv.CAP_PROP_FRAME_COUNT))}
        #     info[k].update({'duration': 1 / info[k]['fs'] * info[k]['length'],
        #                     'output_shape': np.array([info[k]['length'],info[k]['width'],info[k]['length']])})
    elif type is None:
        pass
    else:
        raise NotImplementedError('type should be audio, camera or None')
    return info
