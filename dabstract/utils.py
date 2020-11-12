import yaml
import numpy as np
import os
import sys
import itertools
from pprint import pprint
import copy
import importlib

from typing import Union, Any, List, Callable, Dict, Iterable


def load_yaml_config(
    filename: str,
    path: str,
    walk: bool = False,
    verbose: bool = True,
    post_process: Callable = None,
    unpack: bool = False,
    **kwargs
) -> Union[str, Union[Dict, Any]]:
    """Load a yaml configuration file with some additional functionality
    For example do::
        $  data = load_yaml_config(filename=path_to_dir, path=path_to_yaml, walk=True/False,
        $  post_process=dataset_from_config, **kwargs)

    This loader adds additional constructors which can be used in your yaml file.
    These include the joining of strings witj !join:
    ::
        For example;     !join [str1, str2] -> 'str1tr2'

    Or the joining with underscores with !usjoin:
    ::
        !usjoin[str1, str2] -> 'str1_str2'

    Or the joining of paths with !pathjoin:
    ::
        !pathjoin[str1, str2] -> 'str1/str2'

    Similarly one can inject string or int into the configuration from an external source using !kwargs:
    ::
        In python:
        load_yaml_config(..., **{'item1': {'item12': 2)
        In yaml file
        !kwargs [item1,item12]

    Finally, one can initialise classes using the !class constructor:
    ::
        !class [path.to.module]


    Parameters
    ----------
    filename : str
        filename of the .yaml file
    path : str
        path to a folder containg the .yaml file.
        If walk==True, one can place .yaml in a nested folder
    walk : bool
        Walk through the nested folder
    verbose : bool
        Allow print statements
    post_process : Callable
        Place here a factory function that accepts the yaml input configuration.
        Identical as post_process(load_yaml_config).
    unpack : bool
        Whether or not to unpack to yaml configuration dict before applying it to post_process()
    kwargs : dict
        kwargs will be used by the !kwargs constructor to inject parameters from your code flow
        into the configuration.

    Returns
    -------
    Any
    """

    # search directory
    if walk:
        for root, subFolders, files in os.walk(path):
            if (filename + ".yaml") in files:
                path = os.path.join(root)
                break
    filepath = os.path.join(path, filename + ".yaml")

    # join strings
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return "".join([str(i) for i in seq])

    # join with underscore
    def usjoin(loader, node):
        seq = loader.construct_sequence(node)
        return "_".join([str(i) for i in seq])

    # join path
    def pathjoin(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*[str(i) for i in seq])

    # replace by kwarg value
    def get_kwarg(loader, node):
        seq = loader.construct_sequence(node)
        kwarg = copy.deepcopy(kwargs)
        output_str = str()
        for k, key in enumerate(seq):  # nested dictionary
            output_str = ".".join((output_str, key))
            if key in kwarg:
                kwarg = kwarg[key]
                if (k + 1) == len(seq):
                    output_str = kwarg
        return output_str

    # replace by class
    def get_class(loader, node):
        seq = loader.construct_sequence(node)
        assert len(seq) == 1
        loc_rdot = seq[0].rfind(".")
        module = safe_import_module(seq[0][:loc_rdot])
        return getattr(module, seq[0][loc_rdot + 1 :])

    # add custom custructors
    yaml.add_constructor("!join", join)  # join strings
    yaml.add_constructor("!usjoin", usjoin)  # join strings with underscores
    yaml.add_constructor("!pathjoin", pathjoin)  # join paths
    yaml.add_constructor("!kwarg", get_kwarg)  # join paths
    yaml.add_constructor("!class", get_class)  # join paths

    # load yaml
    with open(filepath, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    # verbose
    if verbose:
        pprint(cfg)

    # custom post process if needed
    if post_process is not None:
        if unpack:
            cfg = post_process(**cfg)
        else:
            cfg = post_process(cfg)

    return cfg


def reformat_yaml(
    cfg: Union[List[Any], Dict[str, Any]]
) -> List[Union[np.ndarray, Any]]:
    """Reformat yaml list to numpy if possible

    Parameters
    ----------
        cfg : List[Any]
            loaded yaml configuration

    Returns
    ----------
        Reformatted yaml
    """
    if isinstance(cfg, list):
        tmp_cfg = cfg
        if any(isinstance(i, list) for i in tmp_cfg):  # nested
            tmp_cfg = [val for i in tmp_cfg for val in i]
        if np.all([(isinstance(i, int) | isinstance(i, float)) for i in tmp_cfg]):
            cfg = np.array(cfg)
    return cfg


def reformat_yaml_iter(
    cfg: Union[List[Any], Dict[str, Any]]
) -> List[Union[np.ndarray, Any]]:
    """Reformat yaml list to numpy if possible in an iterative fasion

    Parameters
    ----------
        cfg : Union[List[Any], Dict[str,Any]]
            loaded yaml configuration

    Returns
    ----------
        Reformatted yaml
    """
    """Reformat yaml list to numpy if possible in iterative fashion"""
    for k, v in cfg.items():
        if isinstance(v, dict):
            reformat_yaml_iter(v)
        else:
            cfg[k] = reformat_yaml(v)
    return cfg


def load_yaml(filepath: str) -> Dict:
    """Load yaml file

    Parameters
    ----------
        filepath : str
            filepath to yaml file

    Returns
    ----------
        Dictionary
    """
    # load file
    with open(filepath, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    return cfg


def str_in_list(lst: List, string: str) -> List[int]:
    """Get indices of a string in a list

    Parameters
    ----------
    lst : list
        list of strings
    string : str
        string to search for

    Returns
    -------
    List[int]
    """
    if isinstance(string, (list, np.ndarray)):
        return [str_in_list(lst, _string) for _string in string]
    indices = [i for i in range(len(lst)) if lst[i] == string]
    return indices


def listnp_combine(
    lst: List, method: str = "concatenate", axis: int = 0, allow_error: bool = False
) -> np.ndarray:
    """Concatenate or stack a list of numpy along with error handling

    Parameters
    ----------
    lst : list
        list of np arrays
    method : str
        'concatenate' or 'stack'
    axis : int
        axis to concat or stack over
    allow_error : bool
        allow for error handling. If op does not succes, list is provided

    Returns
    -------
    np.array OR list of np.array in case of error
    """

    def combine(lst):
        if isinstance(lst, list):
            lst = np.array(lst)
        if method == "concatenate":
            npstack = np.concatenate(lst, axis=axis)
        elif method == "stack":
            npstack = np.stack(lst, axis=axis)
        else:
            print("method for concatenating not supported in listnp_stack in utils.py.")
            sys.exit()
        return npstack

    if len(np.shape(lst[0])) == 0:
        return np.array(lst)
    else:
        if allow_error:
            try:
                npcombine = combine(lst)
            except ValueError:  # if cant combine
                npcombine = copy.deepcopy(lst)
        else:
            npcombine = combine(lst)
    return npcombine


def listdictnp_combine(
    lst: List,
    method: str = "concatenate",
    axis: int = 0,
    keep_nested: bool = False,
    allow_error: bool = False,
) -> Dict[str, Union[np.ndarray, List]]:
    """Concatenate or stack a list of dictionaries contains numpys along with error handling

    Parameters
    ----------
    lst : list
        list of dicts containings np arrays
    method : str
        'concatenate' or 'stack'
    axis : int
        axis to concat or stack over
    keep_nested : bool
        keep nested structure of list or not
    allow_error : bool
        allow for error handling. If op does not succes, list is provided

    Returns
    -------
    np.array OR list of np.array in case of error
    """
    for k in range(len(lst)):
        assert (
            lst[0].keys() == lst[k].keys()
        ), "Dict keys do not match in listdictnp_combine fct"
    # get keys
    keys = lst[0].keys()
    output_dict = dict()

    for key in keys:
        # merge nested list
        if keep_nested:
            tmp = [None] * len(lst)
            for k in range(len(lst)):
                tmp[k] = lst[k][key]
        else:
            tmp = list()
            for k in range(len(lst)):
                tmp = [*tmp, *lst[k][key]]

        # convert to numpy if possible
        output_dict[key] = listnp_combine(
            tmp, method=method, axis=axis, allow_error=allow_error
        )
    return output_dict


def unique_list(seq: List) -> List:
    """Get unique entries in a list

    Parameters
    ----------
        seq: list
            sequence you want to unique-fy

    Returns
    -------
        list of unique values
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def flatten_nested_lst(nested_lst: List[List]) -> List:
    """Flatten a nested list

    Parameters
    ----------
        nested_lst : list
            A nested list

    Returns
    -------
        flattened list
    """
    return [item for sublist in nested_lst for item in sublist]


def any2str(obj: Any) -> str:
    """Convert anything to a string

    Parameters
    ----------
        obj : Any
            anything to convert to a str rep

    Returns
    -------
        string
    """
    if isinstance(obj, str):
        return obj
    elif (
        isinstance(obj, int)
        | isinstance(obj, np.ndarray)
        | isinstance(obj, np.int64)
        | isinstance(obj, dict)
    ):
        return str(obj)
    elif obj is None:
        return "None"
    else:
        print("Instance not supported in any2str()")
        sys.exit()


def filter_data(
    data: Iterable, sel_vect: Union[List[int], np.ndarray], squeeze: bool = False
) -> Iterable:
    """Filter any sequential data object based on indices

    Parameters
    ----------
    data : Iterable
        Iterable object to be filtered
    sel_vect : List[int] or np.ndarray
        indices to filter
    squeeze : bool
        squeeze output when numpy

    Returns
    -------
        Filtered iterable
    """
    # inits
    from dabstract.abstract import abstract

    if isinstance(sel_vect, (np.int64, int)):
        sel_vect = np.array([sel_vect])
    # do different filtering depending on input
    if isinstance(data, list):
        out = [data[k2] for k2 in sel_vect]
        if squeeze:
            out = out[0]
    elif isinstance(data, np.ndarray):
        out = data[
            sel_vect,
        ]
        if squeeze:
            out = out.squeeze()
    elif isinstance(data, tuple):
        tmp = list(data)
        out = [tmp[k2] for k2 in sel_vect]
        if squeeze:
            out = out[0]
    elif isinstance(data, dict):
        out = dict()
        for key in data:
            out[key] = filter_data(data[key], sel_vect, squeeze=squeeze)
    elif isinstance(data, abstract):
        out = data
    elif hasattr(data, "__getitem__"):
        out = data[sel_vect]
    elif data is None:
        out = None
    else:
        print("Not supported data format in filter_data fct")
        sys.exit()
    return out


def safe_import_module(module_string: str) -> object:
    """Import module with error handling

    Parameters
    ----------
        module_string : str
            module string to import

    Returns
    ----------
        imported module
    """
    try:
        return importlib.import_module(module_string)
    except ImportError:
        return object()


def combs_numpy(values: Union[List, np.ndarray]) -> np.ndarray:
    """All possible combinations of numpy entries

    Parameters
    ----------
    values : Union[List, np.ndarray]
        All unique examples to get combinations from

    Returns
    -------
    np.ndarray
    """
    return np.unique(
        np.array(
            list(itertools.product(*[values[:, k] for k in range(values.shape[1])]))
        ),
        axis=0,
    )


def combs_list(values: Union[List, np.ndarray]) -> List:
    """All possible combinations of list entries

    Parameters
    ----------
    values : Union[List, np.ndarray]
        All unique examples to get combinations from

    Returns
    -------
    List
    """
    return list(itertools.product(*values))


# def combs_size_numpy(values: Union[List, np.ndarray], size: int) -> np.ndarray:
#     """Size of all possible combinations of numpy entries
#
#     Parameters
#     ----------
#     values : Union[List, np.ndarray]
#         Any object you want the length from
#
#     Returns
#     -------
#     int
#     """
#
#     return np.array([comb for comb in itertools.combinations(values, size)])


def pprint_ext(str: str, dict: Dict, np_precision: int = 2) -> None:
    """pprint with np precision specification and title

    Parameters
    ----------
    str : str
        title to print
    dict : Dict
        Dictionary to print
    np_precision : int
        precision of numpy print
    """
    np.set_printoptions(precision=np_precision, suppress=True)
    print(str)
    pprint(dict)


def safe_len(var: Any) -> int:
    """safely get length

    Parameters
    ----------
    var : Any
        Any object you want the length from

    Returns
    -------
    int
    """
    try:
        return len(var)
    except TypeError:
        return 1


def stringlist2ind(strlist: List[str]) -> np.ndarray:
    """list to unique indices

    Parameters
    ----------
    strlist : List[str]
        List of strings to indices

    Returns
    -------
    List[int]
    """
    subdb_ext = unique_list(strlist)
    group_np = np.zeros((len(strlist)))
    for k in range(len(strlist)):
        group_np[k] = subdb_ext.index(strlist[k])
    return group_np.astype(np.int)


def list_intersection(lst1: List, lst2: List) -> List:
    """Intersect the values of two lists

    Parameters
    ----------
    lst1 : List
        First list
    lst2 : List
        Second list

    Returns
    -------
    Intersected list
    """
    return [value for k, value in enumerate(lst1) if value in lst2]


def list_difference(lst1: List, lst2: List) -> List:
    """Get difference between the values of two lists

    Parameters
    ----------
    lst1 : List
        First list
    lst2 : List
        Second list

    Returns
    -------
    Difference list
    """
    return [value for k, value in enumerate(lst1) if not value in lst2] + [
        value for k, value in enumerate(lst2) if not value in lst1
    ]


def get_class(name: str, module_location: str, *args, **kwargs):
    """Load a class given the name, module location and it's args and kwargs

    Parameters
    ----------
    name : str
        class name
    module_location : str
        class location

    Returns
    -------
    Instance
    """
    module = safe_import_module(module_location)
    assert hasattr(module, name), name + " is not a part of module " + module_location
    return getattr(module, name)(*args, **kwargs)


def get_fct(name: str, module_location: str):
    """Load a fct given the name, module location and it's args and kwargs

    Parameters
    ----------
    name : str
        function name
    module_location : str
        function location

    Returns
    -------
    Callable
    """
    module = safe_import_module(module_location)
    assert hasattr(module, name), name + " is not a part of module " + module_location
    return getattr(module, name)
