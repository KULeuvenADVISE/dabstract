import yaml
import numpy as np
import os
from datetime import datetime
import sys
import itertools
from pprint import pprint
import copy
import importlib

def load_yaml_config(filename, dir, walk=False, verbose=True, post_process=None, unpack=False, **kwargs):
    """Load a yaml configuration file with some additional functionality

    ...

    Example:
        $  data = load_yaml_config(filename=path_to_dir, dir=path_to_yaml, walk=True/False,
        $  post_process=dataset_from_config, **kwargs)

    ...

    Arguments:
        ...

    Returns:
        ...
    """

    # search directory
    if walk:
        for root, subFolders, files in os.walk(dir):
            if (filename+'.yaml') in files:
                dir = os.path.join(root)
                break
    filepath = os.path.join(dir, filename + '.yaml')

    # join strings
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    # join with underscore
    def usjoin(loader, node):
        seq = loader.construct_sequence(node)
        return '_'.join([str(i) for i in seq])

    # join path
    def pathjoin(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*[str(i) for i in seq])

    # replace by kwarg value
    def get_kwarg(loader, node):
        seq = loader.construct_sequence(node)
        kwarg = copy.deepcopy(kwargs)
        output_str = str()
        for k,key in enumerate(seq): #nested dictionary
            output_str = ".".join((output_str, key))
            if key in kwarg:
                kwarg = kwarg[key]
                if (k+1)==len(seq):
                    output_str = kwarg
        return output_str

    # replace by class
    def get_class(loader, node):
        seq = loader.construct_sequence(node)
        assert len(seq)==1
        loc_rdot = seq[0].rfind(".")
        module = safe_import_module(seq[0][:loc_rdot])
        return getattr(module, seq[0][loc_rdot+1:])

    # add custom custructors
    yaml.add_constructor('!join', join)  # join strings
    yaml.add_constructor('!usjoin', usjoin)  # join strings with underscores
    yaml.add_constructor('!pathjoin', pathjoin)  # join paths
    yaml.add_constructor('!kwarg', get_kwarg)  # join paths
    yaml.add_constructor('!class', get_class)  # join paths

    # load yaml
    with open(filepath, 'r') as ymlfile:
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

def str_in_list(lst,string):
    """Get indices of a string in a list

    Example:
        $  indices = str_in_list(lst,string)

    ...

    Arguments:
        lst (lst): list of strings
        string (str): string to search for

    Returns:
        list of integers
    """
    if isinstance(string,(list,np.ndarray)):
        return [str_in_list(lst,_string) for _string in string]
    indices = [i for i in range(len(lst)) if lst[i] == string]
    return indices

def listnp_combine(lst,method='concatenate',axis=0, allow_error=False):
    """Concatenate or stack a list of numpy along with error handling

    Example:
        $  nparray = listnp_combine(lst,method='concatenate',axis=0, allow_error=False)

    Arguments:
        lst (lst): list of np arrays
        method (str): 'concatenate' or 'stack'
        axis (int): axis to concat or stack over
        allow_error (bool): allow for error handling. If op does not succes, list is provided

    Returns:
        np.array OR list of np.array in case of error
    """
    def combine(lst):
        if isinstance(lst, list):
            lst = np.array(lst)
        if method == 'concatenate':
            npstack = np.concatenate(lst, axis=axis)
        elif method == 'stack':
            npstack = np.stack(lst, axis=axis)
        else:
            print('method for concatenating not supported in listnp_stack in utils.py.')
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

def listdictnp_combine(lst, method='concatenate',axis=0, keep_nested=False, allow_error=False):
    """Concatenate or stack a list of dictionaries contains numpys along with error handling

    Example:
        $  nparray = listdictnp_combine(lst, method='concatenate',axis=0, keep_nested=False, allow_error=False)

    Arguments:
        lst (lst): list of dicts containings np arrays
        method (str): 'concatenate' or 'stack'
        axis (int): axis to concat or stack over
        keep_nested (bool): keep nested structure of list or not
        allow_error (bool): allow for error handling. If op does not succes, list is provided

    Returns:
        np.array OR list of np.array in case of error
    """
    for k in range(len(lst)):
        assert lst[0].keys()==lst[k].keys(), 'Dict keys do not match in listdictnp_combine fct'
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
        output_dict[key] = listnp_combine(tmp, method=method, axis=axis, allow_error=allow_error)
    return output_dict

def unique_list(seq):
    """Get unique entries in a list

    Example:
        $  unique_seq = unique_list(seq)

    Arguments:
        seq (list)
    Returns:
        list of unique values
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def flatten_nested_lst(nested_lst):
    """Flatten a nested list

    Example:
        $  flattened_list = flatten_nested_lst(nested_lst)

    Arguments:
        nested_lst (list)
    Returns:
        flattened list
    """
    return [item for sublist in nested_lst for item in sublist]

def any2str(obj):
    """Convert anything to a string

    Example:
        $  str = any2str(obj)

    Arguments:
        obj: anything to convert to a str rep
    Returns:
        string
    """
    if isinstance(obj,str):
        return obj
    elif isinstance(obj,int) | isinstance(obj,np.ndarray)  | isinstance(obj,np.int64) | isinstance(obj,dict):
        return str(obj)
    elif obj is None:
        return 'None'
    else:
        print('Instance not supported in any2str()')
        sys.exit()

def filter_data(data,sel_vect,squeeze=False):
    """Filter any sequential data object based on indices

    Example:
        $  filtered_data = filter_data(data,sel_vect,squeeze=False)

    Arguments:
        data (obj): seq object to be filtered
        sel_vect: indices to filter
        squeeze: squeeze output when numpy
    Returns:
        filtered data
    """
    # inits
    from dabstract.dataset import abstract
    if isinstance(sel_vect, (np.int64,int)):
        sel_vect = np.array([sel_vect])
    # do different filtering depending on input
    if isinstance(data, list):
        out = [data[k2] for k2 in sel_vect]
        if squeeze:
            out = out[0]
    elif isinstance(data, np.ndarray):
        out = data[sel_vect,]
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
            out[key] = filter_data(data[key],sel_vect, squeeze=squeeze)
    elif isinstance(data, abstract):
        out = data
    elif hasattr(data,'__getitem__'):
        out = data[sel_vect]
    elif data is None:
        out = None
    else:
        print('Not supported data format in filter_data fct')
        sys.exit()
    return out

def safe_import_module(module_string):
    """Import module with error handling

    Example:
        $  module = safe_import_module(module_string)

    Arguments:
        module_string: module string to import
    Returns:
        imported module
    """
    try:
        return importlib.import_module(module_string)
    except ImportError:
        return object()

def combs_numpy(delays):
    """All possible combinations of numpy entries
    """
    return np.unique(np.array(list(itertools.product(*[delays[:,k] for k in range(delays.shape[1])]))),axis=0)

def combs_list(delays):
    """All possible combinations of list entries
    """
    return list(itertools.product(*delays))

def combs_size_numpy(values,size):
    """Size of all possible combinations of numpy entries
    """
    return np.array([comb for comb in itertools.combinations(values, size)])

def reformat_yaml(cfg):
    """Reformat yaml list to numpy if possible
    """
    if isinstance(cfg, list):
        tmp_cfg = cfg
        if any(isinstance(i, list) for i in tmp_cfg): # nested
            tmp_cfg = [val for i in tmp_cfg for val in i]
        if np.all([(isinstance(i, int) | isinstance(i, float)) for i in tmp_cfg]):
            cfg = np.array(cfg)
    return cfg

def reformat_yaml_iter(cfg):
    """Reformat yaml list to numpy if possible in iterative fashion
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            reformat_yaml_iter(v)
        else:
            cfg[k] = reformat_yaml(v)
    return cfg

def load_yaml(filepath):
    """Load yaml file
    """
    # load file
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    return cfg

def pprint_ext(str,dict,np_precision=2):
    """pprint with np precision specification and title
    """
    np.set_printoptions(precision=np_precision, suppress=True)
    print(str)
    pprint(dict)

def safe_len(var):
    """safely get length
    """
    try:
        return len(var)
    except TypeError:
        return 1


def stringlist2ind(strlist):
    """list to unique indices
    """
    subdb_ext = unique_list(strlist)
    group_np = np.zeros((len(strlist)))
    for k in range(len(strlist)):
        group_np[k] = subdb_ext.index(strlist[k])
    return group_np.astype(np.int)

def intersection(lst1, lst2):
    """List intersection
    """
    return [(k,value) for k,value in enumerate(lst1) if value in lst2]

def get_class(name,module_location,*args,**kwargs):
    """Load a class given the name, module location and it's args and kwargs
    """
    module = safe_import_module(module_location)
    assert hasattr(module, name), name + " is not a part of module " + module_location
    return getattr(module, name)(*args, **kwargs)

def get_fct(name,module_location):
    """Load a fct given the name, module location and it's args and kwargs
    """
    module = safe_import_module(module_location)
    assert hasattr(module, name), name + " is not a part of module " + module_location
    return getattr(module, name)