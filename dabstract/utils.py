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
    if isinstance(string,(list,np.ndarray)):
        return [str_in_list(lst,_string) for _string in string]
    indices = [i for i in range(len(lst)) if lst[i] == string]
    return indices

def listnp_combine(lst,method='concatenate',axis=0, allow_error=False):
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
    for k in range(len(lst)):
        if lst[0].keys()!=lst[k].keys():
            print('Dict keys do not match in listdictnp_stack fct')
            sys.error()
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

def dict_listappend(lst):
    # check if keys match in all lists
    for k in range(len(lst)):
        if lst[0].keys()!=lst[k].keys():
            print('Dict keys do not match in listdictnp_stack fct')
            sys.error()
    # get keys
    keys = lst[0].keys()
    output_dict = dict()
    for key in keys:
        output_dict[key] = [lst[k][key] for k in range(len(lst))]
    return output_dict

def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def listfilterdict(lst):
    # get keys
    keys = list()
    for k in range(len(lst)):
        keys.extend(list(lst[k].keys()))
    # get keys to keep
    unique_keys = list(set(keys))
    keep_key = list()
    for k in range(len(unique_keys)):
        if len(np.where([keys[j]==unique_keys[k] for j in range(len(keys))])[0])==len(lst):
            keep_key.append(unique_keys[k])
    # filter out
    lst_out = [None] * len(lst)
    for k in range(len(lst)):
        lst_out[k] = dict()
        for j in range(len(keep_key)):
            lst_out[k][keep_key[j]] = lst[k][keep_key[j]]
    return lst_out

def flatten_nested_lst(nested_lst):
    return [item for sublist in nested_lst for item in sublist]

def flatten_nested_lst_save(nested_lst):
    flat_list = []
    for sublist in nested_lst:
        if isinstance(sublist, list):
            for item in sublist:
                flat_list.append(item)
        else:
            flat_list.append(sublist)
    return flat_list

def ss_type(object, indices = None):
    if isinstance(object, list):
        object = ss_list(object, indices=indices)
    elif isinstance(object,np.ndarray):
        object = object[indices]
    else:
        print('Object not supported.')
        sys.exit()
    return object

def ss_list(lst, indices = None):
    return [lst[k] for k in indices]

def listdir_folder(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

def any2str(obj):
    if isinstance(obj,str):
        return obj
    elif isinstance(obj,int) | isinstance(obj,np.ndarray)  | isinstance(obj,np.int64):
        return str(obj)
    elif obj is None:
        return 'None'
    else:
        print('Instance not supported in any2str()')

def linear_interpolation(new_time, orig_time, orig_data, missing_comp=True):
    new_data = np.zeros(len(new_time))
    for k in range(len(new_time)):
        diff = orig_time - new_time[k]
        diffs = np.abs(diff)
        lower_ids = np.where(diff <= 0)[0]
        higher_ids = np.where(diff >= 0)[0]
        if (missing_comp != 0) & (len(higher_ids) == 0):
            higher_ids = lower_ids
        if len(higher_ids) != 0:
            ids1 = np.argmin(diffs[lower_ids])
            ids2 = np.argmin(diffs[higher_ids])
            ids1 = lower_ids[ids1]
            ids2 = higher_ids[ids2]
            if np.sum(diffs[[ids1, ids2]]) != 0:
                new_value = orig_data[ids1] * diffs[ids2] / np.sum([diffs[[ids1, ids2]]]) + orig_data[ids2] * diffs[ids1] / np.sum([diffs[[ids1, ids2]]])
            else:
                new_value = orig_data[ids1]
        else:
            new_value = np.nan
        new_data[k] = new_value

    return new_data

def wrap_time(timelist):
    wrapped = np.empty(len(timelist))
    for k2 in range(len(timelist)):
        dt_object = datetime.fromtimestamp(timelist[k2])
        wrapped[k2] = dt_object.weekday() + (dt_object.hour * 60 * 60 + dt_object.minute * 60 + dt_object.second) / (24 * 60 * 60)
    return wrapped

def filter_data(data,sel_vect,squeeze=False):
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
    try:
        return importlib.import_module(module_string)
    except ImportError:
        return object()

def any2str(obj):
    if isinstance(obj,str):
        return obj
    elif isinstance(obj,int) | isinstance(obj,np.ndarray)  | isinstance(obj,np.int64) | isinstance(obj,dict):
        return str(obj)
    elif obj is None:
        return 'None'
    else:
        print('Instance not supported in any2str()')
        sys.exit()

def combs_numpy(delays):
    return np.unique(np.array(list(itertools.product(*[delays[:,k] for k in range(delays.shape[1])]))),axis=0)

def combs_list(delays):
    return list(itertools.product(*delays))

def combs_size_numpy(values,size):
    return np.array([comb for comb in itertools.combinations(values, size)])

def reformat_yaml(cfg):
    if isinstance(cfg, list):
        tmp_cfg = cfg
        if any(isinstance(i, list) for i in tmp_cfg): # nested
            tmp_cfg = [val for i in tmp_cfg for val in i]
        if np.all([(isinstance(i, int) | isinstance(i, float)) for i in tmp_cfg]):
            cfg = np.array(cfg)
    return cfg

def reformat_yaml_iter(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            reformat_yaml_iter(v)
        else:
            cfg[k] = reformat_yaml(v)
    return cfg

def load_yaml(filepath):
    # load file
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    return cfg

class circularlist(object):
    def __init__(self, size):
        """Initialization"""
        self.index = 0
        self.size = size
        self._data = []

    def append(self, value):
        """Append an element"""
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def __getitem__(self, key):
        """Get element by index, relative to the current index"""
        if len(self._data) == self.size:
            return(self._data[(key + self.index) % self.size])
        else:
            return(self._data[key])

    def __repr__(self):
        """Return string representation"""
        return self._data.__repr__() + ' (' + str(len(self._data))+' items)'

def pprint_ext(str,dict,np_precision=2):
    np.set_printoptions(precision=np_precision, suppress=True)
    print(str)
    pprint(dict)

def safe_len(var):
    try:
        return len(var)
    except TypeError:
        return 1

def unlink_wrap(dat, lims=[-np.pi, np.pi], thresh = 0.95):
    """
    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an generator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))

class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

#Data/meta: get group
def group_to_ind(group_str):
    subdb_ext = unique_list(group_str)
    group_np = np.zeros((len(group_str)))
    for k in range(len(group_str)):
        group_np[k] = subdb_ext.index(group_str[k])
    return group_np

def intersection(lst1, lst2):
    return [(k,value) for k,value in enumerate(lst1) if value in lst2]

def stringlist2ind(strlist):
    subdb_ext = unique_list(strlist)
    group_np = np.zeros((len(strlist)))
    for k in range(len(strlist)):
        group_np[k] = subdb_ext.index(strlist[k])
    return group_np.astype(np.int)