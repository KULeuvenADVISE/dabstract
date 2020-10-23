import numpy as np
import sklearn.model_selection as modsel

from dabstract.utils import listnp_combine, stringlist2ind

from typing import Union, Any, List, Optional, cast, Type, TypeVar, Callable, Dict, Iterable, Generator, Tuple

tvXvalFunction = TypeVar('cross_val_fct')



def group_random_kfold(folds: int = 4,
                       val_frac: float = 1 / 3,
                       group_key: str = 'group', **kwargs) -> tvXvalFunction:
    """ Crossvalidation fct: group randomk fold
    """
    def get(data):
        # inits
        group = [k for k in data[group_key]]
        xval_folds = random_kfold(folds=folds, val_frac=val_frac)(np.unique(group))
        for key in ('train', 'val', 'test'):
            for f in range(folds):
                xval_folds[key][f] = np.concatenate([np.where(group == k)[0] for k in xval_folds[key][f]])
        return xval_folds

    return get


def sequential_kfold(folds: int = 4,
                     val_frac: float = 1 / 3,
                     group_key: str = 'group', **kwargs) -> tvXvalFunction:
    """ Crossvalidation fct: sequential k fold
    """
    def get(data):
        # inits
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        groups = [None] * folds
        group = stringlist2ind([k for k in data[group_key]])
        for k in np.unique(group):
            tmp_split = np.array_split(np.where(group == k)[0], folds)
            for f in range(folds):
                if k == 0:
                    groups[f] = np.empty(0)
                groups[f] = np.append(groups[f], tmp_split[f]).astype(int)
        for f in range(folds):
            sel = np.setdiff1d(np.arange(folds), f)
            train_index[f] = listnp_combine([groups[k] for k in sel])
            val_inds = np.random.choice(range(len(train_index[f])), int(np.ceil(len(train_index[f]) * val_frac)),
                                        replace=False)
            val_index[f] = train_index[f][val_inds]
            train_index[f] = train_index[f][list(set(range(len(train_index[f]))) - set(val_inds))]
        return {'train': train_index, 'val': val_index, 'test': groups}

    return get


def stratified_kfold(folds: int = 4,
                     val_frac: float = 1 / 3,
                     label: str = None, **kwargs) -> tvXvalFunction:
    """ Crossvalidation fct: Stratified k fold
    """
    assert label is not None, "please provide a label to stratify on"

    def get(data):
        skf = modsel.StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        for k, (train_index_tmp, test_index_tmp) in enumerate(skf.split(np.arange(len(data)), data[label])):
            train_index[k], test_index[k] = train_index_tmp, test_index_tmp
            val_inds = np.random.choice(range(len(train_index[k])), int(np.ceil(len(train_index[k]) * val_frac)),
                                        replace=False)
            val_index[k] = train_index[k][val_inds]
            train_index[k] = train_index[k][list(set(range(len(train_index[k]))) - set(val_inds))]
        return {'train': train_index, 'val': val_index, 'test': test_index}

    return get


def random_kfold(folds: int = 4,
                 val_frac: float = 1 / 3, **kwargs) -> tvXvalFunction:
    """ Crossvalidation fct: random k fold
    """
    def get(data):
        skf = modsel.KFold(n_splits=folds, shuffle=True, random_state=0)
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        for k, (train_index_tmp, test_index_tmp) in enumerate(skf.split(np.arange(len(data)))):
            train_index[k], test_index[k] = train_index_tmp, test_index_tmp
            val_inds = np.random.choice(range(len(train_index[k])), int(np.ceil(len(train_index[k]) * val_frac)),
                                        replace=False)
            val_index[k] = train_index[k][val_inds]
            train_index[k] = train_index[k][list(set(range(len(train_index[k]))) - set(val_inds))]
        return {'train': train_index, 'val': val_index, 'test': test_index}

    return get


def random_split(val_frac: float = 1 / 3,
                 test_frac: float = 1 / 3, **kwargs) -> tvXvalFunction:
    """ Crossvalidation fct: random split
    """
    def get(data):
        train_frac = 1 - val_frac - test_frac
        assert train_frac > 0
        rem_inds = np.arange(len(data))
        val_inds = np.random.choice(rem_inds, int(np.ceil(len(data) * val_frac)), replace=False)
        rem_inds = list(set(rem_inds) - set(val_inds))
        test_inds = np.random.choice(rem_inds, int(np.ceil(len(data) * test_frac)), replace=False)
        train_inds = list(set(rem_inds) - set(test_inds))
        return {'train': [train_inds], 'val': [val_inds], 'test': [test_inds]}

    return get
