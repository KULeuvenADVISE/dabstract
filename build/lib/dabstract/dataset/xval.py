import numpy as np
import sklearn.model_selection as modsel

from dabstract.utils import listnp_combine, filter_data, stringlist2ind

# Group random (random KFolds with groups)
def group_random_kfold(folds=4,val_frac=1/3, **kwargs):
    def get(data):
        # inits
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        group = data['group'][:]
        ugroup, ugroup_index = np.unique(group, return_index=True)
        xval_folds = random_kfold(folds=folds, val_frac=val_frac)(np.unique(group))
        for key in ('train', 'val', 'test'):
            for f in range(folds):
                xval_folds[key][f] = np.concatenate([np.where(group == k)[0] for k in xval_folds[key][f]])
        return xval_folds
    return get

# Time-wise KFold
def sequential_kfold(folds=4,val_frac=1/3, **kwargs):
    def get(data):
        # inits
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        groups = [None] * folds
        group = stringlist2ind(data['group'][:])
        for k in np.unique(group):
            tmp_split = np.array_split(np.where(group == k)[0], folds)
            for f in range(folds):
                if k == 0:
                    groups[f] = np.empty(0)
                groups[f] = np.append(groups[f], tmp_split[f]).astype(int)
        test_index = groups # test index
        for f in range(folds):
            sel = np.setdiff1d(np.arange(folds), f)
            train_index[f] = listnp_combine(filter_data(groups[f], sel))
            val_inds = np.random.choice(range(len(train_index[f])), int(np.ceil(len(train_index[f]) * val_frac)), replace=False)
            val_index[f] = train_index[f][val_inds]
            train_index[f] = train_index[f][list(set(range(len(train_index[f]))) - set(val_inds))]
        return {'train': train_index, 'val': val_index, 'test': test_index}
    return get

# Stratified KFold
def stratified_kfold(folds=4, val_frac=1 / 3, **kwargs):
    def get(data):
        skf = modsel.StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        for k, (train_index_tmp, test_index_tmp) in enumerate(skf.split(np.arange(len(data)), np.arange(len(data)))):
            train_index[k], test_index[k] = train_index_tmp, test_index_tmp
            val_inds = np.random.choice(range(len(train_index[k])), int(np.ceil(len(train_index[k]) * val_frac)), replace=False)
            val_index[k] = train_index[k][val_inds]
            train_index[k] = train_index[k][list(set(range(len(train_index[k]))) - set(val_inds))]
        return {'train': train_index, 'val': val_index, 'test': test_index}
    return get

# random KFold
def random_kfold(folds=4, val_frac=1 / 3, **kwargs):
    def get(data):
        skf = modsel.KFold(n_splits=folds, shuffle=True, random_state=0)
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        for k, (train_index_tmp, test_index_tmp) in enumerate(skf.split(np.arange(len(data)))):
            train_index[k], test_index[k] = train_index_tmp, test_index_tmp
            val_inds = np.random.choice(range(len(train_index[k])), int(np.ceil(len(train_index[k]) * val_frac)), replace=False)
            val_index[k] = train_index[k][val_inds]
            train_index[k] = train_index[k][list(set(range(len(train_index[k]))) - set(val_inds))]
        return {'train': train_index, 'val': val_index, 'test': test_index}
    return get

