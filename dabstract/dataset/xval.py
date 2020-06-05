import numpy as np
import sklearn.model_selection as modsel

from dabstract.utils import listnp_combine, filter_data, stringlist2ind

# Time-wise fold distribution and random train/validation split
def group_random(folds=4,val_frac=1/3, **kwargs):
    def get(data):
        # inits
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        group = data['group'][:]
        ugroups = np.array_split(np.unique(group), folds)
        for f in range(folds):
            test_index[f] = listnp_combine([np.where(group == k)[0] for k in ugroups[f]])
            train_index[f] = np.setdiff1d(np.arange(len(group)), test_index[f])
            val_index[f] = np.random.choice(train_index[f], int(np.ceil(len(train_index[f]) * val_frac)), replace=False)
            train_index[f] = np.setdiff1d(train_index[f], val_index[f])
        return {'train': train_index, 'val': val_index, 'test': test_index}
    return get

# Time-wise fold distribution and random train/validation split
def sequential_random(folds=4,val_frac=1/3, **kwargs):
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

# Random fold distribution and random train/validation split
def random(folds=4, val_frac=1 / 3, **kwargs):
    def get(data):
        skf = modsel.StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        train_index, val_index, test_index = [None] * folds, [None] * folds, [None] * folds
        group = stringlist2ind(data['group'][:])
        for k, (train_index_tmp, test_index_tmp) in enumerate(skf.split(group, group)):
            train_index[k], test_index[k] = train_index_tmp, test_index_tmp
            val_inds = np.random.choice(range(len(train_index[k])), int(np.ceil(len(train_index[k]) * val_frac)),
                                        replace=False)
            val_index[k] = train_index[k][val_inds]
            train_index[k] = train_index[k][list(set(range(len(train_index[k]))) - set(val_inds))]
        return {'train': train_index, 'val': val_index, 'test': test_index}
    return get

