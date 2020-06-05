import numpy as np

def random_subsample(ratio=1, **kwargs):
    def func(data):
        indexes = np.arange(len(data))
        if ratio < 1:
            if isinstance(indexes, np.ndarray):
                indexes = np.random.choice(indexes, int(np.ceil(len(indexes) * ratio)), replace=False)
            elif isinstance(indexes, list):
                for k in range(len(indexes)):
                    indexes[k] = np.random.choice(indexes[k], int(np.ceil(len(indexes[k]) * ratio)), replace=False)
        return indexes
    return func

