import numpy as np
import copy
import pathlib
import pickle
import os

def get_dir_info(path, extension='.wav', save_path='', save=False):
    # get dirs
    filepath = []
    for root, dirs, files in os.walk(path):
        filepath += [os.path.join(root, file) for file in files if extension in file]
    filename = [os.path.relpath(file, path) for file in filepath if extension in file]
    subdb = [os.path.split(file)[0] for file in filename if extension in file]
    # get additional info
    if not os.path.isfile(os.path.join(save_path, 'file_info.pickle')):
        info = [dict()] * len(filepath)
        if extension == '.wav':
            import soundfile as sf
            for k in range(len(filepath)):
                import soundfile as sf
                f = sf.SoundFile(filepath[k])
                info[k]['output_shape'] = np.array([len(f), f.channels])
                info[k]['fs'] = f.samplerate
                info[k]['time_step'] = 1 / f.samplerate
            if save:
                with open(pathlib.Path(save_path, 'file_info.pickle'), "wb") as fp: pickle.dump(info, fp)
    else:
        with open(os.path.join(save_path, 'file_info.pickle'), "rb") as fp:
            info = pickle.load(fp)
        assert len(info) == len(filepath), "info file not of same size as directory"
    return {'filepath': filepath, 'filename': filename, 'subdb': subdb, 'info': info}

# make sure info matches expected format
def clean_info(info, nr_examples, **kwargs):
    if isinstance(info,dict):
        info_out = [dict()] * nr_examples
        for k in range(nr_examples):
            info_out[k].update(info)
    elif info is None:
        info_out = [dict()] * nr_examples
    else:
        info_out = info
    if len(kwargs)>0:
        [(kwargs.pop(key, None) if kwargs[key] is None else 0) for key in copy.deepcopy(kwargs)] # remove none entries
        for k in range(nr_examples):
            info_out[k].update(**kwargs)
    return info_out