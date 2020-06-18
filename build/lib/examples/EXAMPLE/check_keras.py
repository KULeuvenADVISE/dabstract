# ---------- imports ---------- #
# other
from tqdm import tqdm
import pickle
from pprint import pprint
from pylab import show, imshow, plot, xlabel, ylabel, legend, figure, yscale, xscale, xlim, ylim, xticks, title, hist, histogram

# custom
from dabstract.learner import *
from dabstract.evaluation import *
from dabstract.dataset import dataset_wrapper
from dabstract.dataprocessor import Aggregation, Framing, processing_chain
from dabstract.utils import listdictnp_combine, load_yaml_config, unlink_wrap
from dabstract.learner.keras.utils import init_keras
os.environ["dabstract_CUSTOM_DIR"] = "dabstract_custom"
K = init_keras(devices="1")

def main(argv):
    # ---------- main params ---------- #
    # define params
    cg = {'dataset': 'EXAMPLE_anomaly_subsample',
          'features': 'feature_mel',
          'proc_chain_data': 'avg_standard',
          'proc_chain_meta': 'none',
          'model': 'NN_3l_8n_MAE',
          'model_opt': 'EXAMPLE_opt_anomaly'}

    # cg = {'dataset': 'EXAMPLE_classification',
    #       'features': 'feature_mel',
    #       'proc_chain_data': 'avg_minmax',
    #       'proc_chain_meta': 'minmax',
    #       'model': 'NN_3l_64n_Lin',
    #       'model_opt': 'EXAMPLE_opt_classification'}

    # -- other params
    co = {  'dir_conf': 'local',
            'show_model': True,
            'history': True,
            'results': True,
            'output': True}

    # -- get dirs
    dirs = load_yaml_config(filename=co['dir_conf'],
                            dir=os.path.join('configs', 'dirs'),
                            walk=True, **cg)
    # -- get_dataset
    db = load_yaml_config(filename=cg['dataset'],
                          dir=os.path.join('configs', 'db'),
                          walk=True,
                          post_process=dataset_wrapper, **dirs)
    # -- get processing chain
    fe_dp = load_yaml_config(filename=cg['features'],
                             dir=os.path.join('configs', 'dp'),
                             walk=True,
                             post_process=processing_chain)
    # -- get features
    db.prepare_feat(cg['features'], fe_dp)
    # -- get relevant info
    meta = db.get_meta()
    xval = db.get_xval(save_dir=dirs['xval'])
    info = db.get_db_info()
    folds = xval['folds']

    # ---------- model check loop ---------- #
    if co['show_model']:
        # --- get model
        loadfile_model = os.path.join(dirs['results'], "model")
        with open(loadfile_model + '.json') as json_file:
            json_config = json_file.read()
        model = kr.models.model_from_json(json_config)
        with open(os.path.join(dirs['results'], "custom_objects.pickle"), "rb") as f: custom_objects = pickle.load(
            f)  # load
        model = kr.models.model_from_json(json_config, custom_objects=custom_objects)
        model.summary()

    if co['history']:
        for fold in range(folds):
            print('History plot - Fold ' + str(fold))
            # get
            savefile_model = os.path.join(dirs['results'], "model_fold" + str(fold))
            with open(savefile_model + ".pickle", 'rb') as f:
                history = pickle.load(f)
            # plot
            to_plot = ('loss', 'val_loss')
            for key in to_plot:
                plot(history[key])
                print(key + ': ' + str(np.min(history[key])))
            legend(to_plot)
            yscale('log')
            title("Loss - fold " + str(fold))
            xlabel('Epochs')
            show()

    # ---------- Check results ---------- #
    if co['results']:
        metrics_all = [None] * folds
        for fold in range(folds):
            savefile_results = os.path.join(dirs['results'], "testresults_fold" + str(fold) + "_epochBEST")
            with open(savefile_results + ".pickle", 'rb') as f: metrics_all[fold] = pickle.load(f)[2]
        metrics_all_stack = listdictnp_combine(metrics_all, method='stack', keep_nested=True)
        metrics_stats = dict()
        aggregate = Aggregation(methods=['mean', 'std'], axis=0, combine=True)
        for key in metrics_all_stack:
            metrics_stats[key + '_mustd'] = aggregate.process(metrics_all_stack[key])
        # print
        print('Results (all)...')
        pprint(metrics_all)
        print('Results (mu/std)...')
        pprint(metrics_stats)

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))

    except (ValueError, IOError) as e:
        sys.exit(e)