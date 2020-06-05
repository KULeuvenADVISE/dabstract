# ---------- imports ---------- #
# other
import getopt

# custom
from dabstract.learner import *
from dabstract.evaluation import *
from dabstract.dataset import *
from dabstract.dataprocessor.processing_chain import *
from dabstract.utils import pprint_ext, filter_data, load_yaml_config
from dabstract.learner.keras.utils import init_keras

os.environ["dabstract_CUSTOM_DIR"] = "dabstract_custom"

def flow_keras(cg_in=dict(),co_in=dict()):
    # ---------- main params ---------- #
    cg = {'dataset': 'DCASE2020Task1B',
          'features': 'DCASE2020Task1B_avg',
          'proc_chain_data': 'standard',
          'proc_chain_meta': 'none',
          'model': 'NN_3l_64n_Lin',
          'model_opt': 'DCASE2020Task1B'}

    # other params
    co = {'dir_conf': 'local',
          'load_memory': False,
          'multi_processing': False,
          'workers': 5,
          'cuda_devices': "-1",
          'train': True,
          'test': True,
          'overwrite_train': True,
          'overwrite_test': True,
          'keras_verbose': 1}

    # ---------- inits ---------- #
    # --- parameter overloading
    cg.update(cg_in), co.update(co_in)
    # cg, co = parameter_overloading(sys.argv, cg, co) # ToDo: DEPRECATED / change by new argparse
    # -- get dirs
    dirs = load_yaml_config(filename=co['dir_conf'],
                            dir=os.path.join('configs', 'dirs'),
                            walk=True, **cg)
    # -- get_dataset
    data = load_yaml_config(filename=cg['dataset'], dir=os.path.join('configs', 'db'), walk=True,
                            post_process=dataset_from_config, **dirs)
    # -- get processing chain
    fe_dp = load_yaml_config(filename=cg['features'],
                             dir=os.path.join('configs', 'dp'),
                             walk=True,
                             post_process=processing_chain)
    # -- get features
    data.prepare_feat('data',
                      cg['features'],
                      fe_dp,
                      dirs['features'],
                      new_key='features')
    # -- init multi processing
    data.initialize(load_memory=co['load_memory'], multi_processing=co['multi_processing'], workers=co['workers'])

    # -- other
    xval = data.set_xval(save_dir=dirs['xval'], overwrite=co['overwrite_train'])
    info = data.summary()
    pprint(info)
    # --- load processing chain confs
    processor = {'data': load_yaml_config(filename=cg['proc_chain_data'], dir=os.path.join('configs', 'dp'), walk=True,
                                          post_process=processing_chain),
                 'meta': load_yaml_config(filename=cg['proc_chain_meta'], dir=os.path.join('configs', 'dp'), walk=True,
                                          post_process=processing_chain)}
    # --- init keras
    K = init_keras(co['cuda_devices'])

    # ---------- experiment loop ---------- #
    for fold in range(xval['folds']):
        K.clear_session()
        # --- Training
        savefile_model = os.path.join(dirs['results'], "model")
        if (co['train'] & (not os.path.isfile(savefile_model + "_fold" + str(fold) + "_epochBEST.hdf5"))) | co[
            'overwrite_train']:
            # --- create dir
            os.makedirs(dirs['results'], exist_ok=True)

            # --- get opt
            cm = load_yaml_config(filename=cg['model_opt'],
                                  dir=os.path.join('configs', 'learner', 'keras', 'opts'),
                                  walk=True,
                                  post_process=yaml_config_keras_pp,
                                  filepath_mcpcb=savefile_model + "_fold" + str(fold) + "_epochBEST.hdf5")

            # --- init processing chain
            savefile_processor = os.path.join(dirs['results'], "processor_fold" + str(fold))  # save directory
            if (not os.path.isfile(savefile_processor + ".pickle")) or co['overwrite_train']:
                processor['data'].fit(data=data[cm['input']], sel_ind=xval['train'][fold])
                processor['meta'].fit(data=data[cm['output_target']], sel_ind=xval['train'][fold])
                with open(savefile_processor + ".pickle", 'wb') as f:
                    pickle.dump(processor, f)  # save
            else:
                with open(savefile_processor + ".pickle", "rb") as f:
                    processor = pickle.load(f)  # load

            # --- init data generator
            data_generator_train = DataGenerator(data=data[cm['input']],
                                                 target=data[cm['output_target']],
                                                 group=data['group'],
                                                 sel_ind=xval['train'][fold],
                                                 processor_meta=processor['meta'],
                                                 processor_data=processor['data'],
                                                 **cm['keras'])
            data_generator_val = DataGenerator(data=data[cm['input']],
                                               target=data[cm['output_target']],
                                               group=data['group'],
                                               sel_ind=xval['val'][fold],
                                               processor_meta=processor['meta'],
                                               processor_data=processor['data'],
                                               **cm['keras'])
            # --- Init model
            # get
            if (not os.path.isfile(savefile_model + '.json')) or co['overwrite_train']:  # remove later version
                model, custom_objects = get_model(model_name=cg['model'],
                                                  dir=os.path.join('configs', 'learner', 'keras', 'networks'),
                                                  input_shape=data_generator_train.get_output_shape(),
                                                  walk=True)
                with open(os.path.join(dirs['results'], "custom_objects.pickle"), 'wb') as f:
                    pickle.dump(custom_objects, f)
                with open(savefile_model + '.json', 'w') as json_file:
                    json_file.write(model.to_json())
            else:
                with open(savefile_model + '.json') as json_file:
                    json_config = json_file.read()
                with open(os.path.join(dirs['results'], "custom_objects.pickle"), "rb") as f:
                    custom_objects = pickle.load(f)  # load
                model = kr.models.model_from_json(json_config, custom_objects=custom_objects)
            model.compile(**cm['keras']['compile'])
            model.summary()

            # -- fit and save
            history = model.fit(data_generator_train,
                                validation_data=data_generator_val,
                                callbacks=cm['keras']['callbacks'],
                                epochs=cm['keras']['epochs'],
                                verbose=co['keras_verbose'])
            with open(savefile_model + "_fold" + str(fold) + ".pickle", 'wb') as f:
                pickle.dump(history.history, f)

            K.clear_session()

        # ---------- Testing ---------- #
        savefile_results = os.path.join(dirs['results'], "testresults_fold" + str(fold) + "_epochBEST")
        if (co['test'] & (not os.path.isfile(savefile_results + ".pickle"))) | co['overwrite_test']:
            # --- get opt
            cm = load_yaml_config(filename=cg['model_opt'],
                                  dir=os.path.join('configs', 'learner', 'keras', 'opts'),
                                  walk=True,
                                  post_process=yaml_config_keras_pp,
                                  filepath_mcpcb=savefile_model + "_fold" + str(fold) + "_epochBEST.hdf5")

            # # --- init processing chain
            savefile_processor = os.path.join(dirs['results'], "processor_fold" + str(fold))  # save directory
            with open(savefile_processor + ".pickle", "rb") as f:
                processor = pickle.load(f)  # load

            # --- init data generator
            data_generator = DataGenerator(data=data[cm['input']],
                                           target=data[cm['output_target']],
                                           sel_ind=xval['test'][fold],
                                           processor_data=processor['data'],
                                           processor_meta=processor['meta'],
                                           stage='test',
                                           **cm['keras'])
            # get model
            loadfile_model = os.path.join(dirs['results'], "model")
            with open(loadfile_model + '.json') as json_file:
                json_config = json_file.read()
            with open(os.path.join(dirs['results'], "custom_objects.pickle"), "rb") as f:
                custom_objects = pickle.load(f)  # load
            model = kr.models.model_from_json(json_config, custom_objects=custom_objects)
            model.load_weights(loadfile_model + "_fold" + str(fold) + "_epochBEST.hdf5")
            # predict
            estmeta = model.predict(data_generator)
            estmeta = processor['meta'].inv_process(estmeta)
            # get reference
            truemeta = dict()
            for input in np.unique([metric['input'] for metric in cm['test_metrics']]):
                truemeta[input] = data[input][xval['test'][fold]]
            # get metric
            metrics = get_metrics(truemeta, estmeta, metrics=cm['test_metrics'])
            pprint_ext("Result on test set... ", metrics)
            # save
            with open(savefile_results + ".pickle", 'wb') as f:
                pickle.dump((estmeta, truemeta, metrics), f)

            K.clear_session()


def parameter_overloading(arglist, cg, co):
    unixOptions = "dfxpkmouw"
    gnuOptions = ["dataset=", "features=", "xseg=", "proc_chain_data=", "proc_chain_meta=", "model=", "model_opt=",
                  "multi_processing=", "workers="]
    try:
        arguments, values = getopt.getopt(arglist[1:], unixOptions, gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    # evaluate given options
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-d", "--dataset"):
            cg['dataset'] = currentValue
        elif currentArgument in ("-f", "--features"):
            cg['features'] = currentValue
        elif currentArgument in ("-x", "--xseg"):
            cg['xseg'] = currentValue
        elif currentArgument in ("-p", "--proc_chain_data"):
            cg['proc_chain_data'] = currentValue
        elif currentArgument in ("-k", "--proc_chain_meta"):
            cg['proc_chain_meta'] = currentValue
        elif currentArgument in ("-m", "--model"):
            cg['model'] = currentValue
        elif currentArgument in ("-o", "--model_opt"):
            cg['model_opt'] = currentValue
        elif currentArgument in ("-u", "--multi_processing"):
            co['multi_processing'] = currentValue
        elif currentArgument in ("-w", "--workers"):
            co['workers'] = currentValue
        elif currentArgument in ("-w", "--workers"):
            co['gpu_devices'] = currentValue

        print(currentValue)
    return cg, co


if __name__ == "__main__":
    try:
        sys.exit(flow_keras())

    except (ValueError, IOError) as e:
        sys.exit(e)

