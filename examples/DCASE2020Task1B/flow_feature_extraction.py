import os
import sys
import getopt

from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config
from dabstract.dataprocessor import processing_chain
os.environ["dabstract_CUSTOM_DIR"] = "dabstract_custom"

def flow_feature_extraction(cg_in=dict(),co_in=dict()):
    # -- General params
    cg = {'dataset': 'DCASE2020Task1A',
          'key': 'audio',
          'features': 'DCASE2020Task1'}
    # general
    co = {'dir_conf': 'local_server',
          'overwrite': False,
          'verbose': True,
          'multi_processing': True,
          'workers': 5,
          'buffer_len': 5}
    # --- parameter overloading
    cg.update(cg_in), co.update(co_in)
    # -- get dirs
    dirs = load_yaml_config(filename=co['dir_conf'], dir=os.path.join('configs', 'dirs'), walk=True)
    # -- get_dataset
    data = load_yaml_config(filename=cg['dataset'], dir=os.path.join('configs', 'db'), walk=True,
                            post_process=dataset_from_config, **dirs)
    # -- get processing chain
    fe_dp = load_yaml_config(filename=cg['features'], dir=os.path.join('configs', 'dp'), walk=True,
                             post_process=processing_chain)
    # -- get features
    data.prepare_feat(cg['key'],
                      cg['features'],
                      fe_dp,
                      dirs['features'],
                      overwrite=co['overwrite'],
                      verbose=co['verbose'],
                      new_key='feat',
                      multi_processing=co['multi_processing'],
                      workers=co['workers'],
                      buffer_len=co['buffer_len'])

def parameter_overloading(arglist, cg, co):
    unixOptions = "df"
    gnuOptions = ["dataset=", "features="]
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

        print(currentValue)
    return cg, co


if __name__ == "__main__":
    try:
        sys.exit(flow_feature_extraction())

    except (ValueError, IOError) as e:
        sys.exit(e)


