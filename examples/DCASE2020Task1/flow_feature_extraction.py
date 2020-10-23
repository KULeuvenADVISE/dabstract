import os
import sys
import argparse

from dabstract.dataset.helpers import dataset_from_config
from dabstract.utils import load_yaml_config
from dabstract.dataprocessor import ProcessingChain
os.environ["dabstract_CUSTOM_DIR"] = "custom"

cg = {'dataset': 'DCASE2020Task1A',
      'key': 'audio',
      'features': 'DCASE2020Task1'}
# general
co = {'dir_conf': 'local',
      'config_dir': os.path.join('configs'),
      'overwrite': False,
      'verbose': True,
      'multi_processing': False,
      'workers': 5,
      'buffer_len': 5}

def flow_feature_extraction():
    # ---------- params ---------- #
    # -- cmdline parser
    parser = argparse.ArgumentParser(description='JEF Feature extraction')
    # experiment params
    exp = parser.add_argument_group('experiment')
    exp.add_argument('--dataset', type=str, default=cg['dataset'], metavar='D',
                        help='dataset configuration available in config/db (default: %s)' % cg['dataset'])
    exp.add_argument('--key', type=str, default=cg['key'], metavar='F',
                        help='item of the dataset to apply FE on (default: %s)' % cg['key'])
    exp.add_argument('--features', type=str, default=cg['features'], metavar='F',
                        help='feature configuration available in config/dp (default: %s)' % cg['features'])
    # flow params
    flow = parser.add_argument_group('flow')
    flow.add_argument('--dir_conf', type=str, default=co['dir_conf'], metavar='DC',
                        help='dir configuration available in config/dirs (default: %s)' % co['dir_conf'])
    flow.add_argument('--config_dir', type=str, default=co['config_dir'], metavar='CD',
                        help='config directory path (default: %s)' % co['config_dir'])
    flow.add_argument('--verbose', type=int, default=co['verbose'], metavar='VE',
                        help='verbose (bool) (default: %d)' % co['verbose'])
    flow.add_argument('--overwrite', type=int, default=co['overwrite'], metavar='OW',
                        help='overwrite features or not (bool) (default: %d)' % co['overwrite'])
    flow.add_argument('--workers', type=int, default=co['workers'], metavar='WO',
                        help='amount of workers for parallel processing (0: single core, >0: multiprocessing) (default: %d)' % co['workers'])
    flow.add_argument('--buffer_len', type=int, default=co['buffer_len'], metavar='BL',
                        help='length of the worker buffer (default: %d)' % co['buffer_len'])
    args = parser.parse_args()

    # -- get dirs
    dirs = load_yaml_config(filename=args.dir_conf, path=os.path.join(args.config_dir, 'dirs'), walk=True)
    # -- get_dataset
    data = load_yaml_config(filename=args.dataset, path=os.path.join(args.config_dir, 'db'), walk=True,
                            post_process=dataset_from_config, **dirs)
    # -- get processing chain
    # get fe
    fe_dp = load_yaml_config(filename=args.features, path=os.path.join(args.config_dir, 'dp'), walk=True,
                             post_process=processing_chain)
    # fit if needed
    fe_dp.fit(data[cg['key']]['data'])
    # -- get features
    data.prepare_feat(args.key,
                      args.features,
                      fe_dp,
                      overwrite=args.overwrite,
                      verbose=args.verbose,
                      new_key='feat',
                      workers=args.workers,
                      buffer_len=args.buffer_len)

if __name__ == "__main__":
    try:
        sys.exit(flow_feature_extraction())

    except (ValueError, IOError) as e:
        sys.exit(e)



