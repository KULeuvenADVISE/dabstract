from .flow_keras import flow_keras

cg = {'dataset': 'EXAMPLE_classification',
      'features': 'feature_mel_avg',
      'proc_chain_data': 'none',
      'proc_chain_meta': 'none',
      'model': 'NN_3l_64n_Lin',
      'model_opt': 'EXAMPLE_opt_classification'}
flow_keras(cg_in=cg)

