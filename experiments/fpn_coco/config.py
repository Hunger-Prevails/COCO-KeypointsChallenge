

from examples.config_det_resnet18_coco import config as cfg
config = cfg


is_debug = False
if is_debug:
    config.exp = 'exp_debug'
    config.TRAIN.gpus = '0,1,2,3'
    config.TEST.gpus = '0,1,2,3'
    config.TRAIN.image_batch_size = 2
    config.TRAIN.use_prefetchiter = False
    config.TRAIN.filter_strategy.max_num_images = 50
    config.TEST.filter_strategy.max_num_images = 10
    config.TRAIN.solver.frequent = 1
    config.TRAIN.solver.lr_step = '2'
    config.TRAIN.solver.num_epoch = 3
    config.TRAIN.solver.absorb_bn_lr_step = '1'
    config.TRAIN.solver.absorb_bn_num_epoch = 2

    import os
    config.TRAIN.model_prefix = os.path.join(config.hdfs_remote_home_dir, config.root_output_dir, config.exp, 'models') + '/det'
