import os
from easydict import EasyDict as edict
from common.config.config_common import add_home_dir, add_train_params, modify_lr_params, \
    add_test_params, add_network_params, add_dataset_params
from common.config.config_fpn import add_fpn_params
from common.config.config_frcnn import add_frcnn_params

config = edict()
config.person_name = 'xinze.chen'
config.root_output_dir = 'experiments_fpn'
config = add_home_dir(config)

# train params
config = add_train_params(config)
config.TRAIN.gpus = '0,1,2,3'
config.TRAIN.image_batch_size = 2
config.TRAIN.bn_use_global_stats = True
config.TRAIN.do_eval_during_training = True
config.TRAIN.solver.optimizer = 'sgd'
config.TRAIN.solver.lr_step = '14,17'
config.TRAIN.solver.num_epoch = 18
config.TRAIN.filter_strategy.remove_empty_boxes = True
# config.TRAIN.aug_strategy.scales = [(416, 1024), (448, 1024), (480, 1024), (512, 1024), (544, 1024), (576, 1024), (608, 1024)]
# config.TRAIN.aug_strategy.scales = [(608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)]
# config.TRAIN.aug_strategy.scales = [(512, 1173), (544, 1173), (576, 1173), (608, 1173), (640, 1173), (672, 1173), (704, 1173)]
config.TRAIN.aug_strategy.scales = [(600, 1000)]
config.TRAIN.aug_strategy.flip = True
config = modify_lr_params(config, use_warmup=True)

# test params
config = add_test_params(config)
config.TEST.gpus = '0,1,2,3'
config.TEST.aug_strategy.scales = [(600, 1000)]
config.TEST.load_sym_from_file = False

# dataset params
dataset_type = 'det'
config = add_dataset_params(config=config,
                            train_dataset_list=['coco#train2017'],
                            test_dataset_list=['coco#val2017'],
                            dataset_type=dataset_type,  # 'det', 'kps'
                            do_copy_image=True)
config.dataset.num_classes = 81 if dataset_type == 'det' else 2
coco_anno_name = 'instances_val2017.json' if dataset_type == 'det' else 'person_keypoints_val2017.json'
config.dataset.test_coco_annotation_path = config.hdfs_local_home_dir + '/common/dataset/coco2017/annotations/' + coco_anno_name

# net params
config = add_network_params(config, 'resnet#50')
config.network.task_type = 'fpn_rpn_rcnn'  # 'fpn_rpn_rcnn', 'frcnn_rpn_rcnn', 'fpn_only_rpn', 'frcnn_only_rpn'
config.network.sym = 'fpn.symbols.sym_det_e2e.get_symbol'
if 'fpn' in config.network.task_type:
    config.network.sym_body = 'fpn.symbols.sym_body.get_fpn_conv_feat'
    config.network.sym_rpn_head = 'fpn.symbols.sym_rpn.get_fpn_rpn_net'
elif 'frcnn' in config.network.task_type:
    config.network.sym_body = 'fpn.symbols.sym_body.get_frcnn_conv_feat'
    config.network.sym_rpn_head = 'fpn.symbols.sym_rpn.get_frcnn_rpn_net'
config.network.sym_rcnn_head = 'fpn.symbols.sym_rcnn.get_rcnn_subnet_large'

# det params
if 'fpn' in config.network.task_type:
    config = add_fpn_params(config, is_big_net=True)
elif 'frcnn' in config.network.task_type:
    config = add_frcnn_params(config, is_big_net=True)
config.network.rpn_rcnn_num_branch = 1
config.TRAIN.rpn_loss_weights *= config.network.rpn_rcnn_num_branch
config.TRAIN.rcnn_loss_weights *= config.network.rpn_rcnn_num_branch
config.TEST.rpn_do_test = True

# exp param
config.exp = 'sym_det_e2e_fpn_c2_exp_resnet50_big'

config.TRAIN.model_prefix = os.path.join(config.hdfs_remote_home_dir, config.root_output_dir, config.exp, 'models') + '/det'
config.job_list = ['python experiments/fpn_coco/demo_fpn_train.py', ]
config.job_list.append('python experiments/fpn_coco/demo_fpn_test.py')

