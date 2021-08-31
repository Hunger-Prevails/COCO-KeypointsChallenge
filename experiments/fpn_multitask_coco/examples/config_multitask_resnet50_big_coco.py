import os
from easydict import EasyDict as edict
from common.config.config_common import add_home_dir, add_train_params, modify_lr_params, \
    add_test_params, add_network_params, add_dataset_params
from common.config.config_fpn import add_fpn_params
from common.config.config_frcnn import add_frcnn_params
from common.config.config_kps import add_kps_params
from common.config.config_mask import add_mask_params

config = edict()
config.person_name = 'xinze.chen'
config.root_output_dir = 'experiments_fpn_multitask'
config = add_home_dir(config)

# train params
config = add_train_params(config)
config.TRAIN.gpus = '0,1,2,3'
config.TRAIN.image_batch_size = 1
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
config = add_dataset_params(config=config,
                            train_dataset_list=['coco#train2017'],
                            test_dataset_list=['coco#val2017'],
                            dataset_type='kps',
                            do_copy_image=True)
config.dataset.num_classes = 2
config.dataset.test_coco_annotation_path = config.hdfs_local_home_dir + '/common/dataset/coco2017/annotations/person_keypoints_val2017.json'

# net params
# 'fpn_rpn_rcnn', 'fpn_rpn_rcnn_kps', 'fpn_rpn_rcnn_mask', 'fpn_rpn_rcnn_kps_mask'
# 'frcnn_rpn_rcnn', 'frcnn_rpn_rcnn_kps', 'frcnn_rpn_rcnn_mask', 'frcnn_rpn_rcnn_kps_mask'
config = add_network_params(config, 'resnet#50')
config.network.task_type = 'fpn_rpn_rcnn_kps_mask'
config.network.sym = 'fpn_multitask.symbols.sym_multitask_e2e.get_symbol'
if 'fpn' in config.network.task_type:
    config.network.sym_body = 'fpn.symbols.sym_body.get_fpn_conv_feat'
    config.network.sym_rpn_head = 'fpn.symbols.sym_rpn.get_fpn_rpn_net'
elif 'frcnn' in config.network.task_type:
    config.network.sym_body = 'fpn.symbols.sym_body.get_frcnn_conv_feat'
    config.network.sym_rpn_head = 'fpn.symbols.sym_rpn.get_frcnn_rpn_net'
config.network.sym_rcnn_head = 'fpn.symbols.sym_rcnn.get_rcnn_subnet_large'
config.network.sym_kps_head = 'fpn_multitask.symbols.sym_kps.get_kps_subnet_1'
config.network.sym_mask_head = 'fpn_multitask.symbols.sym_mask.get_mask_subnet'

# multitask params
# det params
if 'fpn' in config.network.task_type:
    config = add_fpn_params(config, is_big_net=True)
elif 'frcnn' in config.network.task_type:
    config = add_frcnn_params(config, is_big_net=True)

# kps params
if 'kps' in config.network.task_type:
    config = add_kps_params(config)
    config.network.kps_crop_from_image = False
    config.TRAIN.aug_strategy.kps_jitter_center = False
    config.TEST.aug_strategy.kps_rescale_factor = 0.0
    config.TRAIN.kps_roi_batch_size = int(config.TRAIN.rcnn_batch_rois * config.TRAIN.rcnn_fg_fraction)

    config.network.kps_feat_stride = 16
    config.network.kps_width = config.network.kps_input_width / config.network.kps_feat_stride
    config.network.kps_height = config.network.kps_input_height / config.network.kps_feat_stride

    config.TRAIN.kps_loss_type_list = ['m2_pixel_reg_fixed_smooth_L1', ]
    config.TRAIN.kps_loss_type = config.TRAIN.kps_loss_type_list[0]

    config.TRAIN.aug_strategy.kps_rescale_factor = [-0.1, 0.2]
    config.TRAIN.aug_strategy.kps_do_aspect_ratio = True
    config.TEST.aug_strategy.kps_do_aspect_ratio = True
    config.network.kps_compute_area = True

# mask params
if 'mask' in config.network.task_type:
    config = add_mask_params(config)
    config.TRAIN.mask_roi_batch_size = int(config.TRAIN.rcnn_batch_rois * config.TRAIN.rcnn_fg_fraction)

# exp param
config.exp = 'sym_multitask_e2e_%s_exp_resnet50_big' % config.network.task_type

config.TRAIN.model_prefix = os.path.join(config.hdfs_remote_home_dir, config.root_output_dir, config.exp, 'models') + '/multitask'
config.job_list = ['python experiments/fpn_multitask_coco/demo_fpn_multitask_train.py', ]
config.job_list.append('python experiments/fpn_multitask_coco/demo_fpn_multitask_test.py')
