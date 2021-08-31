import os
from easydict import EasyDict as edict
from common.config.config_common import add_home_dir, add_train_params, modify_lr_params, \
    add_test_params, add_network_params, add_dataset_params
from common.config.config_kps import add_kps_params

config = edict()
config.person_name = 'yinglun.liu'
config.root_output_dir = 'experiments_kps'
config = add_home_dir(config)

# train params
config = add_train_params(config)
config.TRAIN.gpus = '0,1,2,3'
config.TRAIN.image_batch_size = 4
config.TRAIN.bn_use_sync = True
config.TRAIN.bn_use_global_stats = False
config.TRAIN.do_eval_during_training = True
config.TRAIN.solver.optimizer = 'sgd'
config.TRAIN.solver.warmup_lr = 1e-5
config.TRAIN.solver.warmup_epochs = 2
config.TRAIN.solver.lr_decay = 0.2
config.TRAIN.solver.lr_step = [24, 32, 40]
config.TRAIN.solver.num_epoch = 48
config.TRAIN.filter_strategy.remove_empty_boxes = True
config.TRAIN.filter_strategy.remove_empty_kps = True
config.TRAIN.aug_strategy.flip = True
config.TRAIN.aug_strategy.rotated_angle_range = 20
config = modify_lr_params(config, use_warmup=True)

# test params
config = add_test_params(config)
config.TEST.gpus = '0,1,2,3'
config.TEST.use_gt_rois = True
config.TEST.load_sym_from_file = False
config.TEST.load_epoch = 29
config.TEST.filter_strategy.remove_empty_preds = False if config.TEST.use_gt_rois else True
config.TEST.filter_strategy.remove_empty_boxes = True if config.TEST.use_gt_rois else False
config.TEST.filter_strategy.remove_empty_kps = True if config.TEST.use_gt_rois else False

# dataset params
config = add_dataset_params(config=config,
                            train_dataset_list=['coco#train2017'],
                            test_dataset_list=['coco#val2017'],
                            dataset_type='kps',
                            do_copy_image=True)
config.dataset.num_classes = 2
config.dataset.test_coco_annotation_path = config.hdfs_local_home_dir + '/common/dataset/coco2017/annotations/person_keypoints_val2017.json'

# net params
config = add_network_params(config, 'resnet#101')
config.network.task_type = 'kps'
config.network.sym = 'keypoints.symbols.sym_kps_data.deconv_symbol'
config.network.sym_body = 'keypoints.symbols.sym_kps_data.get_conv_feat_res5'

# kps params
config = add_kps_params(config)
config.network.kps_feat_stride = 8
config.network.kps_width = config.network.kps_input_width / config.network.kps_feat_stride
config.network.kps_height = config.network.kps_input_height / config.network.kps_feat_stride
config.network.kps_num_filter = 256
config.network.deformable_units = [0, 0, 0, 3]
config.network.num_deformable_group = [0, 0, 0, 4]
config.network.inc_dilates = [False, False, False, False]

config.TRAIN.kps_loss_type_list = ['pixel_reg_fixed_smooth_L1_aux_suv']
config.TRAIN.kps_loss_type = config.TRAIN.kps_loss_type_list[0]
config.TRAIN.kps_scalar_L1 = 1.0
config.TRAIN.kps_aux_mask_value = 0.2

config.TEST.aug_strategy.kps_flip_test = False
config.TEST.aug_strategy.kps_multiscale = False
config.TEST.aug_strategy.kps_rescale_factor = [0.2, 0.1, 0.0, -0.1]

# exp param
config.exp = 'resnet101_deconv_s8_f256_aux_suv_m0.2'

config.TRAIN.model_prefix = os.path.join(config.hdfs_remote_home_dir, config.root_output_dir, config.exp, 'models') + '/kps'
config.job_list = ['python experiments/keypoints_coco/demo_kps_train.py', ]
config.job_list.append('python experiments/keypoints_coco/demo_kps_test.py')
