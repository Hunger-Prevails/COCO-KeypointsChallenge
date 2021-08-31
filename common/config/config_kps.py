import numpy as np
from easydict import EasyDict as edict


def add_kps_params(config, is_big_input=True):
    if 'TRAIN' not in config:
        config.TRAIN = edict()
    if 'TEST' not in config:
        config.TEST = edict()
    if 'network' not in config:
        config.network = edict()
    if 'dataset' not in config:
        config.dataset = edict()
    if 'aug_strategy' not in config.TRAIN:
        config.TRAIN.aug_strategy = edict()
    if 'aug_strategy' not in config.TEST:
        config.TEST.aug_strategy = edict()

    config.dataset.num_kps = 17
    config.dataset.kps_skeleton = np.array([[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]) - 1

    config.network.kps_crop_from_image = True
    config.network.kps_aux_suv = False

    if is_big_input:
        config.network.kps_input_height = 352
        config.network.kps_input_width = 256
        config.network.kps_pos_distance_x = 25.0
        config.network.kps_pos_distance_y = 25.0
    else:
        config.network.kps_input_height = 176
        config.network.kps_input_width = 128
        config.network.kps_pos_distance_x = 12.0
        config.network.kps_pos_distance_y = 12.0

    config.network.kps_feat_stride = 16
    config.network.kps_width = config.network.kps_input_width / config.network.kps_feat_stride
    config.network.kps_height = config.network.kps_input_height / config.network.kps_feat_stride

    config.TRAIN.kps_roi_batch_size = 4
    config.TRAIN.kps_loss_weights = [1.0, 1.0]
    config.TRAIN.kps_loss_type_list = ['pixel_reg_fixed_smooth_L1', ]
    config.TRAIN.kps_loss_type = config.TRAIN.kps_loss_type_list[0]
    config.TRAIN.kps_keep_invis = True
    config.TRAIN.kps_label_range = 1.0
    config.TRAIN.kps_gauss_sigma = 1.0
    config.TRAIN.kps_scalar_L1 = 1.0
    config.TRAIN.kps_aux_mask_value = 0.6
    config.TRAIN.kps_contra_margin = 0.5

    config.TRAIN.aug_strategy.kps_rescale_factor = [-0.1, 0.2]
    config.TRAIN.aug_strategy.kps_jitter_center = True
    config.TRAIN.aug_strategy.kps_do_aspect_ratio = True

    config.TEST.aug_strategy.kps_multiscale = False
    config.TEST.aug_strategy.kps_flip_test = False
    config.TEST.aug_strategy.kps_rescale_factor = 0.1
    config.TEST.aug_strategy.kps_do_aspect_ratio = True

    config.TEST.kps_iou_thresh = 0.5
    config.TEST.kps_oks_thresh = None
    config.TEST.kps_soft_method = None
    config.TEST.heatmap_fusion = False

    return config



