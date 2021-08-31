import numpy as np
import os
from easydict import EasyDict as edict


def add_home_dir(config):
    file_path = os.path.abspath(__file__)
    config.local_home_dir = file_path.split(config.person_name)[0] + config.person_name
    config.hdfs_local_home_dir = '/opt/hdfs/user/' + config.person_name
    if '/mnt/data-1/data/' in file_path or 'trainvm003.hogpu.cc' in file_path:
        config.hdfs_remote_home_dir = 'hdfs://hobot-mosdata/user/' + config.person_name
    else:
        config.hdfs_remote_home_dir = 'hdfs://hobot-bigdata/user/' + config.person_name
    return config


def add_train_params(config):
    if 'TRAIN' not in config:
        config.TRAIN = edict()
    config.TRAIN.gpus = '0,1,2,3'
    config.TRAIN.image_batch_size = 1
    config.TRAIN.num_workers = 1
    config.TRAIN.use_prefetchiter = 'mosdata' in config.hdfs_remote_home_dir
    config.TRAIN.bn_use_sync = False
    config.TRAIN.bn_use_global_stats = True
    config.TRAIN.do_eval_during_training = False
    # solver params
    config.TRAIN.solver = edict()
    config.TRAIN.solver.optimizer = 'sgd'
    config.TRAIN.solver._lr = 0.00125  # 0.00125 * 16 = 0.02 refer to fpn paper
    config.TRAIN.solver.lr_decay = 0.1
    config.TRAIN.solver.lr_step = [1]
    config.TRAIN.solver.num_epoch = 2
    config.TRAIN.solver.warmup_epochs = 1
    config.TRAIN.solver.load_epoch = None
    # warm up params
    config.TRAIN.solver.warmup = False
    config.TRAIN.solver.warmup_linear = True
    config.TRAIN.solver.warmup_lr = None
    # filter_strategy params
    config.TRAIN.filter_strategy = edict()
    # aug_strategy params
    config.TRAIN.aug_strategy = edict()
    config.TRAIN.aug_strategy.shuffle = True
    config.TRAIN.aug_strategy.aspect_grouping = True
    config.TRAIN.aug_strategy.scales = [(600, 1000)]
    config.TRAIN.aug_strategy.flip = False
    config.TRAIN.aug_strategy.aug_img = False
    config.TRAIN.aug_strategy.rotated_angle_range = 0
    return config


def modify_lr_params(config, use_warmup=True):
    num_gpus = len(config.TRAIN.gpus.split(','))
    config.TRAIN.solver.lr = config.TRAIN.solver._lr * config.TRAIN.image_batch_size * num_gpus * config.TRAIN.num_workers
    if use_warmup and config.TRAIN.solver.warmup_lr:
        config.TRAIN.solver.warmup = True
    else:
        config.TRAIN.solver.warmup = False
    return config


def add_test_params(config):
    if 'TEST' not in config:
        config.TEST = edict()
    config.TEST.gpus = '0,1,2,3'
    config.TEST.use_gt_rois = False
    config.TEST.filter_strategy = edict()
    config.TEST.aug_strategy = edict()   
    config.TEST.aug_strategy.scales = [(600, 1000)]
    config.TEST.load_sym_from_file = False
    return config


def add_network_params(config, net_type_layer):
    if 'network' not in config:
        config.network = edict()
    home_dir = config.hdfs_remote_home_dir
    net_type, num_layer = net_type_layer.split('#')
    config.network.net_type = net_type
    config.network.num_layer = int(num_layer)
    if net_type == 'resnet':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([0, 0, 0], dtype=np.float32)
        config.network.input_scale = 1.0
    elif net_type == 'resnet_v1':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-v1-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
    elif net_type == 'resnet_v2':
        config.network.pretrained_prefix = home_dir + '/common/models/resnet-v2-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
    elif net_type == 'resnext':
        config.network.pretrained_prefix = home_dir + '/common/models/resnext-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
    elif net_type == 'senet':
        config.network.pretrained_prefix = home_dir + '/common/models/senet-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
    elif net_type == 'dpn':
        config.network.pretrained_prefix = home_dir + '/common/models/dpn-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.0167
    elif net_type == 'xception':
        config.network.pretrained_prefix = home_dir + '/common/models/xception-%d' % config.network.num_layer
        config.network.pretrained_epoch = 0
        config.network.input_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        config.network.input_scale = 0.017
    else:
        raise ValueError("unknown net type {}".format(net_type))
    
    config.network.appends_bn = net_type in ['resnet', 'dpn']
    config.network.image_stride = 0
    config.network.num_groups = 1
    config.network.params_allow_missing = False
    return config


def add_dataset_params(config, train_dataset_list=None, test_dataset_list=None, dataset_type='det', do_copy_image=False):
    if 'dataset' not in config:
        config.dataset = edict()
    # train
    config.dataset.train_roidb_path_list = []
    config.dataset.train_imglst_path_list = []
    config.dataset.train_imgidx_path_list = []
    config.dataset.train_imgrec_path_list = []
    if train_dataset_list is not None:
        for dataset in train_dataset_list:
            dataset_name, imageset_name = dataset.split('#')
            roidb_path, imglst_path, imgidx_path, imgrec_path = \
                _get_roidb_image_path(config, dataset_name, imageset_name, dataset_type, do_copy_image=do_copy_image)
            config.dataset.train_roidb_path_list.append(roidb_path)
            config.dataset.train_imglst_path_list.append(imglst_path)
            config.dataset.train_imgidx_path_list.append(imgidx_path)
            config.dataset.train_imgrec_path_list.append(imgrec_path)
    # test
    config.dataset.test_roidb_path_list = []
    config.dataset.test_imglst_path_list = []
    config.dataset.test_imgidx_path_list = []
    config.dataset.test_imgrec_path_list = []
    if test_dataset_list is not None:
        for dataset in test_dataset_list:
            dataset_name, imageset_name = dataset.split('#')
            roidb_path, imglst_path, imgidx_path, imgrec_path = \
                _get_roidb_image_path(config, dataset_name, imageset_name, dataset_type, do_copy_image=do_copy_image)
            config.dataset.test_roidb_path_list.append(roidb_path)
            config.dataset.test_imglst_path_list.append(imglst_path)
            config.dataset.test_imgidx_path_list.append(imgidx_path)
            config.dataset.test_imgrec_path_list.append(imgrec_path)
    return config


def _get_roidb_image_path(config, dataset_name, imageset_name, dataset_type, do_copy_image=False):
    if dataset_name == 'coco':
        dataset_path = 'common/dataset/coco2017'
    elif dataset_name == 'mpi':
        dataset_path = 'common/dataset/mpi'
    elif dataset_name == 'ai_challenge':
        dataset_path = 'common/dataset/ai_challenge'
    else:
        raise ValueError("unknown dataset name {}".format(dataset_name))
    roidb_path = os.path.join(config.hdfs_local_home_dir, dataset_path, 'roidbs', '%s_%s_gt_roidb.pkl' % (imageset_name, dataset_type))
    if do_copy_image:
        imglst_path = os.path.join(config.hdfs_local_home_dir, dataset_path, 'images_lst_rec', '%s.lst' % imageset_name)
        imgidx_path = os.path.join(config.hdfs_local_home_dir, dataset_path, 'images_lst_rec', '%s.idx' % imageset_name)
        imgrec_path = os.path.join(config.hdfs_local_home_dir, dataset_path, 'images_lst_rec', '%s.rec' % imageset_name)
        imglst_path = copy_file_to_local_home(imglst_path, dataset_name, config)
        imgidx_path = copy_file_to_local_home(imgidx_path, dataset_name, config)
        imgrec_path = copy_file_to_local_home(imgrec_path, dataset_name, config)
    else:
        imglst_path = os.path.join(config.hdfs_local_home_dir, dataset_path, 'images_lst_rec', '%s.lst' % imageset_name)
        imgidx_path = os.path.join(config.hdfs_local_home_dir, dataset_path, 'images_lst_rec', '%s.idx' % imageset_name)
        imgrec_path = os.path.join(config.hdfs_local_home_dir, dataset_path, 'images_lst_rec', '%s.rec' % imageset_name)
    return roidb_path, imglst_path, imgidx_path, imgrec_path


def copy_file_to_local_home(src_file, dataset_name, config):
    if '/home/users/gpuwork/mpi_jobs' in config.local_home_dir:
        # train in cluster
        local_dir = '/home/users/gpuwork/mpi_jobs/common' if dataset_name == 'coco' else './data'
    else:
        # train in local
        local_dir = os.path.join(config.local_home_dir, 'common')
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    dst_file = os.path.join(local_dir, dataset_name + '_' + os.path.basename(src_file))
    if not os.path.exists(dst_file):
        copy_commond = 'cp %s %s' % (src_file, dst_file)
        print copy_commond
        os.system(copy_commond)
    return dst_file
