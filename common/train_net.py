import mxnet as mx
import logging
import os
import random
import pprint
import numpy as np
from dataset.load_roidb import load_roidb
from metrics.metric import get_eval_metrics
from utils.utils import create_logger, get_kv, copy_file, load_param
from callback import Speedometer
from lr_scheduler import warmup_scheduler
from module import MutableModule
from common.symbols.sym_common import get_sym_func


def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]


def get_rpn_feat_sym(sym, config):
    rpn_feat_sym = []
    for branch_i in range(config.network.rpn_rcnn_num_branch):
        suffix = '_branch{}'.format(branch_i) if config.network.rpn_rcnn_num_branch > 1 else ''
        if isinstance(config.network.rpn_feat_stride, list):
            feat_sym_each_branch = []
            for stride in config.network.rpn_feat_stride:
                feat_sym_each_branch.append(sym.get_internals()['rpn_conv_stride%d%s_output' % (stride, suffix)])
            rpn_feat_sym.append(feat_sym_each_branch)
        else:
            rpn_feat_sym.append(sym.get_internals()['rpn_conv%s_output' % suffix])
    return rpn_feat_sym


def train_net(config, DataTrainIter, **kwargs):
    ctx = [mx.gpu(int(i)) for i in config.TRAIN.gpus.split(',')]
    seed = 6
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model prefix
    if config.TRAIN.num_workers > 1:
        kv = mx.kvstore.create('dist_sync')
        config.TRAIN.model_prefix += "-%d" % kv.rank
        log_name = 'train_test_%d_pid%d.log' % (kv.rank, os.getpid())
    else:
        kv = get_kv(len(ctx))
        log_name = 'train_test_pid%d.log' % os.getpid()
    config.TRAIN.worker_id = kv.rank if kv is not None else 0
    config.TRAIN.filter_strategy.parts = (config.TRAIN.worker_id, config.TRAIN.num_workers)

    # get logger
    log_path = os.path.join(config.local_home_dir, config.root_output_dir, config.exp + '_' + log_name)
    save_log_path = os.path.join(config.hdfs_local_home_dir, config.root_output_dir, config.exp, log_name)
    # log_path = None
    logger = create_logger(log_path)

    # load symbol
    if config.TRAIN.bn_use_sync:
        assert not config.TRAIN.bn_use_global_stats
        from mxnet.device_sync import init_device_sync
        init_device_sync(ctx)
    sym = get_sym_func(config.network.sym)(config, is_train=True)
    if 'rpn' in config.network.task_type:
        config.network.rpn_feat_sym = get_rpn_feat_sym(sym, config)

    # load training data
    roidb = load_roidb(roidb_path_list=config.dataset.train_roidb_path_list,
                       imglst_path_list=config.dataset.train_imglst_path_list,
                       filter_strategy=config.TRAIN.filter_strategy)
    train_data = DataTrainIter(roidb=roidb, config=config, batch_size=config.TRAIN.image_batch_size * len(ctx), ctx=ctx)
    if config.TRAIN.use_prefetchiter:
        from base_iter import PrefetchingIter
        train_data = PrefetchingIter(data_iter=train_data, num_workers=8, max_queue_size=8)
        train_data.data_iter.load_data()
        train_data.data_iter.get_batch()
    else:
        train_data.load_data()
        train_data.get_batch()
    # test_dataiter_speed(train_data)

    # load initialized params
    if config.TRAIN.solver.load_epoch is not None:
        logger.info('continue training from %s-%04d.params' % (config.TRAIN.model_prefix, config.TRAIN.solver.load_epoch))
        arg_params, aux_params = load_param(config.TRAIN.model_prefix, config.TRAIN.solver.load_epoch)
    elif config.network.pretrained_prefix is not None:
        logger.info('init model from %s-%04d.params' % (config.network.pretrained_prefix, config.network.pretrained_epoch))
        arg_params, aux_params = load_param(config.network.pretrained_prefix, config.network.pretrained_epoch)
    else:
        arg_params = dict()
        aux_params = dict()
    if 'arg_params' in kwargs:
        arg_params.update(kwargs['arg_params'])
    if 'aux_params' in kwargs:
        aux_params.update(kwargs['aux_params'])
    if len(arg_params) == 0:
        arg_params = None
    if len(aux_params) == 0:
        aux_params = None

    # load optimizer
    config.TRAIN.num_examples = train_data.size
    config.TRAIN.batch_size = train_data.batch_size
    config.TRAIN.solver.epoch_size = config.TRAIN.num_examples // config.TRAIN.batch_size
    config.TRAIN.solver.begin_epoch = config.TRAIN.solver.load_epoch if config.TRAIN.solver.load_epoch else 0
    config.TRAIN.solver.begin_num_update = config.TRAIN.solver.epoch_size * config.TRAIN.solver.begin_epoch

    optimizer_params = {'rescale_grad': 1.0 / (config.TRAIN.num_workers * len(ctx)),
                        'learning_rate': config.TRAIN.solver.lr,
                        'lr_scheduler': warmup_scheduler(config),
                        'begin_num_update': config.TRAIN.solver.begin_num_update,
                        'clip_gradient': 2}
    if config.TRAIN.solver.optimizer == 'sgd':
        optimizer_params['momentum'] = 0.9
    if 'wd' in config.TRAIN:
        optimizer_params['wd'] = config.TRAIN.wd
    else:
        if config.TRAIN.solver.optimizer == 'sgd':
            optimizer_params['wd'] = 4e-5
        elif config.TRAIN.solver.optimizer == 'adam':
            optimizer_params['wd'] = 0.000001
        else:
            raise ValueError("unknown optimizer {}".format(config.TRAIN.solver.optimizer))
    logger.info('optimizer_params:{}\n'.format(pprint.pformat(optimizer_params)))

    # load eval_metrics
    if 'eval_info_list' in kwargs:
        eval_info_list = kwargs['eval_info_list']
    else:
        eval_info_list = get_eval_info_list(config)
    eval_metrics = get_eval_metrics(eval_info_list)

    # create module
    mod = MutableModule(symbol=sym,
                        data_names=train_data.data_name,
                        label_names=train_data.label_name,
                        logger=logger,
                        context=ctx,
                        max_data_shapes=train_data.max_data_shape,
                        max_label_shapes=train_data.max_label_shape)

    # get batch end callback
    batch_end_callback = [Speedometer(config.TRAIN.batch_size, 50)]
    if 'batch_end_callback' in kwargs:
        batch_end_callback.extend(_as_list(kwargs['batch_end_callback']))

    # get epoch end callback
    epoch_end_callback = [mx.callback.do_checkpoint(config.TRAIN.model_prefix)]
    if 'epoch_end_callback' in kwargs and config.TRAIN.worker_id == config.TRAIN.num_workers - 1:
        epoch_end_callback.extend(_as_list(kwargs['epoch_end_callback']))

    # change symbol during training
    if 'new_symbol' in config.TRAIN:
        def change_symbol_callback(module, new_sym, epoch):
            def _callback(iter_no, sym, arg, aux):
                if iter_no + 1 == epoch:
                    logging.info('change symbol with epoch %d' % epoch)
                    module._symbol = new_sym
                    module._need_reshape = True
            return _callback
        assert 'new_symbol_epoch' in config.TRAIN
        epoch_end_callback.append(change_symbol_callback(mod, config.TRAIN.new_symbol, config.TRAIN.new_symbol_epoch))

    logger.info('config:{}\n'.format(pprint.pformat(config)))
    mod.fit(train_data=train_data,
            eval_metric=eval_metrics,
            epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback,
            kvstore=kv,
            optimizer=config.TRAIN.solver.optimizer,
            optimizer_params=optimizer_params,
            arg_params=arg_params,
            aux_params=aux_params,
            initializer=mx.init.Mixed(['deconv*weight', '.*'], [mx.init.Bilinear(), mx.init.Normal(sigma=1e-2)]),
            allow_missing=True,
            begin_epoch=config.TRAIN.solver.begin_epoch,
            num_epoch=config.TRAIN.solver.num_epoch,
            force_batch_sync=True)
    if log_path is not None:
        copy_file(log_path, save_log_path)


def get_eval_info_list(config):
    eval_info_list = []
    if 'rpn' in config.network.task_type:
        from fpn.symbols.sym_rpn import get_rpn_eval_info_list
        eval_info_list.extend(get_rpn_eval_info_list(config))
    if 'rpn_rcnn' in config.network.task_type:
        from fpn.symbols.sym_rcnn import get_rcnn_eval_info_list
        eval_info_list.extend(get_rcnn_eval_info_list(config))
    if 'kps' in config.network.task_type:
        from keypoints.symbols.sym_kps_common import get_kps_eval_info_list
        eval_info_list.extend(get_kps_eval_info_list(config))
    if 'mask' in config.network.task_type:
        from fpn_multitask.symbols.sym_mask import get_mask_eval_info_list
        eval_info_list.extend(get_mask_eval_info_list(config))
    return eval_info_list


def test_dataiter_speed(train_data):
    import time
    tic = time.time()
    disp_batches = 1
    batch_list = []
    logging.info('total num batches: %d' % (train_data.size / train_data.batch_size))
    for k in range(2):
        for i, batch in enumerate(train_data):
            for j in batch.data:
                j.wait_to_read()
            for j in batch.label:
                j.wait_to_read()
            indexes = batch.index
            for index in indexes:
                assert index not in batch_list, '{} has in {}'.format(index, batch_list)
                batch_list.append(index)
            if (i + 1) % disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (i, disp_batches * train_data.batch_size / (time.time() - tic)))
                tic = time.time()
        assert len(batch_list) == train_data.size
        train_data.reset()
        batch_list = []
    exit(0)