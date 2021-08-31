import init_paths
import os
import pprint
import mxnet as mx
from config import config as cfg
from common.utils.utils import create_logger, copy_file
from common.dataset.load_roidb_eval import load_coco_test_roidb_eval
from fpn.fpn_iter import FPNTestIter
from fpn.fpn_predict import FPNPredictor


def main():
    config = cfg
    pid = os.getpid()

    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]

    log_name = 'test_pid%d.log' % pid
    log_path = os.path.join(config.local_home_dir, config.root_output_dir, config.exp + '_' + log_name)
    save_log_path = os.path.join(config.hdfs_local_home_dir, config.root_output_dir, config.exp, log_name)
    # log_path = None
    logger = create_logger(log_path)


    config.TEST.filter_strategy.remove_empty_boxes = True
    config.TEST.filter_strategy.max_num_images = 500
    test_roidb, eval_func = load_coco_test_roidb_eval(config, config.dataset.test_coco_annotation_path)

    logger.info('config:{}\n'.format(pprint.pformat(config)))
    test_data = FPNTestIter(roidb=test_roidb, config=config, batch_size=len(test_ctx))
    begin_test_epoch = config.TRAIN.solver.num_epoch
    end_test_epoch = config.TRAIN.solver.num_epoch + 1
    for epoch in range(begin_test_epoch, end_test_epoch):
        predictor = FPNPredictor(config=config,
                                 prefix=config.TRAIN.model_prefix,
                                 epoch=epoch,
                                 provide_data=test_data.provide_data,
                                 max_data_shape=test_data.max_data_shape,
                                 ctx=test_ctx)
        if 'only_rpn' in config.network.task_type:
            predictor.predict_rpn(test_data)
        elif 'rpn_rcnn' in config.network.task_type:
            predictor.predict_rpn_rcnn(test_data,
                                       eval_func=eval_func,
                                       alg='alg-pid%d' % pid)
        else:
            raise ValueError("unknown task type {}".format(config.network.task_type))
    if log_path is not None and save_log_path is not None:
        copy_file(log_path, save_log_path)

if __name__ == '__main__':
    main()

