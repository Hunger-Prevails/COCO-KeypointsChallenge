import init_paths
import os
import sys
import pprint
import numpy as np
import mxnet as mx
from config import config as cfg
from common.dataset.load_roidb_eval import load_coco_test_roidb_eval
from common.utils.utils import create_logger, copy_file, empty_dir
from keypoints.kps_iter import KPSTestIter
from keypoints.kps_predict import KPSPredictor


def get_gt_boxes(test_roidb):
    gt_boxes = [[[] for _ in xrange(len(test_roidb))]
                for _ in xrange(2)]
    for i in range(len(test_roidb)):
        boxes = test_roidb[i]['boxes']
        scores = np.ones((boxes.shape[0], 1), dtype=boxes.dtype)
        gt_boxes[1][i] = np.hstack((boxes, scores))
    return gt_boxes


def get_gt_kps(test_roidb):
    gt_kps = [[[] for _ in xrange(len(test_roidb))]
              for _ in xrange(2)]
    for i in range(len(test_roidb)):
        kps = test_roidb[i]['keypoints']
        scores = np.ones((kps.shape[0], 1), dtype=kps.dtype)
        gt_kps[1][i] = np.ndarray.tolist(np.hstack((kps, scores)))
    return gt_kps[1]


def get_pred_boxes(test_roidb):
    det_boxes = [[[] for _ in xrange(len(test_roidb))]
                 for _ in xrange(2)]
    for i in range(len(test_roidb)):
        det_boxes[1][i] = test_roidb[i]['pred_boxes']
    return det_boxes


def main():
    config = cfg
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    pid = os.getpid()

    log_name = 'test_pid%d.log' % pid
    log_path = os.path.join(config.local_home_dir, config.root_output_dir, config.exp + '_' + log_name)
    save_log_path = os.path.join(config.local_home_dir, config.root_output_dir, config.exp, log_name)
    # log_path = None
    logger = create_logger(log_path)

    test_roidb, eval_func = load_coco_test_roidb_eval(config, config.dataset.test_coco_annotation_path)

    vis = False
    vis_kwargs = dict()
    if vis:
        vis_kwargs['im_save_dir'] = os.path.join(config.local_home_dir, config.root_output_dir, 'images_show_pid%d' % pid)
        vis_kwargs['im_save_max_num'] = 100
        if 'im_save_dir' in vis_kwargs:
            empty_dir(vis_kwargs['im_save_dir'])

    logger.info('config:{}\n'.format(pprint.pformat(config)))
    test_data = KPSTestIter(roidb=test_roidb, config=config, batch_size=len(test_ctx))
    
    predictor = KPSPredictor(config=config,
                             prefix=config.TRAIN.model_prefix,
                             epoch=config.TEST.load_epoch,
                             provide_data=test_data.provide_data,
                             max_data_shape=test_data.max_data_shape,
                             ctx=test_ctx,
                             use_thread=False)

    save_roidb_path = sys.argv[1] if len(sys.argv) == 2 else None

    predictor.predict_data(test_data,
                           eval_func=eval_func,
                           alg='alg-pid%d' % pid,
                           save_roidb_path=save_roidb_path,
                           vis=vis,
                           **vis_kwargs)
    if log_path is not None and save_log_path is not None:
        copy_file(log_path, save_log_path)

if __name__ == '__main__':
    main()

