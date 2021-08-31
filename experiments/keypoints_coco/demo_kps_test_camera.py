import init_paths
import logging
import os
import cPickle
import numpy as np
import mxnet as mx
from config import config as cfg
from common.utils.utils import create_logger, empty_dir, add_music_to_video
from keypoints.kps_iter import KPSTestIter
from keypoints.kps_predict import KPSPredictor


def get_pred_boxes(pred_boxes_path):
    all_images = []
    all_pred_boxes = []
    with open(pred_boxes_path) as fn:
        for line in fn.readlines():
            line = [i.strip() for i in line.strip().split(' ')]
            all_images.append(line[0])
            assert len(line) >= 1
            if len(line) == 1:
                pred_boxes = np.zeros((0, 5), dtype=np.float32)
            else:
                assert (len(line) - 1) % 5 == 0
                pred_boxes = np.array(line)[1:].astype(np.float32).reshape((-1, 5))
            all_pred_boxes.append(pred_boxes)
    logging.info('total %d images' % len(all_pred_boxes))
    return all_images, all_pred_boxes

def make_roidb(all_pred_boxes):
    roidb = []
    for i in range(len(all_pred_boxes)):
        roi_rec = {'image': '/tmp/%d.jpg' % i,
                   'boxes': all_pred_boxes[i],
                   'video_id': 0,
                   'video_index': i}
        roidb.append(roi_rec)
    return roidb

def make_roidb_by_txt(txt_path):
    roidb = []
    with open(txt_path) as fn:
        for line in fn.readlines():
            roi_rec = dict()
            line = [i.strip() for i in line.strip().split(' ')]
            roi_rec['image'] = '/tmp/' + line[0]
            roi_rec['video_id'] = 0
            roi_rec['video_index'] = int(line[0].split('.')[0])
            assert len(line) >= 1
            if len(line) == 1:
                pred_boxes = np.zeros((0, 5), dtype=np.float32)
            else:
                assert (len(line) - 1) % 5 == 0
                pred_boxes = np.array(line)[1:].astype(np.float32).reshape((-1, 5))
            roi_rec['boxes'] = pred_boxes
            roidb.append(roi_rec)
    return roidb

def main():
    import imageio
    config = cfg
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    config.TEST.use_gt_rois = True
    config.dataset.test_imgrec_path_list = []
    create_logger()

    video_path = '/opt/hdfs/user/xinze.chen/common/dataset/deeppose.mp4'

    roidb_path = video_path[:-4] + '.json'
    save_video_path = video_path[:-4] + '_res.mp4'
    save_roidb_path = save_video_path[:-4] + '.json'
    with open(roidb_path, 'rb') as fid:
        test_roidb = cPickle.load(fid)
    config.dataset.video_path = video_path

    vis = True
    vis_kwargs = dict()
    vis_kwargs['writer'] = imageio.get_writer(save_video_path, fps=imageio.get_reader(video_path).get_meta_data()['fps'])
    vis_kwargs['im_save_dir'] = video_path[:-4]
    vis_kwargs['im_save_max_num'] = 100
    if 'im_save_dir' in vis_kwargs:
        empty_dir(vis_kwargs['im_save_dir'])

    begin_test_epoch = config.TRAIN.solver.num_epoch
    end_test_epoch = config.TRAIN.solver.num_epoch + 1
    test_data = KPSTestIter(roidb=test_roidb, config=config, batch_size=len(test_ctx))
    for epoch in range(begin_test_epoch, end_test_epoch):
        predictor = KPSPredictor(config=config,
                                 prefix=config.TRAIN.model_prefix,
                                 epoch=epoch,
                                 provide_data=test_data.provide_data,
                                 max_data_shape=test_data.max_data_shape,
                                 ctx=test_ctx,
                                 use_thread=False)
        predictor.predict_data(test_data=test_data,
                               save_roidb_path=save_roidb_path,
                               vis=vis,
                               **vis_kwargs)

    if vis and 'writer' in vis_kwargs:
        vis_kwargs['writer'].close()
        add_music_to_video(video_path, save_video_path)


if __name__ == '__main__':
    main()