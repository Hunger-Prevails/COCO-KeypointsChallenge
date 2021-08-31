import init_paths
import mxnet as mx
import cPickle
import os
from config import config as cfg
from common.utils.utils import create_logger, empty_dir
from keypoints.kps_iter import KPSTestIter
from keypoints.kps_predict import KPSPredictor


def make_roidb(img_dir, all_images, all_pred_boxes):
    assert len(all_images) == len(all_pred_boxes)
    roidb = []
    for i in range(len(all_images)):
        roi_rec = {'image': img_dir + os.path.basename(all_images[i]),
                   'boxes': all_pred_boxes[i]}
        roidb.append(roi_rec)
    return roidb


def main():
    config = cfg
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    config.TEST.use_gt_rois = True
    config.dataset.test_imgrec_path_list = []
    create_logger()

    roidb_path = '/data-sda/xinze.chen/common/dataset/dance/test.json'
    save_roidb_path = roidb_path[:-4] + '_kps.json'
    with open(roidb_path, 'rb') as fid:
        test_roidb = cPickle.load(fid)

    vis = True
    vis_kwargs = dict()
    vis_kwargs['im_save_dir'] = roidb_path[:-4] + '_res/'
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


if __name__ == '__main__':
    main()