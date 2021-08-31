import init_paths
import os
import mxnet as mx
from config import config as cfg
from common.utils.utils import create_logger, empty_dir
from fpn.fpn_iter import FPNTestIter
from fpn_multitask.fpn_multitask_predict import FPNMultitaskPredictor


def make_roidb(img_dir, all_images):
    roidb = []
    for i in range(len(all_images)):
        roi_rec = {'image': img_dir + os.path.basename(all_images[i])}
        roidb.append(roi_rec)
    return roidb


def main():
    config = cfg
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    config.dataset.test_imgrec_path_list = []
    config.TEST.use_gt_rois = False
    config.TEST.rcnn_use_softnms = False
    config.TEST.rpn_pre_nms_top_n = 1000
    config.TEST.rpn_post_nms_top_n = 100
    create_logger()
    save_roidb_path = None

    img_dir = '/data-sdb/xinze.chen/common/dataset/test_images/20180417_154833/'
    all_images = os.listdir(img_dir)
    test_roidb = make_roidb(img_dir, all_images)

    # config.TEST.use_gt_rois = True
    # roidb_path = '/data-sdb/xinze.chen/common/dataset/baili_ch11/roidb_20171027.json'
    # with open(roidb_path, 'rb') as fid:
    #     test_roidb = cPickle.load(fid)
    # test_roidb = [test_roidb[i] for i in range(10)]

    vis = True
    vis_kwargs = dict()
    vis_kwargs['im_save_dir'] = '/data-sdb/xinze.chen/common/dataset/test_images/20180417_154833_res/'
    vis_kwargs['im_save_max_num'] = -1
    empty_dir(vis_kwargs['im_save_dir'])

    test_data = FPNTestIter(roidb=test_roidb, config=config, batch_size=len(test_ctx))
    begin_test_epoch = config.TRAIN.solver.num_epoch
    end_test_epoch = config.TRAIN.solver.num_epoch + 1
    for epoch in range(begin_test_epoch, end_test_epoch):
        predictor = FPNMultitaskPredictor(config=config,
                                          prefix=config.TRAIN.model_prefix,
                                          epoch=epoch,
                                          provide_data=test_data.provide_data,
                                          max_data_shape=test_data.max_data_shape,
                                          ctx=test_ctx)
        predictor.predict_multitask(test_data=test_data,
                                    alg='alg-pid%d' % os.getpid(),
                                    score_thresh=0.7,
                                    save_roidb_path=save_roidb_path,
                                    vis=vis,
                                    **vis_kwargs)


if __name__ == '__main__':
    main()