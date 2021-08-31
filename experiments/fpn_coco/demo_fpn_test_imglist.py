import init_paths
import mxnet as mx
import glob
from config import config as cfg
from common.utils.utils import create_logger, empty_dir
from fpn.fpn_iter import FPNTestIter
from fpn.fpn_predict import FPNPredictor


def make_roidb(all_images):
    roidb = []
    for i in range(len(all_images)):
        roi_rec = {'image': all_images[i]}
        roidb.append(roi_rec)
    return roidb


def main():
    config = cfg
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    config.dataset.test_imgrec_path_list = []
    config.TEST.rcnn_use_softnms = False
    config.TEST.rcnn_nms = 0.3
    create_logger()

    img_dir = '/opt/hdfs/user/xinze.chen/common/dataset/test_images/'
    all_images = glob.glob(img_dir + '*.jpg')
    test_roidb = make_roidb(all_images)

    vis = True
    vis_kwargs = dict()
    if vis:
        vis_kwargs['im_save_dir'] = img_dir[:-1] + '_res/'
        vis_kwargs['im_save_max_num'] = 100
        if 'im_save_dir' in vis_kwargs:
            empty_dir(vis_kwargs['im_save_dir'])

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
        predictor.predict_rpn_rcnn(test_data=test_data,
                                   score_thresh=0.7,
                                   save_roidb_path=None,
                                   vis=vis,
                                   **vis_kwargs)

if __name__ == '__main__':
    main()