import init_paths
import mxnet as mx
from config import config as cfg
from common.utils.utils import create_logger, empty_dir, add_music_to_video
from fpn.fpn_iter import FPNTestIter
from fpn.fpn_predict import FPNPredictor


def make_roidb(num_images, interval=1):
    roidb = []
    for i in range(0, num_images, interval):
        roi_rec = {'image': '/tmp/%d.jpg' % i,
                   'video_id': 0,
                   'video_index': i}
        roidb.append(roi_rec)
    return roidb

def main():
    import imageio
    config = cfg
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    config.dataset.test_imgrec_path_list = []
    config.TEST.rcnn_use_softnms = False
    config.TEST.rcnn_nms = 0.3
    create_logger()

    video_path = '/opt/hdfs/user/xinze.chen/common/dataset/deeppose.mp4'

    save_video_path = video_path[:-4] + '_res.mp4'
    save_roidb_path = save_video_path[:-4] + '.json'
    video_reader = imageio.get_reader(video_path)
    num_images = len(video_reader)
    test_roidb = make_roidb(num_images, interval=2)
    config.dataset.video_path = video_path

    vis = True
    vis_kwargs = dict()
    if vis:
        vis_kwargs['writer'] = imageio.get_writer(save_video_path, fps=video_reader.get_meta_data()['fps'])
        vis_kwargs['im_save_dir'] = video_path[:-4]
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
                                   save_roidb_path=save_roidb_path,
                                   vis=vis,
                                   **vis_kwargs)

    if vis and 'writer' in vis_kwargs:
        vis_kwargs['writer'].close()
        add_music_to_video(video_path, save_video_path)


if __name__ == '__main__':
    main()