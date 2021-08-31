import init_paths
import mxnet as mx
from config import config as cfg
from common.utils.utils import create_logger, empty_dir, add_music_to_video
from fpn.fpn_iter import FPNTestIter
from fpn_multitask.fpn_multitask_predict import FPNMultitaskPredictor


def make_roidb(num_images):
    roidb = []
    for i in range(num_images):
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
    config.TEST.use_gt_rois = False
    config.TEST.rcnn_use_softnms = False
    config.TEST.rpn_pre_nms_top_n = 1000
    config.TEST.rpn_post_nms_top_n = 100
    create_logger()
    save_roidb_path = None

    vis = True
    vis_kwargs = dict()
    vis_kwargs['do_draw_box'] = False

    config.dataset.is_camera = False
    if config.dataset.is_camera:
        test_roidb = make_roidb(1000000)
        vis_kwargs['show_camera'] = True
    else:
        video_path = '/data-sdb/xinze.chen/common/dataset/test_videos/720p_00_03_00-00_10_42.mp4'
        save_video_path = video_path[:-4] + '_res.mp4'
        video_reader = imageio.get_reader(video_path)
        num_images = len(video_reader)
        test_roidb = make_roidb(num_images)
        config.dataset.video_path = video_path
        save_roidb_path = save_video_path[:-4] + '.json'
        vis_kwargs['writer'] = imageio.get_writer(save_video_path, fps=video_reader.get_meta_data()['fps'])
        vis_kwargs['im_save_dir'] = save_video_path[:-4]
        vis_kwargs['im_save_max_num'] = 100
        if 'im_save_dir' in vis_kwargs:
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
                                    score_thresh=0.7,
                                    save_roidb_path=save_roidb_path,
                                    vis=vis,
                                    **vis_kwargs)

    if 'writer' in vis_kwargs:
        vis_kwargs['writer'].close()
        add_music_to_video(video_path, save_video_path)


if __name__ == '__main__':
    main()