import logging
from common.symbols import sym_common as sym
from fpn.symbols.sym_rcnn import extract_roi
from keypoints.symbols.sym_kps_common import kps_train_test


def get_kps_extract_roi(shared_feat, kps_rois, config, pooled_size, is_train=True, **kwargs):
    return extract_roi(shared_feat=shared_feat,
                       rois=kps_rois,
                       name='kps_feat_rois',
                       roi_extract_method=config.network.roi_extract_method,
                       pooled_size=pooled_size,
                       feat_stride=config.network.rcnn_feat_stride,
                       batch_size=config.TRAIN.image_batch_size if is_train else 1,
                       input_dict=kwargs['input_dict'])


def get_kps_subnet_1(shared_feat, kps_rois, config, is_train=True, **kwargs):
    pooled_size = (config.network.kps_height, config.network.kps_width)
    conv_feat = get_kps_extract_roi(shared_feat, kps_rois, config, pooled_size, is_train, **kwargs)

    for i in range(8):
        _, conv_feat = sym.convrelu(data=conv_feat, num_filter=256, kernel=3, prefix='kps_', suffix='%d' % (i + 1))
    kps_label_pred = sym.conv(data=conv_feat, name='kps_label_pred', num_filter=config.dataset.num_kps, kernel=1)
    kps_pos_offset_pred = sym.conv(data=conv_feat, name='kps_pos_offset_pred', num_filter=config.dataset.num_kps * 2, kernel=1)
    logging.info('kps_label_pred: {}'.format(kps_label_pred.infer_shape(**kwargs['input_dict'])[1]))
    logging.info('kps_pos_offset_pred: {}'.format(kps_pos_offset_pred.infer_shape(**kwargs['input_dict'])[1]))

    return kps_train_test(kps_label_pred=kps_label_pred, kps_pos_offset_pred=kps_pos_offset_pred,
                          config=config, is_train=is_train, **kwargs)


def get_kps_subnet_2(shared_feat, kps_rois, config, is_train=True, **kwargs):
    pooled_size = (config.network.kps_height / 4, config.network.kps_width / 4)
    conv_feat = get_kps_extract_roi(shared_feat, kps_rois, config, pooled_size, is_train, **kwargs)

    for i in range(8):
        _, conv_feat = sym.convrelu(data=conv_feat, num_filter=512, kernel=3, prefix='kps_', suffix='%d' % (i + 1))
    kps_label_pred = sym.deconv(data=conv_feat, name='kps_label_pred_deconv',
                                num_filter=config.dataset.num_kps, kernel=4, stride=2, pad=1)
    kps_label_pred = sym.upsampling_bilinear(data=kps_label_pred, name='kps_label_pred_bilinear',
                                             scale=2, num_filter=config.dataset.num_kps, need_train=True)
    kps_pos_offset_pred = sym.deconv(data=conv_feat, name='kps_pos_offset_pred_deconv',
                                     num_filter=config.dataset.num_kps * 2, kernel=4, stride=2, pad=1)
    kps_pos_offset_pred = sym.upsampling_bilinear(data=kps_pos_offset_pred, name='kps_pos_offset_pred_bilinear',
                                                  scale=2, num_filter=config.dataset.num_kps * 2, need_train=True)
    logging.info('kps_label_pred: {}'.format(kps_label_pred.infer_shape(**kwargs['input_dict'])[1]))
    logging.info('kps_pos_offset_pred: {}'.format(kps_pos_offset_pred.infer_shape(**kwargs['input_dict'])[1]))

    return kps_train_test(kps_label_pred=kps_label_pred, kps_pos_offset_pred=kps_pos_offset_pred,
                          config=config, is_train=is_train, **kwargs)
