import mxnet as mx
import logging
from fpn.operator_py.proposal_target import proposal_target


def get_multitask_labels(rcnn_rois, config, is_train=True, **kwargs):
    input_dict = kwargs['input_dict']

    rcnn_label = None
    rcnn_bbox_target = None
    rcnn_bbox_weight = None

    kps_rois = None
    kps_label = None
    kps_label_weight = None
    kps_pos_offset = None
    kps_pos_offset_weight = None

    mask_rois = None
    mask_label = None

    if is_train:
        gt_roidb = mx.sym.Variable(name='gt_roidb')
        input_dict['gt_roidb'] = (config.TRAIN.image_batch_size, 1000)
        group = proposal_target(rois=rcnn_rois, gt_roidb=gt_roidb, config=config)
        rcnn_rois = group[0]
        rcnn_label = group[1]
        rcnn_bbox_target = group[2]
        rcnn_bbox_weight = group[3]
        logging.info('rcnn_rois: {}'.format(rcnn_rois.infer_shape(**input_dict)[1]))
        logging.info('rcnn_label: {}'.format(rcnn_label.infer_shape(**input_dict)[1]))
        logging.info('rcnn_bbox_target: {}'.format(rcnn_bbox_target.infer_shape(**input_dict)[1]))
        logging.info('rcnn_bbox_weight: {}'.format(rcnn_bbox_weight.infer_shape(**input_dict)[1]))
        idx = 4
        if 'kps' in config.network.task_type:
            kps_rois = group[4]
            kps_label = group[5]
            kps_label_weight = group[6]
            kps_pos_offset = group[7]
            kps_pos_offset_weight = group[8]
            logging.info('kps_rois: {}'.format(kps_rois.infer_shape(**input_dict)[1]))
            logging.info('kps_label: {}'.format(kps_label.infer_shape(**input_dict)[1]))
            logging.info('kps_label_weight: {}'.format(kps_label_weight.infer_shape(**input_dict)[1]))
            logging.info('kps_pos_offset: {}'.format(kps_pos_offset.infer_shape(**input_dict)[1]))
            logging.info('kps_pos_offset_weight: {}'.format(kps_pos_offset_weight.infer_shape(**input_dict)[1]))
            idx = 9
        if 'mask' in config.network.task_type:
            mask_rois = group[idx]
            mask_label = group[idx + 1]
            logging.info('mask_rois: {}'.format(mask_rois.infer_shape(**input_dict)[1]))
            logging.info('mask_label: {}'.format(mask_label.infer_shape(**input_dict)[1]))

    multitask_labels = dict()
    multitask_labels['rcnn_labels'] = [rcnn_rois, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight]
    multitask_labels['kps_labels'] = [kps_rois, kps_label, kps_label_weight, kps_pos_offset, kps_pos_offset_weight]
    multitask_labels['mask_labels'] = [mask_rois, mask_label]

    return multitask_labels







