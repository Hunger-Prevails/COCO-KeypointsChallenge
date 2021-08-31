import mxnet as mx
import logging
from common.symbols import sym_common as sym
from fpn.operator_py.proposal_target import proposal_target
from fpn.operator_py.box_annotator_ohem import box_annotator_ohem


def get_rcnn_grad_scales(config, branch_index=0):
    grad_scale_0 = float(config.TRAIN.rcnn_loss_weights[branch_index * 2 + 0])
    grad_scale_1 = float(config.TRAIN.rcnn_loss_weights[branch_index * 2 + 1]) / (config.TRAIN.rcnn_batch_rois * config.TRAIN.image_batch_size)
    return [grad_scale_0, grad_scale_1]


def get_rcnn_eval_info_list(config):
    eval_info_list = []
    for branch_i in range(config.network.rpn_rcnn_num_branch):
        grad_scales = get_rcnn_grad_scales(config, branch_i)
        suffix = '_branch{}'.format(branch_i) if config.network.rpn_rcnn_num_branch > 1 else ''
        # score
        eval_info = dict()
        eval_info['metric_type'] = 'Softmax'
        eval_info['metric_name'] = 'RCNNLabel%s' % suffix
        eval_info['grad_scale'] = grad_scales[0]
        eval_info['axis'] = -1
        eval_info['eval_fg'] = True
        eval_info_list.append(eval_info)
        # loss
        eval_info = dict()
        eval_info['metric_type'] = 'Sum'
        eval_info['metric_name'] = 'RCNNBBox%s' % suffix
        eval_info['grad_scale'] = grad_scales[1]
        eval_info_list.append(eval_info)
    return eval_info_list


def get_rcnn_labels(rcnn_rois, config, is_train=True, suffix='', **kwargs):
    input_dict = kwargs['input_dict']
    if is_train:
        gt_roidb = mx.sym.Variable(name='gt_roidb%s' % suffix)
        input_dict['gt_roidb%s' % suffix] = (config.TRAIN.image_batch_size, 1000)
        group = proposal_target(rois=rcnn_rois, gt_roidb=gt_roidb, config=config)
        rcnn_rois = group[0]
        rcnn_label = group[1]
        rcnn_bbox_target = group[2]
        rcnn_bbox_weight = group[3]
        logging.info('rcnn_rois{}: {}'.format(suffix, rcnn_rois.infer_shape(**input_dict)[1]))
        logging.info('rcnn_label{}: {}'.format(suffix, rcnn_label.infer_shape(**input_dict)[1]))
        logging.info('rcnn_bbox_target{}: {}'.format(suffix, rcnn_bbox_target.infer_shape(**input_dict)[1]))
        logging.info('rcnn_bbox_weight{}: {}'.format(suffix, rcnn_bbox_weight.infer_shape(**input_dict)[1]))
    else:
        rcnn_label = None
        rcnn_bbox_target = None
        rcnn_bbox_weight = None
    return [rcnn_rois, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight]


def extract_roi(shared_feat, rois, name, roi_extract_method, pooled_size, feat_stride, **kwargs):
    if roi_extract_method == 'roi_align_fpn':
        subnet_shared_feat = dict()
        for stride in feat_stride:
            subnet_shared_feat['data_res%d' % stride] = shared_feat['stride%d' % stride]
        feat_rois = mx.sym.ROIAlignFPN(rois=rois,
                                       name=name,
                                       sample_per_part=2,
                                       pooled_size=pooled_size,
                                       feature_strides=tuple(feat_stride),
                                       **subnet_shared_feat)
    elif roi_extract_method == 'roi_align':
        feat_rois = mx.sym.ROIAlign(data=shared_feat,
                                    rois=rois,
                                    name=name,
                                    sample_per_part=2,
                                    pooled_size=pooled_size,
                                    spatial_scale=1.0 / feat_stride)
    elif roi_extract_method == 'crop_and_resize':
        bbox_id = mx.sym.slice_axis(data=rois, axis=1, begin=0, end=1)
        bbox = mx.sym.slice_axis(data=rois, axis=1, begin=1, end=5)
        feat_rois = mx.contrib.sym.CropAndResize(data=shared_feat,
                                                 box=bbox,
                                                 box_id=bbox_id,
                                                 name=name,
                                                 output_shape=pooled_size,
                                                 spatial_scale=1.0 / feat_stride)
    else:
        raise ValueError("unknown roi extract method {}".format(roi_extract_method))
    logging.info('{}: {}'.format(name, feat_rois.infer_shape(**kwargs['input_dict'])[1]))
    return feat_rois


def get_rcnn_train_test(rcnn_cls_score, rcnn_bbox_pred, config, is_train=True,
                        rcnn_label=None, rcnn_bbox_target=None, rcnn_bbox_weight=None,
                        suffix='', **kwargs):
    input_dict = kwargs['input_dict']
    num_classes = config.dataset.num_classes
    num_bbox_classes = 2 if config.network.rcnn_class_agnostic else config.dataset.num_classes

    if is_train:
        if config.TRAIN.rcnn_enable_ohem:
            rcnn_cls_prob_ohem = mx.sym.SoftmaxActivation(data=rcnn_cls_score)
            rcnn_bbox_loss_ohem = rcnn_bbox_weight * mx.sym.smooth_l1(data=(rcnn_bbox_pred - rcnn_bbox_target), scalar=1.0)
            rcnn_label, rcnn_bbox_weight = box_annotator_ohem(rcnn_cls_prob=rcnn_cls_prob_ohem,
                                                              rcnn_bbox_loss=rcnn_bbox_loss_ohem,
                                                              rcnn_label=rcnn_label,
                                                              rcnn_bbox_weight=rcnn_bbox_weight,
                                                              roi_per_img=config.TRAIN.rcnn_batch_rois,
                                                              batch_size=config.TRAIN.image_batch_size)
            logging.info('rcnn_label_ohem{}: {}'.format(suffix, rcnn_label.infer_shape(**input_dict)[1]))
            logging.info('rcnn_bbox_weight_ohem{}: {}'.format(suffix, rcnn_bbox_weight.infer_shape(**input_dict)[1]))
        grad_scales = get_rcnn_grad_scales(config, 0 if suffix == '' else int(suffix[-1]))
        rcnn_cls_prob = mx.sym.SoftmaxOutput(data=rcnn_cls_score, label=rcnn_label, normalization='valid',
                                             use_ignore=True, ignore_label=-1, grad_scale=grad_scales[0])
        rcnn_bbox_loss_t = rcnn_bbox_weight * mx.sym.smooth_l1(data=(rcnn_bbox_pred - rcnn_bbox_target), scalar=1.0)
        rcnn_bbox_loss = mx.sym.MakeLoss(data=rcnn_bbox_loss_t, grad_scale=grad_scales[1])

        rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(config.TRAIN.image_batch_size, -1))
        rcnn_cls_prob = mx.sym.Reshape(data=rcnn_cls_prob, shape=(config.TRAIN.image_batch_size, -1, num_classes))
        rcnn_bbox_loss = mx.sym.Reshape(data=rcnn_bbox_loss, shape=(config.TRAIN.image_batch_size, -1, num_bbox_classes * 4))
        return [rcnn_label, rcnn_cls_prob, rcnn_bbox_loss]
    else:
        rcnn_cls_prob = mx.sym.SoftmaxActivation(data=rcnn_cls_score)
        return [rcnn_cls_prob, rcnn_bbox_pred]


def get_rcnn_extract_roi(shared_feat, rcnn_rois, config, is_train=True, suffix='', **kwargs):
    return extract_roi(shared_feat=shared_feat,
                       rois=rcnn_rois,
                       name='rcnn_feat%s_rois' % suffix,
                       roi_extract_method=config.network.roi_extract_method,
                       pooled_size=config.network.rcnn_pooled_size,
                       feat_stride=config.network.rcnn_feat_stride,
                       batch_size=config.TRAIN.image_batch_size if is_train else 1,
                       input_dict=kwargs['input_dict'])


def get_rcnn_subnet_small(shared_feat, rcnn_rois, config, is_train=True, suffix='', **kwargs):
    input_dict = kwargs['input_dict']
    num_classes = config.dataset.num_classes
    num_bbox_classes = 2 if config.network.rcnn_class_agnostic else config.dataset.num_classes

    rcnn_feat_rois = get_rcnn_extract_roi(shared_feat, rcnn_rois, config, is_train, suffix, **kwargs)

    _, rcnn_fc6 = sym.fcrelu(data=rcnn_feat_rois, num_hidden=512, prefix='rcnn_', suffix='6%s' % suffix)
    _, rcnn_fc7 = sym.fcrelu(data=rcnn_fc6, num_hidden=256, prefix='rcnn_', suffix='7%s' % suffix)

    rcnn_cls_score = sym.fc(data=rcnn_fc7, name='rcnn_cls_score%s' % suffix, num_hidden=num_classes)
    rcnn_bbox_pred = sym.fc(data=rcnn_fc7, name='rcnn_bbox_pred%s' % suffix, num_hidden=num_bbox_classes * 4)

    logging.info('rcnn_fc6{}: {}'.format(suffix, rcnn_fc6.infer_shape(**input_dict)[1]))
    logging.info('rcnn_fc7{}: {}'.format(suffix, rcnn_fc7.infer_shape(**input_dict)[1]))
    logging.info('rcnn_cls_score{}: {}'.format(suffix, rcnn_cls_score.infer_shape(**input_dict)[1]))
    logging.info('rcnn_bbox_pred{}: {}'.format(suffix, rcnn_bbox_pred.infer_shape(**input_dict)[1]))

    return get_rcnn_train_test(rcnn_cls_score=rcnn_cls_score, rcnn_bbox_pred=rcnn_bbox_pred,
                               config=config, is_train=is_train, suffix=suffix, **kwargs)


def get_rcnn_subnet_large(shared_feat, rcnn_rois, config, is_train=True, suffix='', **kwargs):
    input_dict = kwargs['input_dict']
    num_classes = config.dataset.num_classes
    num_bbox_classes = 2 if config.network.rcnn_class_agnostic else config.dataset.num_classes

    rcnn_feat_rois = get_rcnn_extract_roi(shared_feat, rcnn_rois, config, is_train, suffix, **kwargs)

    _, rcnn_fc6 = sym.fcrelu(data=rcnn_feat_rois, num_hidden=1024, prefix='rcnn_', suffix='6%s' % suffix)
    _, rcnn_fc7 = sym.fcrelu(data=rcnn_fc6, num_hidden=1024, prefix='rcnn_', suffix='7%s' % suffix)

    rcnn_cls_score = sym.fc(data=rcnn_fc7, name='rcnn_cls_score%s' % suffix, num_hidden=num_classes)
    rcnn_bbox_pred = sym.fc(data=rcnn_fc7, name='rcnn_bbox_pred%s' % suffix, num_hidden=num_bbox_classes * 4)

    logging.info('rcnn_fc6{}: {}'.format(suffix, rcnn_fc6.infer_shape(**input_dict)[1]))
    logging.info('rcnn_fc7{}: {}'.format(suffix, rcnn_fc7.infer_shape(**input_dict)[1]))
    logging.info('rcnn_cls_score{}: {}'.format(suffix, rcnn_cls_score.infer_shape(**input_dict)[1]))
    logging.info('rcnn_bbox_pred{}: {}'.format(suffix, rcnn_bbox_pred.infer_shape(**input_dict)[1]))

    return get_rcnn_train_test(rcnn_cls_score=rcnn_cls_score, rcnn_bbox_pred=rcnn_bbox_pred,
                               config=config, is_train=is_train, suffix=suffix, **kwargs)


