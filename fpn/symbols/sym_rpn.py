import mxnet as mx
import logging
from common.symbols import sym_common as sym
from ..operator_py.proposal_rois_fpn import proposal_rois_fpn
from ..operator_py.proposal_rois import proposal_rois


def get_rpn_grad_scales(config, branch_index=0):
    grad_scale_0 = float(config.TRAIN.rpn_loss_weights[branch_index * 2 + 0])
    grad_scale_1 = float(config.TRAIN.rpn_loss_weights[branch_index * 2 + 1]) / (config.TRAIN.rpn_batch_size * config.TRAIN.image_batch_size)
    return [grad_scale_0, grad_scale_1]


def get_rpn_eval_info_list(config):
    eval_info_list = []
    for branch_i in range(config.network.rpn_rcnn_num_branch):
        grad_scales = get_rpn_grad_scales(config, branch_i)
        suffix = '_branch{}'.format(branch_i) if config.network.rpn_rcnn_num_branch > 1 else ''
        # score
        if config.TRAIN.rpn_cls_loss_type == 'softmax':
            eval_info = dict()
            eval_info['metric_type'] = 'Softmax'
            eval_info['metric_name'] = 'RPNLabel%s' % suffix
            eval_info['grad_scale'] = grad_scales[0]
            eval_info['axis'] = 1
            eval_info['eval_fg'] = False
            eval_info_list.append(eval_info)
        elif config.TRAIN.rpn_cls_loss_type == 'cross_entropy':
            eval_info = dict()
            eval_info['metric_type'] = 'CrossEntropy'
            eval_info['metric_name'] = 'RPNLabel%s' % suffix
            eval_info['grad_scale'] = grad_scales[0]
            eval_info['eval_fg'] = False
            eval_info_list.append(eval_info)
        else:
            raise ValueError("unknown rpn cls loss type {}".format(config.TRAIN.rpn_cls_loss_type))
        # loss
        eval_info = dict()
        eval_info['metric_type'] = 'Sum'
        eval_info['metric_name'] = 'RPNBBox%s' % suffix
        eval_info['grad_scale'] = grad_scales[1]
        eval_info_list.append(eval_info)
    return eval_info_list


def get_rpn_train(rpn_cls_score, rpn_bbox_pred, config, suffix=''):
    rpn_label = mx.sym.Variable(name='rpn_label%s' % suffix)
    rpn_bbox_target = mx.sym.Variable(name='rpn_bbox_target%s' % suffix)
    rpn_bbox_weight = mx.sym.Variable(name='rpn_bbox_weight%s' % suffix)
    grad_scales = get_rpn_grad_scales(config, 0 if suffix == '' else int(suffix[-1]))
    # classification
    if config.TRAIN.rpn_cls_loss_type == 'softmax':
        rpn_cls_prob = sym.softmax_out(data=rpn_cls_score, label=rpn_label, ignore_label=-1,
                                       multi_output=True, grad_scale=grad_scales[0])
    elif config.TRAIN.rpn_cls_loss_type == 'cross_entropy':
        from common.operator_py.sigmoid_cross_entropy_loss import sigmoid_cross_entropy_loss
        rpn_cls_prob = sigmoid_cross_entropy_loss(data=rpn_cls_score, label=rpn_label, grad_scale=grad_scales[0],
                                                  use_ignore=True, ignore_label=-1)
    else:
        raise ValueError("unknown rpn cls loss type {}".format(config.TRAIN.rpn_cls_loss_type))
    # bounding box regression
    rpn_bbox_loss_t = rpn_bbox_weight * mx.sym.smooth_l1(data=(rpn_bbox_pred - rpn_bbox_target), scalar=3.0)
    rpn_bbox_loss = mx.sym.MakeLoss(data=rpn_bbox_loss_t, grad_scale=grad_scales[1])
    return [mx.sym.BlockGrad(rpn_label), rpn_cls_prob, rpn_bbox_loss]


def get_fpn_rpn_net(conv_feat, config, is_train=True, suffix='', **kwargs):
    assert config.TRAIN.rpn_cls_loss_type == 'softmax'
    input_dict = kwargs['input_dict']

    rpn_cls_score_list = []
    rpn_bbox_pred_list = []
    rpn_cls_prob_dict = {}
    rpn_bbox_pred_dict = {}
    for i, stride in enumerate(config.network.rpn_feat_stride):
        num_anchors = len(config.network.rpn_anchor_ratios) * len(config.network.rpn_anchor_scales[i])
        rpn_conv = sym.conv(data=conv_feat['stride%d' % stride], name='rpn_conv_stride%d%s' % (stride, suffix),
                            num_filter=config.network.rpn_num_filter, kernel=3)
        rpn_relu = sym.relu(data=rpn_conv, name='rpn_relu_stride%d%s' % (stride, suffix))
        rpn_cls_score = sym.conv(data=rpn_relu, name='rpn_cls_score_stride%d%s' % (stride, suffix),
                                 num_filter=2 * num_anchors, kernel=1)
        rpn_bbox_pred = sym.conv(data=rpn_relu, name='rpn_bbox_pred_stride%d%s' % (stride, suffix),
                                 num_filter=4 * num_anchors, kernel=1)
        logging.info('rpn_conv_stride{}{}: {}'.format(stride, suffix, rpn_conv.infer_shape(**input_dict)[1]))
        logging.info('rpn_cls_score_stride{}{}: {}'.format(stride, suffix, rpn_cls_score.infer_shape(**input_dict)[1]))
        logging.info('rpn_bbox_pred_stride{}{}: {}'.format(stride, suffix, rpn_bbox_pred.infer_shape(**input_dict)[1]))

        rpn_cls_score_list.append(mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1)))
        rpn_bbox_pred_list.append(mx.symbol.Reshape(data=rpn_bbox_pred, shape=(0, -1)))

        # do softmax
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0))
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel")
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0))
        rpn_cls_prob_dict.update({'cls_prob_stride%d' % stride: rpn_cls_prob_reshape})
        rpn_bbox_pred_dict.update({'bbox_pred_stride%d' % stride: rpn_bbox_pred})

    args_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())
    im_info = kwargs['im_info'] if 'im_info' in kwargs else mx.symbol.Variable(name='im_info')
    rois = proposal_rois_fpn(im_info=im_info,
                             config=config,
                             is_train=is_train,
                             **args_dict)
    logging.info('rois{}: {}'.format(suffix, rois.infer_shape(**input_dict)[1]))

    if is_train:
        rpn_cls_score_concat = mx.symbol.concat(*rpn_cls_score_list, dim=2)
        rpn_bbox_pred_concat = mx.symbol.concat(*rpn_bbox_pred_list, dim=1)
        logging.info('rpn_cls_score_concat{}: {}'.format(suffix, rpn_cls_score_concat.infer_shape(**input_dict)[1]))
        logging.info('rpn_bbox_pred_concat{}: {}'.format(suffix, rpn_bbox_pred_concat.infer_shape(**input_dict)[1]))
        return [rois] + get_rpn_train(rpn_cls_score_concat, rpn_bbox_pred_concat, config, suffix=suffix)
    else:
        return [rois[0], rois[1]]


def get_frcnn_rpn_net(conv_feat, config, is_train=True, suffix='', **kwargs):
    assert config.TRAIN.rpn_cls_loss_type == 'softmax'
    input_dict = kwargs['input_dict']

    num_anchors = len(config.network.rpn_anchor_scales) * len(config.network.rpn_anchor_ratios)
    rpn_conv = sym.conv(data=conv_feat, name='rpn_conv%s' % suffix, num_filter=config.network.rpn_num_filter, kernel=3)
    rpn_relu = sym.relu(data=rpn_conv, name='rpn_relu%s' % suffix)
    rpn_cls_score = sym.conv(data=rpn_relu, name='rpn_cls_score%s' % suffix, num_filter=2 * num_anchors, kernel=1)
    rpn_bbox_pred = sym.conv(data=rpn_relu, name='rpn_bbox_pred%s' % suffix, num_filter=4 * num_anchors, kernel=1)
    logging.info('rpn_cls_score{}: {}'.format(suffix, rpn_cls_score.infer_shape(**input_dict)[1]))
    logging.info('rpn_bbox_pred{}: {}'.format(suffix, rpn_bbox_pred.infer_shape(**input_dict)[1]))

    rpn_cls_score = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0))
    rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score, mode="channel")
    rpn_cls_prob_reshape = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0))

    im_info = kwargs['im_info'] if 'im_info' in kwargs else mx.symbol.Variable(name='im_info')
    rois = proposal_rois(cls_prob=rpn_cls_prob_reshape,
                         bbox_pred=rpn_bbox_pred,
                         im_info=im_info,
                         is_train=is_train,
                         config=config)
    logging.info('rois{}: {}'.format(suffix, rois.infer_shape(**input_dict)[1]))

    if is_train:
        return [rois] + get_rpn_train(rpn_cls_score, rpn_bbox_pred, config, suffix=suffix)
    else:
        return [rois[0], rois[1]]