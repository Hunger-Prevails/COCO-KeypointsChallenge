import mxnet as mx
import logging
from common.symbols import sym_common as sym
from common.operator_py.sigmoid_cross_entropy_loss import sigmoid_cross_entropy_loss
from fpn.symbols.sym_rcnn import extract_roi


def get_mask_grad_scales(config):
    grad_scale_0 = float(config.TRAIN.mask_loss_weights[0])
    return [grad_scale_0, ]


def get_mask_eval_info_list(config):
    grad_scales = get_mask_grad_scales(config)
    eval_info_list = []
    eval_info = dict()
    eval_info['metric_type'] = 'CrossEntropy'
    eval_info['metric_name'] = 'MaskLabel'
    eval_info['grad_scale'] = grad_scales[0]
    eval_info_list.append(eval_info)
    return eval_info_list


def get_mask_train_test(mask_label_pred, config, is_train=True, mask_label=None, rcnn_label=None, **kwargs):
    input_dict = kwargs['input_dict']
    num_classes = config.dataset.num_classes
    if is_train:
        if num_classes > 2:
            assert rcnn_label is not None
            rcnn_label_reshape = mx.sym.Reshape(data=rcnn_label, shape=(-1, 1, 1, 1))
            mask_label_pred = mx.contrib.sym.ChannelOperator(data=mask_label_pred, name='mask_label_pred_pick', pick_idx=rcnn_label_reshape,
                                                             group=num_classes, op_type='Group_Pick', pick_type='Label_Pick')
            logging.info('mask_label_pred: {}'.format(mask_label_pred.infer_shape(**input_dict)[1]))
        grad_scales = get_mask_grad_scales(config)
        mask_prob = sigmoid_cross_entropy_loss(data=mask_label_pred, label=mask_label,
                                               use_ignore=True, ignore_label=-1,
                                               grad_scale=grad_scales[0])
        mask_label = mx.sym.Reshape(data=mask_label, shape=(config.TRAIN.image_batch_size, -1, 0, 0))
        mask_prob = mx.sym.Reshape(data=mask_prob, shape=(config.TRAIN.image_batch_size, -1, 0, 0))
        return [mx.sym.BlockGrad(mask_label), mask_prob]
    else:
        mask_prob = sym.sigmoid(data=mask_label_pred, name='mask_label_pred_sigmoid')
        return [mask_prob, ]


def get_mask_extract_roi(shared_feat, mask_rois, config, pooled_size, is_train=True, **kwargs):
    return extract_roi(shared_feat=shared_feat,
                       rois=mask_rois,
                       name='mask_feat_rois',
                       roi_extract_method=config.network.roi_extract_method,
                       pooled_size=pooled_size,
                       feat_stride=config.network.rcnn_feat_stride,
                       batch_size=config.TRAIN.image_batch_size if is_train else 1,
                       input_dict=kwargs['input_dict'])


def get_mask_subnet(shared_feat, mask_rois, config, is_train=True, **kwargs):
    pooled_size = (config.network.mask_pooled_size[0] / 2, config.network.mask_pooled_size[1] / 2)
    conv_feat = get_mask_extract_roi(shared_feat, mask_rois, config, pooled_size, is_train, **kwargs)

    for i in range(4):
        _, conv_feat = sym.convrelu(data=conv_feat, num_filter=256, kernel=3, prefix='mask_', suffix='%d' % (i + 1))
    conv_feat = sym.deconv(data=conv_feat, name='mask_up_deconv', num_filter=256, kernel=2, stride=2)
    conv_feat = sym.relu(data=conv_feat, name='mask_up_relu')
    logging.info('mask_up_deconv: {}'.format(conv_feat.infer_shape(**kwargs['input_dict'])[1]))

    num_filter = 1 if config.dataset.num_classes == 2 else config.dataset.num_classes
    mask_label_pred = sym.conv(data=conv_feat, name='mask_label_pred', num_filter=num_filter, kernel=1)
    logging.info('mask_label_pred: {}'.format(mask_label_pred.infer_shape(**kwargs['input_dict'])[1]))

    return get_mask_train_test(mask_label_pred=mask_label_pred, config=config, is_train=is_train, **kwargs)