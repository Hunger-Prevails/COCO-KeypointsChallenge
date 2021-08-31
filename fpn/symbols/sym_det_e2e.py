import mxnet as mx
import logging
from sym_rcnn import get_rcnn_labels
from common.symbols.sym_common import get_sym_func


def get_symbol_rpn(data, get_body, get_rpn_net, config, is_train=True, **kwargs):
    input_dict = kwargs['input_dict']
    assert config.network.rpn_rcnn_num_branch == 1

    rpn_conv_feat, rcnn_conv_feat = get_body(data=data, config=config, is_train=is_train, input_dict=input_dict)

    logging.info('***************rpn subnet****************')
    rpn_group = get_rpn_net(conv_feat=rpn_conv_feat, config=config, is_train=is_train, input_dict=input_dict)

    if is_train:
        group = mx.symbol.Group(rpn_group[1:])
    else:
        group = mx.symbol.Group(rpn_group)
    logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))
    return group


def get_symbol_e2e(data, get_body, get_rpn_net, get_rcnn_net, config, is_train=True, **kwargs):
    input_dict = kwargs['input_dict'].copy()

    rpn_conv_feat, rcnn_conv_feat = get_body(data=data, config=config, is_train=is_train, input_dict=input_dict)

    all_rpn_group = []
    all_rcnn_group = []
    all_test_group = []
    im_info = mx.symbol.Variable(name='im_info')
    for branch_i in range(config.network.rpn_rcnn_num_branch):
        logging.info('***************branch {}****************'.format(branch_i))
        suffix = '_branch{}'.format(branch_i) if config.network.rpn_rcnn_num_branch > 1 else ''
        input_dict_i = kwargs['input_dict'].copy()

        logging.info('***************rpn subnet****************')
        rpn_group = get_rpn_net(conv_feat=rpn_conv_feat, im_info=im_info, config=config, is_train=is_train,
                                suffix=suffix, input_dict=input_dict_i)

        logging.info('**************all labels*****************')
        rcnn_labels = get_rcnn_labels(rpn_group[0], config=config, is_train=is_train, suffix=suffix, input_dict=input_dict_i)

        logging.info('**************rcnn subnet****************')
        rcnn_rois = rcnn_labels[0]
        logging.info('rcnn_rois{}: {}'.format(suffix, rcnn_rois.infer_shape(**input_dict_i)[1]))
        rcnn_group = get_rcnn_net(shared_feat=rcnn_conv_feat, rcnn_rois=rcnn_rois, config=config, is_train=is_train,
                                  rcnn_label=rcnn_labels[1], rcnn_bbox_target=rcnn_labels[2], rcnn_bbox_weight=rcnn_labels[3],
                                  suffix=suffix, input_dict=input_dict_i)
        if is_train:
            all_rpn_group.extend(rpn_group[1:])
            all_rcnn_group.extend(rcnn_group)
        else:
            if config.TEST.rpn_do_test:
                all_test_group.extend([rpn_group[0], rpn_group[1]] + rcnn_group)
            else:
                all_test_group.extend([rcnn_rois] + rcnn_group)
        input_dict.update(input_dict_i)

    group = mx.symbol.Group(all_rpn_group + all_rcnn_group) if is_train else mx.symbol.Group(all_test_group)
    logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))
    return group


def get_symbol(config, is_train=True):
    data = mx.symbol.Variable(name='data')
    input_dict = {'data': (config.TRAIN.image_batch_size if is_train else 1, 3, 800, 1280)}
    logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))

    if 'only_rpn' in config.network.task_type:
        return get_symbol_rpn(data=data,
                              get_body=get_sym_func(config.network.sym_body),
                              get_rpn_net=get_sym_func(config.network.sym_rpn_head),
                              config=config,
                              is_train=is_train,
                              input_dict=input_dict)
    else:
        return get_symbol_e2e(data=data,
                              get_body=get_sym_func(config.network.sym_body),
                              get_rpn_net=get_sym_func(config.network.sym_rpn_head),
                              get_rcnn_net=get_sym_func(config.network.sym_rcnn_head),
                              config=config,
                              is_train=is_train,
                              input_dict=input_dict)


