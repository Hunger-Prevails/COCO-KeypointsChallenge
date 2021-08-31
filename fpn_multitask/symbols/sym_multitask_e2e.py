import mxnet as mx
import logging
from ..operator_py.box_parser import box_parser
from ..operator_py.generate_rois import generate_rois
from sym_multitask_common import get_multitask_labels
from common.symbols.sym_common import get_sym_func


def get_symbol_e2e(data, get_body, get_rpn_net, get_rcnn_net, get_kps_net, get_mask_net, config, is_train=True, **kwargs):
    input_dict = kwargs['input_dict']
    assert config.network.rpn_rcnn_num_branch == 1

    rpn_conv_feat, rcnn_conv_feat = get_body(data=data, config=config, is_train=is_train, input_dict=input_dict)

    logging.info('***************rpn subnet****************')
    im_info = mx.symbol.Variable(name='im_info')
    rpn_group = get_rpn_net(conv_feat=rpn_conv_feat, config=config, is_train=is_train,
                            im_info=im_info, input_dict=input_dict)

    logging.info('**************all labels*****************')
    multitask_labels = get_multitask_labels(rpn_group[0], config, is_train, input_dict=input_dict)

    logging.info('**************rcnn subnet****************')
    rcnn_labels = multitask_labels['rcnn_labels']
    rcnn_rois = rcnn_labels[0]
    logging.info('rcnn_rois: {}'.format(rcnn_rois.infer_shape(**input_dict)[1]))
    rcnn_group = get_rcnn_net(shared_feat=rcnn_conv_feat, rcnn_rois=rcnn_rois, config=config, is_train=is_train,
                              rcnn_label=rcnn_labels[1], rcnn_bbox_target=rcnn_labels[2], rcnn_bbox_weight=rcnn_labels[3],
                              input_dict=input_dict)

    if not is_train:
        if config.TEST.use_gt_rois:
            rcnn_rois = mx.sym.Variable(name='rois')
            input_dict['rois'] = (1, 8, 5)
            rcnn_rois = mx.sym.Reshape(data=rcnn_rois, shape=(-1, 5))
        else:
            rcnn_rois = box_parser(rois=rcnn_rois, bbox_deltas=rcnn_group[1], cls_prob=rcnn_group[0],
                                   im_info=im_info, bbox_class_agnostic=True, config=config)
        rcnn_group = [rcnn_group[0]]

    kps_group = []
    if 'kps' in config.network.task_type:
        logging.info('**************kps subnet****************')
        kps_labels = multitask_labels['kps_labels']
        if is_train:
            kps_rois = kps_labels[0]
        else:
            if config.TEST.aug_strategy.kps_do_aspect_ratio:
                kps_rois = generate_rois(rois=rcnn_rois,
                                         im_info=im_info,
                                         rescale_factor=config.TEST.aug_strategy.kps_rescale_factor,
                                         jitter_center=False,
                                         aspect_ratio=float(config.network.kps_height) / config.network.kps_width,
                                         compute_area=config.network.kps_compute_area,
                                         do_clip=False)
            else:
                kps_rois = rcnn_rois
        logging.info('kps_rois: {}'.format(kps_rois.infer_shape(**input_dict)[1]))
        kps_group = get_kps_net(shared_feat=rcnn_conv_feat, kps_rois=kps_rois, config=config, is_train=is_train,
                                kps_label=kps_labels[1], kps_label_weight=kps_labels[2],
                                kps_pos_offset=kps_labels[3], kps_pos_offset_weight=kps_labels[4],
                                input_dict=input_dict.copy())
        if not is_train:
            kps_group = [mx.sym.identity(kps_rois)] + kps_group

    mask_group = []
    if 'mask' in config.network.task_type:
        logging.info('**************mask subnet****************')
        mask_labels = multitask_labels['mask_labels']
        if is_train:
            mask_rois = mask_labels[0]
        else:
            mask_rois = rcnn_rois
            # mask_rois = generate_rois(rois=rcnn_rois,
            #                           im_info=im_info,
            #                           rescale_factor=0.0,
            #                           jitter_center=False,
            #                           aspect_ratio=0.0,
            #                           compute_area=True,
            #                           do_clip=True)
        logging.info('mask_rois: {}'.format(mask_rois.infer_shape(**input_dict)[1]))
        mask_group = get_mask_net(shared_feat=rcnn_conv_feat, mask_rois=mask_rois, config=config, is_train=is_train,
                                  mask_label=mask_labels[1], rcnn_label=rcnn_labels[1], input_dict=input_dict)
        if not is_train:
            mask_group = [mx.sym.identity(mask_rois)] + mask_group

    if is_train:
        group = rpn_group[1:] + rcnn_group + kps_group + mask_group
    else:
        group = [rcnn_rois] + rcnn_group + kps_group + mask_group

    group = mx.symbol.Group(group)
    logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))
    return group


def get_symbol(config, is_train=True):
    data = mx.symbol.Variable(name='data')
    input_dict = {'data': (config.TRAIN.image_batch_size if is_train else 1, 3, 800, 1280)}
    logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))

    return get_symbol_e2e(data=data,
                          get_body=get_sym_func(config.network.sym_body),
                          get_rpn_net=get_sym_func(config.network.sym_rpn_head),
                          get_rcnn_net=get_sym_func(config.network.sym_rcnn_head),
                          get_kps_net=get_sym_func(config.network.sym_kps_head),
                          get_mask_net=get_sym_func(config.network.sym_mask_head),
                          config=config,
                          is_train=is_train,
                          input_dict=input_dict)














