import mxnet as mx
import logging
from common.symbols import sym_common as sym


def _get_fpn_feature(conv_feat, feature_dim=256):
    res5 = conv_feat['stride32']
    res4 = conv_feat['stride16']
    res3 = conv_feat['stride8']
    res2 = conv_feat['stride4']
    # lateral connection
    fpn_p5_1x1 = sym.conv(data=res5, name='fpn_p5_1x1', num_filter=feature_dim, kernel=1)
    fpn_p4_1x1 = sym.conv(data=res4, name='fpn_p4_1x1', num_filter=feature_dim, kernel=1)
    fpn_p3_1x1 = sym.conv(data=res3, name='fpn_p3_1x1', num_filter=feature_dim, kernel=1)
    fpn_p2_1x1 = sym.conv(data=res2, name='fpn_p2_1x1', num_filter=feature_dim, kernel=1)
    # top-down connection
    fpn_p5_upsample = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='nearest', name='fpn_p5_upsample')
    fpn_p4_plus = fpn_p5_upsample + fpn_p4_1x1
    fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_plus, scale=2, sample_type='nearest', name='fpn_p4_upsample')
    fpn_p3_plus = fpn_p4_upsample + fpn_p3_1x1
    fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale=2, sample_type='nearest', name='fpn_p3_upsample')
    fpn_p2_plus = fpn_p3_upsample + fpn_p2_1x1
    # FPN feature
    fpn_p5 = sym.conv(data=fpn_p5_1x1, name='fpn_p5', num_filter=feature_dim, kernel=3, stride=1)
    fpn_p4 = sym.conv(data=fpn_p4_plus, name='fpn_p4', num_filter=feature_dim, kernel=3, stride=1)
    fpn_p3 = sym.conv(data=fpn_p3_plus, name='fpn_p3', num_filter=feature_dim, kernel=3, stride=1)
    fpn_p2 = sym.conv(data=fpn_p2_plus, name='fpn_p2', num_filter=feature_dim, kernel=3, stride=1)

    fpn_conv_feat = {'stride32': fpn_p5, 'stride16': fpn_p4, 'stride8': fpn_p3, 'stride4': fpn_p2}
    return fpn_conv_feat


def get_fpn_conv_feat(data, config, is_train=True, **kwargs):
    input_dict = kwargs['input_dict']
    assert config.network.image_stride == 32
    bn_use_global_stats = config.TRAIN.bn_use_global_stats if is_train else True
    from common.symbols.sym_common import cfg
    if is_train and config.TRAIN.bn_use_sync:
        cfg.bn_use_sync = True
    else:
        cfg.bn_use_sync = False
    if 'resnet' in config.network.net_type:
        from common.symbols.sym_resnet import get_symbol as get_resnet_symbol
        in_layer_list = get_resnet_symbol(data=data,
                                          num_layer=config.network.num_layer,
                                          net_type=config.network.net_type,
                                          inc_dilates=config.network.inc_dilates,
                                          deformable_units=config.network.deformable_units,
                                          num_deformable_group=config.network.num_deformable_group,
                                          bn_use_global_stats=bn_use_global_stats)
    else:
        assert not config.TRAIN.bn_use_sync
        if config.network.net_type == 'mobilenet':
            from common.symbols import sym_mobilenet as net
        elif config.network.net_type == 'mobilenet_res':
            from common.symbols import sym_mobilenet_res as net
        elif config.network.net_type == 'mobilenet_v2_w_bypass':
            from common.symbols import sym_mobilenet_v2_w_bypass as net
            net.mirroring_level = config.TRAIN.mirroring_level
        elif config.network.net_type == 'mobilenet_v2_wo_bypass':
            from common.symbols import sym_mobilenet_v2_wo_bypass as net
        elif config.network.net_type == 'tiny_xception_like':
            from common.symbols import sym_tiny_xception as net
        else:
            raise ValueError("unknown net_type {}".format(config.network.net_type))
        net.use_global_stats = bn_use_global_stats
        in_layer_list = net.get_symbol(data=data, inv_resolution=32)

    conv_feat = {'stride4': in_layer_list[0], 'stride8': in_layer_list[1],
                 'stride16': in_layer_list[2], 'stride32': in_layer_list[3]}
    for stride in [4, 8, 16, 32]:
        logging.info('conv_feat_stride{}: {}'.format(stride, conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))
    all_conv_feat = _get_fpn_feature(conv_feat)

    for stride in config.network.rpn_feat_stride:
        logging.info('rpn_conv_feat_stride{}: {}'.format(stride, all_conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))
    for stride in config.network.rcnn_feat_stride:
        logging.info('rcnn_conv_feat_stride{}: {}'.format(stride, all_conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))

    return all_conv_feat, all_conv_feat


def get_frcnn_conv_feat(data, config, is_train=True, **kwargs):
    input_dict = kwargs['input_dict']
    bn_use_global_stats = config.TRAIN.bn_use_global_stats if is_train else True
    from common.symbols.sym_common import cfg
    if is_train and config.TRAIN.bn_use_sync:
        cfg.bn_use_sync = True
    else:
        cfg.bn_use_sync = False
    if 'resnet' in config.network.net_type:
        from common.symbols.sym_resnet import get_symbol
        in_layer_list = get_symbol(data=data,
                                   num_layer=config.network.num_layer,
                                   net_type=config.network.net_type,
                                   inv_resolution=16,
                                   deformable_units=config.network.deformable_units,
                                   num_deformable_group=config.network.num_deformable_group,
                                   bn_use_global_stats=bn_use_global_stats,
                                   input_dict=input_dict)
    else:
        if config.network.net_type == 'mobilenet':
            from common.symbols import sym_mobilenet as net
        elif config.network.net_type == 'mobilenet_res':
            from common.symbols import sym_mobilenet_res as net
        elif config.network.net_type == 'mobilenet_v2_w_bypass':
            from common.symbols import sym_mobilenet_v2_w_bypass as net
        elif config.network.net_type == 'mobilenet_v2_wo_bypass':
            from common.symbols import sym_mobilenet_v2_wo_bypass as net
        elif config.network.net_type == 'tiny_xception_like':
            from common.symbols import sym_tiny_xception as net
        else:
            raise ValueError("unknown net_type {}".format(config.network.net_type))
        net.use_global_stats = bn_use_global_stats
        in_layer_list = net.get_symbol(data=data, inv_resolution=16, input_dict=input_dict)
    res4 = in_layer_list[2]
    res5 = in_layer_list[-1]
    conv_new = sym.conv(data=res5, name='rcnn_conv_new', num_filter=256, kernel=1)
    conv_new_relu = sym.relu(data=conv_new, name='rcnn_conv_new_relu')

    logging.info('rpn_conv_feat: {}'.format(res4.infer_shape(**input_dict)[1]))
    logging.info('rcnn_conv_feat: {}'.format(conv_new_relu.infer_shape(**input_dict)[1]))
    return res4, conv_new_relu