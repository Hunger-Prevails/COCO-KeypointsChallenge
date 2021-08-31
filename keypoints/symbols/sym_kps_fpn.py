import mxnet as mx
import logging
from common.symbols import sym_common as sym

def _get_fpn_feature(conv_feat, feature_dim=256):
    res5 = conv_feat['stride32']
    res4 = conv_feat['stride16']
    res3 = conv_feat['stride8']
    res2 = conv_feat['stride4']
    
    # lateral connection
    fpn_p5_1x1 = sym.bnreluconv(data=res5, prefix='fpn_', suffix='_p5_1x1', num_filter=feature_dim, kernel=1)[2]
    fpn_p4_1x1 = sym.bnreluconv(data=res4, prefix='fpn_', suffix='_p4_1x1', num_filter=feature_dim, kernel=1)[2]
    fpn_p3_1x1 = sym.bnreluconv(data=res3, prefix='fpn_', suffix='_p3_1x1', num_filter=feature_dim, kernel=1)[2]
    fpn_p2_1x1 = sym.bnreluconv(data=res2, prefix='fpn_', suffix='_p2_1x1', num_filter=feature_dim, kernel=1)[2]

    # top-down connection
    fpn_p5_up = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='nearest', name='fpn_p5_up')
    fpn_p4 = fpn_p5_up + fpn_p4_1x1
    fpn_p4_up = mx.symbol.UpSampling(fpn_p4, scale=2, sample_type='nearest', name='fpn_p4_up')
    fpn_p3 = fpn_p4_up + fpn_p3_1x1
    fpn_p3_up = mx.symbol.UpSampling(fpn_p3, scale=2, sample_type='nearest', name='fpn_p3_up')
    fpn_p2 = fpn_p3_up + fpn_p2_1x1

    return {'stride32': fpn_p5_1x1, 'stride16': fpn_p4, 'stride8': fpn_p3, 'stride4': fpn_p2}


def get_fpn_conv_feat(data, config, is_train=True, **kwargs):
    input_dict = kwargs['input_dict']
    assert config.network.image_stride == 32
    
    from common.symbols.sym_common import cfg
    cfg.bn_use_global_stats = config.TRAIN.bn_use_global_stats if is_train else True
    cfg.bn_use_sync = config.TRAIN.bn_use_sync if is_train else False

    if 'resnet' in config.network.net_type:
        from common.symbols.sym_resnet import get_symbol
        in_layer_list = get_symbol(
                        data = data,
                        num_layer = config.network.num_layer,
                        net_type = config.network.net_type,
                        inc_dilates = config.network.inc_dilates,
                        deformable_units = config.network.deformable_units,
                        num_deformable_group = config.network.num_deformable_group)

    elif 'senet' in config.network.net_type:
        from common.symbols.sym_senet import get_symbol
        in_layer_list = get_symbol(data = data, net_depth = config.network.num_layer, num_group = config.network.num_groups)

    elif 'resnext' in config.network.net_type:
        from common.symbols.sym_resnext import get_symbol
        in_layer_list = get_symbol(data = data, net_depth = config.network.num_layer, num_group = config.network.num_groups)

    elif 'xception' in config.network.net_type:
        from common.symbols.sym_xception import get_symbol
        in_layer_list = get_symbol(data = data, inc_dilates = config.network.inc_dilates)

    elif 'dpn' in config.network.net_type:
        from common.symbols.sym_dpn import get_symbol
        in_layer_list = get_symbol(data = data, inc_dilates = config.network.inc_dilates)

    else:
        raise ValueError("unknown net_type {}".format(config.network.net_type))
        
    conv_feat = {'stride4': in_layer_list[0], 'stride8': in_layer_list[1],
                 'stride16': in_layer_list[2], 'stride32': in_layer_list[3]}

    for stride in [4, 8, 16, 32]:
        logging.info('conv_feat_stride{}: {}'.format(stride, conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))

    fpn_conv_feat = _get_fpn_feature(conv_feat)

    for stride in [4, 8, 16, 32]:
        logging.info('fpn_conv_feat_stride{}: {}'.format(stride, fpn_conv_feat['stride%d' % stride].infer_shape(**input_dict)[1]))

    return conv_feat, fpn_conv_feat
