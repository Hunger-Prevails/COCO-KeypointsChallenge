import mxnet as mx
import logging
from common.symbols import sym_common as sym
from sym_kps_common import kps_get_output, kps_aux_output
from common.symbols.sym_common import get_sym_func


def get_conv_feat_res4(data, config, is_train):
    assert config.network.net_type == 'resnet'

    from common.symbols.sym_common import cfg
    cfg.bn_use_global_stats = config.TRAIN.bn_use_global_stats if is_train else True
    cfg.bn_use_sync = config.TRAIN.bn_use_sync if is_train else False

    from common.symbols.sym_resnet import get_symbol
    in_layer_list = get_symbol(
                        data = data,
                        num_layer = config.network.num_layer,
                        net_type = config.network.net_type,
                        inc_dilates = config.network.inc_dilates,
                        deformable_units = config.network.deformable_units,
                        num_deformable_group = config.network.num_deformable_group)

    res2 = in_layer_list[0]
    res3 = in_layer_list[1]
    res4 = in_layer_list[2]
    res5 = in_layer_list[3]
    return res4


def get_conv_feat_res5(data, config, is_train):
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

    res2 = in_layer_list[0]
    res3 = in_layer_list[1]
    res4 = in_layer_list[2]
    res5 = in_layer_list[3]
    return res5


def get_symbol(config, is_train = True):
    data = mx.sym.Variable(name = 'data')
    rois = mx.sym.Variable(name = 'rois')
    rois = mx.sym.Reshape(data = rois, name = 'rois_reshape', shape=(-1, 5))
    if config.network.kps_crop_from_image:
        input_dict = {'data': (config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size, 3,
                               config.network.kps_input_height, config.network.kps_input_width)}
    else:
        input_dict = {'data': (config.TRAIN.image_batch_size, 3, 512, 512),
                      'rois': (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, 5)}
        logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))
        from common.operator_py.crop_and_resize_image import crop_and_resize_image
        data = crop_and_resize_image(
                    data = data,
                    rois = rois,
                    output_height = config.network.kps_input_height,
                    output_width = config.network.kps_input_width)

    logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))

    conv_feat = get_sym_func(config.network.sym_body)(data = data, config = config, is_train = is_train)

    if config.network.appends_bn:
        conv_feat = sym.bn(data = conv_feat, name = 'bn1', fix_gamma = False)
        conv_feat = sym.relu(data = conv_feat, name = 'relu1')
    logging.info('conv_feat: {}'.format(conv_feat.infer_shape(**input_dict)[1]))

    conv_feat = sym.conv(data = conv_feat, name = 'conv_feat', num_filter = config.network.kps_num_filter, kernel = 3)
    conv_feat = sym.relu(data = conv_feat, name = 'conv_feat_relu')
    logging.info('conv_feat: {}'.format(conv_feat.infer_shape(**input_dict)[1]))

    # predicate and make loss
    assert len(config.TRAIN.kps_loss_type_list) == 1
        
    label_estimate = sym.conv(data = conv_feat, name = 'label_estimate', num_filter = config.dataset.num_kps, kernel = 1)
    logging.info('label_estimate: {}'.format(label_estimate.infer_shape(**input_dict)[1]))
    
    offset_estimate = sym.conv(data = conv_feat, name = 'offset_estimate', num_filter = config.dataset.num_kps * 2, kernel = 1)
    logging.info('offset_estimate: {}'.format(offset_estimate.infer_shape(**input_dict)[1]))
        
    group_list = kps_get_output(
                    label_estimate = label_estimate,
                    offset_estimate = offset_estimate,
                    config = config,
                    is_train = is_train,
                    input_dict = input_dict)

    if not is_train:
        input_dict['rois'] = (config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size, 5)
        group_list = [rois] + group_list

    group = mx.symbol.Group(group_list)
    logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))
    return group

def deconv_symbol(config, is_train = True):
    assert config.network.kps_feat_stride in (4, 8)

    data = mx.sym.Variable(name = 'data')
    rois = mx.sym.Variable(name = 'rois')
    rois = mx.sym.Reshape(data = rois, name = 'rois_reshape', shape = (-1, 5))

    if config.network.kps_crop_from_image:
        input_dict = {'data': (config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size, 3,
                               config.network.kps_input_height, config.network.kps_input_width)}
    else:
        input_dict = {'data': (config.TRAIN.image_batch_size, 3, 512, 512),
                      'rois': (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, 5)}
        logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))

        from common.operator_py.crop_and_resize_image import crop_and_resize_image
        data = crop_and_resize_image(
                    data = data,
                    rois = rois,
                    output_height = config.network.kps_input_height,
                    output_width = config.network.kps_input_width)

    logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))

    conv_feat = get_sym_func(config.network.sym_body)(data = data, config = config, is_train = is_train)
    if config.network.appends_bn:
        conv_feat = sym.bn(data = conv_feat, name = 'bn_conv_feat', fix_gamma = False)
        conv_feat = sym.relu(data = conv_feat, name = 'relu_conv_feat')

    logging.info('conv_feat: {}'.format(conv_feat.infer_shape(**input_dict)[1]))

    deconv_feat = sym.deconv(
                        data = conv_feat,
                        name = 'deconv_a',
                        num_filter = config.network.kps_num_filter,
                        kernel = 4,
                        stride = 2,
                        pad = 1)
    deconv_feat = sym.bn(data = deconv_feat, name = 'bn_deconv_a', fix_gamma = False)
    deconv_feat = sym.relu(data = deconv_feat, name = 'relu_deconv_a')

    logging.info('deconv_feat: {}'.format(deconv_feat.infer_shape(**input_dict)[1]))

    deconv_feat = sym.deconv(
                        data = deconv_feat,
                        name = 'deconv_b',
                        num_filter = config.network.kps_num_filter,
                        kernel = 4,
                        stride = 2,
                        pad = 1)
    deconv_feat = sym.bn(data = deconv_feat, name = 'bn_deconv_b', fix_gamma = False)
    deconv_feat = sym.relu(data = deconv_feat, name = 'relu_deconv_b')

    logging.info('deconv_feat: {}'.format(deconv_feat.infer_shape(**input_dict)[1]))

    if config.network.kps_feat_stride == 4:
        deconv_feat = sym.deconv(
                            data = deconv_feat,
                            name = 'deconv_c',
                            num_filter = config.network.kps_num_filter,
                            kernel = 4,
                            stride = 2,
                            pad = 1)
        deconv_feat = sym.bn(data = deconv_feat, name = 'bn_deconv_c', fix_gamma = False)
        deconv_feat = sym.relu(data = deconv_feat, name = 'relu_deconv_c')

        logging.info('deconv_feat: {}'.format(deconv_feat.infer_shape(**input_dict)[1]))

    # predicate and make loss
    assert len(config.TRAIN.kps_loss_type_list) == 1
    
    label_estimate = sym.conv(data = deconv_feat, name = 'label_estimate', num_filter = config.dataset.num_kps, kernel = 1)
    logging.info('label_estimate: {}'.format(label_estimate.infer_shape(**input_dict)[1]))
    
    offset_estimate = sym.conv(data = deconv_feat, name = 'offset_estimate', num_filter = config.dataset.num_kps * 2, kernel = 1)
    logging.info('offset_estimate: {}'.format(offset_estimate.infer_shape(**input_dict)[1]))
    
    group_list = kps_get_output(
                    label_estimate = label_estimate,
                    offset_estimate = offset_estimate,
                    config = config,
                    is_train = is_train,
                    input_dict = input_dict)

    if 'aux_suv' in config.TRAIN.kps_loss_type and is_train:
        shallow_feat = sym.bn(data = conv_feat, name='bn_shallow_' + str(32), fix_gamma=False)
        shallow_feat = sym.relu(data = shallow_feat, name='relu_shallow_' + str(32))

        label_estimate = sym.conv(data = shallow_feat, name = 'label_estimate_' + str(32), num_filter = config.dataset.num_kps, kernel = 1)
        offset_estimate = sym.conv(data = shallow_feat, name = 'offset_estimate_' + str(32), num_filter = config.dataset.num_kps * 2, kernel = 1)

        logging.info('label_estimate_' + str(32) + ': ' + str(label_estimate.infer_shape(**input_dict)[1]))
        logging.info('offset_estimate_' + str(32) + ': ' + str(offset_estimate.infer_shape(**input_dict)[1]))
    
        group_list += kps_aux_output(
                    label_estimate = label_estimate,
                    offset_estimate = offset_estimate,
                    config = config,
                    aux_stride = 32,
                    input_dict = input_dict)

    if not is_train:
        input_dict['rois'] = (config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size, 5)
        group_list = [rois] + group_list

    group = mx.symbol.Group(group_list)
    logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))
    return group

def fpn_symbol(config, is_train = True):
    assert config.network.kps_feat_stride in (4, 8)

    data = mx.sym.Variable(name = 'data')
    rois = mx.sym.Variable(name = 'rois')
    rois = mx.sym.Reshape(data = rois, name = 'rois_reshape', shape = (-1, 5))
    if config.network.kps_crop_from_image:
        input_dict = {'data': (config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size, 3,
                               config.network.kps_input_height, config.network.kps_input_width)}
    else:
        input_dict = {'data': (config.TRAIN.image_batch_size, 3, 512, 512),
                      'rois': (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, 5)}
        logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))
        from common.operator_py.crop_and_resize_image import crop_and_resize_image
        data = crop_and_resize_image(
                    data = data,
                    rois = rois,
                    output_height = config.network.kps_input_height,
                    output_width = config.network.kps_input_width)

    logging.info('data: {}'.format(data.infer_shape(**input_dict)[1]))

    conv_feat, fpn_conv_feat = get_sym_func(config.network.sym_body)(data = data, config = config, is_train = is_train, input_dict = input_dict)

    fpn_p5 = fpn_conv_feat['stride32']
    fpn_p4 = fpn_conv_feat['stride16']
    fpn_p3 = fpn_conv_feat['stride8']
    fpn_p2 = fpn_conv_feat['stride4']

    if config.network.kps_feat_stride == 4:
        feat_p5 = sym.deconv(data=fpn_p5, name='deconv_p5_a', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p5 = sym.bn(data=feat_p5, name='bn_p5_a', fix_gamma=False)
        feat_p5 = sym.relu(data=feat_p5, name='relu_p5_a')

        feat_p5 = sym.deconv(data=feat_p5, name='deconv_p5_b', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p5 = sym.bn(data=feat_p5, name='bn_p5_b', fix_gamma=False)
        feat_p5 = sym.relu(data=feat_p5, name='relu_p5_b')

        feat_p5 = sym.deconv(data=feat_p5, name='deconv_p5_c', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p5 = sym.bn(data=feat_p5, name='bn_p5_c', fix_gamma=False)
        feat_p5 = sym.relu(data=feat_p5, name='relu_p5_c')

        logging.info('feat_p5: {}'.format(feat_p5.infer_shape(**input_dict)[1]))

        feat_p4 = sym.deconv(data=fpn_p4, name='deconv_p4_a', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p4 = sym.bn(data=feat_p4, name='bn_p4_a', fix_gamma=False)
        feat_p4 = sym.relu(data=feat_p4, name='relu_p4_a')

        feat_p4 = sym.deconv(data=feat_p4, name='deconv_p4_b', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p4 = sym.bn(data=feat_p4, name='bn_p4_b', fix_gamma=False)
        feat_p4 = sym.relu(data=feat_p4, name='relu_p4_b')

        logging.info('feat_p4: {}'.format(feat_p4.infer_shape(**input_dict)[1]))

        feat_p3 = sym.deconv(data=fpn_p3, name='deconv_p3', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p3 = sym.bn(data=feat_p3, name='bn_p3', fix_gamma=False)
        feat_p3 = sym.relu(data=feat_p3, name='relu_p3')

        logging.info('feat_p3: {}'.format(feat_p3.infer_shape(**input_dict)[1]))

        feat_p2 = sym.convbnrelu(data=fpn_p2, prefix='feat_', suffix='_p2', num_filter=128, kernel=3)[2];

        logging.info('feat_p2: {}'.format(feat_p2.infer_shape(**input_dict)[1]))

    else:
        feat_p5 = sym.deconv(data=fpn_p5, name='deconv_p5_a', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p5 = sym.bn(data=feat_p5, name='bn_p5_a', fix_gamma=False)
        feat_p5 = sym.relu(data=feat_p5, name='relu_p5_a')

        feat_p5 = sym.deconv(data=feat_p5, name='deconv_p5_b', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p5 = sym.bn(data=feat_p5, name='bn_p5_b', fix_gamma=False)
        feat_p5 = sym.relu(data=feat_p5, name='relu_p5_b')

        logging.info('feat_p5: {}'.format(feat_p5.infer_shape(**input_dict)[1]))

        feat_p4 = sym.deconv(data=fpn_p4, name='deconv_p4_a', num_filter=128, kernel=4, stride=2, pad=1);
        feat_p4 = sym.bn(data=feat_p4, name='bn_p4_a', fix_gamma=False)
        feat_p4 = sym.relu(data=feat_p4, name='relu_p4_a')

        logging.info('feat_p4: {}'.format(feat_p4.infer_shape(**input_dict)[1]))

        feat_p3 = sym.convbnrelu(data=fpn_p3, prefix='feat_', suffix='_p3', num_filter=128, kernel=3)[2]

        logging.info('feat_p3: {}'.format(feat_p3.infer_shape(**input_dict)[1]))

        feat_p2 = sym.convbnrelu(data=fpn_p2, prefix='feat_', suffix='_p2', num_filter=128, kernel=4, stride=2, pad=1)[2];

        logging.info('feat_p2: {}'.format(feat_p2.infer_shape(**input_dict)[1]))

    featmaps = [feat_p5, feat_p4, feat_p3, feat_p2]

    concat_feat = mx.sym.Concat(*featmaps, dim=1, auto_channel_switch=False)
    logging.info('concat_feat: {}'.format(concat_feat.infer_shape(**input_dict)[1]))

    # predicate and make loss
    assert len(config.TRAIN.kps_loss_type_list) == 1
    
    label_estimate = sym.conv(data = concat_feat, name = 'label_estimate', num_filter = config.dataset.num_kps, kernel = 1)
    offset_estimate = sym.conv(data = concat_feat, name = 'offset_estimate', num_filter = config.dataset.num_kps * 2, kernel = 1)

    logging.info('label_estimate: {}'.format(label_estimate.infer_shape(**input_dict)[1]))
    logging.info('offset_estimate: {}'.format(offset_estimate.infer_shape(**input_dict)[1]))
    
    group_list = kps_get_output(
                    label_estimate = label_estimate,
                    offset_estimate = offset_estimate,
                    config = config,
                    is_train = is_train,
                    input_dict = input_dict)

    if 'aux_suv' in config.TRAIN.kps_loss_type and is_train:
        shallow_feat = fpn_conv_feat['stride' + str(4)]
        shallow_feat = sym.bn(data = shallow_feat, name='bn_shallow_' + str(4), fix_gamma=False)
        shallow_feat = sym.relu(data = shallow_feat, name='relu_shallow_' + str(4))

        label_estimate = sym.conv(data = shallow_feat, name = 'label_estimate_' + str(4), num_filter = config.dataset.num_kps, kernel = 1)
        offset_estimate = sym.conv(data = shallow_feat, name = 'offset_estimate_' + str(4), num_filter = config.dataset.num_kps * 2, kernel = 1)

        logging.info('label_estimate_' + str(4) + ': ' + str(label_estimate.infer_shape(**input_dict)[1]))
        logging.info('offset_estimate_' + str(4) + ': ' + str(offset_estimate.infer_shape(**input_dict)[1]))
    
        group_list += kps_aux_output(
                    label_estimate = label_estimate,
                    offset_estimate = offset_estimate,
                    config = config,
                    aux_stride = 4,
                    input_dict = input_dict)

    if not is_train:
        input_dict['rois'] = (config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size, 5)
        group_list = [rois] + group_list

    group = mx.symbol.Group(group_list)
    logging.info('group: {}'.format(group.infer_shape(**input_dict)[1]))
    return group
