import mxnet as mx
from common.symbols import sym_common as sym


def get_kps_grad_scales(config):
    kps_height = config.network.kps_height
    kps_width = config.network.kps_width
    num_kps = config.dataset.num_kps
    kps_loss_type = config.TRAIN.kps_loss_type
    batch_size = config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size
    if 'one_hot' in kps_loss_type:
        grad_scale_0 = config.TRAIN.kps_loss_weights[0]
        grad_scale_1 = config.TRAIN.kps_loss_weights[1] / batch_size
    else:
        grad_scale_0 = config.TRAIN.kps_loss_weights[0] / (batch_size * num_kps)
        grad_scale_1 = config.TRAIN.kps_loss_weights[1] / (batch_size * num_kps)
    return [grad_scale_0, grad_scale_1]


def get_kps_eval_info_list(config):
    kps_loss_type = config.TRAIN.kps_loss_type
    grad_scales = get_kps_grad_scales(config)
    eval_info_list = []
    if 'one_hot' in kps_loss_type:
        if 'softmax' in kps_loss_type:
            eval_info = dict()
            eval_info['metric_type'] = 'Softmax'
            eval_info['metric_name'] = 'KPSLabel'
            eval_info['grad_scale'] = grad_scales[0]
            eval_info['axis'] = 2
            eval_info_list.append(eval_info)
        else:
            raise ValueError("unknown kps loss type {}".format(kps_loss_type))
    else:
        eval_info = dict()
        eval_info['metric_type'] = 'Sum'
        eval_info['metric_name'] = 'KPSLabel'
        eval_info['grad_scale'] = grad_scales[0]
        eval_info_list.append(eval_info)

    eval_info = dict()
    eval_info['metric_type'] = 'Sum'
    eval_info['metric_name'] = 'KPSPosOffset'
    eval_info['grad_scale'] = grad_scales[1]
    eval_info_list.append(eval_info)

    if 'contrast' in kps_loss_type:
        eval_info = dict()
        eval_info['metric_type'] = 'Sum'
        eval_info['metric_name'] = 'KPSContrast'
        eval_info['grad_scale'] = grad_scales[1]
        eval_info_list.append(eval_info)

    return eval_info_list


def kps_train(label_estimate, kps_label, label_mask, offset_estimate, kps_offset, offset_mask, config, aux_stride = None):
    if aux_stride:
        kps_height = config.network.kps_input_height / aux_stride
        kps_width = config.network.kps_input_width / aux_stride
    else:
        kps_height = config.network.kps_height
        kps_width = config.network.kps_width

    num_kps = config.dataset.num_kps
    kps_loss_type = config.TRAIN.kps_loss_type

    grad_scales = get_kps_grad_scales(config)

    if 'one_hot' in kps_loss_type:
        label_estimate = mx.sym.Reshape(data = label_estimate, shape = (-1, kps_height * kps_width, 1, 1))
        kps_label = mx.sym.Reshape(data = kps_label, shape = (-1, 1, 1, 1))
    else:
        kps_label = mx.sym.Reshape(data = kps_label, shape = (-1, num_kps, kps_height, kps_width))
        label_mask = mx.sym.Reshape(data = label_mask, shape = (-1, num_kps, kps_height, kps_width))

    kps_offset = mx.sym.Reshape(data = kps_offset, shape = (-1, num_kps * 2, kps_height, kps_width))
    offset_mask = mx.sym.Reshape(data = offset_mask, shape = (-1, num_kps * 2, kps_height, kps_width))

    if 'one_hot' in kps_loss_type:
        if 'softmax' in kps_loss_type:
            kps_label_loss = mx.sym.SoftmaxOutput(
                                data = label_estimate,
                                label = kps_label,
                                multi_output = True,
                                normalization = 'valid',
                                use_ignore = True,
                                ignore_label = -1,
                                grad_scale = grad_scales[0])
        else:
            raise ValueError("unknown kps loss type {}".format(kps_loss_type))
    else:
        if 'smooth_L1' in kps_loss_type:
            kps_label_loss_t = label_mask * mx.sym.smooth_l1(data = (label_estimate - kps_label), scalar = config.TRAIN.kps_scalar_L1)
        elif 'L2' in kps_loss_type:
            kps_label_loss_t = label_mask * 0.5 * mx.sym.square(label_estimate - kps_label)
        else:
            raise ValueError("unknown kps loss type {}".format(kps_loss_type))

        kps_label_loss = mx.sym.MakeLoss(data = kps_label_loss_t, grad_scale = grad_scales[0])

        if 'contrast' in kps_loss_type:
            margin = config.TRAIN.kps_contra_margin

            keep_tag = sym.pool(data = kps_label, name = 'keep_tag', pool_type = 'max', global_pool = True)  # (num_boxes, num_kps, 1, 1)
            central_peak = sym.pool(data = (label_estimate * kps_label), name = 'central_pool', pool_type = 'max', global_pool = True)  # (num_boxes, num_kps, 1, 1)
            global_peak = sym.pool(data = (label_estimate * (1 - kps_label)), name = 'global_pool', pool_type = 'max', global_pool = True)  # (num_boxes, num_kps, 1, 1)

            contrast_loss_t = mx.sym.relu(margin - (central_peak - global_peak)) * keep_tag
            contrast_loss = mx.sym.MakeLoss(data = contrast_loss_t, grad_scale = grad_scales[1])
    
    kps_offset_loss = offset_mask * mx.sym.smooth_l1(data = (offset_estimate - kps_offset), scalar = 1.0)
    kps_offset_loss = mx.sym.MakeLoss(data = kps_offset_loss, grad_scale = grad_scales[1])

    if 'contrast' in kps_loss_type:
        return kps_label_loss, kps_offset_loss, contrast_loss
    else:
        return kps_label_loss, kps_offset_loss

def kps_get_output(label_estimate, offset_estimate, config, is_train, input_dict):
    kps_height = config.network.kps_height
    kps_width = config.network.kps_width
    num_kps = config.dataset.num_kps
    kps_loss_type = config.TRAIN.kps_loss_type

    if is_train:
        kps_label = mx.sym.Variable(name = 'kps_label')
        label_mask = mx.sym.Variable(name = 'kps_label_mask')
        kps_offset = mx.sym.Variable(name = 'kps_offset')
        offset_mask = mx.sym.Variable(name = 'kps_offset_mask')
        
        if 'one_hot' in kps_loss_type:
            input_dict['kps_label'] = (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, num_kps)
        else:
            input_dict['kps_label'] = (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, num_kps, kps_height * kps_width)
            input_dict['kps_label_mask'] = input_dict['kps_label']

        input_dict['kps_offset'] = (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, num_kps * 2, kps_height * kps_width)
        input_dict['kps_offset_mask'] = input_dict['kps_offset']
        # loss
        if 'contrast' in kps_loss_type:
            kps_label_loss, kps_offset_loss, contrast_loss = kps_train(
                                                                label_estimate = label_estimate,
                                                                kps_label = kps_label,
                                                                label_mask = label_mask,
                                                                offset_estimate = offset_estimate,
                                                                kps_offset = kps_offset,
                                                                offset_mask = offset_mask,
                                                                config = config)
        else:
            kps_label_loss, kps_offset_loss = kps_train(
                                                    label_estimate = label_estimate,
                                                    kps_label = kps_label,
                                                    label_mask = label_mask,
                                                    offset_estimate = offset_estimate,
                                                    kps_offset = kps_offset,
                                                    offset_mask = offset_mask,
                                                    config = config)
        # reshape loss
        if config.network.kps_crop_from_image:
            batch_size = config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size
        else:
            batch_size = config.TRAIN.image_batch_size
        
        if 'one_hot' in kps_loss_type:
            kps_label = mx.sym.Reshape(data = kps_label, shape = (batch_size, -1))
            kps_label_loss = mx.sym.Reshape(data = kps_label_loss, shape = (batch_size, -1, kps_height * kps_width))
    
            group_list = [mx.sym.BlockGrad(kps_label), kps_label_loss, kps_offset_loss]
        elif 'contrast' in kps_loss_type:
            group_list = [kps_label_loss, kps_offset_loss, contrast_loss]
        else:
            group_list = [kps_label_loss, kps_offset_loss]

    else:
        if 'one_hot' in kps_loss_type:
            if 'softmax' in kps_loss_type:
                label_estimate = mx.sym.Reshape(data = label_estimate, shape = (-1, kps_height * kps_width, 1, 1))
                label_estimate = mx.sym.SoftmaxActivation(data = label_estimate, mode = 'channel')
                label_estimate = mx.sym.Reshape(data = label_estimate, shape = (-1, num_kps, kps_height, kps_width))
            else:
                raise ValueError("unknown kps loss type {}".format(kps_loss_type))
        group_list = [label_estimate, offset_estimate]

    return group_list

def kps_aux_output(label_estimate, offset_estimate, config, aux_stride, input_dict):
    kps_height = config.network.kps_input_height / aux_stride
    kps_width = config.network.kps_input_width / aux_stride
    num_kps = config.dataset.num_kps
    kps_loss_type = config.TRAIN.kps_loss_type

    kps_label = mx.sym.Variable(name='kps_aux_label_' + str(aux_stride))
    label_mask = mx.sym.Variable(name='kps_aux_label_mask_' + str(aux_stride))
    kps_offset = mx.sym.Variable(name='kps_aux_offset_' + str(aux_stride))
    offset_mask = mx.sym.Variable(name='kps_aux_offset_mask_' + str(aux_stride))
    
    if 'one_hot' in kps_loss_type:
        input_dict['kps_aux_label_' + str(aux_stride)] = (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, num_kps)
    else:
        input_dict['kps_aux_label_' + str(aux_stride)] = (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, num_kps, kps_height * kps_width)
        input_dict['kps_aux_label_mask_' + str(aux_stride)] = input_dict['kps_aux_label_' + str(aux_stride)]

    input_dict['kps_aux_offset_' + str(aux_stride)] = (config.TRAIN.image_batch_size, config.TRAIN.kps_roi_batch_size, num_kps * 2, kps_height * kps_width)
    input_dict['kps_aux_offset_mask_' + str(aux_stride)] = input_dict['kps_aux_offset_' + str(aux_stride)]
    # loss
    kps_label_loss, kps_offset_loss = kps_train(label_estimate = label_estimate,
                                                    kps_label = kps_label,
                                                    label_mask = label_mask,
                                                    offset_estimate = offset_estimate,
                                                    kps_offset = kps_offset,
                                                    offset_mask = offset_mask,
                                                    config = config,
                                                    aux_stride = aux_stride)
    # reshape loss
    if config.network.kps_crop_from_image:
        batch_size = config.TRAIN.image_batch_size * config.TRAIN.kps_roi_batch_size
    else:
        batch_size = config.TRAIN.image_batch_size
    
    if 'one_hot' in kps_loss_type:
        kps_label = mx.sym.Reshape(data = kps_label, shape = (batch_size, -1))
        kps_label_loss = mx.sym.Reshape(data = kps_label_loss, shape = (batch_size, -1, kps_height * kps_width))

        group_list = [mx.sym.BlockGrad(kps_label), kps_label_loss]
    else:
        group_list = [kps_label_loss, ]
    group_list.append(kps_offset_loss)

    return group_list
