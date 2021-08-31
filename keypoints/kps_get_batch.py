import random
import numpy as np


def kps_generate_new_rois(roi_boxes, roi_batch_size, rescale_factor=0.0, jitter_center=False, aspect_ratio=0.0):
    num_boxes = roi_boxes.shape[0]
    assert num_boxes > 0
    if num_boxes == roi_batch_size:
        all_inds = range(num_boxes)
    elif num_boxes > roi_batch_size:
        all_inds = np.random.choice(range(num_boxes), size=roi_batch_size, replace=False)
    else:
        all_inds = np.array(range(num_boxes))
        while all_inds.shape[0] < roi_batch_size:
            ex_size = np.minimum(num_boxes, roi_batch_size - all_inds.shape[0])
            ex_inds = np.random.choice(range(num_boxes), size=ex_size, replace=False)
            all_inds = np.append(all_inds, ex_inds)

    all_boxes = np.zeros((roi_batch_size, 4), dtype=np.float32)
    for i in xrange(roi_batch_size):
        roi_box = roi_boxes[all_inds[i], :]
        roi_width = roi_box[2] - roi_box[0] + 1
        roi_height = roi_box[3] - roi_box[1] + 1
        if isinstance(rescale_factor, list):
            rand_x1 = random.uniform(rescale_factor[0], rescale_factor[1])
            rand_y1 = random.uniform(rescale_factor[0], rescale_factor[1])
            if jitter_center:
                rand_x2 = random.uniform(rescale_factor[0], rescale_factor[1])
                rand_y2 = random.uniform(rescale_factor[0], rescale_factor[1])
            else:
                rand_x2 = rand_x1
                rand_y2 = rand_y1
        else:
            rand_x1 = rescale_factor
            rand_y1 = rescale_factor
            rand_x2 = rescale_factor
            rand_y2 = rescale_factor
        new_x1 = roi_box[0] - roi_width * rand_x1
        new_y1 = roi_box[1] - roi_height * rand_y1
        new_x2 = roi_box[2] + roi_width * rand_x2
        new_y2 = roi_box[3] + roi_height * rand_y2
        if aspect_ratio > 0:
            new_center_x = (new_x1 + new_x2) / 2
            new_center_y = (new_y1 + new_y2) / 2
            new_width = new_x2 - new_x1 + 1
            new_height = new_y2 - new_y1 + 1
            if aspect_ratio * new_width > new_height:
                new_height = aspect_ratio * new_width
            else:
                new_width = new_height / aspect_ratio
            all_boxes[i, 0] = new_center_x - 0.5 * (new_width - 1)
            all_boxes[i, 1] = new_center_y - 0.5 * (new_height - 1)
            all_boxes[i, 2] = new_center_x + 0.5 * (new_width - 1)
            all_boxes[i, 3] = new_center_y + 0.5 * (new_height - 1)
        else:
            all_boxes[i, 0] = new_x1
            all_boxes[i, 1] = new_y1
            all_boxes[i, 2] = new_x2
            all_boxes[i, 3] = new_y2
    return all_boxes, all_inds


def kps_aux_label_and_offset(bbox, gt_kps, config, aux_stride):
    # gt_box: (4, )
    # gt_kps: (num_kps, 3)
    num_kps = gt_kps.shape[0]
    feat_width = config.network.kps_input_width / aux_stride
    feat_height = config.network.kps_input_height / aux_stride
    kps_loss_type = config.TRAIN.kps_loss_type

    if 'one_hot' in kps_loss_type:
        kps_label = np.full((num_kps,), fill_value=-1, dtype=np.float32)
    else:
        kps_label = np.full((num_kps, feat_height * feat_width), fill_value=-1, dtype=np.float32)

    label_mask = np.zeros((num_kps, feat_height * feat_width), dtype=np.float32)
    kps_offset = np.zeros((num_kps, 2, feat_height * feat_width), dtype=np.float32)
    offset_mask = np.zeros((num_kps, 2, feat_height * feat_width), dtype=np.float32)

    scale_x = feat_width / (bbox[2] - bbox[0] + 1)
    scale_y = feat_height / (bbox[3] - bbox[1] + 1)
    x = (gt_kps[:, 0] - bbox[0]) * scale_x
    y = (gt_kps[:, 1] - bbox[1]) * scale_y
    x_int = np.floor(x)
    y_int = np.floor(y)

    vis = np.logical_and(gt_kps[:, 2] > 0, gt_kps[:, 2] < 3) if config.TRAIN.kps_keep_invis else gt_kps[:, 2] == 2
    valid = np.logical_and(np.logical_and(x_int >= 0, y_int >= 0), np.logical_and(x_int < feat_width, y_int < feat_height))
    valid = np.logical_and(valid, vis)
    keep = np.where(valid == 1)[0]

    if len(keep) > 0:
        if 'one_hot' in kps_loss_type:
            x_offset = x - x_int
            y_offset = y - y_int

            pos = y_int * feat_width + x_int
            keep_pos = pos[keep].astype(np.int32)
            kps_label[keep] = keep_pos

            kps_offset[keep, 0, keep_pos] = x_offset[keep]
            kps_offset[keep, 1, keep_pos] = y_offset[keep]
            offset_mask[keep, 0, keep_pos] = 1
            offset_mask[keep, 1, keep_pos] = 1

        elif 'pixel' in kps_loss_type:
            kps_label[:, :] = 0
            label_mask[:, :] = config.TRAIN.kps_aux_mask_value

            ignore_kps = np.where(gt_kps[:, 2] == 3)[0]
            kps_label[ignore_kps, :] = -1
            label_mask[ignore_kps, :] = 0

            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)

            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.reshape((-1,))
            feat_y_int = feat_y_int.reshape((-1,))

            kps_pos_distance_x = config.network.kps_pos_distance_x / aux_stride
            kps_pos_distance_y = config.network.kps_pos_distance_y / aux_stride

            gauss_sigma = config.TRAIN.kps_gauss_sigma
            label_range = config.TRAIN.kps_label_range

            for keep_i in keep:
                x_offset = (x[keep_i] - feat_x_int) / kps_pos_distance_x
                y_offset = (y[keep_i] - feat_y_int) / kps_pos_distance_y
                
                dist = x_offset ** 2 + y_offset ** 2
                
                neighbourhood = np.where((0 <= dist) & (dist <= label_range))[0]

                if 'reg_fixed' in kps_loss_type:
                    kps_label[keep_i, neighbourhood] = 1
                else:
                    kps_label[keep_i, neighbourhood] = np.exp(dist[neighbourhood] / (-2.0 * gauss_sigma * gauss_sigma))

                kps_offset[keep_i, 0, neighbourhood] = x_offset[neighbourhood]
                kps_offset[keep_i, 1, neighbourhood] = y_offset[neighbourhood]

                offset_mask[keep_i, 0, neighbourhood] = config.TRAIN.kps_aux_mask_value
                offset_mask[keep_i, 1, neighbourhood] = config.TRAIN.kps_aux_mask_value

            assert kps_label.min() >= 0 and kps_label.max() <= 1
        else:
            raise ValueError("unknown kps loss type {}".format(kps_loss_type))

    kps_offset = kps_offset.reshape((num_kps * 2, -1))
    offset_mask = offset_mask.reshape((num_kps * 2, -1))

    return kps_label, label_mask, kps_offset, offset_mask


def kps_compute_label_and_offset(bbox, gt_kps, config):
    # gt_box: (4, )
    # gt_kps: (num_kps, 3)
    num_kps = gt_kps.shape[0]
    feat_width = config.network.kps_width
    feat_height = config.network.kps_height
    kps_loss_type = config.TRAIN.kps_loss_type

    if 'one_hot' in kps_loss_type:
        kps_label = np.full((num_kps,), fill_value=-1, dtype=np.float32)
    else:
        kps_label = np.full((num_kps, feat_height * feat_width), fill_value=-1, dtype=np.float32)

    label_mask = np.zeros((num_kps, feat_height * feat_width), dtype=np.float32)
    kps_offset = np.zeros((num_kps, 2, feat_height * feat_width), dtype=np.float32)
    offset_mask = np.zeros((num_kps, 2, feat_height * feat_width), dtype=np.float32)

    scale_x = feat_width / (bbox[2] - bbox[0] + 1)
    scale_y = feat_height / (bbox[3] - bbox[1] + 1)
    x = (gt_kps[:, 0] - bbox[0]) * scale_x
    y = (gt_kps[:, 1] - bbox[1]) * scale_y
    x_int = np.floor(x)
    y_int = np.floor(y)

    vis = np.logical_and(gt_kps[:, 2] > 0, gt_kps[:, 2] < 3) if config.TRAIN.kps_keep_invis else gt_kps[:, 2] == 2
    valid = np.logical_and(np.logical_and(x_int >= 0, y_int >= 0), np.logical_and(x_int < feat_width, y_int < feat_height))
    valid = np.logical_and(valid, vis)
    keep = np.where(valid == 1)[0]

    if len(keep) > 0:
        if 'one_hot' in kps_loss_type:
            x_offset = x - x_int
            y_offset = y - y_int

            pos = y_int * feat_width + x_int
            keep_pos = pos[keep].astype(np.int32)
            kps_label[keep] = keep_pos

            kps_offset[keep, 0, keep_pos] = x_offset[keep]
            kps_offset[keep, 1, keep_pos] = y_offset[keep]
            offset_mask[keep, 0, keep_pos] = 1
            offset_mask[keep, 1, keep_pos] = 1

        elif 'pixel' in kps_loss_type:
            kps_label[:, :] = 0
            label_mask[:, :] = 1

            ignore_kps = np.where(gt_kps[:, 2] == 3)[0]
            kps_label[ignore_kps, :] = -1
            label_mask[ignore_kps, :] = 0

            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)

            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.reshape((-1,))
            feat_y_int = feat_y_int.reshape((-1,))

            kps_pos_distance_x = config.network.kps_pos_distance_x / config.network.kps_feat_stride
            kps_pos_distance_y = config.network.kps_pos_distance_y / config.network.kps_feat_stride

            gauss_sigma = config.TRAIN.kps_gauss_sigma
            label_range = config.TRAIN.kps_label_range

            for keep_i in keep:
                x_offset = (x[keep_i] - feat_x_int) / kps_pos_distance_x
                y_offset = (y[keep_i] - feat_y_int) / kps_pos_distance_y
                
                dist = x_offset ** 2 + y_offset ** 2
                
                neighbourhood = np.where((0 <= dist) & (dist <= label_range))[0]

                if 'reg_fixed' in kps_loss_type:
                    kps_label[keep_i, neighbourhood] = 1
                else:
                    kps_label[keep_i, neighbourhood] = np.exp(dist[neighbourhood] / (-2.0 * gauss_sigma * gauss_sigma))

                kps_offset[keep_i, 0, neighbourhood] = x_offset[neighbourhood]
                kps_offset[keep_i, 1, neighbourhood] = y_offset[neighbourhood]
                offset_mask[keep_i, 0, neighbourhood] = 1.0
                offset_mask[keep_i, 1, neighbourhood] = 1.0

            assert kps_label.min() >= 0 and kps_label.max() <= 1
        else:
            raise ValueError("unknown kps loss type {}".format(kps_loss_type))

    kps_offset = kps_offset.reshape((num_kps * 2, -1))
    offset_mask = offset_mask.reshape((num_kps * 2, -1))

    return kps_label, label_mask, kps_offset, offset_mask


def kps_get_train_batch(all_boxes, all_kps, config, aux_stride = None):
    # all_boxes: (num_boxes, 4)
    # all_kps: (num_boxes, 51)
    roi_batch_size = all_boxes.shape[0]

    assert all_boxes.shape[0] == all_kps.shape[0]
    assert roi_batch_size > 0

    num_kps = config.dataset.num_kps

    if aux_stride:
        feat_height = config.network.kps_input_height / aux_stride
        feat_width = config.network.kps_input_width / aux_stride
    else:
        feat_height = config.network.kps_height
        feat_width = config.network.kps_width

    kps_label_list = []
    label_mask_list = []
    kps_offset_list = []
    offset_mask_list = []

    for i in range(roi_batch_size):
        if aux_stride:
            kps_label, label_mask, kps_offset, offset_mask = kps_aux_label_and_offset(bbox = all_boxes[i, :], gt_kps = all_kps[i, :, :], config = config, aux_stride = aux_stride)
        else:
            kps_label, label_mask, kps_offset, offset_mask = kps_compute_label_and_offset(bbox = all_boxes[i, :], gt_kps = all_kps[i, :, :], config = config)

        kps_label_list.append(kps_label)
        label_mask_list.append(label_mask)
        kps_offset_list.append(kps_offset)
        offset_mask_list.append(offset_mask)

    if 'one_hot' in config.TRAIN.kps_loss_type:
        kps_labels = np.hstack(kps_label_list)
        kps_labels = kps_labels.reshape((1, roi_batch_size, num_kps))
    else:
        kps_labels = np.vstack(kps_label_list)
        kps_labels = kps_labels.reshape((1, roi_batch_size, num_kps, feat_height * feat_width))

    kps_label_mask = np.vstack(label_mask_list)
    kps_label_mask = kps_label_mask.reshape((1, roi_batch_size, num_kps, feat_height * feat_width))

    kps_offsets = np.vstack(kps_offset_list)
    kps_offset_mask = np.vstack(offset_mask_list)
    kps_offsets = kps_offsets.reshape((1, roi_batch_size, num_kps * 2, feat_height * feat_width))
    kps_offset_mask = kps_offset_mask.reshape((1, roi_batch_size, num_kps * 2, feat_height * feat_width))

    if aux_stride:
        aux_names = ['kps_aux_label_', 'kps_aux_label_mask_', 'kps_aux_offset_', 'kps_aux_offset_mask_']
        aux_names = [name + str(aux_stride) for name in aux_names]
        aux_targets = [kps_labels, kps_label_mask, kps_offsets, kps_offset_mask]
        
        return dict(zip(aux_names, aux_targets))
    else:
        return dict(
                kps_label = kps_labels,
                kps_label_mask = kps_label_mask,
                kps_offset = kps_offsets,
                kps_offset_mask = kps_offset_mask)


def kps_get_test_batch(all_boxes, config):
    # all_boxes: (num_boxes, 4)
    feat_height = config.network.kps_height
    feat_width = config.network.kps_width
    aspect_ratio = float(feat_height) / feat_width if config.TEST.aug_strategy.kps_do_aspect_ratio else 0.0
    
    if config.TEST.aug_strategy.kps_multiscale:
        assert isinstance(config.TEST.aug_strategy.kps_rescale_factor, list)
        num_scales = len(config.TEST.aug_strategy.kps_rescale_factor)

        rois = []

        for scale in config.TEST.aug_strategy.kps_rescale_factor:
            rois.append(
                kps_generate_new_rois(
                    roi_boxes = all_boxes,
                    roi_batch_size = all_boxes.shape[0],
                    rescale_factor = scale,
                    aspect_ratio = aspect_ratio
                )[0]
            )
        rois = np.concatenate(rois, axis = 1)
        rois = rois.reshape((num_scales * all_boxes.shape[0], 4))

    else:
        rois = kps_generate_new_rois(
                roi_boxes = all_boxes,
                roi_batch_size = all_boxes.shape[0],
                rescale_factor = config.TEST.aug_strategy.kps_rescale_factor,
                aspect_ratio = aspect_ratio)[0]

    if config.TEST.aug_strategy.kps_flip_test:
        rois = np.tile(rois, (1, 2))
        rois = rois.reshape((all_boxes.shape[0] * 2, 4))
    
    return rois
