import numpy as np
from fpn.fpn_get_batch import sample_rois
from keypoints.kps_get_batch import kps_get_train_batch
from keypoints.kps_get_batch_2 import kps_get_train_batch_2
from common.processing.mask_transform import polys_or_rles_to_masks


def sample_kps_from_rpn(rois, gt_kps, config, extra_vars):
    fg_gt_indexes = extra_vars['fg_gt_indexes']

    # kps
    kps_height = config.network.kps_height
    kps_width = config.network.kps_width
    num_kps = config.dataset.num_kps
    kps_roi_batch_size = config.TRAIN.kps_roi_batch_size
    if 'one_hot' in config.TRAIN.kps_loss_type:
        kps_label_shape = (kps_roi_batch_size, num_kps)
    else:
        kps_label_shape = (kps_roi_batch_size, num_kps, kps_height * kps_width)
    kps_label_weight_shape = (kps_roi_batch_size, num_kps, kps_height * kps_width)
    kps_pos_offset_shape = (kps_roi_batch_size, num_kps * 2, kps_height * kps_width)
    kps_pos_offset_weight_shape = (kps_roi_batch_size, num_kps * 2, kps_height * kps_width)

    kps_labels = np.full(kps_label_shape, fill_value=-1, dtype=np.float32)
    kps_labels_weights = np.zeros(kps_label_weight_shape, dtype=np.float32)
    kps_pos_offsets = np.zeros(kps_pos_offset_shape, dtype=np.float32)
    kps_pos_offset_weights = np.zeros(kps_pos_offset_weight_shape, dtype=np.float32)
    if config.network.kps_compute_area:
        kps_rois = np.array([[0, 0.0, 0.0, 15.0, 15.0, 256.0]] * kps_roi_batch_size, dtype=np.float32)
    else:
        kps_rois = np.array([[0, 0.0, 0.0, 15.0, 15.0]] * kps_roi_batch_size, dtype=np.float32)

    if len(gt_kps) > 0:
        assert len(fg_gt_indexes) > 0
        if len(fg_gt_indexes) > kps_roi_batch_size:
            fg_gt_indexes = fg_gt_indexes[:kps_roi_batch_size]
        num_fg_gt_indexes = len(fg_gt_indexes)
        fg_gt_kps = gt_kps[fg_gt_indexes, :]
        fg_rois = rois[:num_fg_gt_indexes, :]
        if 'm2' in config.TRAIN.kps_loss_type:
            kps_rois_sub, _, kps_labels_sub, kps_labels_weights_sub, kps_pos_offsets_sub, kps_pos_offset_weights_sub = \
                kps_get_train_batch_2(all_boxes=fg_rois[:, 1:],
                                      all_kps=fg_gt_kps,
                                      base_anchor_centers=config.base_anchor_centers,
                                      roi_batch_size=num_fg_gt_indexes,
                                      config=config,
                                      change_box=True)
        else:
            kps_rois_sub, _, kps_labels_sub, kps_labels_weights_sub, kps_pos_offsets_sub, kps_pos_offset_weights_sub = \
                kps_get_train_batch(all_boxes=fg_rois[:, 1:],
                                    all_kps=fg_gt_kps,
                                    roi_batch_size=num_fg_gt_indexes,
                                    config=config,
                                    change_box=True)
        # assert np.sum(kps_rois_sub - fg_rois[:, 1:]) == 0
        kps_rois[:num_fg_gt_indexes, 0] = fg_rois[:, 0]
        kps_rois[:num_fg_gt_indexes, 1:5] = kps_rois_sub
        if config.network.kps_compute_area:
            kps_rois[:num_fg_gt_indexes, 5] = (fg_rois[:, 3] - fg_rois[:, 1] + 1) * (fg_rois[:, 4] - fg_rois[:, 2] + 1)
        kps_labels[:num_fg_gt_indexes, :] = kps_labels_sub[0, :]
        kps_labels_weights[:num_fg_gt_indexes, :] = kps_labels_weights_sub[0, :]
        kps_pos_offsets[:num_fg_gt_indexes, :] = kps_pos_offsets_sub[0, :]
        kps_pos_offset_weights[:num_fg_gt_indexes, :] = kps_pos_offset_weights_sub[0, :]
    else:
        assert not config.TRAIN.filter_strategy.remove_empty_boxes
        assert len(fg_gt_indexes) == 0
    return kps_rois, kps_labels, kps_labels_weights, kps_pos_offsets, kps_pos_offset_weights


def sample_kps_from_gt_boxes(gt_boxes, gt_kps, image_id, config):
    kps_roi_batch_size = config.TRAIN.kps_roi_batch_size
    if len(gt_kps) > 0:
        all_boxes, _, kps_labels, kps_labels_weights, kps_pos_offsets, kps_pos_offset_weights = \
            kps_get_train_batch(all_boxes=gt_boxes,
                                all_kps=gt_kps,
                                roi_batch_size=kps_roi_batch_size,
                                config=config,
                                change_box=True)
        kps_rois = np.zeros((kps_roi_batch_size, 5), dtype=np.float32)
        kps_rois[:, 0] = image_id
        kps_rois[:, 1:] = all_boxes
        kps_labels = kps_labels[0, :]
        kps_labels_weights = kps_labels_weights[0, :]
        kps_pos_offsets = kps_pos_offsets[0, :]
        kps_pos_offset_weights = kps_pos_offset_weights[0, :]
        return kps_rois, kps_labels, kps_labels_weights, kps_pos_offsets, kps_pos_offset_weights
    else:
        assert not config.TRAIN.filter_strategy.remove_empty_boxes


def sample_mask_from_rpn(rois, gt_polys, config, extra_vars):
    fg_gt_indexes = extra_vars['fg_gt_indexes']

    # masks
    mask_height = config.network.mask_pooled_size[0]
    mask_width = config.network.mask_pooled_size[1]
    mask_roi_batch_size = config.TRAIN.mask_roi_batch_size
    mask_labels = -np.ones((mask_roi_batch_size, 1, mask_height, mask_width))
    mask_rois = np.array([[0, 0.0, 0.0, 15.0, 15.0]] * mask_roi_batch_size, dtype=np.float32)

    if len(gt_polys) > 0:
        assert len(fg_gt_indexes) > 0
        if len(fg_gt_indexes) > mask_roi_batch_size:
            fg_gt_indexes = fg_gt_indexes[:mask_roi_batch_size]
        num_fg_gt_indexes = len(fg_gt_indexes)
        fg_gt_polys = [gt_polys[_] for _ in fg_gt_indexes]
        fg_rois = rois[:num_fg_gt_indexes, :]
        mask_labels[:num_fg_gt_indexes, 0, :, :] = polys_or_rles_to_masks(polys_or_rles=fg_gt_polys,
                                                                          boxes=fg_rois[:, 1:],
                                                                          mask_height=mask_height,
                                                                          mask_width=mask_width)
        mask_rois[:num_fg_gt_indexes, :] = fg_rois
    else:
        assert not config.TRAIN.filter_strategy.remove_empty_boxes
        assert len(fg_gt_indexes) == 0
    return mask_rois, mask_labels


def sample_rois_multitask(rois, fg_rois_per_image, rois_per_image, num_classes, config, gt_boxes, ignore_regions,
                          gt_kps=None, gt_polys=None):
    bbox_rois, bbox_labels, bbox_targets, bbox_weights, extra_vars = sample_rois(rois=rois,
                                                                                 fg_rois_per_image=fg_rois_per_image,
                                                                                 rois_per_image=rois_per_image,
                                                                                 num_classes=num_classes,
                                                                                 config=config,
                                                                                 gt_boxes=gt_boxes,
                                                                                 ignore_regions=ignore_regions,
                                                                                 need_extra_vars=True)
    res_list = [bbox_rois, bbox_labels, bbox_targets, bbox_weights]
    if 'kps' in config.network.task_type:
        res_list.extend(sample_kps_from_rpn(bbox_rois, gt_kps, config, extra_vars))
    if 'mask' in config.network.task_type:
        res_list.extend(sample_mask_from_rpn(bbox_rois, gt_polys, config, extra_vars))
    return res_list






