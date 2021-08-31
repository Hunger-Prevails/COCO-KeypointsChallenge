import mxnet as mx
import math
import numpy as np
from common.processing.kps_transform import merge_kps
from kps_nms_utils import nms_composite


def append_instance_score(kps_results, local_vars, config):
    """
    calculates composite instance score and appends it to the end of the final pose estimate
    :param kps_results: (num_boxes, num_kps * 3)
    :param local_vars: extra variables provided by test iterator
    :return: final pose estimate with instance scores enclosed
    """
    num_kps = config.dataset.num_kps
    score_kps = kps_results[:, 2::3].copy()  # (num_boxes, num_kps)

    if 'one_hot' in config.TRAIN.kps_loss_type:
        kps_scores = np.mean(score_kps, axis = 1)  # (num_boxes,)
    else:
        score_kps[score_kps < 0.1] = 0

        kps_scores = np.sum(score_kps, axis = 1) / (np.sum(score_kps != 0, axis = 1) + np.spacing(1)) # (num_boxes,)

    if config.TEST.use_gt_rois:
        return np.hstack((kps_results, kps_scores.reshape((-1, 1)))), np.arange(kps_results.shape[0])  # (num_boxes, num_kps * 3 + 1)

    kps_rois = local_vars['rois']
    kps_roi_scores = local_vars['roi_scores']

    composite_scores = kps_scores * kps_roi_scores  # (num_boxes,)

    x_estimates = kps_results[:, 0::3].reshape((-1, 1))  # (num_boxes * num_kps,)
    y_estimates = kps_results[:, 1::3].reshape((-1, 1))  # (num_boxes * num_kps,)

    estimates = np.hstack((x_estimates, y_estimates)).reshape(-1, num_kps, 2)  # (num_boxes, num_kps, 2)

    keep = nms_composite(
                estimates = estimates,
                score_kps = score_kps,
                scores = composite_scores,
                rois = kps_rois,
                thresh_oks = config.TEST.kps_oks_thresh,
                thresh_iou = config.TEST.kps_iou_thresh,
                method = config.TEST.kps_soft_method)
    
    kps_results = np.hstack((kps_results, composite_scores.reshape((-1, 1))))

    return kps_results[keep], keep


def hough_vote(heatmaps, offsets):
    # heatmaps: (num_boxes * num_kps, 1, feat_height, feat_width)
    # offsets: (num_boxes * num_kps, 2, feat_height, feat_width)
    import mxnet as mx
    grid = mx.nd.GridGenerator(data = mx.nd.array(-offsets), transform_type = 'warp')
    return mx.nd.BilinearSampler(mx.nd.array(heatmaps), grid).asnumpy()


def soft_argmax(hough_map, max_inds_xy, config):
    # hough_map: (num_boxes * num_kps, feat_height * feat_width)
    # max_inds_xy: (num_boxes * num_kps, 2)
    soft_inds_xy = np.zeros(max_inds_xy.shape)

    grid_x = np.arange(config.network.kps_width)
    grid_y = np.arange(config.network.kps_height)

    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    grid_x = grid_x.reshape((-1,))  # (feat_height * feat_width,)
    grid_y = grid_y.reshape((-1,))  # (feat_height * feat_width,)

    for idx in xrange(hough_map.shape[0]):
        offset_x = grid_x - max_inds_xy[idx, 0]  # (feat_height * feat_width,)
        offset_y = grid_y - max_inds_xy[idx, 1]  # (feat_height * feat_width,)

        dist = offset_x ** 2 + offset_y ** 2

        neighbourhood = np.where((dist <= 4.0) & (hough_map[idx] >= 0.1))[0]
        if neighbourhood.size == 0:
            neighbourhood = np.where((dist <= 4.0) & (hough_map[idx] >= 0.0))[0]

        assert neighbourhood.size != 0

        sum_score = hough_map[idx, neighbourhood].sum()
        
        x_offset = np.sum(offset_x[neighbourhood] * hough_map[idx, neighbourhood]) / sum_score
        y_offset = np.sum(offset_y[neighbourhood] * hough_map[idx, neighbourhood]) / sum_score

        soft_inds_xy[idx, 0] = max_inds_xy[idx, 0] + x_offset
        soft_inds_xy[idx, 1] = max_inds_xy[idx, 1] + y_offset

    return soft_inds_xy


def smoothen(kps_scores, feat_height, feat_width, config):
    # kps_scores: (num_boxes * num_kps, feat_height * feat_width)
    kps_scores = kps_scores.reshape((-1, 1, feat_height, feat_width))

    kps_scores = mx.nd.array(kps_scores)
    
    kernel_x = np.arange(-1, 2)
    kernel_y = np.arange(-1, 2)

    kernel_x, kernel_y = np.meshgrid(kernel_x, kernel_y)

    kernel_x = kernel_x.reshape((-1,))
    kernel_y = kernel_y.reshape((-1,))

    dist = kernel_x ** 2 + kernel_y ** 2

    gauss_sigma = config.TRAIN.kps_gauss_sigma

    kernel = np.exp(dist / (-2.0 * gauss_sigma * gauss_sigma))

    # kernel = mx.nd.array(np.ones((1, 1, 3, 3)))
    kernel = mx.nd.array(kernel.reshape(1, 1, 3, 3))

    kps_scores = mx.nd.Convolution(
                        data = kps_scores,
                        weight = kernel,
                        kernel = (3, 3),
                        num_filter = 1,
                        pad = (1, 1),
                        stride = (1, 1),
                        no_bias = True)

    return kps_scores.asnumpy().reshape((-1, feat_height * feat_width))


def kps_predict(outputs, local_vars, config):
    do_vote = 'hough_vote' in config.TRAIN.kps_loss_type
    do_fix = 'reg_fixed' in config.TRAIN.kps_loss_type
    do_pixel = 'pixel' in config.TRAIN.kps_loss_type
    do_approx = 'approx' in config.TRAIN.kps_loss_type

    assert do_vote <= do_fix
    assert do_approx <= do_fix
    assert do_fix <= do_pixel

    rois = outputs[0][:, 1:]  # rois: (num_boxes, 4)
    kps_scores = outputs[1]  # kps_scores: (num_boxes, num_kps, feat_height, feat_width)
    kps_deltas = outputs[2]  # kps_deltas: (num_boxes, num_kps * 2, feat_height, feat_width)

    num_kps = config.dataset.num_kps

    if config.TEST.aug_strategy.kps_flip_test:
        rois = rois[::2]
        kps_deltas = kps_deltas[::2]

        native_scores = kps_scores[::2]  # (num_boxes, num_kps, feat_height, feat_width)
        _mirror_scores = kps_scores[1::2]  # (num_boxes, num_kps, feat_height, feat_width)
        
        mirror_scores = np.zeros(_mirror_scores.shape)

        mirror_scores[:, 0, :, :] = _mirror_scores[:, 0, :, ::-1]
        mirror_scores[:, 1::2, :, :] = _mirror_scores[:, 2::2, :, ::-1]
        mirror_scores[:, 2::2, :, :] = _mirror_scores[:, 1::2, :, ::-1]

        kps_scores = native_scores / 2 + mirror_scores / 2

    num_boxes = kps_scores.shape[0]
    feat_height = kps_scores.shape[2]
    feat_width = kps_scores.shape[3]

    origin_xy = rois[:, :2]  # (num_boxes, 2)
    origin_xy = np.tile(origin_xy, (1, num_kps))  # (num_boxes, num_kps * 2)
    origin_xy = origin_xy.reshape((num_boxes * num_kps, 2))  # (num_boxes * num_kps, 2)

    scales_xy = np.zeros((num_boxes, 2), dtype = np.float32)
    scales_xy[:, 0] = feat_width / (rois[:, 2] - rois[:, 0] + 1)
    scales_xy[:, 1] = feat_height / (rois[:, 3] - rois[:, 1] + 1)
    scales_xy = np.tile(scales_xy, (1, num_kps))  # (num_boxes, num_kps * 2)
    scales_xy = scales_xy.reshape((num_boxes * num_kps, 2))  # (num_boxes * num_kps, 2)

    kps_scores = kps_scores.reshape((num_boxes * num_kps, -1))  # (num_boxes * num_kps, feat_height * feat_width)
    kps_deltas = kps_deltas.reshape((num_boxes * num_kps, 2, -1))  # (num_boxes * num_kps, 2, feat_height * feat_width)

    if do_pixel:
        kps_scores = smoothen(kps_scores, feat_height, feat_width, config)

    if do_vote:
        kps_pos_distance_x = config.network.kps_pos_distance_x / config.network.kps_feat_stride
        kps_pos_distance_y = config.network.kps_pos_distance_y / config.network.kps_feat_stride
        kps_deltas[:, 0] *= kps_pos_distance_x
        kps_deltas[:, 1] *= kps_pos_distance_y

        kps_scores = kps_scores.reshape((-1, 1, feat_height, feat_width))  # (num_boxes * num_kps, 1, feat_height, feat_width)
        kps_deltas = kps_deltas.reshape((-1, 2, feat_height, feat_width))  # (num_boxes * num_kps, 2, feat_height, feat_width)

        hough_map = hough_vote(kps_scores, kps_deltas);  # (num_boxes * num_kps, 1, feat_height, feat_width)
        hough_map = hough_map.reshape((-1, feat_height * feat_width))  # (num_boxes * num_kps, feat_height * feat_width)
        kps_scores = kps_scores.reshape((-1, feat_height * feat_width))  # (num_boxes * num_kps, feat_height * feat_width)

        all_inds = np.arange(num_boxes * num_kps)
        max_inds_xy = np.zeros((num_boxes * num_kps, 2))

        max_inds = hough_map.argmax(axis = 1)
        max_inds_xy[:, 0] = max_inds % feat_width
        max_inds_xy[:, 1] = max_inds / feat_width

        max_scores = kps_scores[all_inds, max_inds].copy()
        max_inds_xy = soft_argmax(hough_map, max_inds_xy, config)

        pred_kps = max_inds_xy / scales_xy + origin_xy  # (num_boxes * num_kps, 2)
        pred_kps /= local_vars['im_scale']

    elif do_approx:
        kps_pos_distance_x = config.network.kps_pos_distance_x / config.network.kps_feat_stride
        kps_pos_distance_y = config.network.kps_pos_distance_y / config.network.kps_feat_stride
        kps_deltas[:, 0] *= kps_pos_distance_x
        kps_deltas[:, 1] *= kps_pos_distance_y

        all_inds = np.arange(num_boxes * num_kps)
        max_inds_xy = np.zeros((num_boxes * num_kps, 2))
        sec_inds_xy = np.zeros((num_boxes * num_kps, 2))

        top_inds = np.argsort(kps_scores, axis = 1)[:, -2:]  # (num_boxes * num_kps, 2)
        top_inds = top_inds[:, ::-1]
        max_inds_xy[:, 0] = top_inds[:, 0] % feat_width
        max_inds_xy[:, 1] = top_inds[:, 0] / feat_width
        sec_inds_xy[:, 0] = top_inds[:, 1] % feat_width
        sec_inds_xy[:, 1] = top_inds[:, 1] / feat_width

        max_scores = kps_scores[all_inds, top_inds[:, 0]]  # (num_boxes * num_kps,)
        max_deltas_xy = kps_deltas[all_inds, :, top_inds[:, 0]]  # (num_boxes * num_kps, 2)
        sec_deltas_xy = kps_deltas[all_inds, :, top_inds[:, 1]]  # (num_boxes * num_kps, 2)

        pred_kps = 0.75 * (max_inds_xy + max_deltas_xy) + 0.25 * (sec_inds_xy + sec_deltas_xy)
        pred_kps = pred_kps / scales_xy + origin_xy
        pred_kps /= local_vars['im_scale']

    else:
        all_inds = np.arange(num_boxes * num_kps)
        max_inds_xy = np.zeros((num_boxes * num_kps, 2))

        max_inds = kps_scores.argmax(axis = 1)  # (num_boxes * num_kps,)
        max_inds_xy[:, 0] = max_inds % feat_width
        max_inds_xy[:, 1] = max_inds / feat_width

        max_scores = kps_scores[all_inds, max_inds]  # (num_boxes * num_kps,)
        max_deltas_xy = kps_deltas[all_inds, :, max_inds]  # (num_boxes * num_kps, 2)

        if do_pixel:
            kps_pos_distance_x = config.network.kps_pos_distance_x / config.network.kps_feat_stride
            kps_pos_distance_y = config.network.kps_pos_distance_y / config.network.kps_feat_stride
            max_deltas_xy[:, 0] *= kps_pos_distance_x
            max_deltas_xy[:, 1] *= kps_pos_distance_y
        
        pred_kps = (max_inds_xy + max_deltas_xy) / scales_xy + origin_xy  # (num_boxes * num_kps, 2)
        pred_kps /= local_vars['im_scale']

    kps_results = np.hstack((pred_kps, max_scores.reshape((-1, 1))))  # (num_boxes * num_kps, 3)
    kps_results = kps_results.reshape((num_boxes, -1))  # (num_boxes, num_kps * 3)

    # merge kps_results when multi-scale test is applied
    if config.TEST.aug_strategy.kps_multiscale:
        num_scales = len(config.TEST.aug_strategy.kps_rescale_factor)
        
        coords_x = kps_results[:, 0::3]  # (num_boxes, num_kps)
        coords_y = kps_results[:, 1::3]  # (num_boxes, num_kps)
        score_kps = kps_results[:, 2::3]  # (num_boxes, num_kps)

        final_x = []
        final_y = []
        final_s = []
        assert kps_results.shape[0] % num_scales == 0

        for i in xrange(kps_results.shape[0] / num_scales):
            _coords_x = coords_x[i * num_scales:i * num_scales + num_scales]  # (num_scales, num_kps)
            _coords_y = coords_y[i * num_scales:i * num_scales + num_scales]  # (num_scales, num_kps)
            _score_kps = score_kps[i * num_scales:i * num_scales + num_scales]  # (num_scales, num_kps)

            _score_kps = _score_kps * (_score_kps >= 0)
            _score_valid = _score_kps * (_score_kps >= 0.1)
            _score_max = np.argmax(_score_kps, axis = 0)  # (num_kps,)

            _sum_valid = np.sum(_score_valid, axis = 0)  # (num_kps,)
            _idx_neg = np.where(_sum_valid == 0)[0]
            _sum_valid[_idx_neg] = 1.0

            _final_x = np.sum(_coords_x * _score_valid, axis = 0) / _sum_valid  # (num_kps,)
            _final_y = np.sum(_coords_y * _score_valid, axis = 0) / _sum_valid  # (num_kps,)

            _final_x[_idx_neg] = _coords_x[_score_max[_idx_neg], _idx_neg]
            _final_y[_idx_neg] = _coords_y[_score_max[_idx_neg], _idx_neg]
            _final_s = _score_kps[_score_max, np.arange(num_kps)]  # (num_kps,)

            final_x.append(_final_x)
            final_y.append(_final_y)
            final_s.append(_final_s)

        final_x = np.stack(final_x, axis = 0).reshape((-1, 1))  # (num_boxes * num_kps / num_scales, 1)
        final_y = np.stack(final_y, axis = 0).reshape((-1, 1))  # (num_boxes * num_kps / num_scales, 1)
        final_s = np.stack(final_s, axis = 0).reshape((-1, 1))  # (num_boxes * num_kps / num_scales, 1)

        kps_results = np.hstack((final_x, final_y, final_s)).reshape((-1, num_kps * 3))  # (num_boxes / num_scales, num_kps * 3)

    kps_results, keep = append_instance_score(kps_results, local_vars, config)
    return kps_results.tolist(), keep
