import math
import numpy as np


def compute_oks(alpha_kps, alpha_roi, score_kps, beta_kps, beta_score_kps):
    """
    computes oks between the estimate under question and other estimates
    :param alpha_kps: (num_kps, 2)
    :param alpha_roi: (4,)
    :param score_kps: (num_kps,)
    :param beta_kps: (num_boxes, num_kps, 2)
    """
    sigmas = np.array(
                        [
                            .026, .025, .025, .035, .035, .079,
                            .079, .072, .072, .062, .062, .107,
                            .107, .087, .087, .089, .089
                        ]
                    )
    variances = (sigmas * 2.0) ** 2

    area = (alpha_roi[2] - alpha_roi[0] + 1) * (alpha_roi[3] - alpha_roi[1] + 1)

    dx = beta_kps[:, :, 0] - alpha_kps[:, 0]  # (num_boxes - 1, num_kps)
    dy = beta_kps[:, :, 1] - alpha_kps[:, 1]  # (num_boxes - 1, num_kps)

    similarity = (dx ** 2 + dy ** 2) / variances / (area + np.spacing(1)) / 2  # (num_boxes - 1, num_kps)
    similarity = np.exp(- similarity)
    similarity *= (score_kps >= 0.1) & (beta_score_kps >= 0.1)

    return np.sum(similarity, axis = 1) / (np.sum(similarity != 0, axis = 1) + np.spacing(1))  # (num_boxes,)


def soft_nms_oks(estimates, score_kps, scores, rois, thresh, method='gaussian'):
    """Nms based on kp predictions."""
    order = scores.argsort()[::-1]
    scores = scores[order]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        ovr = compute_oks(
                estimates[i],
                rois[i],
                score_kps[i],
                estimates[order[1:]])

        order = order[1:]
        scores = scores[1:]
        if method == 'linear':
            inds = np.where(ovr >= thresh)[0]
            scores[inds] *= (1 - ovr[inds])
        elif method == 'quadratic':
            inds = np.where(ovr >= thresh)[0]
            scores[inds] *= (ovr[inds] - 1) / (thresh - 1)
        else:
            scores *= np.exp(- ovr ** 2 / thresh)
        inds = np.where(scores >= 2e-3)[0]
        order = order[inds]
        scores = scores[inds]

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

    return np.array(keep)


def nms_oks(estimates, score_kps, scores, rois, thresh):
    """
    :param estimates: (num_boxes, num_kps, 2)
    :param score_kps: (num_boxes, num_kps)
    :param scores: (num_boxes,)
    :param rois: (num_boxes, 4)
    :param thresh: retain oks < thresh
    :return: indices to keep
    """
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        if scores[i] == 0:
            break
        keep.append(i)
        
        ovr = compute_oks(
                estimates[i],
                rois[i],
                score_kps[i],
                estimates[order[1:]],
                score_kps[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def soft_nms_iou(dets, scores, thresh, method='gaussian'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:]
        scores = scores[1:]

        if method == 'linear':
            inds = np.where(ovr >= thresh)[0]
            scores[inds] *= (1 - ovr[inds])
        elif method == 'quadratic':
            inds = np.where(ovr >= thresh)[0]
            scores[inds] *= (ovr[inds] - 1) / (thresh - 1)
        else:
            scores *= np.exp(- ovr ** 2 / thresh)

        inds = np.where(scores >= 2e-3)[0]
        order = order[inds]
        scores = scores[inds]

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

    return np.array(keep)


def nms_iou(dets, scores, thresh):
    """
    :param dets: (num_boxes, 4)
    :paran scores: (num_boxes,)
    :param thresh: retain overlap < thresh
    :return: indices to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def nms_composite(estimates, score_kps, scores, rois, thresh_oks, thresh_iou, method):
    """
    composite nms by applying consecutive suppressing progress
    :param estimates: (num_boxes, num_kps, 2)
    :param score_kps: (num_boxes, num_kps)
    :param scores: (num_boxes,)
    :param rois: (num_boxes, 4)
    """
    assert not (thresh_oks is None and thresh_iou is None)

    if thresh_iou:
        if method:
            iou_keep = soft_nms_iou(rois, scores.copy(), thresh_iou, method)
        else:
            iou_keep = nms_iou(rois, scores, thresh_iou)
    else:
        iou_keep = np.arange(scores.shape[0])

    estimates = estimates[iou_keep]
    score_kps = score_kps[iou_keep]
    scores = scores[iou_keep]
    rois = rois[iou_keep]

    if thresh_oks:
        if method:
            oks_keep = soft_nms_oks(estimates, score_kps, scores.copy(), rois, thresh_oks, method)
        else:
            oks_keep = nms_oks(estimates, score_kps, scores, rois, thresh_oks)
    else:
        oks_keep = np.arange(scores.shape[0])

    return iou_keep[oks_keep]
