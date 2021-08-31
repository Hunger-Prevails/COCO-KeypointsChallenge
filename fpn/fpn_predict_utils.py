import numpy as np
from common.processing.bbox_transform import bbox_pred, clip_boxes


def rpn_predict_single_branch(outputs, local_vars, thresh=0):
    pred = ['rois', 'rois_score']
    pred_boxes = outputs[pred.index('rois')][:, 1:]
    scores = outputs[pred.index('rois_score')]
    pred_boxes = pred_boxes / local_vars['im_scale']
    proposals = np.hstack((pred_boxes, scores))
    keep = np.where(proposals[:, 4] > thresh)[0]

    proposals = proposals[keep, :]
    return proposals


def rpn_predict_multi_branch(outputs, local_vars, thresh=0):
    pred = ['rois', 'rois_score']
    num_branch = len(outputs) / len(pred)
    num_boxes = outputs[pred.index('rois')].shape[0]
    all_proposals = np.zeros((num_boxes, 5 * num_branch), dtype=np.float32)
    for branch_i in range(num_branch):
        rois = outputs[branch_i * len(pred) + pred.index('rois')]
        scores = outputs[branch_i * len(pred) + pred.index('rois_score')]
        proposals = rpn_predict_single_branch([rois, scores], local_vars, thresh=thresh)
        all_proposals[:, 5 * branch_i: 5 * branch_i + 5] = proposals
    return all_proposals


def rcnn_predict_single_branch(outputs, local_vars, config):
    pred = ['rois', 'rcnn_cls_prob', 'rcnn_bbox_pred']
    rois = outputs[pred.index('rois')]
    rois = rois[:, 1:]
    scores = outputs[pred.index('rcnn_cls_prob')].reshape((rois.shape[0], -1))
    bbox_deltas = outputs[pred.index('rcnn_bbox_pred')].reshape((rois.shape[0], -1))
    scores = scores[:, 1:]
    bbox_deltas = bbox_deltas[:, 4:]
    if config.network.rcnn_bbox_normalization_precomputed:
        rcnn_bbox_stds = np.tile(np.array(config.network.rcnn_bbox_stds), bbox_deltas.shape[1] / 4)
        rcnn_bbox_means = np.tile(np.array(config.network.rcnn_bbox_means), bbox_deltas.shape[1] / 4)
        bbox_deltas = bbox_deltas * rcnn_bbox_stds + rcnn_bbox_means
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = pred_boxes / local_vars['im_scale']
    pred_boxes = clip_boxes(pred_boxes, (local_vars['im_height'], local_vars['im_width']))

    return pred_boxes, scores


def rcnn_predict_multi_branch(outputs, local_vars, config):
    pred = ['rois', 'rcnn_cls_prob', 'rcnn_bbox_pred']
    num_branch = len(outputs) / len(pred)
    assert num_branch == config.network.rpn_rcnn_num_branch
    num_boxes = outputs[pred.index('rois')].shape[0]
    all_pred_boxes = np.zeros((num_boxes, 4 * num_branch), dtype=np.float32)
    all_scores = np.zeros((num_boxes, num_branch), dtype=np.float32)
    for branch_i in range(num_branch):
        rois = outputs[branch_i * len(pred) + pred.index('rois')]
        scores = outputs[branch_i * len(pred) + pred.index('rcnn_cls_prob')]
        bbox_deltas = outputs[branch_i * len(pred) + pred.index('rcnn_bbox_pred')]
        pred_boxes, scores = rcnn_predict_single_branch([rois, scores, bbox_deltas], local_vars, config)
        all_pred_boxes[:, 4 * branch_i: 4 * branch_i + 4] = pred_boxes
        all_scores[:, branch_i] = scores[:, 0]
    return all_pred_boxes, all_scores


def rpn_predict(outputs, local_vars, config, thresh=0):
    if config.network.rpn_rcnn_num_branch > 1:
        return rpn_predict_multi_branch(outputs, local_vars, thresh=thresh)
    else:
        return rpn_predict_single_branch(outputs, local_vars, thresh=thresh)


def rcnn_predict(outputs, local_vars, config):
    if config.network.rpn_rcnn_num_branch > 1:
        return rcnn_predict_multi_branch(outputs, local_vars, config)
    else:
        return rcnn_predict_single_branch(outputs, local_vars, config)


def rpn_rcnn_predict(outputs, local_vars, config, rpn_thresh=0):
    pred = ['rois', 'rois_score', 'rcnn_cls_prob', 'rcnn_bbox_pred']
    rpn_outputs = []
    rcnn_outputs = []
    for branch_i in range(config.network.rpn_rcnn_num_branch):
        rpn_outputs.append(outputs[branch_i * len(pred) + pred.index('rois')])
        rpn_outputs.append(outputs[branch_i * len(pred) + pred.index('rois_score')])
        rcnn_outputs.append(outputs[branch_i * len(pred) + pred.index('rois')])
        rcnn_outputs.append(outputs[branch_i * len(pred) + pred.index('rcnn_cls_prob')])
        rcnn_outputs.append(outputs[branch_i * len(pred) + pred.index('rcnn_bbox_pred')])
    proposals = rpn_predict(rpn_outputs, local_vars, config, thresh=rpn_thresh)
    pred_boxes, scores = rcnn_predict(rcnn_outputs, local_vars, config)
    return proposals, pred_boxes, scores

