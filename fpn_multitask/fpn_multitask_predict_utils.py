import numpy as np
import time
import mxnet as mx
from keypoints.kps_predict_utils import kps_predict
from common.processing.bbox_transform import bbox_pred, clip_boxes
from keypoints.kps_get_batch import kps_generate_new_rois


def e2e_multitask_predict(outputs, local_vars, config, nms_det, score_thresh=1e-3):
    res_dict = dict()
    res_dict['det_results'] = []
    res_dict['kps_results'] = []
    res_dict['mask_boxes'] = []
    res_dict['masks'] = []
    if len(outputs) == 0:
        return res_dict

    pred = ['rcnn_rois', 'rcnn_cls_prob']
    rcnn_rois = outputs[pred.index('rcnn_rois')]
    rcnn_scores = outputs[pred.index('rcnn_cls_prob')]
    pred_boxes = rcnn_rois[:, 1:] / local_vars['im_scale']
    if not config.TEST.use_gt_rois:
        pred_boxes = clip_boxes(pred_boxes, (local_vars['im_height'], local_vars['im_width']))
        pred_scores = rcnn_scores[:, 1:]
        keep_1 = np.where(pred_scores > score_thresh)[0]
        if len(keep_1) == 0:
            return res_dict
        pred_dets = np.hstack((pred_boxes, pred_scores)).astype(np.float32)[keep_1, :]
        keep_2 = nms_det(pred_dets)
        if len(keep_2) == 0:
            return res_dict
        cls_dets = pred_dets[keep_2, :]
    else:
        cls_dets = np.ones((pred_boxes.shape[0], 5), dtype=pred_boxes.dtype)
        cls_dets[:, :4] = pred_boxes

    if 'kps' in config.network.task_type:
        pred.extend(['kps_rois', 'kps_scores', 'kps_deltas'])
        kps_rois = outputs[pred.index('kps_rois')]
        kps_scores = outputs[pred.index('kps_scores')]
        kps_deltas = outputs[pred.index('kps_deltas')]
        if not config.TEST.use_gt_rois:
            kps_rois = kps_rois[keep_1[keep_2], :]
            kps_scores = kps_scores[keep_1[keep_2], :]
            kps_deltas = kps_deltas[keep_1[keep_2], :]
            local_vars['pred_boxes'] = cls_dets[:, :4]
            local_vars['pred_scores'] = cls_dets[:, 4, np.newaxis]
        if 'm2' in config.TRAIN.kps_loss_type:
            from keypoints.kps_predict_utils_2 import kps_predict_2
            kps_results = kps_predict_2([kps_rois[:, :5], kps_scores, kps_deltas], local_vars, config)
        else:
            kps_results = kps_predict([kps_rois[:, :5], kps_scores, kps_deltas], local_vars, config)
    else:
        kps_results = []

    if 'mask' in config.network.task_type:
        pred.extend(['mask_rois', 'mask_prob'])
        mask_rois = outputs[pred.index('mask_rois')]
        mask_scores = outputs[pred.index('mask_prob')]
        if mask_scores.shape[1] == 2:
            mask_scores = mask_scores[:, 1, :, :]
        else:
            assert mask_scores.shape[1] == 1
            mask_scores = mask_scores[:, 0, :, :]
        if not config.TEST.use_gt_rois:
            mask_scores = mask_scores[keep_1[keep_2], :]
            mask_rois = mask_rois[keep_1[keep_2], :]
            mask_boxes = mask_rois[:, 1:5] / local_vars['im_scale']
            mask_boxes = clip_boxes(mask_boxes, (local_vars['im_height'], local_vars['im_width']))
            mask_boxes = np.hstack((mask_boxes, cls_dets[:, 4, np.newaxis]))
        else:
            assert False
            # mask_boxes = mask_rois[:, 1:5] / local_vars['im_scale']
    else:
        mask_boxes = []
        mask_scores = []

    res_dict['det_results'] = cls_dets
    res_dict['kps_results'] = kps_results
    res_dict['mask_boxes'] = mask_boxes
    res_dict['masks'] = mask_scores
    return res_dict


def stage_rcnn_predict(outputs, local_vars, config, nms_det, score_thresh=1e-3):
    pred = ['rois', 'rcnn_cls_prob', 'rcnn_bbox_pred']
    rois = outputs[pred.index('rois')][:, 1:]
    scores = outputs[pred.index('rcnn_cls_prob')]
    bbox_deltas = outputs[pred.index('rcnn_bbox_pred')]
    if config.network.rcnn_bbox_normalization_precomputed:
        rcnn_bbox_stds = np.tile(np.array(config.network.rcnn_bbox_stds), bbox_deltas.shape[1] / 4)
        rcnn_bbox_means = np.tile(np.array(config.network.rcnn_bbox_means), bbox_deltas.shape[1] / 4)
        bbox_deltas = bbox_deltas * rcnn_bbox_stds + rcnn_bbox_means
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = pred_boxes / local_vars['im_scale']
    pred_boxes = clip_boxes(pred_boxes, (local_vars['im_height'], local_vars['im_width']))
    cls_boxes = pred_boxes[:, 4:]
    cls_scores = scores[:, 1, np.newaxis]
    keep_1 = np.where(cls_scores > score_thresh)[0]
    cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep_1, :]
    keep_2 = nms_det(cls_dets)
    cls_dets = cls_dets[keep_2, :]
    return cls_dets


def stage_kps_predict(rcnn_boxes, shared_feat_outputs, kps_mod, local_vars, config):
    if config.network.kps_compute_area:
        kps_rois = np.zeros((1, rcnn_boxes.shape[0], 6), dtype=np.float32)
        kps_rois[0, :, 5] = (rcnn_boxes[:, 2] - rcnn_boxes[:, 0] + 1) * (rcnn_boxes[:, 3] - rcnn_boxes[:, 1] + 1)
    else:
        kps_rois = np.zeros((1, rcnn_boxes.shape[0], 5), dtype=np.float32)
    aspect_ratio = float(config.network.kps_height) / config.network.kps_width if config.TEST.aug_strategy.kps_do_aspect_ratio else 0.0
    kps_rois[0, :, 1:5], _ = kps_generate_new_rois(roi_boxes=rcnn_boxes,
                                                   roi_batch_size=rcnn_boxes.shape[0],
                                                   rescale_factor=config.TEST.aug_strategy.kps_rescale_factor,
                                                   jitter_center=False,
                                                   aspect_ratio=aspect_ratio)
    kps_data_batch = mx.io.DataBatch(data=[], pad=0)
    kps_data_batch.data = [mx.nd.array(kps_rois), ]
    kps_data_batch.provide_data = [('kps_rois', kps_rois.shape)]
    for i, stride in enumerate(config.network.rcnn_feat_stride):
        kps_data_batch.data.append(shared_feat_outputs[i])
        kps_data_batch.provide_data.append(('data_res%d' % stride, shared_feat_outputs[i].shape))
    kps_net_time = time.time()
    kps_mod.forward(kps_data_batch)
    kps_outputs = kps_mod.get_outputs()
    kps_outputs = [kps_rois[0, :, :5]] + [kps_outputs[i].asnumpy() for i in range(len(kps_outputs))]
    kps_net_time = time.time() - kps_net_time
    if 'm2' in config.TRAIN.kps_loss_type:
        from keypoints.kps_predict_utils_2 import kps_predict_2
        kps_results = kps_predict_2(kps_outputs, local_vars, config)
    else:
        kps_results = kps_predict(kps_outputs, local_vars, config)

    return kps_results, kps_net_time


def stage_mask_predict(rcnn_boxes, shared_feat_outputs, mask_mod, config):
    mask_rois = np.zeros((1, rcnn_boxes.shape[0], 5), dtype=np.float32)
    mask_rois[0, :, 1:] = rcnn_boxes
    mask_data_batch = mx.io.DataBatch(data=[], pad=0)
    mask_data_batch.data = [mx.nd.array(mask_rois), ]
    mask_data_batch.provide_data = [('mask_rois', mask_rois.shape)]
    for i, stride in enumerate(config.network.rcnn_feat_stride):
        mask_data_batch.data.append(shared_feat_outputs[i])
        mask_data_batch.provide_data.append(('data_res%d' % stride, shared_feat_outputs[i].shape))
    mask_net_time = time.time()
    mask_mod.forward(mask_data_batch)
    mask_outputs = mask_mod.get_outputs()
    mask_scores = mask_outputs[0].asnumpy()
    mask_net_time = time.time() - mask_net_time
    if mask_scores.shape[1] == 2:
        cls_masks = mask_scores[:, 1, :, :]
    else:
        assert mask_scores.shape[1] == 1
        cls_masks = mask_scores[:, 0, :, :]
    return cls_masks, mask_net_time


# from common.processing.mask_transform import gpu_mask_voting
# def frcnn_multitask_predict(outputs, local_vars, config, nms_det, score_thresh=1e-3):
#     res_dict = dict()
#     res_dict['cls_dets'] = []
#     res_dict['kps_results'] = []
#     res_dict['cls_masks'] = []
#     if len(outputs) == 0:
#         return res_dict
#     pred = ['rcnn_rois', 'rcnn_cls_prob']
#     rcnn_rois = outputs[pred.index('rcnn_rois')]
#     rcnn_scores = outputs[pred.index('rcnn_cls_prob')]
#
#     pred_boxes = rcnn_rois[:, 1:] / local_vars['im_scale']
#     pred_boxes = clip_boxes(pred_boxes, (local_vars['im_height'], local_vars['im_width']))
#     pred_scores = rcnn_scores
#
#     if 'mask' in config.network.task_type:
#         pred.extend(['mask_prob'])
#         mask_scores = outputs[pred.index('mask_prob')]
#         if mask_scores.shape[1] == 2:
#             mask_scores = mask_scores[:, 1:, :, :]
#         else:
#             assert mask_scores.shape[1] == 1
#         result_mask, result_box = gpu_mask_voting(masks=mask_scores,
#                                                   boxes=pred_boxes,
#                                                   scores=pred_scores,
#                                                   num_classes=config.dataset.num_classes,
#                                                   max_per_image=100,
#                                                   im_width=local_vars['im_width'],
#                                                   im_height=local_vars['im_height'],
#                                                   nms_thresh=config.TEST.rcnn_nms,
#                                                   merge_thresh=0.5,
#                                                   binary_thresh=config.TEST.mask_binary_thresh)
#         cls_dets = result_box[1]
#         cls_masks = result_mask[1][:, 0, :, :]
#     else:
#         cls_dets = []
#         cls_masks = []
#
#     res_dict['cls_dets'] = cls_dets
#     # res_dict['kps_results'] = kps_results
#     res_dict['cls_masks'] = cls_masks
#     return res_dict
