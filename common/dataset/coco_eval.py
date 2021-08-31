import os
import cv2
import json
import numpy as np
import logging
from ..pycocotools.coco import COCO
from ..pycocotools.cocoeval import COCOeval
from ..pycocotools.mask import encode as encodeMask_c
from load_roidb import filter_roidb


def mask_voc2coco(voc_masks, voc_boxes, im_height, im_width, binary_thresh=0.5):
    num_pred = len(voc_masks)
    assert(num_pred == voc_boxes.shape[0])
    mask_img = np.zeros((im_height, im_width, num_pred), dtype=np.uint8, order='F')
    for i in xrange(num_pred):
        pred_box = np.round(voc_boxes[i, :4]).astype(int)
        pred_mask = voc_masks[i]
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (pred_box[2] - pred_box[0] + 1, pred_box[3] - pred_box[1] + 1))
        mask_img[pred_box[1]:pred_box[3]+1, pred_box[0]:pred_box[2]+1, i] = pred_mask >= binary_thresh
    coco_mask = encodeMask_c(mask_img)
    return coco_mask


class COCOEval(object):
    def __init__(self, annotation_path):
        self.coco = COCO(annotation_path)

        self.imageset_name = annotation_path[:-5].split('_')[-1]
        self.imageset_index = self.coco.getImgIds()
        self.num_images = len(self.imageset_index)

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

    def sample_on_imdb(self, roidb, filter_strategy):
        roidb, choose_inds = filter_roidb(roidb, filter_strategy, need_inds=True)
        self.imageset_index = [self.imageset_index[i] for i in choose_inds]
        self.num_images = len(self.imageset_index)
        return roidb

    def evaluate_detections(self, detections, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'detections_%s_%s_results.json' % (self.imageset_name, alg))
        self.write_coco_det_results(detections, res_file)
        if 'test' not in self.imageset_name:
            ann_type = 'bbox'
            coco_dt = self.coco.loadRes(res_file)
            coco_eval = COCOeval(self.coco, coco_dt)
            coco_eval.params.useSegm = (ann_type == 'segm')
            coco_eval.params.imgIds = self.imageset_index
            coco_eval.evaluate()
            coco_eval.accumulate()
            logging.info('detection result:')
            coco_eval.summarize()

    def write_coco_det_results(self, detections, res_file):
        """ example results
        [{"image_id": 42,
          "category_id": 18,
          "bbox": [258.15,41.29,348.26,243.78],
          "score": 0.236}, ...]
        """
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1)
            coco_cat_id = self._class_to_coco_ind[cls]
            results.extend(self._coco_det_results_one_category(detections[cls_ind], coco_cat_id))
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_det_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.imageset_index):
            if len(boxes[im_ind]) == 0:
                continue
            dets = boxes[im_ind].astype(np.float)
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'bbox': [xs[k], ys[k], ws[k], hs[k]],
                       'score': scores[k]} for k in xrange(dets.shape[0])]
            results.extend(result)
        return results

    def evaluate_keypoints(self, keypoints, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'person_keypoints_%s_%s_result.json' % (self.imageset_name, alg))
        self.write_coco_kps_results(keypoints, res_file)
        if 'test' not in self.imageset_name:
            ann_type = 'keypoints'
            coco_kps = self.coco.loadRes(res_file)
            coco_eval = COCOeval(self.coco, coco_kps, ann_type)
            coco_eval.params.imgIds = self.imageset_index
            coco_eval.evaluate()
            coco_eval.accumulate()
            logging.info('keypoint result:')
            coco_eval.summarize()

    def write_coco_kps_results(self, keypoints, res_file):
        """ example results
        [{'image_id' : 42,
          'category_id' : 1,
          'keypoints' : [x1, y1, v1, ..., xk, yk, vk]
          'score' : 0.98}]
        """
        results = []
        for im_ind, index in enumerate(self.imageset_index):
            points = keypoints[im_ind]
            if len(points) == 0:
                continue
            result = [{'image_id': index,
                       'category_id': 1,
                       'keypoints': point[0:-1],
                       'score': point[-1]} for point in points]
            results.extend(result)
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def evalute_sds(self, detections, masks, binary_thresh=0.5, alg='alg', res_folder=''):
        res_file = os.path.join(res_folder, 'detections_%s_%s_results.json' % (self.imageset_name, alg))
        self.write_coco_sds_results(detections, masks, binary_thresh, res_file)
        if 'test' not in self.imageset_name:
            ann_type = 'segm'
            coco_dt = self.coco.loadRes(res_file)
            coco_eval = COCOeval(self.coco, coco_dt)
            coco_eval.params.useSegm = (ann_type == 'segm')
            coco_eval.params.imgIds = self.imageset_index
            coco_eval.evaluate()
            coco_eval.accumulate()
            logging.info('sds result:')
            coco_eval.summarize()

    def write_coco_sds_results(self, detections, masks, binary_thresh, res_file):
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Collecting %s results (%d/%d)' % (cls, cls_ind, self.num_classes - 1)
            coco_cat_id = self._class_to_coco_ind[cls]
            results.extend(self._coco_sds_results_one_category(detections[cls_ind], masks[cls_ind], binary_thresh, coco_cat_id))
        print 'Writing results json to %s' % res_file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_sds_results_one_category(self, boxes, masks, binary_thresh, cat_id):
        results = []
        for im_ind, index in enumerate(self.imageset_index):
            if len(boxes[im_ind]) == 0:
                continue
            dets = boxes[im_ind].astype(np.float)
            scores = dets[:, -1]
            height = self.coco.loadImgs(index)[0]['height']
            width = self.coco.loadImgs(index)[0]['width']
            mask_encode = mask_voc2coco(masks[im_ind], dets[:, :4], height, width, binary_thresh)
            result = [{'image_id': index,
                       'category_id': cat_id,
                       'segmentation': mask_encode[k],
                       'score': scores[k]} for k in xrange(dets.shape[0])]
            results.extend(result)
        return results
