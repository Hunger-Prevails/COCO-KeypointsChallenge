import logging
import mxnet as mx
from common.processing.nms import py_nms_wrapper, py_softnms_wrapper
from common.base_predictor import BasePredictor
from fpn_multitask_predict_utils import e2e_multitask_predict
from fpn.fpn_iter import FPNTestIter


def eval_during_training_func(roidb, eval_func, config):
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    test_data = FPNTestIter(roidb=roidb, config=config, batch_size=len(test_ctx))
    def _callback(iter_no, sym, arg, aux):
        predictor = FPNMultitaskPredictor(config, config.TRAIN.model_prefix, iter_no + 1,
                                          test_data.provide_data, test_data.max_data_shape, test_ctx)
        predictor.predict_multitask(test_data, eval_func)
    return _callback


class FPNMultitaskPredictor(BasePredictor):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shape, ctx=mx.cpu()):
        super(FPNMultitaskPredictor, self).__init__(config, prefix, epoch, provide_data, max_data_shape, ctx=ctx,
                                                    allow_missing=config.network.params_allow_missing)
        self.new_roidb = []
        if 'kps' in config.network.task_type and 'm2' in config.TRAIN.kps_loss_type:
            from keypoints.kps_get_batch_2 import kps_compute_base_anchor_centers
            self.config.base_anchor_centers = kps_compute_base_anchor_centers(feat_height=config.network.kps_height,
                                                                              feat_width=config.network.kps_width,
                                                                              feat_stride=config.network.kps_feat_stride)   # (num_anchors, 2)

    def predict_multitask(self, test_data, eval_func=None, alg='alg', score_thresh=1e-3,
                          save_roidb_path=None, vis=False, **vis_kwargs):
        k = 0
        num_classes = self.config.dataset.num_classes
        assert num_classes == 2
        if self.config.TEST.rcnn_use_softnms:
            nms_det = py_softnms_wrapper(self.config.TEST.rcnn_softnms)
        else:
            nms_det = py_nms_wrapper(self.config.TEST.rcnn_nms)
        all_boxes = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_classes)]
        all_kps_results = []
        all_mask_boxes = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_classes)]
        all_masks = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_classes)]
        for data_batch, need_forward in test_data:
            if k % 100 == 0:
                logging.info('{}/{}'.format(k, test_data.size))
            outputs = self.predict(data_batch, need_forward)
            for i in range(len(outputs)):
                outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]
                local_vars_i = test_data.extra_local_vars[i]
                res_dict = e2e_multitask_predict(outputs_i, local_vars_i, self.config, nms_det, score_thresh)
                all_boxes[1][k + i] = res_dict['det_results']
                all_kps_results.append(res_dict['kps_results'])
                all_mask_boxes[1][k + i] = res_dict['mask_boxes']
                all_masks[1][k + i] = res_dict['masks']
                if save_roidb_path is not None:
                    roi_rec = local_vars_i['roi_rec'].copy()
                    for name in res_dict:
                        roi_rec[name] = res_dict[name]
                    self.new_roidb.append(roi_rec)
                if vis:
                    self.vis_results(local_vars=local_vars_i,
                                     det_results=res_dict['det_results'],
                                     kps_results=res_dict['kps_results'],
                                     mask_results={'mask_boxes': res_dict['mask_boxes'],
                                                   'masks': res_dict['masks']},
                                     **vis_kwargs)
            k += test_data.batch_size
        test_data.reset()
        if save_roidb_path is not None:
            self.save_roidb(self.new_roidb, save_roidb_path)
            self.new_roidb = []
        if eval_func is not None:
            eval_func(all_boxes=all_boxes,
                      all_kps_results=all_kps_results,
                      all_mask_boxes=all_mask_boxes,
                      all_masks=all_masks,
                      alg=alg)
