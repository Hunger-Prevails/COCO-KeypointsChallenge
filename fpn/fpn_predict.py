import logging
import cPickle
import numpy as np
import mxnet as mx
from fpn_iter import FPNTestIter
from fpn_predict_utils import rpn_predict, rcnn_predict, rpn_rcnn_predict
from common.processing.nms import py_nms_wrapper, py_softnms_wrapper
from common.base_predictor import BasePredictor
from common.dataset.det_eval import evaluate_recall


def eval_during_training_func(roidb, eval_func, config):
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    test_data = FPNTestIter(roidb=roidb, config=config, batch_size=len(test_ctx))
    def _callback(iter_no, sym, arg, aux):
        predictor = FPNPredictor(config, config.TRAIN.model_prefix, iter_no + 1,
                                 test_data.provide_data, test_data.max_data_shape, test_ctx)
        if 'only_rpn' in config.network.task_type:
            predictor.predict_rpn(test_data)
        elif 'rpn_rcnn' in config.network.task_type:
            predictor.predict_rpn_rcnn(test_data, eval_func)
        else:
            raise ValueError("unknown task type {}".format(config.network.task_type))
    return _callback


class FPNPredictor(BasePredictor):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shape, ctx=mx.cpu()):
        super(FPNPredictor, self).__init__(config, prefix, epoch, provide_data, max_data_shape, ctx=ctx,
                                           allow_missing=config.network.params_allow_missing)
        self.new_roidb = []

    def predict_rpn(self, test_data):
        k = 0
        thresh = 0
        pred_boxes = list()
        for data_batch, need_forward in test_data:
            if k % 100 == 0:
                logging.info('{}/{}'.format(k, test_data.size))
            k += test_data.batch_size
            outputs = self.predict(data_batch, need_forward)
            for i in range(len(outputs)):
                outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]
                pred_boxes.append(rpn_predict(outputs_i, test_data.extra_local_vars[i], self.config, thresh=thresh))
        test_data.reset()
        evaluate_recall(test_data.roidb, candidate_boxes=pred_boxes)

    def predict_rpn_rcnn(self, test_data, eval_func=None, alg='alg', score_thresh=1e-3,
                         save_roidb_path=None, vis=False, **vis_kwargs):
        assert len(self.config.TEST.aug_strategy.scales) == 1
        k = 0
        if self.config.TEST.rcnn_use_softnms:
            nms = py_softnms_wrapper(self.config.TEST.rcnn_softnms)
        else:
            nms = py_nms_wrapper(self.config.TEST.rcnn_nms)

        if self.config.network.rpn_rcnn_num_branch > 1:
            assert self.config.dataset.num_classes == 2
            assert self.config.network.rcnn_class_agnostic is False
            num_rpn_classes = self.config.network.rpn_rcnn_num_branch + 1
            num_rcnn_classes = self.config.network.rpn_rcnn_num_branch + 1
        else:
            num_rpn_classes = 2
            num_rcnn_classes = self.config.dataset.num_classes
        all_proposals = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_rpn_classes)]
        all_boxes = [[[] for _ in xrange(test_data.size)] for _ in xrange(num_rcnn_classes)]

        for data_batch, need_forward in test_data:
            if k % 100 == 0:
                logging.info('{}/{}'.format(k, test_data.size))
            outputs = self.predict(data_batch, need_forward)
            for i in range(len(outputs)):
                outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]
                local_vars_i = test_data.extra_local_vars[i]
                if self.config.TEST.rpn_do_test:
                    proposals, pred_boxes, scores = rpn_rcnn_predict(outputs_i, local_vars_i, self.config)
                    for j in range(1, num_rpn_classes):
                        all_proposals[j][k + i] = proposals[:, (j - 1) * 5:j * 5]
                else:
                    pred_boxes, scores = rcnn_predict(outputs_i, local_vars_i, self.config)
                for j in range(1, num_rcnn_classes):
                    if self.config.network.rcnn_class_agnostic:
                        cls_boxes = pred_boxes
                    else:
                        cls_boxes = pred_boxes[:, (j - 1) * 4:j * 4]
                    cls_scores = scores[:, j - 1, np.newaxis]
                    keep = np.where(cls_scores > score_thresh)[0]
                    cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                    keep = nms(cls_dets)
                    all_boxes[j][k + i] = cls_dets[keep, :]
                if save_roidb_path is not None:
                    roi_rec = local_vars_i['roi_rec'].copy()
                    roi_rec['boxes'] = all_boxes[1][k + i]
                    self.new_roidb.append(roi_rec)
                if vis:
                    det_results = []
                    for j in range(1, num_rcnn_classes):
                        det_results.append(all_boxes[j][k + i])
                    self.vis_results(local_vars=local_vars_i,
                                     multiclass_det=True,
                                     det_results=all_boxes,
                                     **vis_kwargs)
                # max_per_image = 100
                # if max_per_image > 0:
                #     image_scores = np.hstack([all_boxes[j][k + i][:, -1] for j in range(1, num_classes)])
                #     if len(image_scores) > max_per_image:
                #         image_thresh = np.sort(image_scores)[-max_per_image]
                #         for j in range(1, num_classes):
                #             keep = np.where(all_boxes[j][k + i][:, -1] >= image_thresh)[0]
                #             all_boxes[j][k + i] = all_boxes[j][k + i][keep, :]
            k += test_data.batch_size
        test_data.reset()
        if save_roidb_path is not None:
            self.save_roidb(self.new_roidb, save_roidb_path)
            self.new_roidb = []
        if eval_func is not None:
            eval_func(all_proposals=all_proposals, all_boxes=all_boxes, alg=alg)

    def predict_rcnn_mst(self, test_data, eval_func=None, alg='alg'):
        thresh = 1e-3
        num_classes = self.config.dataset.num_classes
        num_images = test_data.size
        if self.config.TEST.rcnn_use_softnms:
            nms = py_softnms_wrapper(self.config.TEST.rcnn_softnms)
        else:
            nms = py_nms_wrapper(self.config.TEST.rcnn_nms)
        all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(num_classes)]

        for s in range(len(self.config.TEST.aug_strategy.scales)):
            k = 0
            test_data.kwargs['scales'] = [self.config.TEST.aug_strategy.scales[s]]
            all_boxes_single_scale = [[[] for _ in xrange(num_images)] for _ in xrange(num_classes)]
            for data_batch, need_forward in test_data:
                if k % 100 == 0:
                    logging.info('{}/{}'.format(k, test_data.size))
                outputs = self.predict(data_batch, need_forward)
                for i in range(len(outputs)):
                    outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]
                    pred_boxes, scores = rcnn_predict(outputs_i, test_data.extra_local_vars[i], self.config)
                    for j in range(1, num_classes):
                        if self.config.network.rcnn_class_agnostic:
                            cls_boxes = pred_boxes[:, 4:]
                        else:
                            cls_boxes = pred_boxes[:, j * 4:(j + 1) * 4]
                        cls_scores = scores[:, j, np.newaxis]
                        keep = np.where(cls_scores > thresh)[0]
                        cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                        all_boxes_single_scale[j][k + i] = cls_dets
                k += test_data.batch_size
            test_data.reset()
            res_file = 'detections_%s-scale%d_results.pkl' % (alg, s)
            with open(res_file, 'wb') as f:
                cPickle.dump(all_boxes_single_scale, f, cPickle.HIGHEST_PROTOCOL)

        logging.info('merge results from all test scales')
        for s in range(len(self.config.TEST.aug_strategy.scales)):
            res_file = 'detections_%s-scale%d_results.pkl' % (alg, s)
            with open(res_file, 'rb') as fid:
                all_boxes_single_scale = cPickle.load(fid)
            for idx_class in range(1, num_classes):
                for idx_im in range(num_images):
                    if len(all_boxes[idx_class][idx_im]) == 0:
                        all_boxes[idx_class][idx_im] = all_boxes_single_scale[idx_class][idx_im]
                    else:
                        all_boxes[idx_class][idx_im] = np.vstack((all_boxes[idx_class][idx_im], all_boxes_single_scale[idx_class][idx_im]))

        logging.info('do test nms')
        for idx_class in range(1, num_classes):
            for idx_im in range(num_images):
                keep = nms(all_boxes[idx_class][idx_im])
                all_boxes[idx_class][idx_im] = all_boxes[idx_class][idx_im][keep, :]

        # max_per_image = 100
        # if max_per_image > 0:
        #     for idx_im in range(num_images):
        #         image_scores = np.hstack([all_boxes[idx_class][idx_im][:, -1] for idx_class in range(1, num_classes)])
        #         if len(image_scores) > max_per_image:
        #             image_thresh = np.sort(image_scores)[-max_per_image]
        #             for idx_class in range(1, num_classes):
        #                 keep = np.where(all_boxes[idx_class][idx_im] >= image_thresh)[0]
        #                 all_boxes[idx_class][idx_im] = all_boxes[idx_class][idx_im][keep, :]

        if eval_func is not None:
            eval_func(all_boxes=all_boxes, alg=alg)

        # if do_save:
        #     rcnn_dir = os.path.join(self.config.hdfs_local_output_dir, 'rcnn_data')
        #     if not os.path.exists(rcnn_dir):
        #         os.makedirs(rcnn_dir)
        #     rcnn_file_name = 'detections_%s_%s_results.json' % (test_imdb.image_set, alg)
        #     rcnn_path = os.path.join(rcnn_dir, rcnn_file_name)
        #     if os.path.exists(rcnn_path):
        #         os.remove(rcnn_path)
        #     move_file(rcnn_file_name, rcnn_path)
        #
        #     rcnn_file_name = test_imdb.name + '_%s_rcnn.pkl' % alg
        #     with open(rcnn_file_name, 'wb') as f:
        #         cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
        #     rcnn_path = os.path.join(rcnn_dir, rcnn_file_name)
        #     if os.path.exists(rcnn_path):
        #         os.remove(rcnn_path)
        #     move_file(rcnn_file_name, rcnn_path)