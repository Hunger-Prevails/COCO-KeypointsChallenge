import os
import cPickle
import logging
import multiprocessing
import numpy as np
import mxnet as mx
from common.base_predictor import BasePredictor
from kps_iter import KPSTestIter
from kps_predict_utils import kps_predict


def eval_during_training_func(roidb, eval_func, config):
    test_ctx = [mx.gpu(int(i)) for i in config.TEST.gpus.split(',')]
    test_data = KPSTestIter(roidb=roidb, config=config, batch_size=len(test_ctx))
    def _callback(iter_no, sym, arg, aux):
        predictor = KPSPredictor(config, config.TRAIN.model_prefix, iter_no + 1,
                                 test_data.provide_data, test_data.max_data_shape, test_ctx)
        predictor.predict_data(test_data, eval_func=eval_func)
    return _callback


class KPSPredictor(BasePredictor):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shape, ctx=mx.cpu(), use_thread=False):
        super(KPSPredictor, self).__init__(config, prefix, epoch, provide_data, max_data_shape, ctx=ctx)

        self.new_roidb = []
        self.use_thread = use_thread

        if use_thread:
            self.result_list = []
            self.pool_handler = multiprocessing.Pool(processes = len(self.mod_list) * 2)
            logging.info('use thread for test')

    def predict_data(self, test_data, eval_func=None, alg='alg', save_roidb_path=None, vis=False, **vis_kwargs):
        all_kps_results = []
        k = 0
        for data_batch, need_forward in test_data:
            if k % 100 == 0:
                logging.info('{}/{}'.format(k, test_data.size))
            k += test_data.batch_size
            outputs = self.predict(data_batch, need_forward)  # list of outputs for each image

            for i in xrange(len(outputs)):
                outputs_i = [outputs[i][j].asnumpy() for j in range(len(outputs[i]))]  # outputs for a single image
                local_vars_i = test_data.extra_local_vars[i]  # extra vars for a single image

                if self.config.TEST.heatmap_fusion:
                    roi_rec = local_vars_i['roi_rec']
                    image_id = int(os.path.split(roi_rec['image'])[1].split('.')[0])
                    rois = outputs_i[0][:, 1:]
                    heatmaps = outputs_i[1]
                    offsets = outputs_i[2]

                    if config.TEST.aug_strategy.kps_flip_test:
                        rois = rois[::2]
                        offsets = offsets[::2]

                        native_scores = heatmaps[::2]  # (num_boxes, num_kps, feat_height, feat_width)
                        _mirror_scores = heatmaps[1::2]  # (num_boxes, num_kps, feat_height, feat_width)
                        
                        mirror_scores = np.zeros(_mirror_scores.shape)

                        mirror_scores[:, 0, :, :] = _mirror_scores[:, 0, :, ::-1]
                        mirror_scores[:, 1::2, :, :] = _mirror_scores[:, 2::2, :, ::-1]
                        mirror_scores[:, 2::2, :, :] = _mirror_scores[:, 1::2, :, ::-1]

                        heatmaps = native_scores / 2 + mirror_scores / 2

                    new_roidb = dict(
                                    image_id = image_id,
                                    rois = rois,
                                    heatmaps = heatmaps,
                                    offsets = offsets)

                    if '/opt' in save_roidb_path:
                        with open(str(image_id) + '.pkl', 'wb') as fid:
                            cPickle.dump(new_roidb, fid, cPickle.HIGHEST_PROTOCOL)
                        cmd = 'cp -f %s %s' % (str(image_id) + '.pkl', os.path.join(save_roidb_path, str(image_id) + '.pkl'))
                        os.system(cmd)
                        os.system(cmd)
                    else:
                        with open(os.path.join(save_roidb_path, str(image_id) + '.pkl'), 'wb') as fid:
                            cPickle.dump(new_roidb, fid, cPickle.HIGHEST_PROTOCOL)
                    continue

                if self.use_thread:
                    self.result_list.append(self.pool_handler.apply_async(kps_predict, (outputs_i, local_vars_i, self.config)))
                else:
                    kps_results, keep = kps_predict(outputs_i, local_vars_i, self.config)
                    all_kps_results.append(kps_results)
                    if save_roidb_path is not None:
                        roi_rec = local_vars_i['roi_rec'].copy()
                        roi_rec['kps_pred'] = np.array(kps_results)
                        roi_rec['box_keep'] = keep
                        self.new_roidb.append(roi_rec)
                    if vis:
                        self.vis_results(local_vars = local_vars_i,
                                         det_results = local_vars_i['roi_rec']['boxes'],
                                         kps_results = kps_results,
                                         **vis_kwargs)
        if self.config.TEST.heatmap_fusion:
            return

        if self.use_thread:
            self.pool_handler.close()
            self.pool_handler.join()
            for result in self.result_list:
                all_kps_results.append(result.get())

        test_data.reset()
        if save_roidb_path is not None:
            self.save_roidb(self.new_roidb, save_roidb_path)
            self.new_roidb = []
        if eval_func is not None:
            eval_func(all_kps_results = all_kps_results, alg = alg)
