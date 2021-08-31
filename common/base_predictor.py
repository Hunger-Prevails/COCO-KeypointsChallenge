from importlib import import_module
import os
import logging
import cPickle
import cv2
import numpy as np
import mxnet as mx
from common.module import MutableModule
from common.utils.utils import load_param
from common.processing.image_draw import draw_all
from common.symbols.sym_common import get_sym_func


class BasePredictor(object):
    def __init__(self, config, prefix, epoch, provide_data, max_data_shapes=None, ctx=mx.cpu(), allow_missing=False):
        self.config = config
        logging.info('load model from %s-%04d.params' % (prefix, epoch))
        if self.config.TEST.load_sym_from_file:
            symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        else:
            symbol = get_sym_func(config.network.sym)(config, is_train=False)
            arg_params, aux_params = load_param(prefix, epoch)
        if not isinstance(ctx, list):
            ctx = [ctx]
        data_names = [k[0] for k in provide_data]
        self.mod_list = []
        for a_ctx in ctx:
            mod = MutableModule(symbol, data_names, None, context=a_ctx, max_data_shapes=max_data_shapes)
            mod.bind(provide_data, for_training=False)
            mod.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=allow_missing)
            self.mod_list.append(mod)
        self.count = 0

    def predict(self, data_batch, need_forward):
        for i in range(len(data_batch)):
            if need_forward[i]:
                self.mod_list[i].forward(data_batch[i])
        outputs = []
        for i in range(len(data_batch)):
            if need_forward[i]:
                outputs.append(self.mod_list[i].get_outputs())
            else:
                outputs.append([])
        return outputs

    def save_roidb(self, roidb, save_roidb_path):
        if os.path.exists(save_roidb_path):
            os.system('rm %s' % save_roidb_path)
        if '/opt' in save_roidb_path:
            local_save_roidb_path = os.path.basename(save_roidb_path)
            with open(local_save_roidb_path, 'wb') as fid:
                cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
            cmd = 'cp -f %s %s' % (local_save_roidb_path, save_roidb_path)
            os.system(cmd)
            os.system(cmd)
        else:
            with open(save_roidb_path, 'wb') as fid:
                cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)

    def vis_results(self, local_vars, multiclass_det=False, det_results=None, kps_results=None, mask_results=None,
                    box_color=None, point_color=None, skeleton_color=None, mask_color=None,
                    do_draw_box=True, im_save_dir=None, im_save_max_num=-1,
                    writer=None, show_camera=False):
        if box_color is None:
            box_color = (0, 255, 255)
        if point_color is None:
            point_color = (0, 255, 255)
        if skeleton_color is None:
            skeleton_color = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255]]
        if mask_color is None:
            mask_color = (240, 240, 240)  # (60, 20, 220)
        if multiclass_det:
            assert kps_results is None
            assert mask_results is None
            im_draw = local_vars['im']
            for j in range(1, self.config.dataset.num_classes):
                all_boxes = det_results[j][:, :4]
                if len(all_boxes) > 0:
                    im_draw = draw_all(im=local_vars['im'],
                                       all_boxes=all_boxes,
                                       box_color=box_color)
        else:
            all_boxes = det_results[:, :4] if det_results is not None and len(det_results) > 0 else None
            all_kps = np.array(kps_results)[:, :-1] if kps_results is not None and len(kps_results) > 0 else None
            if mask_results is not None:
                mask_boxes = mask_results['mask_boxes'] if len(mask_results['mask_boxes']) > 0 else None
                masks = mask_results['masks'] if len(mask_results['masks']) > 0 else None
            else:
                mask_boxes = None
                masks = None
            im_draw = draw_all(im=local_vars['im'],
                               all_boxes=all_boxes,
                               all_kps=all_kps,
                               skeleton=self.config.dataset.kps_skeleton,
                               all_mask_boxes=mask_boxes,
                               all_masks=masks,
                               box_color=box_color,
                               point_color=point_color,
                               skeleton_color=skeleton_color,
                               mask_color=mask_color,
                               kps_thresh=0.2,
                               show_num=False,
                               do_draw_box=do_draw_box)

        if im_save_dir is not None:
            if im_save_max_num == -1 or self.count < im_save_max_num:
                im_name = os.path.splitext(os.path.basename(local_vars['roi_rec']['image']))[0]
                im_save_path = os.path.join(im_save_dir, im_name + '_draw.jpg')
                cv2.imwrite(im_save_path, im_draw)
                self.count += 1
        if writer is not None:
            writer.append_data(im_draw[:, :, ::-1])
        if show_camera:
            cv2.imshow('test', im_draw)
            cv2.waitKey(1)