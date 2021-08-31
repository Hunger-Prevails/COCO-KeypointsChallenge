import numpy as np
import mxnet as mx
import copy
from mxnet.executor_manager import _split_input_slice
from fpn_get_batch import assign_anchor_fpn, assign_anchor
from common.processing.image import tensor_vstack, transform
from common.base_iter import BaseIter, BaseTestIter
from common.processing.image_aug import get_image, aug_data_func
from common.utils.utils import makediv, serialize


class FPNIter(BaseIter):
    def __init__(self, roidb, config, batch_size, ctx=None):
        super(FPNIter, self).__init__(roidb, config, batch_size, ctx)
        self.rpn_feat_sym = config.network.rpn_feat_sym

        self.data_name = ['data']
        self.label_name = []
        if 'only_rpn' not in self.config.network.task_type:
            self.data_name.append('im_info')
        for branch_i in range(self.config.network.rpn_rcnn_num_branch):
            suffix = '_branch{}'.format(branch_i) if self.config.network.rpn_rcnn_num_branch > 1 else ''
            self.label_name.extend(['rpn_label%s' % suffix, 'rpn_bbox_target%s' % suffix, 'rpn_bbox_weight%s' % suffix])
            if 'only_rpn' not in self.config.network.task_type:
                self.data_name.append('gt_roidb%s' % suffix)
        self.max_data_shape, self.max_label_shape = self.infer_max_shape()

        self.kwargs = dict()
        self.kwargs['flip'] = self.config.TRAIN.aug_strategy.flip
        self.kwargs['aug_img'] = self.config.TRAIN.aug_strategy.aug_img
        self.kwargs['rotated_angle_range'] = self.config.TRAIN.aug_strategy.rotated_angle_range
        self.kwargs['scales'] = self.config.TRAIN.aug_strategy.scales
        self.kwargs['image_stride'] = self.config.network.image_stride

    def infer_max_shape(self):
        data_channel = 3
        max_data_height = makediv(max([v[0] for v in self.config.TRAIN.aug_strategy.scales]), self.config.network.image_stride)
        max_data_width = makediv(max([v[1] for v in self.config.TRAIN.aug_strategy.scales]), self.config.network.image_stride)
        max_data_shape = [('data', (self.batch_size, data_channel, max_data_height, max_data_width))]
        max_label_shape = self.infer_max_label_shape(max_data_shape)
        if 'only_rpn' not in self.config.network.task_type:
            for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                suffix = '_branch{}'.format(branch_i) if self.config.network.rpn_rcnn_num_branch > 1 else ''
                max_data_shape.append(('gt_roidb%s' % suffix, (self.batch_size, 50000)))
        return max_data_shape, max_label_shape

    def infer_max_label_shape(self, max_data_shape):
        label = dict()
        if 'fpn' in self.config.network.task_type:
            for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                suffix = '_branch{}'.format(branch_i) if self.config.network.rpn_rcnn_num_branch > 1 else ''
                feat_shape_list = []
                for i in range(len(self.config.network.rpn_feat_stride)):
                    feat_shape = self.rpn_feat_sym[branch_i][i].infer_shape(**dict(max_data_shape))[1][0]
                    feat_shape_list.append(feat_shape)
                label.update(assign_anchor_fpn(feat_shape=feat_shape_list,
                                               gt_boxes=np.zeros((0, 5)),
                                               config=self.config,
                                               feat_strides=self.config.network.rpn_feat_stride,
                                               scales=self.config.network.rpn_anchor_scales,
                                               ratios=self.config.network.rpn_anchor_ratios,
                                               suffix=suffix))
        elif 'frcnn' in self.config.network.task_type:
            max_data_height = max_data_shape[0][1][2]
            max_data_width = max_data_shape[0][1][3]
            for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                suffix = '_branch{}'.format(branch_i) if self.config.network.rpn_rcnn_num_branch > 1 else ''
                feat_shape = self.rpn_feat_sym[branch_i].infer_shape(**dict(max_data_shape))[1][0]
                im_info = [[max_data_height, max_data_width, 1.0]]
                label.update(assign_anchor(feat_shape=feat_shape,
                                           gt_boxes=np.zeros((0, 5)),
                                           im_info=im_info,
                                           config=self.config,
                                           feat_stride=self.config.network.rpn_feat_stride,
                                           scales=self.config.network.rpn_anchor_scales,
                                           ratios=self.config.network.rpn_anchor_ratios,
                                           suffix=suffix))
        else:
            raise ValueError("unknown task type {}".format(self.config.network.task_type))
        label = [label[k] for k in self.label_name]
        max_label_shape = [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_label_shape

    def get_label(self, data_tensor, im_info_list, gt_roidb_list):
        label_list = []
        if 'fpn' in self.config.network.task_type:
            feat_shape_list = []
            for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                feat_shape_each_branch = []
                for i in range(len(self.config.network.rpn_feat_stride)):
                    feat_shape = self.rpn_feat_sym[branch_i][i].infer_shape(data=data_tensor.shape)[1][0]
                    feat_shape_each_branch.append(feat_shape)
                feat_shape_list.append(feat_shape_each_branch)
            for i in range(self.batch_size):
                batch_labels = dict()
                for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                    suffix = '_branch{}'.format(branch_i) if self.config.network.rpn_rcnn_num_branch > 1 else ''
                    label = assign_anchor_fpn(feat_shape=feat_shape_list[branch_i],
                                              gt_boxes=gt_roidb_list[i][branch_i]['boxes'],
                                              ignore_regions=gt_roidb_list[i][branch_i]['ignore_regions'],
                                              config=self.config,
                                              feat_strides=self.config.network.rpn_feat_stride,
                                              scales=self.config.network.rpn_anchor_scales,
                                              ratios=self.config.network.rpn_anchor_ratios,
                                              suffix=suffix)
                    batch_labels.update(label)
                label_list.append(batch_labels)
        elif 'frcnn' in self.config.network.task_type:
            feat_shape_list = []
            for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                feat_shape = self.rpn_feat_sym[branch_i].infer_shape(data=data_tensor.shape)[1][0]
                feat_shape_list.append(feat_shape)
            for i in range(self.batch_size):
                batch_labels = dict()
                for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                    suffix = '_branch{}'.format(branch_i) if self.config.network.rpn_rcnn_num_branch > 1 else ''
                    label = assign_anchor(feat_shape=feat_shape_list[branch_i],
                                          im_info=im_info_list[i],
                                          gt_boxes=gt_roidb_list[i][branch_i]['boxes'],
                                          ignore_regions=gt_roidb_list[i][branch_i]['ignore_regions'],
                                          config=self.config,
                                          feat_stride=self.config.network.rpn_feat_stride,
                                          scales=self.config.network.rpn_anchor_scales,
                                          ratios=self.config.network.rpn_anchor_ratios,
                                          suffix=suffix)
                    batch_labels.update(label)
                label_list.append(batch_labels)
        else:
            raise ValueError("unknown task type {}".format(self.config.network.task_type))
        return label_list

    def get_gt_roidb(self, gt_boxes, gt_classes, res_dict):
        gt_boxes = np.hstack((gt_boxes, gt_classes.reshape((-1, 1))))
        gt_roidb = []
        for branch_i in range(self.config.network.rpn_rcnn_num_branch):
            gt_roidb_i = dict()
            if self.config.network.rpn_rcnn_num_branch == 1:
                gt_roidb_i['boxes'] = gt_boxes[gt_classes > 0, :]
                gt_roidb_i['ignore_regions'] = gt_boxes[gt_classes == -1, :]
                if 'kps' in self.config.network.task_type:
                    gt_roidb_i['keypoints'] = res_dict['all_kps']
                if 'mask' in self.config.network.task_type:
                    gt_roidb_i['polys'] = res_dict['all_polys']
            else:
                gt_roidb_i['boxes'] = gt_boxes[gt_classes == (branch_i + 1), :]
                gt_roidb_i['ignore_regions'] = gt_boxes[gt_classes == -(branch_i + 1), :]
            gt_roidb.append(gt_roidb_i)
        return gt_roidb

    def serialize_gt_roidb(self, gt_roidb_list):
        max_roidb_len = 0
        seri_gt_roidb_list = []
        for i in range(len(gt_roidb_list)):
            gt_roidb = serialize(gt_roidb_list[i])
            max_roidb_len = gt_roidb.shape[0] if gt_roidb.shape[0] > max_roidb_len else max_roidb_len
            seri_gt_roidb_list.append(gt_roidb)
        seri_gt_roidb = np.zeros((len(seri_gt_roidb_list), max_roidb_len + 1), dtype=np.float32)
        for i in range(len(seri_gt_roidb_list)):
            roidb_len = seri_gt_roidb_list[i].shape[0]
            seri_gt_roidb[i, 0] = roidb_len
            seri_gt_roidb[i, 1:roidb_len + 1] = seri_gt_roidb_list[i]
        return seri_gt_roidb

    def get_batch(self):
        index_start = self.cur
        index_end = min(index_start + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(index_start, index_end)]
        slices = _split_input_slice(index_end - index_start, self.work_load_list)

        data_list = []
        im_info_list = []
        gt_roidb_list = []
        for i_slice in slices:
            i_roidb = [roidb[i] for i in range(i_slice.start, i_slice.stop)]
            for j in range(len(i_roidb)):
                roidb_j = i_roidb[j]
                im = get_image(roidb_j, self.imgrec)
                all_boxes = roidb_j['boxes'].copy()
                gt_classes = roidb_j['gt_classes']
                keep = np.where(gt_classes > 0)[0]
                all_kps = roidb_j['keypoints'][keep, :].copy() if 'kps' in self.config.network.task_type else None
                all_polys = copy.deepcopy([roidb_j['gt_masks'][_] for _ in keep]) if 'mask' in self.config.network.task_type else None
                res_dict = aug_data_func(im, all_boxes=all_boxes, all_kps=all_kps, all_polys=all_polys, **self.kwargs)
                im_tensor = transform(res_dict['img'], pixel_means=self.config.network.input_mean, scale=self.config.network.input_scale)  # (1, 3, h, w)
                data_list.append(im_tensor)
                im_info_list.append(np.array([[im_tensor.shape[2], im_tensor.shape[3], res_dict['img_scale']]], dtype=np.float32))
                gt_roidb = self.get_gt_roidb(res_dict['all_boxes'], gt_classes, res_dict)
                gt_roidb_list.append(gt_roidb)
        data_tensor = tensor_vstack(data_list)

        all_data = dict()
        all_data['data'] = data_tensor

        if 'only_rpn' not in self.config.network.task_type:
            all_data['im_info'] = tensor_vstack(im_info_list)
            for branch_i in range(self.config.network.rpn_rcnn_num_branch):
                suffix = '_branch{}'.format(branch_i) if self.config.network.rpn_rcnn_num_branch > 1 else ''
                gt_roidb_i = [gt_roidb_list[i][branch_i] for i in range(len(gt_roidb_list))]
                all_data['gt_roidb%s' % suffix] = self.serialize_gt_roidb(gt_roidb_i)

        label_list = self.get_label(data_tensor, im_info_list, gt_roidb_list)

        all_label = dict()
        for name in self.label_name:
            pad = -1 if 'label' in name else 0
            all_label[name] = tensor_vstack([batch[name] for batch in label_list], pad=pad)

        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]


class FPNTestIter(BaseTestIter):
    def __init__(self, roidb, config, batch_size):
        super(FPNTestIter, self).__init__(roidb, config, batch_size)
        self.data_name = ['data', 'im_info']
        if self.config.TEST.use_gt_rois:
            self.data_name.append('rois')

        self.label_name = None
        self.max_data_shape, self.max_label_shape = self.infer_max_shape()

        self.kwargs = dict()
        self.kwargs['scales'] = self.config.TEST.aug_strategy.scales
        self.kwargs['image_stride'] = self.config.network.image_stride

        self.get_batch()

    def infer_max_shape(self):
        data_channel = 3
        max_data_height = makediv(max([v[0] for v in self.config.TEST.aug_strategy.scales]), self.config.network.image_stride)
        max_data_width = makediv(max([v[1] for v in self.config.TEST.aug_strategy.scales]), self.config.network.image_stride)
        max_data_shape = [('data', (1, data_channel, max_data_height, max_data_width)), ('im_info', (1, 3))]
        if self.config.TEST.use_gt_rois:
            max_data_shape.append(('rois', (1, 100, 5)))
        return max_data_shape, None

    def get_one_roidb(self, roidb_j, j=0):
        data = dict()
        local_vars = dict()
        im = get_image(roidb_j, self.imgrec, self.video_reader, self.camera_reader)
        local_vars['im'] = im.copy()
        local_vars['roi_rec'] = roidb_j
        need_forward = True
        if self.config.TEST.use_gt_rois:
            all_boxes = roidb_j['boxes'].copy()
            if all_boxes.shape[0] > 0:
                res_dict = aug_data_func(im, all_boxes, **self.kwargs)
                im = res_dict['img']
                im_scale = res_dict['img_scale']
                aug_all_boxes = res_dict['all_boxes']
                data['rois'] = np.full((1, aug_all_boxes.shape[0], 5), fill_value=j, dtype=all_boxes.dtype)
                data['rois'][:, :, 1:] = aug_all_boxes
            else:
                data['rois'] = np.full((1, 0, 5), fill_value=j, dtype=all_boxes.dtype)
                im_scale = 1.0
                need_forward = False
        else:
            local_vars['im_height'] = im.shape[0]
            local_vars['im_width'] = im.shape[1]
            res_dict = aug_data_func(im, **self.kwargs)
            im = res_dict['img']
            im_scale = res_dict['img_scale']
        local_vars['im_scale'] = im_scale
        im_tensor = transform(im, pixel_means=self.config.network.input_mean, scale=self.config.network.input_scale)  # (1, 3, h, w)
        data['data'] = im_tensor
        data['im_info'] = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]])
        self.extra_local_vars.append(local_vars)
        return data, need_forward