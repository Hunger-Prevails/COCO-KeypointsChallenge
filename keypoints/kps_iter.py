import numpy as np
from kps_get_batch import kps_get_train_batch, kps_get_test_batch, kps_generate_new_rois
from common.utils.utils import makediv
from common.processing.image_aug import get_image, aug_data_func
from common.base_iter import BaseIter, BaseTestIter
from common.processing.image_roi import get_roi_images
from common.processing.image import transform


class KPSIter(BaseIter):
    def __init__(self, roidb, config, batch_size, ctx=None, work_load_list=None):
        super(KPSIter, self).__init__(roidb, config, batch_size, ctx, work_load_list)
        data_channel = 3
        if config.network.kps_crop_from_image:
            self.data_name = ['data']
            self.max_data_shape = [('data', (self.batch_size * config.TRAIN.kps_roi_batch_size, data_channel,
                                             config.network.kps_input_height, config.network.kps_input_width))]
        else:
            self.data_name = ['data', 'rois']
            self.max_data_shape = [('data', (self.batch_size, data_channel,
                                             makediv(max([v[0] for v in config.TRAIN.aug_strategy.scales]), config.network.image_stride),
                                             makediv(max([v[1] for v in config.TRAIN.aug_strategy.scales]), config.network.image_stride)))]

        assert len(config.TRAIN.kps_loss_type_list) == 1

        if 'pixel' in config.TRAIN.kps_loss_type:
            self.label_name = ['kps_label', 'kps_label_mask', 'kps_offset', 'kps_offset_mask']

            if 'aux_suv' in config.TRAIN.kps_loss_type:
                aux_names = ['kps_aux_label_', 'kps_aux_label_mask_', 'kps_aux_offset_', 'kps_aux_offset_mask_']

                if 'fpn' in config.network.sym:
                    self.label_name += [name + str(4) for name in aux_names]
                elif 'deconv' in config.network.sym:
                    self.label_name += [name + str(32) for name in aux_names]
                else:
                    raise ValueError('invalid network symbol for auxiliary supervision')

        elif 'one_hot' in config.TRAIN.kps_loss_type:
            self.label_name.extend(['kps_label', 'kps_offset', 'kps_offset_mask'])
        else:
            raise ValueError("unknown kps loss type {}".format(kps_loss_type))

        self.max_label_shape = None

        self.kwargs = dict()
        self.kwargs['flip'] = config.TRAIN.aug_strategy.flip
        self.kwargs['aug_img'] = config.TRAIN.aug_strategy.aug_img
        self.kwargs['rotated_angle_range'] = config.TRAIN.aug_strategy.rotated_angle_range
        self.kwargs['scales'] = None if config.network.kps_crop_from_image else config.TRAIN.aug_strategy.scales
        self.kwargs['image_stride'] = config.network.image_stride

    def get_one_roidb(self, roidb_j, idx = 0):
        config = self.config
        im = get_image(roidb_j, self.imgrec)
        res_dict = aug_data_func(im, roidb_j['boxes'].copy(), roidb_j['keypoints'].copy(), **self.kwargs)
        im = res_dict['img']
        gt_boxes = res_dict['all_boxes']
        gt_kps = res_dict['all_kps']
    
        assert len(config.TRAIN.kps_loss_type_list) == 1
        assert gt_boxes.shape[0] == gt_kps.shape[0]
        assert gt_boxes.shape[0] > 0

        num_kps = config.dataset.num_kps
        feat_height = config.network.kps_height
        feat_width = config.network.kps_width

        aspect_ratio = float(feat_height) / feat_width if config.TRAIN.aug_strategy.kps_do_aspect_ratio else 0.0

        gt_boxes, all_inds = kps_generate_new_rois(
                                roi_boxes = gt_boxes,
                                roi_batch_size = config.TRAIN.kps_roi_batch_size,
                                rescale_factor = config.TRAIN.aug_strategy.kps_rescale_factor,
                                jitter_center = config.TRAIN.aug_strategy.kps_jitter_center,
                                aspect_ratio = aspect_ratio)
        gt_kps = gt_kps[all_inds, :]
        gt_kps = gt_kps.reshape((-1, num_kps, 3))

        label = kps_get_train_batch(all_boxes = gt_boxes, all_kps = gt_kps, config = config)

        if 'aux_suv' in config.TRAIN.kps_loss_type:
            if 'fpn' in config.network.sym:
                label.update(kps_get_train_batch(all_boxes = gt_boxes, all_kps = gt_kps, config = config, aux_stride = 4))
            elif 'deconv' in config.network.sym:
                label.update(kps_get_train_batch(all_boxes = gt_boxes, all_kps = gt_kps, config = config, aux_stride = 32))
            else:
                raise ValueError('invalid network symbol for auxiliary supervision')

        data = dict()
        if config.network.kps_crop_from_image:
            data['data'] = get_roi_images(im, gt_boxes.tolist(),
                                          [config.network.kps_input_height, config.network.kps_input_width],
                                          input_mean=config.network.input_mean,
                                          scale=config.network.input_scale)   # (roi_batch_size, 3, h, w)
            for label_name in label:
                label[label_name] = label[label_name].reshape((-1,) + label[label_name].shape[2:])
        else:
            data['data'] = transform(im, pixel_means=config.network.input_mean, scale=config.network.input_scale)  # (1, 3, h, w)
            data['rois'] = np.full((1, gt_boxes.shape[0], 5), fill_value=idx, dtype=gt_boxes.dtype)
            data['rois'][0, :, 1:] = gt_boxes

        return data, label


class KPSTestIter(BaseTestIter):
    def __init__(self, roidb, config, batch_size):
        super(KPSTestIter, self).__init__(roidb, config, batch_size)
        data_channel = 3
        num_rois = 30 if config.TEST.use_gt_rois else 100
        
        if config.network.kps_crop_from_image:
            self.data_name = ['data', 'rois']   # 'rois' for post process , not for network
            self.max_data_shape = [('data', (num_rois, data_channel,
                                             config.network.kps_input_height, config.network.kps_input_width))]
            self.max_data_shape.append(('rois', (num_rois, 5)))
        else:
            self.data_name = ['data', 'rois']
            self.max_data_shape = [('data', (1, data_channel,
                                             makediv(max([v[0] for v in config.TEST.aug_strategy.scales]), config.network.image_stride),
                                             makediv(max([v[1] for v in config.TEST.aug_strategy.scales]), config.network.image_stride)))]
            self.max_data_shape.append(('rois', (1, num_rois, 5)))

        self.label_name = None
        self.max_label_shape = None

        self.kwargs = dict()
        self.kwargs['scales'] = None if config.network.kps_crop_from_image else config.TEST.aug_strategy.scales
        self.kwargs['image_stride'] = config.network.image_stride

        self.get_batch()

    def get_data(self, im, all_boxes, idx=0):
        config = self.config
        data = dict()
        res_dict = aug_data_func(im, all_boxes, **self.kwargs)
        
        aug_im = res_dict['img']
        aug_all_boxes = res_dict['all_boxes']
        im_scale = res_dict['img_scale']
        
        rois = kps_get_test_batch(all_boxes=aug_all_boxes, config=config)

        if config.network.kps_crop_from_image:
            crop_images = get_roi_images(aug_im, rois.tolist(),
                                          [config.network.kps_input_height, config.network.kps_input_width],
                                          input_mean=config.network.input_mean,
                                          scale=config.network.input_scale)  # (num_gt_boxes, 3, h, w)

            if config.TEST.aug_strategy.kps_flip_test:
                crop_images[1::2] = crop_images[1::2, :, :, ::-1]

            data['data'] = crop_images
            data['rois'] = np.full((rois.shape[0], 5), fill_value=idx, dtype=all_boxes.dtype)
            data['rois'][:, 1:] = rois  # (num_gt_boxes, 5)  [idx, x1, y1, x2, y2]
        else:
            data['data'] = transform(aug_im, pixel_means=config.network.input_mean, scale=config.network.input_scale)  # (1, 3, h, w)
            data['rois'] = np.full((1, rois.shape[0], 5), fill_value=idx, dtype=all_boxes.dtype)
            data['rois'][0, :, 1:] = rois  # (1, num_gt_boxes, 5)  [idx, x1, y1, x2, y2]
        
        return data, im_scale

    def get_one_roidb(self, roidb_j, idx=0):
        config = self.config
        local_vars = dict()
        if config.TEST.use_gt_rois:
            all_boxes = roidb_j['boxes'].copy()
        else:
            all_boxes = roidb_j['pred_boxes'][:, :4].copy()
            local_vars['rois'] = all_boxes
            local_vars['roi_scores'] = roidb_j['pred_boxes'][:, 4].copy()

        im = get_image(roidb_j, imgrec=self.imgrec, video_reader=self.video_reader, camera_reader=self.camera_reader)
        data, im_scale = self.get_data(im, all_boxes, idx=idx)
        
        local_vars['im_scale'] = im_scale
        local_vars['im'] = im
        local_vars['roi_rec'] = roidb_j

        self.extra_local_vars.append(local_vars)
        return data, all_boxes.shape[0] > 0
