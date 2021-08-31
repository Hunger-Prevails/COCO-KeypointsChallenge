import mxnet as mx
import numpy as np
import cPickle
from fpn.fpn_get_batch import sample_rois
from common.utils.utils import deserialize


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, config):
        super(ProposalTargetOperator, self).__init__()
        self._config = config
        self._fg_rois_per_image = np.round(config.TRAIN.rcnn_batch_rois * config.TRAIN.rcnn_fg_fraction).astype(int)
        self._rois_per_image = config.TRAIN.rcnn_batch_rois
        self._num_classes = 2 if config.network.rcnn_class_agnostic else config.dataset.num_classes
        self._num_images = config.TRAIN.image_batch_size

        if 'kps' in config.network.task_type and 'm2' in config.TRAIN.kps_loss_type:
            from keypoints.kps_get_batch_2 import kps_compute_base_anchor_centers
            self._config.base_anchor_centers = kps_compute_base_anchor_centers(feat_height=config.network.kps_height,
                                                                               feat_width=config.network.kps_width,
                                                                               feat_stride=config.network.kps_feat_stride)

    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy().reshape((self._num_images, -1, 5))
        gt_roidb = in_data[1].asnumpy()

        all_res_list = []
        for n in range(self._num_images):
            all_rois_n = all_rois[n, :]
            gt_roidb_n = gt_roidb[n, :]
            gt_roidb_n = gt_roidb_n[1:int(gt_roidb_n[0]) + 1]
            gt_roidb_n = deserialize(gt_roidb_n)

            gt_boxes_n = gt_roidb_n['boxes']
            if len(gt_boxes_n) > 0:
                image_ids = np.full((gt_boxes_n.shape[0], 1), n, dtype=gt_boxes_n.dtype)
                all_rois_n = np.vstack((all_rois_n, np.hstack((image_ids, gt_boxes_n[:, :-1]))))
            assert np.all(all_rois_n[:, 0] == n)

            sample_rois_params = dict()
            sample_rois_params['rois'] = all_rois_n
            sample_rois_params['fg_rois_per_image'] = self._fg_rois_per_image
            sample_rois_params['rois_per_image'] = self._rois_per_image
            sample_rois_params['num_classes'] = self._num_classes
            sample_rois_params['config'] = self._config
            sample_rois_params['gt_boxes'] = gt_boxes_n
            sample_rois_params['ignore_regions'] = gt_roidb_n['ignore_regions']

            if 'kps' in self._config.network.task_type or 'mask' in self._config.network.task_type:
                from fpn_multitask.fpn_multitask_get_batch import sample_rois_multitask
                sample_rois_params['gt_kps'] = gt_roidb_n['keypoints'] if 'keypoints' in gt_roidb_n else []
                sample_rois_params['gt_polys'] = gt_roidb_n['polys'] if 'polys' in gt_roidb_n else []
                res_list = sample_rois_multitask(**sample_rois_params)
            else:
                res_list = sample_rois(**sample_rois_params)
            all_res_list.append(res_list)

        for i in range(len(all_res_list[0])):
            res_i = [all_res_list[j][i] for j in range(len(all_res_list))]
            if len(res_i) == 1:
                res_i = res_i[0]
            elif res_i[0].ndim == 1:
                res_i = np.hstack(res_i)
            else:
                res_i = np.vstack(res_i)
            self.assign(out_data[i], req[i], res_i)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, config):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._config = cPickle.loads(config)

    def list_arguments(self):
        args = ['rois', 'gt_roidb']
        return args

    def list_outputs(self):
        outputs = ['bbox_rois', 'bbox_label', 'bbox_target', 'bbox_weight']
        if 'kps' in self._config.network.task_type:
            outputs.extend(['kps_rois', 'kps_label', 'kps_label_weight', 'kps_pos_offset', 'kps_pos_offset_weight'])
        if 'mask' in self._config.network.task_type:
            outputs.extend(['mask_rois', 'mask_label'])
        return outputs

    def infer_shape(self, in_shape):
        num_classes = 2 if self._config.network.rcnn_class_agnostic else self._config.dataset.num_classes
        num_images = self._config.TRAIN.image_batch_size

        num_rcnn_rois = num_images * self._config.TRAIN.rcnn_batch_rois
        bbox_rois_shape = (num_rcnn_rois, 5)
        bbox_label_shape = (num_rcnn_rois, )
        bbox_target_shape = (num_rcnn_rois, num_classes * 4)
        bbox_weight_shape = (num_rcnn_rois, num_classes * 4)
        output_shape = [bbox_rois_shape, bbox_label_shape, bbox_target_shape, bbox_weight_shape]

        if 'kps' in self._config.network.task_type:
            kps_height = self._config.network.kps_height
            kps_width = self._config.network.kps_width
            num_kps = self._config.dataset.num_kps
            num_kps_rois = num_images * self._config.TRAIN.kps_roi_batch_size
            if self._config.network.kps_compute_area:
                kps_rois_shape = (num_kps_rois, 6)
            else:
                kps_rois_shape = (num_kps_rois, 5)
            if 'one_hot' in self._config.TRAIN.kps_loss_type:
                kps_label_shape = (num_kps_rois, num_kps)
            else:
                kps_label_shape = (num_kps_rois, num_kps, kps_height * kps_width)
            kps_label_weight_shape = (num_kps_rois, num_kps, kps_height * kps_width)
            kps_pos_offset_shape = (num_kps_rois, num_kps * 2, kps_height * kps_width)
            kps_pos_offset_weight_shape = (num_kps_rois, num_kps * 2, kps_height * kps_width)
            output_shape.extend([kps_rois_shape, kps_label_shape, kps_label_weight_shape,
                                 kps_pos_offset_shape, kps_pos_offset_weight_shape])

        if 'mask' in self._config.network.task_type:
            mask_height = self._config.network.mask_pooled_size[0]
            mask_width = self._config.network.mask_pooled_size[1]
            num_mask_rois = num_images * self._config.TRAIN.mask_roi_batch_size
            mask_rois_shape = (num_mask_rois, 5)
            mask_label = (num_mask_rois, 1, mask_height, mask_width)
            output_shape.extend([mask_rois_shape, mask_label])
        return in_shape, output_shape

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._config)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


def proposal_target(rois, gt_roidb, config):
    group = mx.sym.Custom(rois=rois,
                          gt_roidb=gt_roidb,
                          op_type='proposal_target',
                          config=cPickle.dumps(config))
    return group


from mxnet.test_utils import default_context, set_default_context, assert_almost_equal, check_numeric_gradient
def test_proposal_target_multitask():
    from common.utils.load_data import load_imageset
    from experiments.fpn_multitask_coco.config import config
    from fpn.fpn_iter import FPNIter
    config.network.task_type = 'rpn_rcnn_kps'
    config.TRAIN.image_batch_size = 2
    config.feat_sym = None
    ctx = mx.gpu(0)

    roidb = load_imageset(dataset_list=config.dataset.train_dataset,
                          imageset_list=config.dataset.train_imageset,
                          dataset_root_path_list=config.dataset.train_dataset_path,
                          cache_path_list=config.dataset.cache_path,
                          imglst_path_list=config.dataset.train_imglst_path,
                          filter_strategy=config.TRAIN.filter_strategy,
                          is_kps=True)
    train_data = FPNIter(roidb=roidb, config=config, batch_size=config.TRAIN.image_batch_size, ctx=[ctx])

    all_data = train_data.get_batch()
    gt_roidb = all_data['gt_roidb']

    rois_shape = (config.TRAIN.image_batch_size, 20, 5)
    rois = mx.random.uniform(0, 1, rois_shape, ctx=mx.cpu()) * 500
    for i in range(config.TRAIN.image_batch_size):
        rois[i, :, 0] = i
    in_shapes = [rois_shape, gt_roidb.shape]
    in_data = [rois, mx.nd.array(gt_roidb)]

    prop = ProposalTargetProp(config=cPickle.dumps(config))
    _, output_shapes = prop.infer_shape(in_shapes)

    out_data = []
    req = []
    for output_shape in output_shapes:
        out_data.append(mx.nd.zeros(output_shape, ctx=ctx))
        req.append('write')

    op = prop.create_operator(None, None, None)
    op.forward(is_train=True, req=req, in_data=in_data, out_data=out_data, aux=[])

    for a_out_data in out_data:
        print a_out_data.asnumpy()
        print '------------------------------------------------'


if __name__ == '__main__':
    ctx = mx.gpu(0)
    set_default_context(ctx)
    test_proposal_target_multitask()