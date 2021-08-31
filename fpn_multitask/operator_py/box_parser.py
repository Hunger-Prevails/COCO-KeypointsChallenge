import mxnet as mx
import numpy as np
import cPickle
from distutils.util import strtobool
from common.processing.bbox_transform import bbox_pred, clip_boxes


class BoxParserOperator(mx.operator.CustomOp):
    def __init__(self, bbox_class_agnostic, config):
        super(BoxParserOperator, self).__init__()
        self._bbox_class_agnostic = bbox_class_agnostic
        self._config = config

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()
        bbox_deltas = in_data[1].asnumpy()
        cls_prob = in_data[2].asnumpy()
        im_info = in_data[3].asnumpy()

        if self._bbox_class_agnostic:
            bbox_deltas = bbox_deltas[:, 4:]
        else:
            bbox_class_idx = np.argmax(cls_prob[:, 1:], axis=1) + 1
            bbox_class_idx = bbox_class_idx[:, np.newaxis] * 4
            bbox_class_idx = np.hstack((bbox_class_idx, bbox_class_idx + 1, bbox_class_idx + 2, bbox_class_idx + 3))
            rows = np.arange(rois.shape[0], dtype=np.int32)
            bbox_deltas = bbox_deltas[rows[:, np.newaxis], bbox_class_idx.astype(np.int32)]
        assert bbox_deltas.shape[1] == 4

        if self._config.network.rcnn_bbox_normalization_precomputed:
            bbox_deltas = bbox_deltas * np.array(self._config.network.rcnn_bbox_stds) + np.array(self._config.network.rcnn_bbox_means)

        pred_boxes = bbox_pred(rois[:, 1:], bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_info[0, :2])

        output = rois
        output[:, 1:] = pred_boxes

        for ind, val in enumerate([output]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('box_parser')
class BoxParserProp(mx.operator.CustomOpProp):
    def __init__(self, bbox_class_agnostic, config, **kwargs):
        super(BoxParserProp, self).__init__(need_top_grad=False)
        self._bbox_class_agnostic = strtobool(bbox_class_agnostic)
        self._config = cPickle.loads(config)

    def list_arguments(self):
        return ['rois', 'bbox_deltas', 'cls_prob', 'im_info']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]
        return in_shape, [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BoxParserOperator(self._bbox_class_agnostic, self._config)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


def box_parser(rois, bbox_deltas, cls_prob, im_info, bbox_class_agnostic, config):
    group = mx.sym.Custom(rois=rois,
                          bbox_deltas=bbox_deltas,
                          cls_prob=cls_prob,
                          im_info=im_info,
                          op_type='box_parser',
                          bbox_class_agnostic=bbox_class_agnostic,
                          config=cPickle.dumps(config))
    return group