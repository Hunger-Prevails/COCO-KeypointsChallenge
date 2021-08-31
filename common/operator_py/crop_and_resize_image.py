import mxnet as mx
import numpy as np
import cv2

def limit_roi(roi, im_height, im_width):
    """limit roi to image border"""
    l = max(0, roi[0])
    t = max(0, roi[1])
    r = min(im_width-1, roi[2])
    b = min(im_height-1, roi[3])
    return [l, t, r, b]

def _crop_and_resize_image_py(im, roi, roi_shape):
    roi = [int(v) for v in roi]
    cut_roi = limit_roi(roi, im.shape[0], im.shape[1])
    im_roi = im[cut_roi[1]:cut_roi[3]+1, cut_roi[0]:cut_roi[2]+1, :]
    im_roi = cv2.copyMakeBorder(im_roi,
                                cut_roi[1] - roi[1], roi[3] - cut_roi[3],
                                cut_roi[0] - roi[0], roi[2] - cut_roi[2],
                                cv2.BORDER_CONSTANT)
    roi_shape = (roi_shape[-1], roi_shape[-2])
    im_roi = cv2.resize(im_roi, roi_shape, interpolation=cv2.INTER_LINEAR)
    im_roi = im_roi.astype(np.float32)
    return im_roi

def crop_and_resize_image_py(data, all_rois, output_height, output_width):
    num_rois = all_rois.shape[0]
    rois_output = np.zeros((num_rois, data.shape[1], output_height, output_width), dtype=np.float32)
    for i in range(num_rois):
        roi_batch_ind = int(all_rois[i][0])
        rois = all_rois[i][1:]
        im = data[roi_batch_ind].transpose(1, 2, 0)  # (R, G, B)
        croped_im = _crop_and_resize_image_py(im, rois, [output_height, output_width])
        rois_output[i, :, :, :] = croped_im.transpose(2, 0, 1)
    return rois_output


class CropAndResizeImageOperator(mx.operator.CustomOp):
    def __init__(self, output_height, output_width):
        super(CropAndResizeImageOperator, self).__init__()
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0].asnumpy()
        all_rois = in_data[1].asnumpy()
        rois_output = crop_and_resize_image_py(data, all_rois, self.output_height, self.output_width)
        self.assign(out_data[0], req[0], rois_output)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('CropAndResizeImage')
class CropAndResizeImageProp(mx.operator.CustomOpProp):
    def __init__(self, output_height, output_width):
        super(CropAndResizeImageProp, self).__init__(need_top_grad=False)
        self.output_height = int(output_height)
        self.output_width = int(output_width)

    def list_arguments(self):
        return ['data', 'rois']

    def list_outputs(self):
        return ['rois_output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]   # (n, 3, h, w)
        rois_shape = in_shape[1]    # (num_rois, 5)
        output_rois_shape = (rois_shape[0], data_shape[1], self.output_height, self.output_width)
        return [data_shape, rois_shape], [output_rois_shape, ]

    def create_operator(self, ctx, shapes, dtypes):
        return CropAndResizeImageOperator(self.output_height, self.output_width)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []


def crop_and_resize_image(data, rois, output_height, output_width):
    output = mx.sym.Custom(data=data,
                           rois=rois,
                           op_type='CropAndResizeImage',
                           output_height=output_height,
                           output_width=output_width)
    return output