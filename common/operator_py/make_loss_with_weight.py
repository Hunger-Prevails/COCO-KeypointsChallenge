import mxnet as mx
import numpy as np


class MakeLossWithWeightOperator(mx.operator.CustomOp):
    def __init__(self, grad_scale=1.0):
        super(MakeLossWithWeightOperator, self).__init__()
        self._grad_scale = float(grad_scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        num_output = np.sum(in_data[1].asnumpy())
        if num_output == 0:
            self.assign(in_grad[0], req[0], 0.)
        else:
            self.assign(in_grad[0], req[0], self._grad_scale / float(num_output))


@mx.operator.register('MakeLossWithWeight')
class MakeLossWithWeightProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0):
        super(MakeLossWithWeightProp, self).__init__(need_top_grad=False)
        self._grad_scale = grad_scale

    def list_arguments(self):
        return ['data', 'weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]]

    def create_operator(self, ctx, shapes, dtypes):
        return MakeLossWithWeightOperator(self._grad_scale)


def make_loss_with_weight(data, weight, grad_scale=1.0):
    return mx.sym.Custom(data=data,
                         weight=weight,
                         op_type='MakeLossWithWeight',
                         grad_scale=grad_scale)

