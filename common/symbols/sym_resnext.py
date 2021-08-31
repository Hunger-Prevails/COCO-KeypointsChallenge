import mxnet as mx
eps = 2e-5
mirroring_level=2
use_sync_bn = False

def residual_unit(data,
                  num_filter,
                  stride,
                  dim_match,
                  bottle_neck=True,
                  kernel_size=(3,3),
                  dilate=1,
                  num_group=1,
                  bn_mom=0.9,
                  name=None,
                  workspace=512):

    pad_size = (((kernel_size[0] - 1) * dilate + 1) // 2,
                ((kernel_size[1] - 1) * dilate + 1) // 2)
    if num_group > 1:
        assert bottle_neck
    if bottle_neck:
        ratio_1 = 0.25
        ratio_2 = 0.25
        if num_group == 32:
            ratio_1 = 0.5
            ratio_2 = 0.5
        if num_group == 64:
            ratio_1 = 1
            ratio_2 = 1

        conv1 = mx.sym.Convolution(data=data,
                                   num_filter=int(num_filter*ratio_1),
                                   kernel=(1,1),
                                   stride=(1,1),
                                   pad=(0,0),
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               eps=eps,
                               momentum=bn_mom,
                               sync= True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1,
                                 act_type='relu',
                                 name=name + '_relu1')

        if mirroring_level >= 1:
            act1._set_attr(force_mirroring='True')

        conv2 = mx.sym.Convolution(data=act1,
                                   num_filter=int(num_filter*ratio_2),
                                   kernel=kernel_size,
                                   stride=stride,
                                   dilate=(dilate,dilate),
                                   pad=pad_size,
                                   num_group = num_group,
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               eps=eps,
                               momentum=bn_mom,
                               sync= True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2,
                                 act_type='relu',
                                 name=name + '_relu2')
        if mirroring_level >= 1:
            act2._set_attr(force_mirroring='True')


        conv3 = mx.sym.Convolution(data=act2,
                                   num_filter=num_filter,
                                   kernel=(1,1),
                                   stride=(1,1),
                                   pad=(0,0),
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv3')

        bn3 = mx.sym.BatchNorm(data=conv3,
                               fix_gamma=False,
                               eps=eps,
                               momentum=bn_mom,
                               sync= True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               name=name + '_bn3')

              
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=data,
                                          num_filter=num_filter,
                                          kernel=(1,1),
                                          stride=stride,
                                          no_bias=True,
                                          workspace=workspace,
                                          name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut,
                                        fix_gamma=False,
                                        eps=eps,
                                        momentum=bn_mom,
                                        sync= True if use_sync_bn else False,
                                        attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                        if mirroring_level >= 2 else {},
                                        name=name + '_sc_bn')


        data_out = bn3 + shortcut
        data_out = mx.sym.Activation(data=data_out,
                                     act_type='relu',
                                     name=name + '_out_relu')
        if mirroring_level >= 1:
            data_out._set_attr(force_mirroring='True')
        return data_out
    else:
        conv1 = mx.sym.Convolution(data=data,
                                   num_filter=num_filter,
                                   kernel=kernel_size,
                                   stride=stride,
                                   dilate=(dilate, dilate),
                                   pad=pad_size,
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               momentum=bn_mom,
                               eps=eps,
                               sync= True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1,
                                 act_type='relu',
                                 name=name + '_relu1')
        if mirroring_level >= 1:
            act1._set_attr(force_mirroring='True')
        conv2 = mx.sym.Convolution(data=act1,
                                   num_filter=num_filter,
                                   kernel=kernel_size,
                                   stride=(1,1),
                                   dilate=(dilate, dilate),
                                   pad=pad_size,
                                   no_bias=True,
                                   workspace=workspace,
                                   name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               momentum=bn_mom,
                               eps=eps,
                               sync= True if use_sync_bn else False,
                               attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                               if mirroring_level >= 2 else {},
                               name=name + '_bn2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=data,
                                          num_filter=num_filter,
                                          kernel=(1,1),
                                          stride=stride,
                                          no_bias=True,
                                          workspace=workspace,
                                          name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut,
                                        fix_gamma=False,
                                        momentum=bn_mom,
                                        eps=eps,
                                        sync= True if use_sync_bn else False,
                                        attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                        if mirroring_level >= 2 else {},
                                        name=name + '_sc_bn')
        data_out = bn2 + shortcut
        data_out = mx.sym.Activation(data=data_out,
                                     act_type='relu',
                                     name=name + '_out_relu')
        if mirroring_level >= 1:
            data_out._set_attr(force_mirroring='True')
        return data_out


def residual_backbone(data,
                      units,
                      num_stage,
                      filter_list,
                      bottle_neck=True,
                      num_group=1,
                      bn_mom=0.9,
                      workspace=512):
    num_unit = len(units)
    assert(num_unit == num_stage)
    #data = mx.sym.Variable(name='data')
    body = mx.sym.Convolution(data=data,
                              num_filter=filter_list[0],
                              kernel=(7, 7),
                              stride=(2,2),
                              pad=(3, 3),
                              no_bias=True,
                              name="conv0",
                              workspace=workspace)
    body = mx.sym.BatchNorm(data=body,
                            fix_gamma=False,
                            eps=eps,
                            momentum=bn_mom,
                            sync= True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            name='bn0')
    body = mx.sym.Activation(data=body,
                             act_type='relu',
                             name='relu0')
    if mirroring_level >= 1:
        body._set_attr(force_mirroring='True')
    body = mx.symbol.Pooling(data=body,
                             kernel=(3, 3),
                             stride=(2,2),
                             pad=(1,1),
                             pool_type='max')

    in_layer_list = []

    for i in range(num_stage):
        body = residual_unit(data=body,
                             num_filter=filter_list[i+1], stride=(1 if i==0 else 2, 1 if i==0 else 2),
                             dim_match=False,
                             name='stage%d_unit%d' % (i + 1, 1),
                             bottle_neck=bottle_neck,
                             num_group=num_group,
                             workspace=workspace)
        for j in range(units[i]-1):
            body = residual_unit(data=body,
                                 num_filter=filter_list[i+1], stride=(1,1),
                                 dim_match=True,
                                 name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck,
                                 num_group=num_group,
                                 workspace=workspace)

        in_layer_list.append(body)

    return in_layer_list


def get_symbol(data,
               net_depth,
               num_group=1,
               bn_mom=0.9,
               workspace=512):
    
    if net_depth == 18:
         units = [2, 2, 2, 2]
    elif net_depth == 34:
         units = [3, 4, 6, 3]
    elif net_depth == 50:
         units = [3, 4, 6, 3]
    elif net_depth == 101:
        units = [3, 4, 23, 3]
    elif net_depth == 152:
        units = [3, 8, 36, 3]
    elif net_depth == 200:
        units = [3, 24, 36, 3]
    elif net_depth == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(net_depth))

    num_stage = 4
    filter_list =[64,256, 512, 1024, 2048] if net_depth>=50 else [64, 64, 128, 256, 512]
    bottle_neck = True if net_depth >= 50 else False
    
    in_layer_list = residual_backbone(data=data,
                             units=units,
                             num_stage = num_stage,
                             filter_list = filter_list,
                             bottle_neck = bottle_neck,
                             num_group = num_group,
                             bn_mom = bn_mom,
                             workspace = workspace)
    return in_layer_list
