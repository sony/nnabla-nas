import copy
import numpy as np

import nnabla as nn
import nnabla.functions as F

from ..... import module as Mo
from .....utils.helper import AverageMeter
from .dynamic_op import DynamicBatchNorm2d
from .my_random_resize_crop import MyResize


def set_running_statistics(model, forward_model, dataloader, data_size, batch_size, ):
    bn_mean = {}
    bn_var = {}

    # [load parameters]
    model_params = model.get_parameters()
    forward_model.set_parameters(model_params.copy())

    for name, m in forward_model.get_modules():
        if isinstance(m, Mo.BatchNormalization):
            bn_mean[name] = AverageMeter(name)
            bn_var[name] = AverageMeter(name)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = F.mean(x, axis=(0, 2, 3), keepdims=True)
                    # batch_var = (x - batch_mean) * (x - batch_mean)
                    # batch_var = F.mean(batch_var, axis=(0, 2, 3), keepdims=True)
                    batch_var = F.mean(x ** 2, axis=(0, 2, 3), keepdims=True) - batch_mean ** 2

                    """batch_mean.apply(persistent=True)
                    batch_var.apply(persistent=True)
                    batch_mean_tmp = batch_mean.get_unlinked_variable()
                    batch_var_tmp = batch_var.get_unlinked_variable()
                    mean_est.update(batch_mean_tmp, x.shape[0])
                    var_est.update(batch_var_tmp, x.shape[0])"""
                    mean_est.update(batch_mean.d, x.shape[0])
                    var_est.update(batch_var.d, x.shape[0])

                    _feature_dim = batch_mean.shape[1]
                    return F.batch_normalization(
                        x, bn._beta[:, :_feature_dim, :, :], bn._gamma[:, :_feature_dim, :, :],
                        batch_mean, batch_var, decay_rate=1, eps=1e-5, batch_stat=False
                    )
                return lambda_forward

            m.call = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        return

    with nn.no_grad():
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        nn.set_auto_forward(True)
        forward_model.apply(training='valid')
        transform = MyResize()
        x = nn.Variable(shape=(batch_size, 3, 224, 224))
        for i in range(data_size // batch_size):
            # output = forward_model(*[transform(x)])
            # output.apply(persistent=True)
            images, _ = dataloader.next()
            if isinstance(images, nn.NdArray):
                x.data = images
            else:
                x.d = images
            forward_model(*[transform(x)])
            # output.forward(clear_buffer=True)

            # inputs = images_list[MyResize.ACTIVE_SIZE_NUM]
            # forward_model(inputs)
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False
        nn.set_auto_forward(False)

    for name, m in model.get_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.shape[1]
            assert isinstance(m, Mo.BatchNormalization)
            new_mean = np.concatenate((bn_mean[name].avg, m._mean.d[:, feature_dim:, :, :]), axis=1)
            new_var = np.concatenate((bn_var[name].avg, m._var.d[:, feature_dim:, :, :]), axis=1)
            m._mean.d = new_mean
            m._var.d = new_var
            """new_mean = nn.Variable(shape=m._mean.shape)
            new_var = nn.Variable(shape=m._var.shape)
            new_mean.data[:, :feature_dim, :, :] = bn_mean[name].avg.data
            new_var.data[:, :feature_dim, :, :] = bn_var[name].avg.data
            #new_mean.d[:, :feature_dim, :, :] = bn_mean[name].avg
            #new_var.d[:, :feature_dim, :, :] = bn_var[name].avg
            if feature_dim < m._mean.shape[1]:
                new_mean.data[:, feature_dim:, :, :] = m._mean.data[:, feature_dim:, :, :]
                new_var.data[:, feature_dim:, :, :] = m._var.data[:, feature_dim:, :, :]
            m._mean.data = new_mean.data
            m._var.data = new_var.data"""
