import numpy as np

import nnabla as nn
import nnabla.functions as F

def label_smoothing_loss(pred, label, label_smoothing=0.1):
    loss = F.softmax_cross_entropy(pred, label)
    if label_smoothing <= 0:
        return loss
    return (1 - label_smoothing) * loss - label_smoothing \
        * F.mean(F.log_softmax(pred), axis=1, keepdims=True)

def label_smooth(target, n_classes: int, label_smoothing=0.1):
	soft_target = F.one_hot(target, shape=(n_classes, ))
	"""batch_size = target.shape[0]
	soft_target = nn.Variable((batch_size, n_classes))
	for i in range(batch_size):
		target_tmp = target[i].reshape((1, 1))
		ones = nn.Variable.from_numpy_array(np.array([1]))
		soft_target[i] = F.scatter_nd(ones, target_tmp, shape=(n_classes,))"""
	soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
	return soft_target

def cross_entropy_loss_with_soft_target(pred, soft_target):
	return F.mean(F.sum(- soft_target * F.log_softmax(pred), axis=1))

def cross_entropy_loss_with_label_smoothing(pred, target, label_smoothing=0.1):
	soft_target = label_smooth(target, pred.shape[1], label_smoothing)
	return cross_entropy_loss_with_soft_target(pred, soft_target)

def val2list(val, repeat_time=1):
	if isinstance(val, list) or isinstance(val, np.ndarray):
		return val
	elif isinstance(val, tuple):
		return list(val)
	else:
		return [val for _ in range(repeat_time)]

def make_divisible(v, divisor=8, min_val=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	:param v:
	:param divisor:
	:param min_val:
	:return:
	"""
	if min_val is None:
		min_val = divisor
	new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v

def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2

def sub_filter_start_end(kernel_size, sub_kernel_size):
	center = kernel_size // 2
	dev = sub_kernel_size // 2
	start, end = center - dev, center + dev + 1
	assert end - start == sub_kernel_size
	return start, end

def min_divisible_value(n1, v1):
	""" make sure v1 is divisible by n1, otherwise decrease v1 """
	if v1 >= n1:
		return n1
	while n1 % v1 != 0:
		v1 -= 1
	return v1

"""class AverageMeter(object):

	def __init__(self):
		self.val = 0
		self.avg = 0
		#self.sum = nn.Variable.from_numpy_array(np.zeros(shape))
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		#self.sum = nn.Variable.from_numpy_array(np.zeros(shape))
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.count += n
		self.sum += val * n
		self.avg = self.sum / self.count
		#self.sum.forward()
		#self.avg.forward()

class DistributedTensor(object):

	def __init__(self, name, comm):
		self.name = name
		self.sum = None
		self.count = 0
		self.synced = False

	def update(self, val, delta_n=1):
		val *= delta_n
		if self.sum is None:
			self.sum = val
		else:
			self.sum += val
		self.count += delta_n

	@property
	def avg(self):
		if not self.synced:
			#self.sum = hvd.allreduce(self.sum, name=self.name)
			self.comm.all_reduce(self.sum, division=True, inplace=False)
			self.synced = True
		return self.sum / self.count"""

