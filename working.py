from nnabla_nas.contrib.mobilenet import SearchNet
import nnabla as nn

net = SearchNet(
    num_classes=100,
    mode="max",
    # skip_connect=False
)
net.load_parameters(
    '/home/denguyeb/Desktop/nnabla_nas_working/nnabla_nas/log/mobilenet/imagenet/songhan/search/arch.h5')

net(nn.Variable([1, 3, 224, 224]))
net.save_parameters('weights.h5')
