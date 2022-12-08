#---------------------------- DARTS --------------------------------#
# search DARTS
python main.py experiment=classification/darts/cifar10_search

# train DARTS
python main.py experiment=classification/darts/cifar10_train
#------------------------------------------------------------------#


#---------------------------- PNAS --------------------------------#
# search PNAS
python main.py experiment=classification/pnas/cifar10_search

# train PNAS
python main.py experiment=classification/pnas/cifar10_train

# search PNAS with latency constraints
python main.py experiment=classification/pnas/cifar10_search_latency

# train PNAS with latency constraints
python main.py experiment=classification/pnas/cifar10_train_latency             
#------------------------------------------------------------------#



#-------------------------MobileNet V2-----------------------------#
# on cifar10 data set

# search MobileNet
python main.py experiment=classification/mobilenet/cifar10_search

# train MobileNet
python main.py experiment=classification/mobilenet/cifar10_train

# search MobileNet with latency
python main.py experiment=classification/mobilenet/cifar10_search_latency
# train MobileNet with latancy
python main.py experiment=classification/mobilenet/cifar10_train_latency

# reference MobileNet V2
python main.py experiment=classification/mobilenet/cifar10_reference
#------------------------------------------------------------------#



#----------------------------Zoph---------------------------------#
# search zoph search space with pnas and without latency constraint
python main.py experiment=classification/zoph/pnas_zoph_search

# train zoph network
python main.py experiment=classification/zoph/zoph_train
#------------------------------------------------------------------#


# train random zoph network
python main.py experiment=classification/zoph/zoph_train_random

# train random zoph network, using additions for merging
python main.py experiment=classification/zoph/zoph_train_random_merge_add

# continue training from checkpoint
python main.py experiment=classification/zoph/zoph_train_continue
#------------------------------------------------------------------#



#---------------------------- RandomlyWired------------------------#
# train randomly wired network
python main.py experiment=classification/random_wired/cifar10_train
#------------------------------------------------------------------#



#---------------------- ImageNet (PNAS)----------------------------#
# search MobileNet network
mpirun -n 4 python main.py experiment=classification/mobilenet/imagenet_search

# train MobileNet network
mpirun -n 4 python main.py experiment=classification/mobilenet/imagenet_train

# search MobileNet network with latency
mpirun -n 4 python main.py experiment=classification/mobilenet/imagenet_search_latency

# train MobileNet network with latency
mpirun -n 4 python main.py experiment=classification/mobilenet/imagenet_train_latency

# ref MobileNet network
mpirun -n 4 python main.py experiment=classification/mobilenet/imagenet_reference
#------------------------------------------------------------------#



#---------------------- FAIR NAS ----------------------------#
mpirun -n 4 python main.py experiment=classification/fairnas/cifar10_search

mpirun -n 4 python main.py experiment=classification/fairnas/cifar10_train

mpirun -n 4 python main.py experiment=classification/fairnas/imagenet_search

mpirun -n 4 python main.py experiment=classification/fairnas/imagenet_train

#---------------------- OFA-MobileNetV3 (ImageNet) ----------------------------#
mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_fullnet

mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_kernel

mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_depth_phase1

mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_depth_phase2

mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_expand_phase1

mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_expand_phase2

mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_train_subnet
#------------------------------------------------------------------#


#---------------------- OFA-XCEPTION(ImageNet) ----------------------------#
mpirun -n 4 python main.py experiment=classification/ofa/ofa_xception/imagenet_search_fullnet

mpirun -n 4 python main.py experiment=classification/ofa/ofa_xception/imagenet_search_kernel

mpirun -n 4 python main.py experiment=classification/ofa/ofa_xception/imagenet_search_depth_phase1

mpirun -n 4 python main.py experiment=classification/ofa/ofa_xception/imagenet_search_depth_phase2

mpirun -n 4 python main.py experiment=classification/ofa/ofa_xception/imagenet_search_expand_phase1

mpirun -n 4 python main.py experiment=classification/ofa/ofa_xception/imagenet_search_expand_phase2

mpirun -n 4 python main.py experiment=classification/ofa/ofa_xception/imagenet_train_subnet
#------------------------------------------------------------------#

#---------------------- OFA-ResNet50 (ImageNet) ----------------------------#
mpirun -n 4 python main.py experiment=classification/ofa/ofa_resnet50/imagenet_search_fullnet

mpirun -n 4 python main.py experiment=classification/ofa/ofa_resnet50/imagenet_search_depth

mpirun -n 4 python main.py experiment=classification/ofa/ofa_resnet50/imagenet_search_expand_phase1

mpirun -n 4 python main.py experiment=classification/ofa/ofa_resnet50/imagenet_search_expand_phase2

mpirun -n 4 python main.py experiment=classification/ofa/ofa_resnet50/imagenet_search_width_phase1

mpirun -n 4 python main.py experiment=classification/ofa/ofa_resnet50/imagenet_search_width_phase2

mpirun -n 4 python main.py experiment=classification/ofa/ofa_resnet50/imagenet_train_subnet
#------------------------------------------------------------------#

#---------------------- CompOFA-MobileNetV3 (ImageNet) ----------------------------#
mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_comp_phase1

mpirun -n 8 python main.py experiment=classification/ofa/ofa_mbv3/imagenet_search_comp_phase2
