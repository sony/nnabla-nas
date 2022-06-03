#---------------------------- DARTS --------------------------------#
# search DARTS
python main.py --search \
               -f examples/classification/darts/cifar10_search.json  \
               -a DartsSearcher \
               -o log/classification/darts/cifar10/search

# train DARTS
python main.py -f examples/classification/darts/cifar10_train.json \
               -a Trainer \
               -o log/classification/darts/cifar10/train
#------------------------------------------------------------------#


#---------------------------- PNAS --------------------------------#
# search PNAS
python main.py --search \
               -f examples/classification/pnas/cifar10_search.json \
               -a ProxylessNasSearcher \
               -o log/classification/pnas/cifar10/search

# train PNAS
python main.py -f examples/classification/pnas/cifar10_train.json \
               -a Trainer \
               -o log/classification/pnas/cifar10/train

# search PNAS with latency constraints
python main.py --search \
               -f examples/classification/pnas/cifar10_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/classification/pnas/cifar10/constrained/search

# train PNAS with latency constraints
python main.py -f examples/classification/pnas/cifar10_train_latency.json \
               -a Trainer \
               -o log/classification/pnas/cifar10/constrained/train               
#------------------------------------------------------------------#



#-------------------------MobileNet V2-----------------------------#
# on cifar10 data set

# search MobileNet
python main.py --search \
               -f examples/classification/mobilenet/cifar10_search.json \
               -a ProxylessNasSearcher \
               -o log/classification/mobilenet/cifar10/search

# train MobileNet
python main.py -f examples/classification/mobilenet/cifar10_train.json \
               -a Trainer \
               -o log/classification/mobilenet/cifar10/train

# search MobileNet with latency
python main.py --search \
               -f examples/classification/mobilenet/cifar10_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/classification/mobilenet/cifar10/constrained/search

# train MobileNet with latancy
python main.py -f examples/classification/mobilenet/cifar10_train_latency.json \
               -a Trainer \
               -o log/classification/mobilenet/cifar10/constrained/train

# reference MobileNet V2
python main.py -f examples/classification/mobilenet/cifar10_reference.json  \
               -a Trainer \
               -o log/classification/mobilenet/cifar10/reference
#------------------------------------------------------------------#



#----------------------------Zoph---------------------------------#
# search zoph search space with pnas and without latency constraint
python main.py --search \
               -f examples/classification/zoph/pnas_zoph_search.json \
               -a ProxylessNasSearcher \
               -o log/classification/zoph/search

# train zoph network
python main.py -f examples/classification/zoph/zoph_train.json \
               -a Trainer \
               -o log/classification/zoph/train
#------------------------------------------------------------------#


# train random zoph network
python main.py -f examples/classification/zoph/zoph_train_random.json \
               -a Trainer \
               -o log/classification/zoph/train_random

# train random zoph network, using additions for merging
python main.py -f examples/classification/zoph/zoph_train_random_merge_add.json \
               -a Trainer \
               -o log/classification/zoph/train_random_merge_add

# continue training from checkpoint
python main.py -f examples/zoph/zoph_train_continue.json \
               -a Trainer \
               -o log/classification/zoph/train_continue
#------------------------------------------------------------------#



#---------------------------- RandomlyWired------------------------#
# train randomly wired network
python main.py -f examples/classification/random_wired/random_wired_train.json \
               -a Trainer \
               -o log/classification/random_wired/train
#------------------------------------------------------------------#



#---------------------- ImageNet (PNAS)----------------------------#
# search MobileNet network
mpirun -n 4 python main.py --search \
               -f examples/classification/mobilenet/imagenet_search.json \
               -a ProxylessNasSearcher \
               -o log/classification/mobilenet/imagenet/search

# train MobileNet network
mpirun -n 4 python main.py\
               -f examples/classification/mobilenet/imagenet_train.json \
               -a Trainer \
               -o log/classification/mobilenet/imagenet/train

# search MobileNet network with latency
mpirun -n 4 python main.py --search \
               -f examples/classification/mobilenet/imagenet_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/classification/mobilenet/imagenet/latency/search

# search MobileNet network with latency
mpirun -n 4 python main.py \
               -f examples/classification/mobilenet/imagenet_train_latency.json \
               -a ProxylessNasSearcher \
               -o log/classification/mobilenet/imagenet/latency/train

# ref MobileNet network
mpirun -n 4 python main.py\
               -f examples/classification/mobilenet/imagenet_reference.json \
               -a Trainer \
               -o log/classification/mobilenet/imagenet/reference
#------------------------------------------------------------------#



#---------------------- FAIR NAS ----------------------------#
mpirun -n 4 python main.py --search \
               -f examples/classification/fairnas/fairnas_cifar10_search.json \
               -a FairNasSearcher \
               -o log/classification/fairnas/cifar10/search


mpirun -n 4 python main.py --search \
               -f examples/classification/fairnas/fairnas_imagenet_search.json \
               -a FairNasSearcher \
               -o log/classification/fairnas/imagenet/search

mpirun -n 4 python main.py -f examples/classification/fairnas/fairnas_imagenet_train.json \
               -a Trainer \
               -o log/classification/fairnas/imagenet/retrain/

#---------------------- OFA ----------------------------#
mpirun -n 4 python main.py \
               -f examples/classification/ofa/ofa_imagenet_search_fullnet.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K7_E6_D4/

mpirun -n 4 python main.py --search \
               -f examples/classification/ofa/ofa_imagenet_search_kernel.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K357_E6_D4/

mpirun -n 4 python main.py --search \
               -f examples/classification/ofa/ofa_imagenet_search_depth_phase1.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K357_E6_D34/

mpirun -n 4 python main.py --search \
               -f examples/classification/ofa/ofa_imagenet_search_depth_phase2.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K357_E6_D234/

mpirun -n 4 python main.py --search \
               -f examples/classification/ofa/ofa_imagenet_search_expand_phase1.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K357_E46_D234/

mpirun -n 4 python main.py --search \
               -f examples/classification/ofa/ofa_imagenet_search_expand_phase2.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K357_E346_D234/

mpirun -n 4 python main.py \
               -f examples/classification/ofa/ofa_imagenet_train_subnet.json \
               -a OFATrainer \
               -o log/classification/ofa/imagenet/train_subnet/

#---------------------- CompOFA ----------------------------#
mpirun -n 4 python main.py --search \
               -f examples/classification/ofa/ofa_imagenet_search_comp_phase1.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K357_E346_D234/phase1/

mpirun -n 4 python main.py --search \
               -f examples/classification/ofa/ofa_imagenet_search_comp_phase2.json \
               -a OFASearcher \
               -o log/classification/ofa/imagenet/search/K357_E346_D234/phase2/