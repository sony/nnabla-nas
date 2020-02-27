#---------------------------- DARTS --------------------------------#
# search DARTS
python main.py -d 1 --search \
               -f examples/darts_search.json  \
               -a DartsSearcher \
               -o log/darts/search

# train DARTS
python main.py -d 1 \
               -f examples/darts_train.json \
               -a Trainer -o log/darts/train

# search PNAS
python main.py -d 2 --search \
               -f examples/pnas_search.json \
               -a ProxylessNasSearcher \
               -o log/pnas/search

# train PNAS
python main.py -d 2 \
               -f examples/pnas_train.json \
               -a Trainer \
               -o log/pnas/train

# search PNAS with latency constraints
python main.py -d 3 --search \
               -f examples/pnas_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/pnas-lat/search

# train PNAS with latency constraints
python main.py -d 3 \
               -f examples/pnas_train_latency.json \
               -a Trainer \
               -o log/pnas-lat/train               
#------------------------------------------------------------------#



#-------------------------MobileNet V2-----------------------------#
# on cifar10 data set

# search MobileNet
python main.py -d 0 --search \
               -f examples/mobilenet_cifar10_search.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/cifar10/search

# train MobileNet
python main.py -d 0 \
               -f examples/mobilenet_cifar10_train.json \
               -a Trainer \
               -o log/mobilenet/cifar10/train

# search MobileNet with latency
python main.py -d 0 --search \
               -f examples/mobilenet_cifar10_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/cifar10/latency/search

# train MobileNet with latancy
python main.py -d 0 \
               -f examples/mobilenet_cifar10_train_latency.json \
               -a Trainer \
               -o log/mobilenet/cifar10/latency/train

# train reference MobileNet V2
python main.py -d 1\
               -f examples/mobilenet_cifar10_reference.json  \
               -a Trainer \
               -o log/mobilenet/cifar10/reference
#------------------------------------------------------------------#



#------------------MobileNet V2 (experimental)---------------------#
# search MobileNet
python main.py -d 1 --search \
               -f examples/mnv2_search.json  \
               -a DartsSearcher \
               -o log/mnv2/search

python main.py -d 1 \
               -f examples/mnv2_train.json \
               -a Trainer -o log/mnv2/train
#------------------------------------------------------------------#



#----------------------------Zoph---------------------------------#
# search zoph search space with pnas and without latency constraint
python main.py -d 0 --search \
               -f examples/pnas_zoph_search.json \
               -a ProxylessNasSearcher \
               -o log/zoph/search

# train zoph network
python main.py -d 0 \
               -f examples/zoph_train.json \
               -a Trainer \
               -o log/zoph/train
#------------------------------------------------------------------#


# train random zoph network
python main.py -d 0 \
               -f examples/zoph_train_random.json \
               -a Trainer \
               -o log/zoph/train_random

# train random zoph network, using additions for merging
python main.py -d 0 \
               -f examples/zoph_train_random_merge_add.json \
               -a Trainer \
               -o log/zoph/train_random_merge_add

# continue training from checkpoint
python main.py -d 0 \
               -f examples/zoph_train_continue.json \
               -a Trainer \
               -o log/zoph/train_continue

#---------------------------- RandomlyWired------------------------#
# train randomly wired network
python main.py -d 0 \
               -f examples/random_wired_train.json \
               -a Trainer \
               -o log/random_wired/train

#---------------------------- ImageNet ----------------------------#
# module load mpi/openmpi-x86_64
# /home/decardif/local/bin/mpirun

# search MobileNet network
mpirun -n 4 python main.py --search \
               -f examples/mobilenet_imagenet_search.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/imagenet/search

# train MobileNet network
mpirun -n 4 python main.py\
               -f examples/mobilenet_imagenet_train.json \
               -a Trainer \
               -o log/mobilenet/imagenet/train

# ref MobileNet network
mpirun -n 4 python main.py\
               -f examples/mobilenet_imagenet_reference.json \
               -a Trainer \
               -o log/mobilenet/imagenet/reference
#------------------------------------------------------------------#
