#---------------------------- DARTS --------------------------------#
# search DARTS
python main.py --search \
               -f examples/darts/search.json  \
               -a DartsSearcher \
               -o log/darts/search

# train DARTS
python main.py  -f examples/darts/train.json \
               -a Trainer -o log/darts/train
#------------------------------------------------------------------#


#---------------------------- PNAS --------------------------------#
# search PNAS
python main.py  --search \
               -f examples/pnas/search.json \
               -a ProxylessNasSearcher \
               -o log/pnas/search

# train PNAS
python main.py -f examples/pnas/train.json \
               -a Trainer \
               -o log/pnas/train

# search PNAS with latency constraints
python main.py  --search \
               -f examples/pnas/search_latency.json \
               -a ProxylessNasSearcher \
               -o log/pnas-lat/search

# train PNAS with latency constraints
python main.py -f examples/pnas/train_latency.json \
               -a Trainer \
               -o log/pnas-lat/train               
#------------------------------------------------------------------#



#-------------------------MobileNet V2-----------------------------#
# on cifar10 data set

# search MobileNet
python main.py  --search \
               -f examples/mobilenet/cifar10_search.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/cifar10/search

# train MobileNet
python main.py -f examples/mobilenet/cifar10_train.json \
               -a Trainer \
               -o log/mobilenet/cifar10/train

# search MobileNet with latency
python main.py  --search \
               -f examples/mobilenet/cifar10_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/cifar10/latency/search

# train MobileNet with latancy
python main.py -d 0 \
               -f examples/mobilenet/cifar10_train_latency.json \
               -a Trainer \
               -o log/mobilenet/cifar10/latency/train

# train reference MobileNet V2
python main.py -f examples/mobilenet/cifar10_reference.json  \
               -a Trainer \
               -o log/mobilenet/cifar10/reference
#------------------------------------------------------------------#



#----------------------------Zoph---------------------------------#
# search zoph search space with pnas and without latency constraint
python main.py  --search \
               -f examples/zoph/pnas_zoph_search.json \
               -a ProxylessNasSearcher \
               -o log/zoph/search

# train zoph network
python main.py -f examples/zoph/zoph_train.json \
               -a Trainer \
               -o log/zoph/train
#------------------------------------------------------------------#


# train random zoph network
python main.py -f examples/zoph/zoph_train_random.json \
               -a Trainer \
               -o log/zoph/train_random

# train random zoph network, using additions for merging
python main.py -f examples/zoph/zoph_train_random_merge_add.json \
               -a Trainer \
               -o log/zoph/train_random_merge_add

# continue training from checkpoint
python main.py -f examples/zoph/zoph_train_continue.json \
               -a Trainer \
               -o log/zoph/train_continue
#------------------------------------------------------------------#



#---------------------------- RandomlyWired------------------------#
# train randomly wired network
python main.py -f examples/random_wired/random_wired_train.json \
               -a Trainer \
               -o log/random_wired/train
#------------------------------------------------------------------#



#---------------------------- ImageNet ----------------------------#
# search MobileNet network
mpirun -n 4 python main.py --search \
               -f examples/mobilenet/imagenet_search.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/imagenet/search

# train MobileNet network
mpirun -n 4 python main.py\
               -f examples/mobilenet/imagenet_train.json \
               -a Trainer \
               -o log/mobilenet/imagenet/train

# search MobileNet network with latency
mpirun -n 4 python main.py --search \
               -f examples/mobilenet/imagenet_search_latency.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/imagenet/latency/search

# search MobileNet network with latency
mpirun -n 4 python main.py \
               -f examples/mobilenet/imagenet_train_latency.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/imagenet/latency/train

# search MobileNet network Songhan
mpirun -n 4 python main.py --search \
               -f examples/mobilenet/imagenet_search_songhan.json \
               -a ProxylessNasSearcher \
               -o log/mobilenet/imagenet/songhan/search

# train MobileNet network Songhan
mpirun -n 4 python main.py \
               -f examples/mobilenet/imagenet_train_songhan.json \
               -a Trainer \
               -o log/mobilenet/imagenet/songhan/train

# ref MobileNet network
mpirun -n 4 python main.py\
               -f examples/mobilenet/imagenet_reference.json \
               -a Trainer \
               -o log/mobilenet/imagenet/reference
#------------------------------------------------------------------#
