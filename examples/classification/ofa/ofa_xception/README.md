# Model Zoo - OFA Xception - Imagenet

### Common settings and environment setup

- The experiments are run with:
```
NNABLA_CUDA_EXT_VERSION  1.28.0
CUDA                     10.2
CUDNN                    8.3
PYTHON                   3.8
NCCL_VERSION             2.7.8-1+cuda10.2
OPENMPI_VERSION          3.1.6
```
- Training and testing is done on 4 V100 GPUs (32GB Memory each).

### Fullnet 

#### Results:

| Model                    | GPUs | Epochs | Train time(h)|Top1-accuracy (img_size=160)|Top1-accuracy (img_size=224)| 
|--------------------------|------|-----------|--------|---------------------|-----------|
|Xception41 Fullnet `['XP1 7x7 3']` |  4  | 50 | 33     | 0.6491 | 0.6667 |
|Xception41 Fullnet `['XP1 3x3 3']` |  4  | 50 | 27.9     | 0.7257 | 0.7358 |

#### Hyperparameter descriptions

- For the fullnet training we follow the same hyperparameter specification as given in the original [`Xception paper`](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf).
```
On ImageNet:
    – Optimizer: SGD
    – Momentum: 0.9
    – Initial learning rate: 0.045
    – Learning rate decay: decay of rate 0.94 every 2 epochs
```
- We train to support 4 elastic-image sizes: `[128, 160, 192, 224]`
- We follow the same `lr_scheduler` for all the other search spaces as well.

---- 

### Search Kernel Space

`Kernel Search Space = [3x3, 5x5, 7x7]`

#### Results on validation models:

|Models validated on|
|----------------|
|`['XP1 7x7 3', 'XP1 5x5 3', 'XP1 3x3 3', 'XP1 5x5 3', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 3x3 3']`|
|`['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3']`|
|`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

`Results coming soon!`

#### Hyperparameter descriptions:

- For the `base learning rate` in the rest of the search spaces we follow a similar trend as in [`OFA GitHub`](https://github.com/mit-han-lab/once-for-all/blob/master/train_ofa_net.py). Please note that these hyperparameters are essentially meant for `MobilenetV3` and not `Xception`, so we just follow the trend instead of exactly replicating them.

```
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.03
– Learning rate decay: decay of rate 0.94 every 2 epochs
```

----

### Search Depth Space  1

`Depth Search Space = [2, 3]`

#### Results on validation models:

|Models validated on|
|----------------|
|`['XP1 7x7 3', 'XP1 5x5 3', 'XP1 7x7 3', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 3x3 2', 'XP1 7x7 2', 'XP1 3x3 3']`|
|`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 2', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 5x5 3', 'XP1 7x7 3', 'XP1 3x3 3']`|
|`['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 2', 'XP1 3x3 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 3x3 3']`|
|`['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3']`|
|`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

`Results coming soon!`

#### Hyperparameter descriptions:
```
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.025
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

### Search Depth Space 2

`Depth Search Space = [1, 2, 3]`

#### Results on validation models:

|Models validated on|
|----------------|
|`['XP1 7x7 3', 'XP1 5x5 3', 'XP1 7x7 3', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 5x5 1', 'XP1 7x7 2', 'XP1 3x3 3']`|
|`['XP1 7x7 3', 'XP1 5x5 2', 'XP1 7x7 3', 'XP1 5x5 3', 'XP1 3x3 3', 'XP1 3x3 2', 'XP1 7x7 2', 'XP1 5x5 2']`|
|`['XP1 5x5 3', 'XP1 3x3 1', 'XP1 7x7 3', 'XP1 3x3 3', 'XP1 7x7 1', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 3x3 3']`|
|`['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3']`|
|`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

`Results coming soon!`

#### Hyperparameter descriptions:
```
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.03
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

### Search Expand Ratio Space 1

`Expand Ratio Search Space = [0.8, 1]`

Note: If depth of a block==1, expand_ratio will be ignored since we just need in_channels and out_channels for a block with a single layer. So blocks: ["XP0.8 KxK 1", "XP1 KxK 1"] are equivalent in this architecture design.

#### Results on validation models:
|Models validated on|
|-------------------|
|`['XP1 7x7 1', 'XP1 7x7 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP1 3x3 2', 'XP0.8 3x3 2', 'XP0.8 3x3 3', 'XP1 5x5 3']`|        
|`['XP1 3x3 3', 'XP1 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 1', 'XP0.8 5x5 3']`|          
|`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 3', 'XP0.8 5x5 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3']`|
|`['XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 7x7 3', 'XP0.8 5x5 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 2', 'XP0.8 3x3 2']`|
|`['XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP0.8 5x5 2', 'XP0.8 5x5 3', 'XP1 7x7 1']`|            
|`['XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP0.8 5x5 1', 'XP1 3x3 3', 'XP1 3x3 3']`|              
|`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP1 7x7 3', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1']`|  
|`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

`Results coming soon!`

#### Hyperparameter descriptions:
```
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.02
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

### Search Expand Ratio Space 2
`Expand Ratio Search Space = [0.6, 0.8, 1]`

Note: If depth of a block==1, expand_ratio will be ignored since we just need in_channels and out_channels for a block with a single layer. So blocks: ["XP0.6 KxK 1", "XP0.8 KxK 1", "XP1 KxK 1"] are equivalent in this architecture design.

#### Results on validation models:
|Models validated on|
|-------------------|
|`['XP1 7x7 3', 'XP1 7x7 3', 'XP0.6 3x3 1', 'XP0.6 3x3 1', 'XP1 3x3 2', 'XP0.8 3x3 2', 'XP0.6 3x3 1', 'XP0.6 5x5 2']`|
|`['XP1 3x3 3', 'XP1 3x3 3', 'XP0.6 3x3 1', 'XP0.6 3x3 1', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 1', 'XP1 7x7 3']`|
|`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.6 3x3 1', 'XP0.6 3x3 2', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.6 3x3 1', 'XP0.6 3x3 1']`|
|`['XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.6 3x3 1', 'XP0.6 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.6 3x3 1', 'XP0.6 3x3 1']`|
|`['XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 5x5 2', 'XP1 3x3 1', 'XP1 3x3 1']`|
|`['XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 5x5 1', 'XP1 3x3 3', 'XP1 3x3 3']`|
|`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 5x5 3', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1']`|
|`['XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 5x5 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3']`|
|`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

`Results coming soon!`

#### Hyperparameter descriptions:
```
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.03
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

### Subnet extracted from Full-OFA Net

In this configuration we can specify any subnet architecture we wish to extract from the trained Full-OFA Net K357_E0.6+0.8+1_D123.

#### Results:

| Models extracted and validation on |
|------------------------------------|
|`['XP1 7x7 2', 'XP1 3x3 3', 'XP1 7x7 3', 'XP0.8 3x3 3', 'XP0.6 3x3 3', 'XP1 5x5 2', 'XP0.6 7x7 3', 'XP0.6 7x7 2']`|

`Results coming soon!`

#### Hyperparameter descriptions for finetuning:
```
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.005
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

#### Notes

- All models are trained with `augment_valid=False`, `colortwist=True` and `train-test split=0.99`.