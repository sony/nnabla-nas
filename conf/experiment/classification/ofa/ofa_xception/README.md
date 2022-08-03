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
|Xception41 Fullnet `['XP1 7x7 3']` |  4  | 50 | 33     | 0.6876 | 0.7211 |
|Xception41 Fullnet `['XP1 3x3 3']` |  4  | 50 | 27.9     | 0.7257 | 0.7358 |
|Xception65 Fullnet `['XP1 3x3 3']` |  4  | 50 | 42.8     | 0.7102 | 0.7421 |

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

**Models validated on:**
| Model No. | Model Genotype |
|-------|---------|
| 1. | `['XP1 7x7 3', 'XP1 5x5 3', 'XP1 3x3 3', 'XP1 5x5 3', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 3x3 3']`|
| 2. | `['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3']`|
| 3. | `['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

**Results:**
| Model No. | Top1-accuracy (img_size=160) | Top1-accuracy (img_size=224) |
|--------|--------|------------|
| 1. | 0.6787 | 0.6924 |
| 2. | 0.6795 | 0.6963 |
| 3. | 0.6762 | 0.6990 |

#### Hyperparameter descriptions:

- For the `base learning rate` in the rest of the search spaces we follow a similar trend as in [`OFA GitHub`](https://github.com/mit-han-lab/once-for-all/blob/master/train_ofa_net.py). Please note that these hyperparameters are essentially meant for `MobilenetV3` and not `Xception`, so we just follow the trend instead of exactly replicating them.

```
- Epochs: 20
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.03
– Learning rate decay: decay of rate 0.94 every 2 epochs
```

----

### Search Depth Space  1

`Depth Search Space = [2, 3]`

#### Results on validation models:

**Models validated on:**
| Model No. | Model Genotype |
|--------|--------|
| 1. |`['XP1 7x7 3', 'XP1 5x5 3', 'XP1 7x7 3', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 3x3 2', 'XP1 7x7 2', 'XP1 3x3 3']`|
| 2. |`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 2', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 5x5 3', 'XP1 7x7 3', 'XP1 3x3 3']`|
| 3. |`['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 2', 'XP1 3x3 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 3x3 3']`|
| 4. |`['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3']`|
| 5. |`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

**Results:**
| Model No. | Top1-accuracy (img_size=160) | Top1-accuracy (img_size=224) |
|-------|---------|------------|
| 1. | 0.6500 | 0.6675 |
| 2. | 0.6556 | 0.6711 |
| 3. | 0.6573 | 0.6690 |
| 4. | 0.6495 | 0.6700 |
| 5. | 0.6642 | 0.6847 |

#### Hyperparameter descriptions:
```
- Epochs: 15
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.025
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

### Search Depth Space 2

`Depth Search Space = [1, 2, 3]`

#### Results on validation models:

**Models validated on:**
| Model No. | Model Genotype |
|-------|---------|
| 1. |`['XP1 7x7 3', 'XP1 5x5 3', 'XP1 7x7 3', 'XP1 3x3 3', 'XP1 7x7 3', 'XP1 5x5 1', 'XP1 7x7 2', 'XP1 3x3 3']`|
| 2. |`['XP1 7x7 3', 'XP1 5x5 2', 'XP1 7x7 3', 'XP1 5x5 3', 'XP1 3x3 3', 'XP1 3x3 2', 'XP1 7x7 2', 'XP1 5x5 2']`|
| 3. |`['XP1 5x5 3', 'XP1 3x3 1', 'XP1 7x7 3', 'XP1 3x3 3', 'XP1 7x7 1', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 3x3 3']`|
| 4. |`['XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3', 'XP1 5x5 3']`|
| 5. |`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

**Results:**
| Model No. | Top1-accuracy (img_size=160) | Top1-accuracy (img_size=224) |
|-------|------------|---------|
| 1. | 0.6620 | 0.6801 |
| 2. | 0.6594 | 0.6789 |
| 3. | 0.6614 | 0.6699 |
| 4. | 0.6668 | 0.6806 |
| 5. | 0.6683 | 0.6875 |

#### Hyperparameter descriptions:
```
- Epochs: 30
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
**Models validated on:**
| Model No. | Model Genotype |
|---------|----------|
| 1. |`['XP1 7x7 1', 'XP1 7x7 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP1 3x3 2', 'XP0.8 3x3 2', 'XP0.8 3x3 3', 'XP1 5x5 3']`|        
| 2. |`['XP1 3x3 3', 'XP1 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 1', 'XP0.8 5x5 3']`|          
| 3. |`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 3', 'XP0.8 5x5 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3']`|
| 4. |`['XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 7x7 3', 'XP0.8 5x5 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 2', 'XP0.8 3x3 2']`|
| 5. |`['XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP0.8 5x5 2', 'XP0.8 5x5 3', 'XP1 7x7 1']`|            
| 6. |`['XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP0.8 5x5 1', 'XP1 3x3 3', 'XP1 3x3 3']`|              
| 7. |`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP1 7x7 3', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1']`|  
| 8. |`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

**Results:**
| Model No. | Top1-accuracy (img_size=160) | Top1-accuracy (img_size=224) |
|--------|----------|----------|
| 1. | 0.6633 | 0.6729 |
| 2. | 0.6508 | 0.6579 |
| 3. | 0.6581 | 0.6685 |
| 4. | 0.6621 | 0.6732 |
| 5. | 0.6573 | 0.6697 |
| 6. | 0.6707 | 0.6843 |
| 7. | 0.6493 | 0.6544 |
| 8. | 0.6777 | 0.6855 |

#### Hyperparameter descriptions:
```
- Epochs: 15
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
**Models validated on:**
| Model No. | Model Genotype |
|--------|-----------|
| 1. |`['XP1 7x7 3', 'XP1 7x7 3', 'XP0.6 3x3 1', 'XP0.6 3x3 1', 'XP1 3x3 2', 'XP0.8 3x3 2', 'XP0.6 3x3 1', 'XP0.6 5x5 2']`|
| 2. |`['XP1 3x3 3', 'XP1 3x3 3', 'XP0.6 3x3 1', 'XP0.6 3x3 1', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 1', 'XP1 7x7 3']`|
| 3. |`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.6 3x3 1', 'XP0.6 3x3 2', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.6 3x3 1', 'XP0.6 3x3 1']`|
| 4. |`['XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.6 3x3 1', 'XP0.6 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.6 3x3 1', 'XP0.6 3x3 1']`|
| 5. |`['XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 3x3 1', 'XP1 5x5 2', 'XP1 3x3 1', 'XP1 3x3 1']`|
| 6. |`['XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 3x3 3', 'XP1 5x5 1', 'XP1 3x3 3', 'XP1 3x3 3']`|
| 7. |`['XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 5x5 3', 'XP0.8 3x3 1', 'XP0.8 3x3 1', 'XP0.8 3x3 1']`|
| 8. |`['XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 5x5 1', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3', 'XP0.8 3x3 3']`|
| 9. |`['XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3', 'XP1 7x7 3']`|

**Results:**
| Model No. | Top1-accuracy (img_size=160) | Top1-accuracy (img_size=224) |
|---------|----------|---------|
| 1. | 0.7006 | 0.7271 |
| 2. | 0.7012 | 0.7273 |
| 3. | 0.6921 | 0.7161 |
| 4. | 0.6999 | 0.7246 |
| 5. | 0.6933 | 0.7182 |
| 6. | 0.7008 | 0.7284 |
| 7. | 0.6959 | 0.7187 |
| 8. | 0.7020 | 0.7279 |
| 9. | 0.7005 | 0.7286 |

#### Hyperparameter descriptions:
```
- Epochs: 30
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.03
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

### Subnet extracted from Full-OFA Net

In this configuration we can specify any subnet architecture we wish to extract from the trained Full-OFA Net K357_E0.6+0.8+1_D123. The subnet models should perform quite well just after extraction, without any further finetuning.

#### Results:

| Example subnet extracted and validation on: |
|------------------------------------|
|`['XP1 7x7 2', 'XP1 3x3 3', 'XP1 7x7 3', 'XP0.8 3x3 3', 'XP0.6 3x3 3', 'XP1 5x5 2', 'XP0.6 7x7 3', 'XP0.6 7x7 2']`|

**Result of extracted subnet without any finetuning:**
Top1-accuracy (img_size=224)|
|-------------------|
| 0.7363 |

**Result of extracted subnet after 5 epochs of finetuning:**
Top1-accuracy (img_size=224)|
|-------------------|
| 0.7402 |

#### Hyperparameter descriptions for finetuning:
```
- Epochs: 5 
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.005
– Learning rate decay: decay of rate 0.94 every 2 epochs
```
----

#### Notes

- All models are trained with `augment_valid=False`, `colortwist=True` and `train-test split=0.99`.