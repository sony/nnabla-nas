# Model Zoo - OFA ResNet50 - Imagenet

OFA ResNet50 search space includes elastic `depth` (D), `expand_ratio` (E), and `width_mult` (W).
`expand_ratio` and `width_mult` adjust the channel size of the bottleneck residual blocks. 
While `expand ratio` changes the middle layer's channel size, `width_mult` is the multiplication ratio for the output layers.

## Search fullnet
| Model                    | GPUs | Epochs | Train time(h)|Top1-accuracy (img_size=160)|Top1-accuracy (img_size=224)| 
|--------------------------|------|-----------|--------|---------------------|-----------|
|Fullnet `D2-E0.35-W1.0` |  8  | 185 | -     | 0.7707 | 0.7941 |

## Search elastic depth
| Model                    | GPUs | Epochs | Train time(h)|Top1-accuracy (img_size=160)|Top1-accuracy (img_size=224)| 
|--------------------------|------|-----------|--------|---------------------|-----------|
|Subnet `D0-E0.35-W1.0` |  8  | 125 | -     | 0.7611 | 0.7824 |
|Fullnet `D2-E0.35-W1.0` |  8  | 125 | -     | 0.7761 | 0.7972 |

## Search elastic expand
| Model                    | GPUs | Epochs | Train time(h)|Top1-accuracy (img_size=160)|Top1-accuracy (img_size=224)| 
|--------------------------|------|-----------|--------|---------------------|-----------|
|Subnet `D0-E0.2-W1.0` |  8  | 25 (stage1) + 125 (stage2) | -     | 0.7471 | 0.7677 |
|Subnet `D0-E0.35-W1.0` |  8  | 25 (stage1) + 125 (stage2) | -     | 0.7634 | 0.7821 |
|Fullnet `D2-E0.35-W1.0` |  8  | 25 (stage1) + 125 (stage2) | -     | 0.7763 | 0.7985 |

## Search elastic width_mult
| Model                    | GPUs | Epochs | Train time(h)|Top1-accuracy (img_size=160)|Top1-accuracy (img_size=224)| 
|--------------------------|------|-----------|--------|---------------------|-----------|
|Subnet `D0-E0.2-W0.65` |  8  | 25 (stage1) + 125 (stage2) | -     | 0.6873 | 0.7145 |
|Subnet `D0-E0.2-W1.0 ` |  8  | 25 (stage1) + 125 (stage2) | -     | 0.7454 | 0.7658 |
|Subnet `D0-E0.35-W0.65`|  8  | 25 (stage1) + 125 (stage2) | -     | 0.7311 | 0.7541 |
|Subnet `D0-E0.35-W1.0` |  8  | 25 (stage1) + 125 (stage2) | -     | 0.7564 | 0.7782 |
|Fullnet `D2-E0.35-W1.0`|  8  | 25 (stage1) + 125 (stage2) | -     | 0.7763 | 0.7947 |
