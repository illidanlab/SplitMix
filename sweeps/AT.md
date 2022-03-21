Joint Customization of Model Sizes & Robustness
===============================================

For each experiment, we use one sweep file, e.g., `fed_iid/cifar10.yaml`, for training and one,
e.g., `fed_iid/cifar10_test.yaml`, for testing.

## Full-width Net w\ Customizable Robustness

Results in Fig 9 (a,b).

Plots:
* Visualize RA-SA trade-off: [../ipynb/RA SA Trade-off.ipynb](../ipynb/RA%20SA%20Trade-off.ipynb)

### FedAvg + Re-training for individual $\lambda$

Runs are included in the next section of FedAvg.

### Split-Mix + DAT

DAT means DBN-based AT. For Split-Mix, we use the layer-wise mixing to customize robustness.

* Cifar10 full set, 10%val To run with DBN 
```shell
wandb sweep sweeps/fed_at_iid/cifar10_dbn.yaml
# => [03/04] https://wandb.ai/jyhong/SplitMix_release/sweeps/a9hteskz
#    @GPU8
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/ct5xsy5e
#    @GPU10 lbn
wandb sweep sweeps/fed_at_iid/cifar10_dbn_test.yaml
# => https://wandb.ai/jyhong/SplitMix_release/sweeps/dsmxxbkc
```
* Digits + class non-iid; SplitMix full-width, DAT
```shell
wandb sweep sweeps/fed_at_niid/digits_SplitMix.yaml
# => [03/04] https://wandb.ai/jyhong/SplitMix_release/sweeps/x3rdoj6w
#    @GPU8
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/2ythpq4u
#    @GPU10 lbn
wandb sweep sweeps/fed_at_niid/digits_SplitMix_test.yaml
# => https://wandb.ai/jyhong/FOAL_Slim_AT_Digits_niid/sweeps/jwvlr8ww
#    @GPU10 lbn
# => https://wandb.ai/jyhong/FOAL_Slim_AT_Digits_niid/sweeps/da0cbvcx
#    @GPU8 lbn=False
```
* Digits; SplitMix full-width, DAT
```shell
wandb sweep sweeps/fed_at_niid/digits_SplitMix.yaml
# => [03/14] https://wandb.ai/jyhong/SplitMix_release/sweeps/i3y775om
#    @GPU9 lbn=True/False
wandb sweep sweeps/fed_at_niid/digits_SplitMix_test.yaml
# => https://wandb.ai/jyhong/SplitMix_release/sweeps/zql6s714
#    @GPU9 no lbn
# => https://wandb.ai/jyhong/SplitMix_release/sweeps/f2932fw1
#    @GPU9 lbn
```

## Customizable Robustness + Customizable Model Sizes

Results in Fig 9 (b,c,d). Check [ipynb](../ipynb/Joint%20customization.ipynb) for figure reproducing.

### FedAvg

Including full-width results for Digits.

* Cifar10 full set, 10%val
```shell
wandb sweep sweeps/fed_at_iid/cifar10_width_scale.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/141b9oxe
#    @GPU10 width_scale=0.125, 1, adv_lmbd=0.3,0.5,1.
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/ke98rg80
#    @GPU10 width_scale=0.125, 1, adv_lmbd=0
wandb sweep sweeps/fed_at_iid/cifar10_width_scale_test.yaml
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/g8xmc74v
#    @GPU10 width_scale=0.125, 1, adv_lmbd=0,0.3,0.5,1.
```
* DomainNet
```shell
wandb sweep sweeps/fed_at_niid/domainnet.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/wg5c6cwq
#    [CANCELED] @GPU9
# => [03/04] https://wandb.ai/jyhong/SplitMix_release/sweeps/u4xw539p
#    @GPU9 width_scale=0.125, 1, adv_lmbd=0.3,0.5,1.
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/w7hmo1qm
#    @GPU9 width_scale=0.125, 1, adv_lmbd=0
wandb sweep sweeps/fed_at_niid/domainnet_test.yaml
# => [03/13] https://wandb.ai/jyhong/SplitMix_release/sweeps/tft5h80j
#    @GPU9 width_scale=0.125, 1, adv_lmbd=0,0.3,0.5,1.
```
* Digits + class non-iid
```shell
wandb sweep sweeps/fed_at_niid/digits.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/bu48bn86
#    @GPU8 adv_lmbd > 0
# => [03/04] https://wandb.ai/jyhong/SplitMix_release/sweeps/lcg4j04w
#    @GPU8 adv_lmbd=0
# => [03/04] https://wandb.ai/jyhong/SplitMix_release/sweeps/halznf37
#    @GPU8 adv_lmbd=0.2,0.8, width=1
wandb sweep sweeps/fed_at_niid/digits_test.yaml
# => https://wandb.ai/jyhong/SplitMix_release/sweeps/27d2wanw
#    @GPU8
```
* Digits
```shell
wandb sweep sweeps/fed_at_niid/digits.yaml
# => [03/14] https://wandb.ai/jyhong/SplitMix_release/sweeps/kdv3o8ym
#    @GPU9 adv_lmbd=0,0.3,0.5, 1.
# => [03/14] https://wandb.ai/jyhong/SplitMix_release/sweeps/mumax35r
#    @GPU10 adv_lmbd=0.2,0.8
wandb sweep sweeps/fed_at_niid/digits_test.yaml
# => https://wandb.ai/jyhong/SplitMix_release/sweeps/d3gmza1k
#    @GPU9 adv_lmbd=0,0.2,0.3,0.5, 0.8, 1.
```


### SplitMix + DAT

* Cifar10 full set, 10%val To run with DBN 
```shell
wandb sweep sweeps/fed_at_iid/cifar10_SplitMix_dbn.yaml
# => [03/04] https://wandb.ai/jyhong/SplitMix_release/sweeps/q02jj60c
#    @GPU8 track, no rescale init/layer
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/3u998q7z
#    @GPU8 track, no rescale init/layer, use lbn
wandb sweep sweeps/fed_at_iid/cifar10_SplitMix_dbn_test.yaml
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/d26ifudn
#    @GPU8 lbn=False
```
* DomainNet
```shell
wandb sweep sweeps/fed_at_niid/domainnet_SplitMix_dbn.yaml
# => [03/06] https://wandb.ai/jyhong/SplitMix_release/sweeps/ok1qel67
#    @GPU8 track, no rescale init/layer
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/xx08vrju
#    @GPU8 track, no rescale init/layer, use lbn
wandb sweep sweeps/fed_at_niid/domainnet_SplitMix_dbn_test.yaml
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/me0jmpi4
#    @GPU8 lbn=False
# => [03/12] https://wandb.ai/jyhong/SplitMix_release/sweeps/ctuur0ey
#    @GPU8 lbn=True
```