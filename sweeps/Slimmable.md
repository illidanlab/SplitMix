Customize Model Sizes
=====================

For each experiment, we use one sweep file, e.g., `fed_iid/digits.yaml`, for training and one,
e.g., `fed_iid/digits_test.yaml`, for testing.

# Benchmarks

## Baseline with a single slim_ratio

* Cifar10 100% or 50% train data
```shell
wandb sweep sweeps/fed_iid/cifar10.yaml
# => [02/25] https://wandb.ai/jyhong/SplitMix_release/sweeps/m8tc32eg
#    @GPU8 4 width_scale, 100%
# => [03/01] https://wandb.ai/jyhong/SplitMix_release/sweeps/biuwy5xb
#    @GPU9 50% data
wandb sweep sweeps/fed_iid/cifar10_test.yaml
# => [02/26] https://wandb.ai/jyhong/SplitMix_release/sweeps/d6ua8kbt
#    @GPU8 4 width_scale, 100%
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/jbn34q4n
#    @GPU9 50% data
```
* Cifar10 full set class-niid
```shell
wandb sweep sweeps/fed_class_niid/cifar10.yaml
# => [02/27] https://wandb.ai/jyhong/SplitMix_release/sweeps/dlxe994l
#    @GPU8
wandb sweep sweeps/fed_class_niid/cifar10_test.yaml
# => [02/28] https://wandb.ai/jyhong/SplitMix_release/sweeps/6ua8jh9x
```
* Digits 
```shell
wandb sweep sweeps/fed_niid/digits.yaml
# => [02/24] https://wandb.ai/jyhong/SplitMix_release/sweeps/057l05ow
#    @GPU8 4 width_scale
wandb sweep sweeps/fed_niid/digits_test.yaml
# => [02/26] https://wandb.ai/jyhong/SplitMix_release/sweeps/8g8s7kp4
```
* DomainNet
```shell
wandb sweep sweeps/fed_niid/domainnet.yaml
# => [02/25] https://wandb.ai/jyhong/SplitMix_release/sweeps/wf20oh8r
#    @GPU8 4 width_scale
wandb sweep sweeps/fed_niid/domainnet_test.yaml
# => [03/01] https://wandb.ai/jyhong/SplitMix_release/sweeps/y489wn02
```

## SHeteroFL (group_slim schedule)

* Cifar10 100% or 50% train data
```shell
wandb sweep sweeps/fed_iid/cifar10_SHeteroFL.yaml
# => [02/25] https://wandb.ai/jyhong/SplitMix_release/sweeps/35idaxc0
#    @GPU8
# => [03/01] https://wandb.ai/jyhong/SplitMix_release/sweeps/l0w7i3qn
#    @GPU8 record all width acc. 100%
# => [03/01] https://wandb.ai/jyhong/SplitMix_release/sweeps/8bia6u1y
#    @GPU9 Use 50% training data
wandb sweep sweeps/fed_iid/cifar10_SHeteroFL_test.yaml
# => [02/26] https://wandb.ai/jyhong/SplitMix_release/sweeps/13li9grh
#    @GPU8 100%
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/6bbo3mwi
#    @GPU9 50%
```
* Cifar10 full set, 10%val, class nidd
```shell
wandb sweep sweeps/fed_class_niid/cifar10_SHeteroFL.yaml
# => [02/26] https://wandb.ai/jyhong/SplitMix_release/sweeps/i2bp6jtm 
#    @GPU8 
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/54zqzzno
#    @GPU8 to record all width performance
wandb sweep sweeps/fed_class_niid/cifar10_SHeteroFL_test.yaml
# => [02/26] https://wandb.ai/jyhong/SplitMix_release/sweeps/fvg0045z
```
* Digits
```shell
wandb sweep sweeps/fed_niid/digits_SHeteroFL.yaml
# => [02/25] https://wandb.ai/jyhong/SplitMix_release/sweeps/zujysna2
#    @GPU8
# => [03/01] https://wandb.ai/jyhong/SplitMix_release/sweeps/ufwuoldc
#    @GPU8 record all-width val acc
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/wdu8cof1
#    @GPU8 repeat 3 times
wandb sweep sweeps/fed_niid/digits_SHeteroFL_test.yaml
# => [02/25] https://wandb.ai/jyhong/SplitMix_release/sweeps/0lh7d73x
```
* DomainNet (From group_size)
```shell
wandb sweep sweeps/fed_niid/domainnet_SHeteroFL.yaml
# => [02/25] https://wandb.ai/jyhong/SplitMix_release/sweeps/cs6y3ir0
#    @GPU8
# => [03/01] https://wandb.ai/jyhong/SplitMix_release/sweeps/dqfo7crn
#    @GPU8 extend to 1k iters and record all width val_acc
wandb sweep sweeps/fed_niid/domainnet_SHeteroFL_test.yaml
# => [02/28] https://wandb.ai/jyhong/SplitMix_release/sweeps/shs7yw8p
```

## SplitMix

We shift the atom model in budget-sufficient users until reach the max width.
* Cifar10 full set
```shell
wandb sweep sweeps/fed_iid/cifar10_SplitMix.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/a7oujc4r
#    @GPU8 try lr_sch = multi_step100, 100%
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/yypftd0t
#    @GPU8 try lr_sch = multi_step100, 50%
wandb sweep sweeps/fed_iid/cifar10_SplitMix_test.yaml
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/fjt4nczs
#    @GPU8 multi_step50
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/y6e7r33c
#    @GPU9 multi_step50, 50%
# => https://wandb.ai/jyhong/SplitMix_release/sweeps/32jld004
#    @GPU9 multi_step100, 50%
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/u0g831ex
#    @GPU8 multi_step100, 100%
```
* Cifar10 class niid
```shell
wandb sweep sweeps/fed_class_niid/cifar10_SplitMix.yaml
# => [02/26] https://wandb.ai/jyhong/SplitMix_release/sweeps/80p3ha4u
#    @GPU8 ens_preresnet18
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/np49m5hp
#    @GPU8 to record all
wandb sweep sweeps/fed_class_niid/cifar10_SplitMix_test.yaml
# => [02/28] https://wandb.ai/jyhong/SplitMix_release/sweeps/g71nb2yv
#    @GPU8
```
* Digits
```shell
wandb sweep sweeps/fed_niid/digits_SplitMix.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/9ecf1cy1
#    @GPU8 repeat 3 times
# ens_digit, seed=1 is runned in ablation study.
wandb sweep sweeps/fed_niid/digits_SplitMix_test.yaml
# => [02/28] https://wandb.ai/jyhong/SplitMix_release/sweeps/3wr7bsxb
```
* DomainNet
```shell
wandb sweep sweeps/fed_niid/domainnet_SplitMix.yaml
# => [02/23] https://wandb.ai/jyhong/SplitMix_release/sweeps/naglzvcl 
#    @GPU8 re-run with new ens_net
wandb sweep sweeps/fed_niid/domainnet_SplitMix_test.yaml
# => [02/28] https://wandb.ai/jyhong/SplitMix_release/sweeps/2kxrau5h
```

Plots:
* Visualize convergence curves (Fig 1,4,13): [../ipynb/Convergence curve.ipynb](../ipynb/Convergence%20curves.ipynb)
* Benchmark tables (Table 1): [../ipynb/Convergence%20curve.ipynb](../ipynb/Benchmarks.ipynb)
* DomainNet per domain acc (Fig. 6): [../ipynb/DomainNet%20per%20domain%20acc.ipynb](../ipynb/DomainNet%20per%20domain%20acc.ipynb)
* Client-wise statistics of test acc, training and communication efficiency (Fig. 5): [../ipynb/client_stat.ipynb](../ipynb/client_stat.ipynb)

# Ablation Study

## SplitMix Ablation of rescale_init, rescale_layer, loss_temp

* Digits
```shell
wandb sweep sweeps/ablation/digits_SplitMix.yaml
# => [02/23] https://wandb.ai/jyhong/SplitMix_release/sweeps/ybief82d
#    @GPU8 All-in-one ablation
# => [03/06] https://wandb.ai/jyhong/SplitMix_release/sweeps/3oe7oq7t
#    @GPU8 All-in-one ablation + track bn stat
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/accu8vm4
#    @GPU10 All-in-one ablation + track bn stat + LBN
wandb sweep sweeps/ablation/digits_SplitMix_test.yaml
# => [02/23] https://wandb.ai/jyhong/SplitMix_release/sweeps/cpzoxxq9
#    @GPU8 All-in-one ablation
# => [04/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/tr78ctgv
#    @GPU8 All-in-one ablation + track bn stat
# => [04/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/2di8eygl
#    @GPU8 All-in-one ablation + refresh BN before test
# => [04/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/x11krsq4
#    @GPU8 All-in-one ablation + track bn stat + LBN
```

Plots
* Ablation table (Table 4): [../ipynb/Ablation%20study.ipynb](../ipynb/Ablation%20study.ipynb)

## Vary budget distributions

Results in Sec. A.5. Vary the users per round (`pr_nuser`).

**FedAvg**: As FedAvg are individually trained for each width, there is no need to test with varying budgets.

* Skewed groups: We let more groups to be very sufficient or insufficient.
* Non-exp distribution: For example, step increase by 0.25

**SplitMix**:
* Digits
```shell
wandb sweep sweeps/vary_budget_dist/digits_SplitMix.yaml
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/0bzucbfp
#    @GPu10
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/wrp3hoh9
#    @GPu10 slim_ratios=ln0.5_0.4
wandb sweep sweeps/vary_budget_dist/digits_SplitMix_test.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/8g0irs68
#    @GPU10
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/wz10puq8
#    @GPu10 slim_ratios=ln0.5_0.4
```

**SHeteroFL**:
* Digits
```shell
wandb sweep sweeps/vary_budget_dist/digits_SHeteroFL.yaml
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/aym3td6d
#    @GPU10
# => [03/07] https://wandb.ai/jyhong/SplitMix_release/sweeps/f87p0v6r
#    @GPU10 slim_ratios=ln0.5_0.4
wandb sweep sweeps/vary_budget_dist/digits_SHeteroFL_test.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/jbak4jzs
#    @GPU10
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/a36ramy7
#    @GPU10 slim_ratios=ln0.5_0.4
```

Plots:
* Budget distribution and per-width performance comparisons (Fig 11): [../ipynb/Digits%20vary%20budget%20distribution.ipynb](../ipynb/Digits%20vary%20budget%20distribution.ipynb)

## Effect of Lower Contact Rates (`pr_nuser`)

Results in Sec. A.6. Vary the users per round (`pr_nuser`).

**SplitMix**:
* Digits
```shell
wandb sweep sweeps/vary_pr_nuser/digits_SplitMix.yaml
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/cw23nwuv
#    @GPU8 
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/9ecf1cy1
#    @GPU8 pr_nuser=-1
wandb sweep sweeps/vary_pr_nuser/digits_SplitMix_test.yaml
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/80ewd3yq
#    @GPU8
```

**SHeteroFL**:
* Digits
```shell
wandb sweep sweeps/vary_pr_nuser/digits_SHeteroFL.yaml
# => [03/02] https://wandb.ai/jyhong/SplitMix_release/sweeps/hpij2kgi
#    @GPU8
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/wdu8cof1
#    @GPU8 repeat 3 times, pr_nuser=-1
wandb sweep sweeps/vary_pr_nuser/digits_SHeteroFL_test.yaml
# => [03/03] https://wandb.ai/jyhong/SplitMix_release/sweeps/la52g84k
#    @GPU8  missing pr_nuser=-1 (seed>1)
# => [03/09] https://wandb.ai/jyhong/SplitMix_release/sweeps/0qjd6qdr
#    @GPU8
```

Plots:
* Budget distribution and per-width performance comparisons (Fig 11): [../ipynb/Digits%20vary%20budget%20distribution.ipynb](../ipynb/Digits%20vary%20pr%20nuser.ipynb)
