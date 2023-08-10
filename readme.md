# 第二届广州·琶洲算法大赛-智能交通CV模型赛题第2名方案

## 1.算法结构框图、思路步骤详述、代码组织结构介绍

### 1.1 算法结构框图

![model](https://www.zjunjie.top/mdimages/model.png)

### 1.2 思路步骤详述

​		此任务为多任务联合训练，由于不同任务共享backbone，在实际训练中，不同任务间存在一定的compete，在模型选型过程中，我们认为更大的模型和参数量更有利于网络在不同任务域上寻找到一个合适的收敛点，此外，在训练上，我们着重关注不同任务间的收敛速度，尽量使得不同head在相近的时间到达收敛点。更大的Backbone具有更强的特征提取能力，我们测试了ConvNeXt-L和ConvNeXt-XL后选择以ConvNeXt-XL作为backbone，DINO已经是SOTA级的检测头，我们选择保留分割头，我们测试了UPerNet和Mask2Former后选择Mask2Former，分类任务由于相对较为简单，我们直接将backbone输出的feature pooling后分类。在解决不同任务compete上，我们提出Multi-branch FFN结构，通过解耦backbone的FFN层，促进不同任务更快的收敛；此外，我们通过Multi-branch augmentation策略扩充数据集，增强网络的泛化能力，具体到训练阶段时，我们注意到不同head的收敛速度差异巨大，其中分类和检测收敛较快，分割收敛较慢，我们不对分割pipeline做过多的数据增强。在推理环节，我们实现三任务的TTA，由于训练阶段在分割和检测的精度上做了trade-off，我们在推理时对检测做了更多的调参。具体包括推理时使用更多的query，聚合框时后处理方法由nms到soft-nms再到wbf，再通过clip bbox对越界框进行限制。实际训练过程中，我们探索了更多的优化方案，包括arcface loss、model soup（uniform soup）等，前者由于收敛失败，多次优化后未果放弃，后者在检测和分类上有显著提升，但分割影响较大，虽然有很大的潜力，但由于时间原因未做过多尝试。

### 1.3 代码组织结构介绍

```shell
config：训练推理所用网络配置文件
evaluation-evaluator.py：检测、分类任务TTA
evaluation-seg_evaluator.py：分割TTA
modeling-heads-simple_cls_head.py：分类头
modeling-heads-mask2former：分割头
script：训练测试脚本
```

## 2.数据增强/清洗策略

```python
# seg
transforms=[
    L(ResizeStepScaling)(min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25), 
    L(RandomPaddingCrop)(crop_size=[1280, 720]), 
    L(RandomHorizontalFlip)(), 
    L(One_of_aug)(method = [  # transform img and mask, rorate(low rato) or shift(high rato)
                A.Rotate (limit=5, p=0.1), 
                A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.0,rotate_limit=0,interpolation=1,p=0.5),
                ],p = 0.5,only_img = False
    ),
    L(One_of_aug)(method = [  # transform img, blur(low rato)
                A.Blur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3,p = 1),
                ],p = 0.1 , only_img = True
     ), 
    L(RandomDistort)(brightness_range=0.4, contrast_range=0.4, saturation_range=0.4),]

# cls
transforms=L(build_transforms_lazy)(
    is_train=True,
    size_train=[448, 448],
    do_rea=True,
    rea_prob=0.5, # higher ratio is harmful
    do_flip=True,
    do_autoaug=True,
    autoaug_prob=0.5, # higher ratio is harmful
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
),

# det
transforms=[
    dict(Decode=dict(),),
    dict(RandomSelect =dict(
        transforms1 = [ # mosaic and mixup, low ratio mixup for more stable training
                    dict(Mosaic=dict(
                                    input_dim=[1024, 1024],
                                    degrees = [-2,2], 
                                    translate = [-0.02,0.02], 
                                    scale = [0.4,1.2],
                                    enable_mixup = True,
                                    mixup_prob = 0.5)),],
        transforms2 = [
                    dict(RandomSelect=dict(
                        transforms1=[
                            dict(RandomShortSideResize=dict(
                                # short_side_sizes=list(range(800,1280,64)), # no use
                                short_side_sizes=[800, 896, 1024, 1120, 1280], 
                                max_size=1280)
                                ),
                        ],
                        transforms2=[
                            dict(RandomShortSideResize=dict(short_side_sizes=[800, 1000, 1200]),),
                            dict(RandomSizeCrop=dict(min_size=736, max_size=1200),),
                            dict(RandomShortSideResize=dict(
                                # short_side_sizes=list(range(800,1280,64)), # no use
                                short_side_sizes=[800, 896, 1024, 1120, 1280], 
                                max_size=1280)
                                ),
                        ],
                    ),),],
                    p=0.2)),
    dict(RandomFlip=dict(prob=0.5),), # after mosaic
    dict(AutoAugment=dict(autoaug_type="v2"),), # v2 may be the best, no time left to validate v0\v1\v3 
```

## 3.调参优化策略

我们的模型有着非常快的收敛速度，最佳训练epochs在60，具体训练中，我们根据batch size的变化线性scale down我们的学习率

```python
optimizer = L(build_lr_optimizer_lazy)(
    optimizer_type='AdamW',
    base_lr=1e-4,
    weight_decay=1e-4,
    grad_clip_enabled=True,
    grad_clip_norm=0.1,
    apply_decay_param_fun=None,
    lr_multiplier=L(build_lr_scheduler_lazy)(
        max_iters=900000,
        warmup_iters=200,
        solver_steps=[720000],
        solver_gamma=0.1,
        base_lr=1e-4,
        sched='CosineAnnealingLR',
    ),
)
# data settings
sample_num = 7000
epochs=60 # e60 is enough for converging
dataloader.train.task_loaders.segmentation.total_batch_size = 1 * 8   # 7k samples 100e 
dataloader.train.task_loaders.fgvc.total_batch_size = 8 * 8  # 8.1k 300e
dataloader.train.task_loaders.trafficsign.total_batch_size = 1 * 8  # 6.1k  240e

iters_per_epoch = sample_num // dataloader.train.task_loaders.segmentation.total_batch_size

max_iters = iters_per_epoch * epochs

# optimizer
optimizer.lr_multiplier.max_iters = max_iters
optimizer.base_lr = optimizer.lr_multiplier.learning_rate = 0.5*1e-4 # scale down the lr
optimizer.lr_multiplier.solver_steps = [int(max_iters * 0.8)]
```

消融实验过程 待补充

### 3.训练/测试脚本

best模型地址：链接：https://pan.baidu.com/s/1gcLch2TtU38ZSoh929ND7Q 提取码：h8n3 

```shell
# train
bash scripts/train.sh
# test
bash scripts/test.sh
```