# 第二届广州·琶洲算法大赛-智能交通CV模型赛题第2名方案

## 1.算法结构框图、思路步骤详述、代码组织结构介绍

### 1.1 算法结构框图

![model](https://www.zjunjie.top/mdimages/model.png)

### 1.2 思路步骤详述

​		此任务为多任务联合训练，由于不同任务共享backbone，在实际训练中，不同任务间存在一定的compete，在模型选型过程中，我们认为更大的模型和参数量更有利于网络在不同任务域上寻找到一个合适的收敛点，此外，在训练上，我们着重关注不同任务间的收敛速度，尽量使得不同head在相近的时间到达收敛点。在模型选取上，更大的Backbone具有更强的特征提取能力，我们测试了ConvNeXt-L和ConvNeXt-XL后选择以ConvNeXt-XL作为backbone，DINO已经是SOTA级的检测头，我们选择保留检测头，分割方面我们测试了UPerNet和Mask2Former后选择Mask2Former，分类任务由于相对较为简单，我们直接将backbone输出的feature pooling后分类。在解决不同任务compete上，我们提出Multi-branch FFN结构，通过解耦backbone的FFN层，促进不同任务更快的收敛；此外，我们通过Multi-branch augmentation策略扩充数据集，增强网络的泛化能力，具体到训练阶段时，我们注意到不同head的收敛速度差异巨大，其中分类和检测收敛较快，分割收敛较慢，我们不对分割pipeline做过多的数据增强。在推理环节，我们实现三任务的TTA，由于训练阶段在分割和检测的精度上做了trade-off，我们在推理时对检测做了更多的调参。具体包括推理时使用更多的query，聚合框时后处理方法由nms到soft-nms再到wbf，再通过clip bbox对越界框进行限制。实际训练过程中，我们探索了更多的优化方案，包括arcface loss、model soup（uniform soup）等，前者由于收敛失败，多次优化后未果放弃，后者在检测和分类上有显著提升，但分割影响较大，虽然有很大的潜力，但由于时间原因未做过多尝试。

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

我们使用的数据增强包括ResizeStepScaling、RandomPaddingCrop、RandomHorizontalFlip、Rotate、ShiftScaleRotate、Blur、MedianBlur、RandomDistort、Mosaic、AutoAugment v2，具体参数配置如下：

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

推理过程中，我们对分割、检测、分类任务分别进行TTA，具体实现为multi-scale、flip后进行结果聚合，具体如下：

```python
aug_params = {
    'fgvc':{            # 分类
        's' : [0.93, 1, 1.2, 1.4, 0.93, 1, 1.2, 1.4],   # 放缩比例
        'f' : [0,    0,  0,  0,   1,    1,   1,  1]}      # 是否水平翻转，1是，0 否
    'trafficsign':{     # 检测
        's' : [0.8, 0.9, 1, 1.1, 0.8, 0.9, 1, 1.1 ],   # 放缩比例
        'f' : [0,   0,   0, 0,   1,   1,   1, 1]}      # 是否水平翻转，1是，0 否
    'seg':{     # 分割
        's' : [0.9,1.0,1.2,1.5,1.6, 0.9,1.0,1.2,1.5,1.6 ],   # 放缩比例
        'f' : [0,   0,  0, 0,   0,  1,  1,  1,  1,  1]      # 是否水平翻转，1是，0 否
    }
}
```



## 3.调参优化策略

我们的模型有着非常快的收敛速度，最佳训练epochs在60附近，具体训练中，我们根据batch size的变化线性scale down我们的学习率。

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
dataloader.train.task_loaders.segmentation.total_batch_size = 1 * 8 
dataloader.train.task_loaders.fgvc.total_batch_size = 8 * 8 
dataloader.train.task_loaders.trafficsign.total_batch_size = 1 * 8 

iters_per_epoch = sample_num // dataloader.train.task_loaders.segmentation.total_batch_size

max_iters = iters_per_epoch * epochs

# optimizer
optimizer.lr_multiplier.max_iters = max_iters
optimizer.base_lr = optimizer.lr_multiplier.learning_rate = 0.5*1e-4 # scale down the lr
optimizer.lr_multiplier.solver_steps = [int(max_iters * 0.8)]
```

`消融实验`

| **model**                                               | **epoch** | **score** |
| ------------------------------------------------------- | --------- | --------- |
| baseline（vit-b）                                       | 120e      | 0.7329    |
| Vit-B+setr+det800                                       | 120e      | 0.7698    |
| Vit-B+setr+det1120                                      | 120e      | 0.7935    |
| Vit-B+setr+det1120+high dims                            | 120e      | 0.7961    |
| Vit-L+setr                                              | 120e      | 0.7811    |
| Convnext-Xl+msf+upernet                                 | 40e       | 0.8307    |
| Convnext-Xl+msf+upernet                                 | 100e      | 0.8429    |
| Convnext-Xl+msf+upernet+segcrop                         | 100e      | 0.8452    |
| Convnext-Xl+msf+upernet+segcrop+multi-branch ffn        | 100e      | 0.8524    |
| Convnext-Xl+msf+mask2former+segcrop                     | 60e       | 0.8599    |
| Convnext-Xl+msf+mask2former+segcrop+aug                 | 60e       | 0.8622    |
| Convnext-Xl+msf+mask2former+segcrop+aug1                | 60e       | 0.8611    |
| Convnext-Xl+msf+mask2former+segcrop+aug1+two            | 60e       | 0.8553    |
| Convnext-Xl+msf+mask2former+segcrop+aug1+multi          | 60e       | 0.8633    |
| Convnext-Xl+msf+mask2former+segcrop+aug1+v2             | 60e       | 0.8628    |
| Convnext-Xl+msf+mask2former+segcrop+aug1+v2+1280        | 60e       | 0.8643    |
| Convnext-Xl+msf+mask2former+segcrop+aug1+v2+1280+mosaic | 60e       | 0.8655    |

| **strategy**     | **postprocess**  | **score** |
| ---------------- | ---------------- | --------- |
| +det tta         | nms              | 0.8659    |
| +det seg cls tta | nms              | 0.8699    |
| +det seg cls tta | more scale+wbf   | 0.8709    |
| +det seg cls tta | low thr+wbf+clip | 0.8712    |

## 4.训练/测试脚本

best模型地址：链接：https://pan.baidu.com/s/1gcLch2TtU38ZSoh929ND7Q 提取码：h8n3 

预训练权重：链接：https://pan.baidu.com/s/1Zf4uowNNiOv9L07N86Wd2A 提取码：rnbh 

convext-xl预训练模型下载地址https://github.com/BR-IDL/PaddleViT，convert代码在my_tools/convert.py

训练测试的log在track1/outputs

`训练脚本`

正式训练前将预训练权重放在pretrained/

训练结果保存在outputs/train_convxl_m2f_e60_revise_dino256_pre_aug1_v2_1280_mosaic_noseg/

```shell
# build env
cd track1
pip3 install -r requirements.txt

# build ms_deform_attn
cd ./PaddleSeg/contrib/PanopticSeg/paddlepanseg/models/ops/ms_deform_attn
python3 setup.py install

# train
cd track1
bash scripts/train.sh
```

`测试脚本`

推理结果保存在outputs/tta/last_e60_noseg_det_mosaic_seg0.9_1_1.2_1.5_1.6_cls_0.93_1_1.2_1.4_det0.8_0.9_1_1.1_q100_flip3/wbf_0.55filter_pred_results.json

```shell
# test
cd track1
bash scripts/test.sh
# wbf2submit
# 最终结果必须运行该脚本
python3 tools/wbf.py

```

## 5.感谢

https://github.com/BR-IDL/PaddleViT

https://github.com/PaddlePaddle/PaddleSeg

https://github.com/xiteng01/CVPR2023_foundation_model_Track1

https://github.com/HandsLing/CVPR2023-Track1-2rd-Solution

https://github.com/Traffic-X/Open-TransMind/tree/main/PAZHOU