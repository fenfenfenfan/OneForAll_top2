# import paddle 
# paddle.disable_signal_handler()
from .common import train
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from modeling.meta_arch.multitask_v2 import MultiTaskBatchFuse

# segmentation
from data.transforms.seg_transforms import ResizeStepScaling, RandomPaddingCrop, Mosaic, \
    RandomHorizontalFlip, RandomDistort, Normalize,GenerateInstanceTargets, One_of_aug,RandomSelectAug,RandomWeather
from data.build_segmentation import build_segmentation_dataset, build_segmentation_trainloader, \
    build_segementation_test_dataset
from evaluation.segmentation_evaluator import SegEvaluator

# classification
from data.build import build_reid_test_loader_lazy
from data.transforms.build import build_transforms_lazy

from data.build_cls import build_hierachical_softmax_train_set, \
    build_hierachical_test_set, build_vehiclemulti_train_loader_lazy
from evaluation.common_cls_evaluator import CommonClasEvaluatorSingleTask

# detection
from data.build_trafficsign import build_cocodet_set, build_cocodet_loader_lazy
from evaluation.cocodet_evaluator import CocoDetEvaluatorSingleTask
from solver.build import build_lr_optimizer_lazy, build_lr_scheduler_lazy
import albumentations as A
dataloader=OmegaConf.create()
_root = os.getenv("FASTREID_DATASETS", "datasets")

seg_num_classes=19


dataloader.train=L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',),
    task_loaders=L(OrderedDict)(
        segmentation=L(build_segmentation_trainloader)(
            data_set=L(build_segmentation_dataset)(
                    dataset_name="BDD100K",
                    dataset_root=_root + '/track1_train_data/seg/', 
                    transforms=[
                        # L(Mosaic)(prob=0.2, input_dim=[1280, 720]), # seg head no need to use mosaic
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
                        L(RandomDistort)(brightness_range=0.4, contrast_range=0.4, saturation_range=0.4),
                        L(GenerateInstanceTargets)(num_classes = seg_num_classes),
                        L(Normalize)()],
                    mode='train',
                    # mosaic=True
                    ),
            total_batch_size=16, 
            worker_num=4, 
            drop_last=True, 
            shuffle=True,
            num_classes=seg_num_classes,
            is_train=True,
        ),

        fgvc=L(build_vehiclemulti_train_loader_lazy)(
            sampler_config={'sampler_name': 'ClassAwareSampler'},
            train_set=L(build_hierachical_softmax_train_set)(
                names = ("FGVCDataset",),
                train_dataset_dir = _root + '/track1_train_data/cls/train/',
                test_dataset_dir = _root + '/track1_train_data/cls/val/',
                train_label = _root + '/track1_train_data/cls/train.txt',
                test_label = _root + '/track1_train_data/cls/val.txt',
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

                num_classes=196,
            ),
            total_batch_size=16,
            num_workers=4,
        ),

        trafficsign=L(build_cocodet_loader_lazy)(
            data_set=L(build_cocodet_set)(
                dataset_name="COCODataSet",
                transforms=[
                    dict(Decode=dict(),),
                    # dict(RandomSelect=dict(
                    #     transforms1=[
                    #          dict(RandomShortSideResize=dict(
                    #             short_side_sizes=[736, 800, 896, 1024, 1120], 
                    #             max_size=1120)
                    #             ),
                    #     ],
                    #     transforms2=[
                    #         dict(RandomShortSideResize=dict(short_side_sizes=[700, 900, 1100]),),
                    #         dict(RandomSizeCrop=dict(min_size=684, max_size=1100),),
                    #         dict(RandomShortSideResize=dict(
                    #             short_side_sizes=[736, 800, 896, 1024, 1120], 
                    #             max_size=1120)
                    #             ),
                    #     ],
                    # ),),
                    dict(RandomSelect =dict(
                        transforms1 = [ # mosaic and mixup, low ratio mixup for more stable training
                                    dict(Mosaic=dict(
                                                    input_dim=[1024, 1024],
                                                    degrees = [-2,2], 
                                                    translate = [-0.02,0.02], 
                                                    scale = [0.4,1.2],
                                                    enable_mixup = True,
                                                    mixup_prob = 0.5)),],
                        # transforms1=[
                        #             dict(RandomShortSideResize=dict(
                        #                 short_side_sizes=list(range(800,1280,64)), 
                        #                 max_size=1280)
                        #                 ),
                        #                 ],
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
                    dict(NormalizeImage=dict(
                        is_scale=True, 
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
                        ),
                    dict(NormalizeBox=dict()),
                    dict(BboxXYXY2XYWH=dict()),
                    dict(Permute=dict()),
                ],
                image_dir='train',
                anno_path='merge_train.json',
                dataset_dir= _root + '/track1_train_data/dec/',
                data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'],
                mosaic_epoch = 0,
            ),
            total_batch_size=8,
            num_workers=4,
            batch_transforms=[
                dict(PadMaskBatch=dict(pad_to_stride=-1, return_pad_mask=True),),
            ],
            is_train=True,
            shuffle=True, 
            drop_last=True, 
            collate_batch=False,
        ),
    ),
)

# NOTE
# trian/eval模式用于构建对应的train/eval Dataset, 需提供样本及标签;
# infer模式用于构建InferDataset, 只需提供测试数据, 最终生成结果文件用于提交评测, 在训练时可将该部分代码注释减少不必要评测

dataloader.test = [
    
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            segmentation=L(build_segmentation_trainloader)(
                data_set=L(build_segementation_test_dataset)(
                        dataset_name="BDD100K",
                        dataset_root=_root + '/track1_train_data/seg/', 
                        transforms=[L(Normalize)()],
                        mode='val',
                        is_padding=True),
                total_batch_size=8, 
                worker_num=4, 
                drop_last=False, 
                shuffle=False,
                num_classes=seg_num_classes,
                is_train=False,
            ),
        ),
    ),

    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            fgvc=L(build_reid_test_loader_lazy)(
                test_set=L(build_hierachical_test_set)(
                    dataset_name = "FGVCDataset",
                    train_dataset_dir = _root + '/track1_train_data/cls/train/',
                    test_dataset_dir = _root + '/track1_train_data/cls/val/',
                    train_label = _root + '/track1_train_data/cls/train.txt',
                    test_label = _root + '/track1_train_data/cls/val.txt',
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[448, 448],
                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    ),
                    is_train=True  # eval mode 
                ),
                test_batch_size=8,
                num_workers=8,
            ),
        ),
    ),
    
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(           
            trafficsign=L(build_cocodet_loader_lazy)(
                data_set=L(build_cocodet_set)(
                    is_padding=True,
                    dataset_name="COCODataSet",
                    transforms=[
                        dict(Decode=dict(),),
                        dict(Resize=dict(
                            target_size=[608, 608], 
                            keep_ratio=False)
                            ),
                        dict(NormalizeImage=dict(
                            is_scale=True, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
                            ),
                        dict(Permute=dict()),
                    ],
                    image_dir='val',
                    anno_path='val.json',
                    dataset_dir= _root + '/track1_train_data/dec/',
                    data_fields=['image', 'gt_bbox', 'gt_class', 'difficult'],
                ),
                total_batch_size=8,
                num_workers=4,
                batch_transforms=[
                    dict(PadMaskBatch=dict(pad_to_stride=32, return_pad_mask=False),),
                ],
                is_train=False,
                shuffle=False, 
                drop_last=False, 
                collate_batch=False,
            ),
        ),    
    ),
]

# NOTE
# trian/eval模式用于eval;
# infer模式则用于生成测试集预测结果(可直接提交评测), 在训练时可注释该部分代码减少不必要评测

# dataloader.evaluator = [
#     # segmentation
#     L(SegEvaluator)(
#     ),  # train/eval mode

#     # classification
#     L(CommonClasEvaluatorSingleTask)(
#         cfg=dict(), num_classes=196
#     ),   # train/eval mode

#     # detection
#     L(CocoDetEvaluatorSingleTask)(
#         classwise=False, 
#         output_eval=None,
#         bias=0, 
#         IouType='bbox', 
#         save_prediction_only=False,
#         parallel_evaluator=True,
#         num_valid_samples=3067, 
#     ),  # train/eval mode
# ]



from ppdet.modeling import ShapeSpec
from modeling.backbones.vit import ViT
from modeling.backbones.convnext_one import ConvNeXt
from modeling.heads.simple_cls_head import ClsHead
from modeling.heads.setr_head import SegmentationTransformer

from modeling.heads.mask2former.mask2former import Mask2Former
from modeling.losses.mask2former_loss import Mask2FormerLoss
from modeling.losses.seg_loss import SegSETRLoss, UPerLoss

from modeling.heads.detr import DETR
from ppdet.modeling.transformers.dino_transformer import DINOTransformer
from ppdet.modeling.transformers.matchers import HungarianMatcher
from ppdet.modeling.heads.detr_head import DINOHead
from ppdet.modeling.post_process import DETRBBoxPostProcess
from ppdet.modeling.losses.detr_loss import DINOLoss
from modeling.heads.upernet import UPerNet


# cls
backbone=L(ConvNeXt)(
    in_channels=3,
    num_classes=1000,
    global_pool=True,
    output_stride=32,
    patch_size=4,
    depths=(3, 3, 27, 3),
    dims=(256, 512, 1024, 2048), # large
    ls_init_value=1e-6,
    conv_mlp=False,
    stem_type='patch',
    head_init_scale=1.,
    head_norm_first=False,
    norm_layer=None,
    dropout=0.,
    droppath=0.1
)

trafficsign_num_classes=45
use_focal_loss=True

model=L(MultiTaskBatchFuse)(
    backbone=backbone,
    # heads=L(OrderedDict)(
    #     segmentation=L(UPerNet)(
    #         num_classes=seg_num_classes,
    #         backbone_indices=[0, 1, 2, 3],
    #         in_channels=[256, 512, 1024, 2048],
    #         channels=512,
    #         dropout_prob=0.1,
    #         enable_auxiliary_loss=True,
    #         loss=L(UPerLoss)(
    #             weights=[1.0, 0.4]  # 第二个是辅助分支
    #         ),
    #     ),

    heads=L(OrderedDict)(
        segmentation=L(Mask2Former)(
            in_channels= [256, 512, 1024, 2048],
            num_classes=seg_num_classes,
            maskformer_num_feature_levels = 4,
            loss=L(Mask2FormerLoss)(num_classes = seg_num_classes, 
                                    loss_ce = 1.0,    
                                    mask_weight = 3.0,
                                    dice_weight = 2.0,
                                    cost_loss_ce = 2.0,    
                                    cost_mask_weight = 5.0,
                                    cost_dice_weight = 5.0,
                                    seg_loss_weight = 1.0, 
                                    ),
        ),

        fgvc=L(ClsHead)(
            embedding_size=2048, 
            class_num=196,
        ),

        trafficsign=L(DETR)(
            transformer=L(DINOTransformer)(
                            num_classes=trafficsign_num_classes,
                            hidden_dim=256,
                            num_queries=900,
                            position_embed_type='sine',
                            return_intermediate_dec=True,
                            backbone_feat_channels=[256, 512, 1024, 2048],
                            num_levels=4,
                            num_encoder_points=4,
                            num_decoder_points=4,
                            nhead=8,
                            num_encoder_layers=6,
                            num_decoder_layers=6,
                            dim_feedforward=2048,
                            dropout=0.0,
                            activation="relu",
                            num_denoising=100,
                            label_noise_ratio=0.5,
                            box_noise_scale=1.0,
                            learnt_init_query=True,
                            eps=1e-2),
            detr_head=L(DINOHead)(loss=L(DINOLoss)(
                            num_classes=trafficsign_num_classes,
                            loss_coeff={"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1},
                            aux_loss=True,
                            use_focal_loss=use_focal_loss,
                            matcher=L(HungarianMatcher)(
                                matcher_coeff={"class": 2, "bbox": 5, "giou": 2},
                                use_focal_loss=use_focal_loss,)   
                            )
           ),
            post_process=L(DETRBBoxPostProcess)(
                            num_classes=trafficsign_num_classes,
                            num_top_queries=50,
                            use_focal_loss=use_focal_loss,
                            ),
        ),
    ),
    pixel_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    pixel_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    multi_branch='one' # control the multi-branch ffn
)


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

train.amp.enabled = False

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


train.max_iter = max_iters
train.eval_period = int(iters_per_epoch * epochs)
train.checkpointer.period = int(iters_per_epoch * 20)
train.checkpointer.max_to_keep=20
train.init_checkpoint = 'pretrained/convert_model_x_dino_resnet_m2f.pdparams'

train.output_dir = 'outputs/train_convxl_m2f_e60_revise_dino256_pre_aug1_v2_1280_mosaic_noseg'

# resume settings (remember last_checkpoint and --resume)
train.log_period = 20