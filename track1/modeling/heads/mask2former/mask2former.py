# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.utils import utils


from .pixel_decoder import MSDeformAttnPixelDecoder
from .transformer_decoder import MultiScaleMaskedTransformerDecoder
import paddle
from paddleseg.cvlibs import param_init
from modeling.initializer import xavier_uniform_



__all__ = ['Mask2Former']


def sub_weight_init(m):
    if isinstance(m, nn.Conv2D):
        xavier_uniform_(m.weight)
        if m.bias is not None:
            param_init.constant_init(m.bias, value = 0)
    elif isinstance(m, nn.BatchNorm2D):
        param_init.constant_init(m.weight, value=1)
        param_init.constant_init(m.bias,value = 0)
            
class Mask2Former(nn.Layer):
    """
    The Mask2Former implementation based on PaddlePaddle.

    The original article refers to
     Bowen Cheng, et, al. "Masked-attention Mask Transformer for Universal Image Segmentation"
     (https://arxiv.org/abs/2112.01527)

    Args:
        num_classes (int): The number of target semantic classes.
        backbone (paddle.nn.Layer): The backbone network. Currently supports ResNet50-vd/ResNet101-vd/Xception65.
        backbone_indices (tuple|None): The indices of backbone output feature maps to use.
        backbone_feat_os (tuple|None): The output strides of backbone output feature maps.
        num_queries (int): The number of queries to use in the decoder.
        pd_num_heads (int): The number of heads of the multi-head attention modules used in the pixel decoder.
        pd_conv_dim (int): The number of convolutional filters used for input projection in the pixel decoder.
        pd_mask_dim (int): The number of convolutional filters used to produce mask features in the pixel decoder.
        pd_ff_dim (int): The number of feature channels in the feed-forward networks used in the pixel decoder.
        pd_num_layers (int): The number of basic layers used in the pixel decoder.
        pd_common_stride (int): The base output stride of feature maps in the pixel decoder.
        td_hidden_dim (int): The dimension of the hidden features in the transformer decoder.
        td_num_head (int): The number of heads of the multi-head attention modules used in the transformer decoder.
        td_ff_dim (int): The number of feature channels in the feed-forward networks used in the transformer decoder.
        td_num_layers (int): The number of basic layers used in the transformer decoder.
        td_pre_norm (bool): Whether or not to normalize features before the attention operation in the transformer 
            decoder.
        td_mask_dim (bool): The number of convolutional filters used for mask prediction in the transformer decoder.
        td_enforce_proj (bool): Whether or not to use an additional input projection layer in the transformer decoder.
        pretrained (str|None, optional): The path or url of pretrained model. If None, no pretrained model will be used.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 loss = None,
                 neck = False,
                 num_classes = 2,
                 backbone_indices = [0, 1, 2, 3],
                 backbone_feat_os = [4, 8, 16, 32],
                 maskformer_num_feature_levels  =3,
                 num_queries = 100,
                 pd_num_heads = 8,
                 pd_conv_dim = 256,
                 pd_mask_dim = 256,
                 pd_ff_dim= 1024,
                 pd_num_layers= 6,
                 pd_common_stride = 4,
                 td_hidden_dim = 256,
                 td_num_heads = 8,
                 td_ff_dim = 2048,
                 td_num_layers = 9,
                 td_pre_norm = False,
                 td_mask_dim = 256,
                 td_enforce_proj = False,
                 pretrained=None):
        super().__init__()

        self.neck = neck
        use_neck = False
        if self.neck:
            print('=============================>use ASPP neck in seg!')
            use_neck = True
            self.neck.apply(sub_weight_init)
            
            
        self.num_queries = num_queries
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_feat_strides=backbone_feat_os,
            in_feat_chns=in_channels,
            feat_indices=backbone_indices,
            num_heads=pd_num_heads,
            ff_dim=pd_ff_dim,
            num_enc_layers=pd_num_layers,
            conv_dim=pd_conv_dim,
            mask_dim=pd_mask_dim,
            common_stride=pd_common_stride,
            maskformer_num_feature_levels = maskformer_num_feature_levels,
            use_neck= use_neck)
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=pd_conv_dim,
            num_classes=num_classes,
            hidden_dim=td_hidden_dim,
            num_queries=self.num_queries,
            num_heads=td_num_heads,
            ff_dim=td_ff_dim,
            num_dec_layers=td_num_layers,
            pre_norm=td_pre_norm,
            mask_dim=td_mask_dim,
            num_feature_levels = maskformer_num_feature_levels,
            enforce_input_proj=td_enforce_proj)
        self.loss = loss

 
        self.pretrained = pretrained
        self.sem_seg_postprocess_before_inference = False
        self.init_weight()


    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls)[..., :-1]
        mask_pred = F.sigmoid(mask_pred)
        semseg = paddle.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg


    def sem_seg_postprocess(self, result, img_size, output_height,
                            output_width):
        """
        Return semantic segmentation predictions in the original resolution.

        The input images are often resized when entering semantic segmentor. Moreover, in same
        cases, they also padded inside segmentor to be divisible by maximum network stride.
        As a result, we often need the predictions of the segmentor in a different
        resolution from its inputs.

        Args:
            result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
                where C is the number of classes, and H, W are the height and width of the prediction.
            img_size (tuple): image size that segmentor is taking as input.
            output_height, output_width: the desired output resolution.

        Returns:
            semantic segmentation prediction (Tensor): A tensor of the shape
                (C, output_height, output_width) that contains per-pixel soft predictions.
        """
        result = paddle.unsqueeze(result[:, :img_size[0], :img_size[1]], axis=0)
        result = F.interpolate(
            result,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False)[0]
        return result


    def forward(self, x,  inputs = None, current_iter = None):
        if self.neck:
            feats = self.neck(x)    #特征图输出从大到小
        else:
            if isinstance(x, list):
                feats = x[0] # 直接取feature形式
        x_shape = inputs['image'].shape[2:] 
        multi_scale_features, mask_features = self.pixel_decoder(feats)
        pred_logits, pred_masks, aux_logits, aux_masks = self.transformer_decoder(
            multi_scale_features, mask_features)
        #输出构成
        outputs = {}
        outputs['pred_logits'] = pred_logits
        outputs['pred_masks'] = pred_masks
        outputs['aux_outputs'] = []
        for aux_logit,aux_mask  in zip(aux_logits, aux_masks):
            outputs['aux_outputs'].append({'pred_logits':aux_logit,'pred_masks':aux_mask})
        if self.training:
            return self.loss(outputs, inputs['instances'])
        else:
            mask_cls_results = outputs["pred_logits"]  
            mask_pred_results = outputs["pred_masks"]  

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=x_shape,
                mode="bilinear",
                align_corners=False, )  
            processed_results = []

            for mask_cls_result, mask_pred_result in zip(mask_cls_results,
                                                         mask_pred_results):  
                image_size = x_shape
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = self.sem_seg_postprocess(
                        mask_pred_result, image_size, image_size[0],
                        image_size[1])

                r = self.semantic_inference(mask_cls_result, mask_pred_result)

                if not self.sem_seg_postprocess_before_inference:
                    r = self.sem_seg_postprocess(r, image_size, image_size[0],
                                                 image_size[1])
                processed_results.append(paddle.unsqueeze(r,axis = 0))

            # r = r[None, ...]
            processed_results = paddle.concat(processed_results, axis = 0)
            return [processed_results]


    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
