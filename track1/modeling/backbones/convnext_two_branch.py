from functools import partial
import paddle
import paddle.nn as nn
FLAG=''
class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        output = inputs.divide(keep_prob) * random_tensor # divide to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)

class LayerNorm2D(nn.LayerNorm):
    """ LayerNorm for channels-fisrt tensors with 2d spatial dimension, e.g., (N, C, H, W)"""
    def __init__(self, normalized_shape, epsilon=1e-6):
        super().__init__(normalized_shape, epsilon=epsilon)

    def forward(self, x):
        return nn.functional.layer_norm(x.transpose([0, 2, 3, 1]),
                                        self._normalized_shape,
                                        self.weight,
                                        self.bias,
                                        self._epsilon).transpose([0, 3, 1, 2])

class Identity(nn.Layer):
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class ConvMlp(nn.Layer):
    """ ConvMLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: conv2d -> act -> dropout -> conv2d -> dropout

    Attributes:
        fc1: nn.Conv2D
        fc2: nn.Conv2D
        act: GELU
        dropout: dropout after fc1 and fc2
    """

    def __init__(self, in_features, hidden_features, dropout=0.):
        super().__init__()
        # cls
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        # seg
        w_attr_3, b_attr_3 = self._init_weights()
        self.fc3 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_3,
                             bias_attr=b_attr_3)

        w_attr_4, b_attr_4 = self._init_weights()
        self.fc4 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_4,
                             bias_attr=b_attr_4)
        # # det
        # w_attr_5, b_attr_5 = self._init_weights()
        # self.fc5 = nn.Linear(in_features,
        #                      hidden_features,
        #                      weight_attr=w_attr_5,
        #                      bias_attr=b_attr_5)

        # w_attr_6, b_attr_6 = self._init_weights()
        # self.fc6 = nn.Linear(hidden_features,
        #                      in_features,
        #                      weight_attr=w_attr_6,
        #                      bias_attr=b_attr_6)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        if not FLAG=='seg':
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
        # elif FLAG=='seg':
        else:
            x = self.fc3(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc4(x)
            x = self.dropout(x)
        # else:
        #     x = self.fc5(x)
        #     x = self.act(x)
        #     x = self.dropout(x)
        #     x = self.fc6(x)
        #     x = self.dropout(x)
        return x


class Mlp(nn.Layer):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout

    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout: dropout after fc1 and fc2
    """

    def __init__(self, in_features, hidden_features, dropout=0.):
        super().__init__()
        # cls
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        # seg
        w_attr_3, b_attr_3 = self._init_weights()
        self.fc3 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_3,
                             bias_attr=b_attr_3)

        w_attr_4, b_attr_4 = self._init_weights()
        self.fc4 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_4,
                             bias_attr=b_attr_4)
        # det
        # w_attr_5, b_attr_5 = self._init_weights()
        # self.fc5 = nn.Linear(in_features,
        #                      hidden_features,
        #                      weight_attr=w_attr_5,
        #                      bias_attr=b_attr_5)

        # w_attr_6, b_attr_6 = self._init_weights()
        # self.fc6 = nn.Linear(hidden_features,
        #                      in_features,
        #                      weight_attr=w_attr_6,
        #                      bias_attr=b_attr_6)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        if not FLAG=='seg':
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
        # elif FLAG=='seg':
        else:
            x = self.fc3(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc4(x)
            x = self.dropout(x)
        # else:
        #     x = self.fc5(x)
        #     x = self.act(x)
        #     x = self.dropout(x)
        #     x = self.fc6(x)
        #     x = self.dropout(x)
        return x


class ConvNeXtBlock(nn.Layer):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 ls_init_value=1e-6,
                 conv_mlp=False,
                 mlp_ratio=4,
                 norm_layer=None):
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2D, epsilon=1e-6) if conv_mlp else partial(nn.LayerNorm, epsilon=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv2D(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim))
        self.gamma = paddle.create_parameter(
            shape=[dim], dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(ls_init_value))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.transpose([0, 2, 3, 1])
            x = self.norm(x)
            x = self.mlp(x)
            x = x.transpose([0, 3, 1, 2])
        if self.gamma is not None:
            x = x.multiply(self.gamma.reshape([1, -1, 1, 1]))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Layer):
    def __init__(self,
                 in_chs,
                 out_chs,
                 stride=2,
                 depth=2, 
                 dp_rates=None,
                 ls_init_value=1.0,
                 conv_mlp=False,
                 norm_layer=None,
                 cl_norm_layer=None,
                 cross_stage=False):
        super().__init__()
        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2D(in_chs, out_chs, kernel_size=stride, stride=stride))
        else:
            self.downsample = nn.Identity()
        dp_rates = dp_rates or [0.] * depth
        blocks = []
        for j in range(depth):
            blocks.append(ConvNeXtBlock(dim=out_chs,
                                        drop_path=dp_rates[j],
                                        ls_init_value=ls_init_value,
                                        conv_mlp=conv_mlp,
                                        norm_layer=norm_layer if conv_mlp else cl_norm_layer))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 num_classes=1000,
                 global_pool=True,
                 output_stride=32,
                 patch_size=4,
                 depths=[3, 3, 27, 3],
                 dims=[192, 384, 768, 1536],
                 ls_init_value=1e-6,
                 conv_mlp=False,
                 stem_type='patch',
                 head_init_scale=1.,
                 head_norm_first=False,
                 norm_layer=None,
                 dropout=0.5,
                 droppath=0.):
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = partial(LayerNorm2D, epsilon=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, epsilon=1e-6)
        else:
            assert conv_mlp
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.dropout = dropout
        self.feature_info = []

        if stem_type == 'patch':
            self.stem = nn.Sequential(
                nn.Conv2D(in_channels, dims[0], kernel_size=patch_size, stride=patch_size),
                norm_layer(dims[0]))
            curr_stride = patch_size
            prev_chs = dims[0]
        else:
            self.stem = nn.Sequential(
                nn.Conv2D(in_channels, 32, kernel_size=3, stride=2, padding=1),
                norm_layer(32),
                nn.GELU(),
                nn.Conv2D(32, 64, kernel_size=3, padding=1))
            curr_stride = 2
            prev_chs = 64

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in paddle.linspace(0, droppath, sum(list(depths))).split(list(depths))]
        stages = []
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            curr_stride *= stride
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs, out_chs, stride=stride,
                depth=depths[i], dp_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                norm_layer=norm_layer, cl_norm_layer=cl_norm_layer)
            )
            prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, layer=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs
        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(
                ('global_pool', nn.AdaptiveAvgPool2D(1)) if global_pool else Identity(),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else Identity()),
                ('drop', nn.Dropout(self.dropout)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else Identity()))



    def forward(self, x,flag):
        x = self.stem(x["image"])
        # if 'gt_bbox' in x:
        #     FLAG='det'
        # elif 'gt_field' in x:
        #     FLAG='seg'
        # else:
        #     FLAG='cls'
        global FLAG
        FLAG = flag
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)

        b = [outs[-1] for i in range(4)]
        return [tuple(outs),b]