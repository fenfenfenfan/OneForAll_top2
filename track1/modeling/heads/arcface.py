
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
import math
from modeling.losses import cross_entropy_loss
from paddle.nn import AdaptiveAvgPool2D
class GeM(nn.Layer):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()

        self.p = paddle.create_parameter(shape=[1], dtype='float32',default_initializer=paddle.nn.initializer.Assign(paddle.ones([1])*p))
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clip(min=eps).pow(p), (x.shape[-2], x.shape[-1])).pow(1./p)
        
class NormLinear(nn.Layer):
    """An enhanced linear layer, which could normalize the input and the linear
    weight.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample
        bias (bool): Whether there is bias. If set to ``False``, the
            layer will not learn an additive bias. Defaults to ``True``.
        feature_norm (bool): Whether to normalize the input feature.
            Defaults to ``True``.
        weight_norm (bool):Whether to normalize the weight.
            Defaults to ``True``.
    """

    def __init__(self,
                 in_features,
                 out_features):

        super().__init__()
        self.weight = paddle.create_parameter(shape=[in_features, out_features], dtype='float32')

    def forward(self, input):
        return F.linear(F.normalize(input), F.normalize(self.weight))


class SubCenterNormLinear(nn.Layer):
    """An enhanced linear layer, which could normalize the input and the linear
    weight.
    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample
        bias (bool): Whether there is bias. If set to ``False``, the
            layer will not learn an additive bias. Defaults to ``True``.
        feature_norm (bool): Whether to normalize the input feature.
            Defaults to ``True``.
        weight_norm (bool):Whether to normalize the weight.
            Defaults to ``True``.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 k = 3):

        super().__init__()
        self.out_features = out_features
        self.k = k

        self.weight = paddle.create_parameter(shape=[in_features, out_features * k], dtype='float32')

    def forward(self, input):

        input = F.normalize(input)
        weight = F.normalize(self.weight)

        cosine_all = F.linear(input, weight)
        cosine_all = cosine_all.reshape([-1, self.out_features, self.k])
        cosine = paddle.max(cosine_all, axis=2)
        return cosine

                
class ArcMarginProduct(nn.Layer):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, number_sub_center=1, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        # self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        
        # nn.init.xavier_uniform_(self.weight)
        if number_sub_center == 1:
            self.norm_linear = NormLinear(in_features, out_features)
        else:
            self.norm_linear = SubCenterNormLinear(
                    in_features, out_features, k=number_sub_center)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------

        cosine = self.norm_linear(input)
        if not self.training:
            return cosine
        # cosine: [1, 196]
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        # sine: [1, 196]
        phi = cosine * self.cos_m - sine * self.sin_m
        # phi: [1, 196]
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------

        one_hot = paddle.zeros(cosine.shape)
        
        one_hot = paddle.nn.functional.one_hot(label,num_classes=self.out_features)
        # one_hot: [1, 196]
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output: [1, 196]
        output *= self.s

        return output
    
def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

class ArcFaceModel(nn.Layer):
    def __init__(self, in_features, mid_features=1024, out_features=512,num_classes=192, s=30.0, 
                 m=0.50, easy_margin=False,number_sub_center=1, ls_eps=0.0, adacos=False,weight=1.0,shortcut=False):
        super().__init__()

        # in_features = self.model.classifier.in_features
        self.classifier = nn.Identity()
        self.global_pool = nn.Identity()
        self.pooling = GeM()
        if shortcut:
            self.MLP = nn.Linear(in_features, out_features)
        self.shortcut = shortcut
        self.embedding = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.BatchNorm1D(mid_features),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mid_features,out_features),
        )

        if adacos:
            s = math.sqrt(2)*math.log(num_classes-1)

        self.fc = ArcMarginProduct(
            # Embedding size
            out_features,
            # Number of output classes 
            num_classes,
            number_sub_center,
            s=s, 
            m=m, 
            easy_margin=easy_margin, 
            ls_eps=ls_eps
            )
        self.weight = weight
        self.avg_pool = AdaptiveAvgPool2D(1, data_format="NCHW")
        self.flatten = nn.Flatten()

    def forward(self, features, labels):
        feature = features[0][-1]
        # pooled_features = self.model(images).flatten(1)
        # pooled_features = self.pooling(feature).flatten(1)
        # pooled_features = F.avg_pool2d(feature,(feature.shape[-2], feature.shape[-1]))
        pooled_features = self.avg_pool(feature)
        pooled_features = self.flatten(pooled_features)
        # pooled_features: [1, 1536]
        embedding = self.embedding(pooled_features)
        # embedding: [1, 512]
        if self.shortcut:
            embedding += self.MLP(pooled_features)

        labels = labels['targets'].astype('int64')
        output = self.fc(embedding, labels)
        # output: [1, 196]
        if self.training:
            loss = dict()
            loss['crossentropy_loss'] = self.weight * nn.CrossEntropyLoss()(output, labels)
            return loss
        else:
            return output
    