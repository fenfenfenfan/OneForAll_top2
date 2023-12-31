English | [简体中文](./README_cn.md)

# MobileSeg

These semantic segmentation models are designed for mobile and edge devices.

MobileSeg models adopt encoder-decoder architecture and use lightweight models as encoder.

## Reference

> Sandler, Mark, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. "Mobilenetv2: Inverted residuals and linear bottlenecks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4510-4520. 2018.

> Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314-1324. 2019.

> Ma, Ningning, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." In Proceedings of the European conference on computer vision (ECCV), pp. 116-131. 2018.

> Yu, Changqian, Bin Xiao, Changxin Gao, Lu Yuan, Lei Zhang, Nong Sang, and Jingdong Wang. "Lite-hrnet: A lightweight high-resolution network." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10440-10450. 2021.

> Han, Kai, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, and Chang Xu. "Ghostnet: More features from cheap operations." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1580-1589. 2020.

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (flip) | mIoU (ms+flip) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|MobileSeg|MobileNetV2|1024x512|80000|73.94%|74.32%|75.38%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_mobilenetv2_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_mobilenetv2_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=f210c79b6fd52f5135cf2f238e9d678d)|
|MobileSeg|MobileNetV3_large_x1_0|1024x512|80000|73.47%|73.72%|74.72%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_mobilenetv3_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_mobilenetv3_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=28c57d0e666337ea98a1046160ef95d2)|
|MobileSeg|Lite_HRNet_18|1024x512|80000|70.75%|71.62%|72.53%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_litehrnet18_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_litehrnet18_cityscapes_1024x512_80k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=02706145c7c463f3c76a0cb9d54728b8)|
|MobileSeg|ShuffleNetV2_x1_0|1024x512|80000|69.46%|70.00%|70.90%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_shufflenetv2_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_shufflenetv2_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=3d83c00cf9b90f2446959e8c97a4fb7a)|
|MobileSeg|GhostNet_x1_0|1024x512|80000|71.88%|72.22%|73.08%|[model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_ghostnet_cityscapes_1024x512_80k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/mobileseg_ghostnet_cityscapes_1024x512_80k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=73a6b325c0ae941a40746d53911c03bc)|


## Inference Speed

| Model    | Backbone |  V100 TRT Inference Speed(FPS)  | Snapdragon 855 Inference Speed(FPS) |
|:-------- |:--------:|:-------------------------------:|:-----------------------------------:|
| MobileSeg      | MobileNetV2              | 67.57          | 27.01   |
| **MobileSeg**  | MobileNetV3_large_x1_0   | 67.39          | 32.90   |
| MobileSeg      | Lite_HRNet_18            | *10.5*         | 13.05   |
| MobileSeg      | ShuffleNetV2_x1_0        | *37.09*        | 39.61   |
| MobileSeg      | GhostNet_x1_0            | *35.58*        | 38.74   |

Note that:
* Test the inference speed on Nvidia GPU V100: use PaddleInference Python API, enable TensorRT, the data type is FP32, the dimension of input is 1x3x1024x2048.
* Test the inference speed on Snapdragon 855: use PaddleLite CPP API, 1 thread, the dimension of input is 1x3x256x256.
