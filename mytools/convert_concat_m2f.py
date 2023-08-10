import paddle
from collections import OrderedDict

m2f = paddle.load("track1/pretrained/r50_m2f.pdparams")
convnext_dino = paddle.load("track1/pretrained/convert_model_x_dino.pdparams")


covert_model_list = []
for k,v in convnext_dino.items():
    covert_model_list.append((k,v))
for k,v in m2f.items():
    print(k)
    if 'backbone' in k:continue
    covert_model_list.append(('heads.segmentation.'+k,v))

convert_model = OrderedDict(covert_model_list)


paddle.save(convert_model,"track1/pretrained/convert_model_x_dino_resnet_m2f.pdparams")
