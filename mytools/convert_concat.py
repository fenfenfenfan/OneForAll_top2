import paddle
from collections import OrderedDict

dino_vit_base = paddle.load("track1/pretrained/dino_vit-base_paddle.pdmodel")
convnext = paddle.load("track1/pretrained/convert_model_x.pdparams")


covert_model_list = []
for k,v in convnext.items():
    covert_model_list.append((k,v))
for k,v in dino_vit_base.items():
    if 'backbone' in k:continue
    covert_model_list.append((k,v))

convert_model = OrderedDict(covert_model_list)


paddle.save(convert_model,"track1/pretrained/convert_model_x_dino.pdparams")
