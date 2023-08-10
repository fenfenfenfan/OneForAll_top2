import paddle
from collections import OrderedDict

paddleseg = paddle.load("track1/pretrained/convnext_xlarge.pdparams")


covert_model_list = []
for k,v in paddleseg.items():
    covert_model_list.append(("backbone."+k,v))


convert_model = OrderedDict(covert_model_list)


paddle.save(convert_model,"track1/pretrained/convert_model_x.pdparams")
