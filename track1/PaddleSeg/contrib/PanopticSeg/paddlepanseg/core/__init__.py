# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# In this package we provide PaddleSeg-style functions for training, 
# validation, prediction, and inference.
# We do not re-use the APIs of PaddleSeg because there is a large gap
# between semantic segmentation and panoptic segmentation.

from .train import train
from .val import evaluate
from .predict import predict
from . import infer

__all__ = ['train', 'evaluate', 'predict', 'infer']
