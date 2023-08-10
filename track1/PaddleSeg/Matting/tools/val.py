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

import argparse
import os
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

import paddle
import paddleseg
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger, utils

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.core import evaluate, evaluate_ml
from ppmatting.utils import Config, MatBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--opts',
        help='Update the key-value pairs of all options.',
        default=None,
        nargs='+')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output/results')
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--save_results',
        dest='save_results',
        help='save prediction alpha while evaluating',
        action='store_true')
    parser.add_argument(
        '--metrics',
        dest='metrics',
        nargs='+',
        help='The metrics to evaluate, it may be the combination of ("sad", "mse", "grad", "conn")',
        type=str,
        default='sad')
    parser.add_argument(
        '--device',
        dest='device',
        help='Set the device type, which may be GPU, CPU or XPU.',
        default='gpu',
        type=str)

    return parser.parse_args()


def main(args):
    assert args.cfg is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.cfg, opts=args.opts)
    builder = MatBuilder(cfg)

    paddleseg.utils.show_env_info()
    paddleseg.utils.show_cfg_info(cfg)
    paddleseg.utils.set_device(args.device)

    model = builder.model
    val_dataset = builder.val_dataset

    if isinstance(model, paddle.nn.Layer):
        if args.model_path:
            utils.load_entire_model(model, args.model_path)
            logger.info('Loaded trained params of model successfully')
        evaluate(
            model,
            val_dataset,
            num_workers=args.num_workers,
            save_dir=args.save_dir,
            save_results=args.save_results,
            metrics=args.metrics)
    else:
        evaluate_ml(
            model,
            val_dataset,
            save_dir=args.save_dir,
            save_results=args.save_results)


if __name__ == '__main__':
    args = parse_args()
    main(args)
