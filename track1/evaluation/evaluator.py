"""evaluation.evaluator
"""
import logging
import time
import datetime
from contextlib import contextmanager

import paddle
from utils import comm
from utils.logger import log_every_n_seconds

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def preprocess_inputs(self, inputs):
        """preprocess_inputs
        """
        
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.
        Args:
            inputs: the inputs that's used to call the model.
            outputs: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass

from paddleseg.transforms import functional
import cv2
import copy 
import math
import numpy as np
import os
def cal_real_params(inputs, aug_params, stride):
    '''
    args:
        inputs: inputs(dict)
        s: scale(list),
        f: flip(list),1 means flip, 0 means not flip
    '''
    s = aug_params['s']
    f = aug_params['f']
    assert len(s)==len(f), f"s and f should have the same length, but got {len(s)} and {len(f)}"

    if 'fgvc' in inputs.keys():
        task = 'fgvc'
    elif 'trafficsign' in inputs.keys():
        task = 'trafficsign'
    h,w = inputs[task]["image"].shape[-2:]
    for i, scale in enumerate(s):
        nh = math.ceil(h * scale / stride) * stride
        aug_params['s'][i] = nh/h
    return aug_params
        

def img_aug(inputs, aug_params, stride):
    '''
    args:
        inputs: inputs(dict)
        s: scale(list),
        f: flip(list),1 means flip, 0 means not flip
    '''
    s = aug_params['s']
    f = aug_params['f']

    inputs_list = []
    if 'fgvc' in inputs.keys():
        task = 'fgvc'
    elif 'trafficsign' in inputs.keys():
        task = 'trafficsign'
    img = copy.deepcopy(inputs[task]["image"][0]).numpy().transpose((1,2,0))

    for scale, flip in zip(s, f):
        h, w = img.shape[:2]
        nh = math.ceil(h * scale / stride) * stride
        nw = math.ceil(w * scale / stride) * stride
        sw = nw/w
        sh = nh/h
        new_img = functional.resize(img, (nh, nw), cv2.INTER_LINEAR)
        
        if flip:
            new_img = cv2.flip(new_img, 1)
        new_img = new_img.transpose((2,0,1))[None,:,:,:]
        new_input = copy.deepcopy(inputs)
        new_input[task]["image"] = paddle.to_tensor(new_img)
        new_input[task]["scale_factor"]*=paddle.to_tensor([sh, sw],place=new_input[task]["scale_factor"].place)
        new_input[task]["im_shape"]*=paddle.to_tensor([sh, sw],place=new_input[task]["im_shape"].place)
        inputs_list.append(new_input)
    return inputs_list

def nms(bbox, thresh):

    scores = bbox[:,1]

    categories = list(range(45))
    category_idxs = bbox[:,0].astype('int32')
    out = paddle.vision.ops.nms(bbox[:,2:6],
                                thresh,
                                scores,
                                category_idxs,
                                categories)
    return out
        
def custom_nms(out, thresh):
    nms_after_result = []
    for i in range(45):
    
        box = out[out[:,0]==i]
        if len(box)==0:
            continue
        index = nms(box,thresh)
        cat_final_result = box[index].reshape([-1,6])
        nms_after_result.append(cat_final_result)
    nms_after_result = paddle.concat(nms_after_result,axis=0)
    return  nms_after_result

def matrix2list(matrix, img_id):
    '''
    matrix.shape : (n,6)
    {"image_id": 1, "category_id": 39, "bbox": [167.82223510742188, 1248.7110595703125, 204.79998779296875, 204.800048828125], "score": 0.07833300530910492}
    '''
    res_list = []
    for box in matrix:
        box = box.tolist()
        x1,y1,x2,y2 = box[2:]
        x,y,w,h = x1, y1, (x2-x1), (y2-y1)
        
        d = dict(
            image_id=img_id,
            category_id = int(box[0]),
            bbox = [x,y,w,h],
            score = box[1]
        )
        res_list.append(d)
    return res_list

def inverse_sf_box(boxes, image_size, f):
    # x1,y1,x2,y2
    h,w = image_size
    # print("inverse_sf_box:",image_size)
    # boxes[:,2:]/=s
    if f:
        x1, x2 = boxes[:,2], boxes[:,4]
        # print(x1[0],x2[0])
        boxes[:,2] = w-x2
        boxes[:,4] = w-x1
        # print(boxes[:,2][0],boxes[:,4][0])
    # 边界处理
    boxes[:,2] = boxes[:,2].clip(min=0, max=w-1)
    boxes[:,3] = boxes[:,3].clip(min=0, max=h-1)
    boxes[:,4] = boxes[:,4].clip(min=0, max=w-1)
    boxes[:,5] = boxes[:,5].clip(min=0, max=h-1)
    # # 去掉面积为0的  这一步操作貌似特别慢
    # size = (boxes[:,4] - boxes[:,2]) * (boxes[:,5] - boxes[:,3])
    # boxes=boxes[size>0]
    # print(boxes.shape)
    return boxes

import random
def save_det_vis(bboxes, img_dict, debug_puth, task, image_id, color):
    '''
        bboxes: (n,6), cat,score,x1,y1,x2,y2
    '''
    
    save_path = os.path.join(debug_puth,task)
    os.makedirs(save_path, exist_ok=True)
    img = np.array(img_dict[image_id].tolist(),dtype=np.uint8) # 这里不知道为什么，很怪，只能这样搞
    for box in bboxes:
        box = box.tolist()
        if box[1]<0.1:continue
        s = 1280/2048
        cv2.rectangle(img,(int(box[2]*s), int(box[3]*s)), (int(box[4]*s), int(box[5]*s)),color[int(box[0])], 2)
        cv2.putText(img, str(int(box[0]))+":"+str(box[1])[:4], (int(box[2]*s), int(box[3]*s)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(save_path, f"{image_id}.jpg"),img)

def save_cls_vis(cat_dict, img_dict, debug_puth, task):
    k = list(cat_dict.keys())[0]
    v = list(cat_dict.values())[0]
    save_path = os.path.join(debug_puth,task)
    os.makedirs(save_path, exist_ok=True)
    img = np.array(img_dict[k].tolist(),dtype=np.uint8) # 这里不知道为什么，很怪，只能这样搞

    cv2.putText(img, str(v), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(save_path, k),img)

def inference_on_dataset(model, data_loader, evaluator, flip_test=False, moe_group=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
        flip_test (bool): If get features with flipped images
    Returns:
        The return value of `evaluator.evaluate()`
    """
    ############### 超参 ###############
    # det_image_size = [1120,1120]   # 检测任务原图的h,w
    stride = 32
    use_nms = False
    nms_iou_thresh = 0.6
    nms_score_thre = 0.0
    visualization = False    # 推理结果可视化
    debug_puth = "debug" # 可视化保存路径
    simple_test = False      # 只推理少量的图
    simple_test_per_gpu = 5     # simple_test 模式下每张卡推理多少图
    # aug_params = {
    #     'fgvc':{            # 分类
    #         's' : [1, 1.2, 1.4, 1, 1.2, 1.4],   # 放缩比例
    #         'f' : [0,   0, 0, 1,   1, 1]      # 是否水平翻转，1是，0 否
    #     },
    #     'trafficsign':{     # 检测
    #         's' : [0.9, 1, 1.1, 0.9, 1, 1.1 ],   # 放缩比例
    #         'f' : [0,   0, 0, 1,   1, 1]      # 是否水平翻转，1是，0 否
    #     }
    # }
    # aug_params = {
    #     'fgvc':{            # 分类
    #         's' : [0.93, 1, 1.1, 1.2, 1.4, 0.93, 1, 1.1, 1.2, 1.4],   # 放缩比例
    #         'f' : [0,    0,   0,   0, 0,   1,    1,   1,  1,  1]      # 是否水平翻转，1是，0 否
    #     },
    #     'trafficsign':{     # 检测
    #         's' : [0.7, 0.8, 0.9, 1, 1.1, 0.7, 0.8, 0.9, 1, 1.1 ],   # 放缩比例
    #         'f' : [0,   0,   0,   0, 0,   1,   1,   1,   1, 1]      # 是否水平翻转，1是，0 否
    #     }
    # }
    aug_params = {
        'fgvc':{            # 分类
            's' : [0.93, 1, 1.2, 1.4, 0.93, 1, 1.2, 1.4],   # 放缩比例
            'f' : [0,    0,  0,  0,   1,    1,   1,  1]      # 是否水平翻转，1是，0 否
        },
        # 'fgvc':{            # 分类
        #     's' : [1],   # 放缩比例
        #     'f' : [1]      # 是否水平翻转，1是，0 否
        # },
        'trafficsign':{     # 检测
            's' : [0.8, 0.9, 1, 1.1, 0.8, 0.9, 1, 1.1 ],   # 放缩比例
            'f' : [0,   0,   0, 0,   1,   1,   1, 1]      # 是否水平翻转，1是，0 否
        }
    }
    # aug_params = {
    #     'fgvc':{            # 分类
    #         's' : [1, 1.2, 1, 1.2],   # 放缩比例
    #         'f' : [0, 0,   1, 1]      # 是否水平翻转，1是，0 否
    #     },
    #     'trafficsign':{     # 检测
    #         's' : [0.9, 1, 1.1, 0.9, 1, 1.1 ],   # 放缩比例
    #         'f' : [0,   0, 0, 1,   1, 1]      # 是否水平翻转，1是，0 否
    #     }
    # }
    print(aug_params)
    # 用于还原图像
    mean=np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape((1,1,3))
    std=np.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape((1,1,3))
    ############### 超参 ###############
    r = lambda: random.randint(0,255)
    color = [[r(),r(),r()] for _ in range(45)]
    img_dict = {}
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    if hasattr(data_loader, 'dataset'):
        logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    model.eval()
    
    task = 'fgvc'
    with paddle.no_grad():
        for idx, inputs in enumerate(data_loader):
            # print('inputs: ', inputs)
            
            if simple_test and idx >= simple_test_per_gpu:
                break
            
            if 'fgvc' in inputs.keys():
                task = 'fgvc'
            elif 'trafficsign' in inputs.keys():
                task = 'trafficsign'
            det_image_size = paddle.floor(inputs[task]["im_shape"] / inputs[task]["scale_factor"] + 0.5).tolist()[0]
            # print('det_image_size',det_image_size)
            # det_image_size = inputs[task]["image"]
            if visualization:
                img = inputs[task]["image"].numpy()[0].transpose((1,2,0))
                img=(img*std+mean)
                os.makedirs(os.path.join(debug_puth, task), exist_ok=True)
                if task == 'fgvc':
                    path = inputs[task]["img_paths"][0]
                    img_dict[path[path.rfind('/')+1:]] = img.astype(np.uint8)
                    cv2.imwrite(os.path.join(debug_puth, task, f"orig_"+path[path.rfind('/')+1:]), img.astype(np.uint8))
                elif task == 'trafficsign':
                    img_dict[int(inputs[task]["im_id"][0][0])] = img.astype(np.uint8)
                    cv2.imwrite(os.path.join(debug_puth, task, f"orig_{int(inputs[task]['im_id'][0][0])}.jpg"), img.astype(np.uint8))

            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            
            start_compute_time = time.perf_counter()
            
            # real_aug_params = cal_real_params(inputs, aug_params[task], stride)
            
            new_inputs_list = img_aug(inputs, aug_params[task], stride)
            bbox = []
            box_num = 0
            scores = None
            for i, new_inputs in enumerate(new_inputs_list):
                if visualization:
                    img = new_inputs[task]["image"].numpy()[0].transpose((1,2,0))
                    img=(img*std+mean)
                    if task == 'fgvc':
                        path = new_inputs[task]["img_paths"][0]
                        cv2.imwrite(os.path.join(debug_puth, task, f"aug_{i}_"+path[path.rfind('/')+1:]), img.astype(np.uint8))
                    elif task == 'trafficsign':
                        # img_dict[int(new_inputs[task]["im_id"][0][0])] = img.astype(np.uint8)
                        cv2.imwrite(os.path.join(debug_puth, task, f"aug_{i}_{int(new_inputs[task]['im_id'][0][0])}.jpg"), img.astype(np.uint8))
                # print(new_inputs[task].keys())
                outputs = model(new_inputs, moe_group) if moe_group is not None else model(new_inputs)
                
                if task == 'trafficsign':
                    # x1 y1 x2 y2
                    f = aug_params[task]['f'][i]
                    bbox.append(inverse_sf_box(outputs[task]["bbox"], det_image_size, f))
                    box_num += outputs[task]["bbox_num"]
                    # print(outputs[task].keys())
                elif task == 'fgvc':
                    if scores is not None:
                        scores += outputs[task]
                    else:
                        scores = outputs[task]
                
                total_compute_time += time.perf_counter() - start_compute_time
            if task == 'trafficsign':
                outputs = {
                    'trafficsign':{
                        "bbox":paddle.concat(bbox,axis=0),
                        "bbox_num":box_num
                    }
                }
                # print("orig out:",outputs[task]["bbox"])
            elif task=='fgvc':
                scores /= len(aug_params[task]['s'])
                outputs[task] = scores

            evaluator.process(inputs, outputs)


            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )
            
            if comm.get_world_size() > 1:
                comm.synchronize()
    
    model.train()
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    if results is None or results=={}:
        return {}

    if task == 'fgvc':
        key = 'cls'
        if visualization:
            cls_list = results[key]
            for cat_dict in cls_list:
                if list(cat_dict.keys())[0] in img_dict.keys():
                    save_cls_vis(cat_dict, img_dict, debug_puth, task)
    else:
        
        key = 'dec'
    
        boxes_list = results[key]
        new_boxes_list = []
        single_img_box_list = []
        
        last_img_id = 1
        box_len = len(boxes_list)
        boxes_list.sort(key = lambda x: x["image_id"] )
        for i, box in enumerate(boxes_list):

            if box["image_id"]!=last_img_id:
                out = paddle.to_tensor(single_img_box_list)

                if use_nms:
                    # print("before nms", out)
                    out = custom_nms(out, nms_iou_thresh)
                    # print("after nms", out)
                    out = out[out[:,1]>=nms_score_thre]
                if visualization and last_img_id in img_dict.keys():
                    save_det_vis(out, img_dict, debug_puth, task, last_img_id,color)
                new_boxes_list.extend(matrix2list(out, last_img_id))
                
                last_img_id = box["image_id"]
                single_img_box_list = []

            x,y,w,h = box["bbox"]
            x2,y2 = x+w, y+h
            single_img_box_list.append([box["category_id"],box["score"], x,y,x2,y2])

            if i==box_len-1:
                out = paddle.to_tensor(single_img_box_list)
                if use_nms:
                    out = custom_nms(out, nms_iou_thresh)
                    out = out[out[:,1]>=nms_score_thre]
                
                if visualization and last_img_id in img_dict.keys():
                    save_det_vis(out, img_dict, debug_puth, task, last_img_id,color)
                new_boxes_list.extend(matrix2list(out, last_img_id))
                

        results[key] = new_boxes_list

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
