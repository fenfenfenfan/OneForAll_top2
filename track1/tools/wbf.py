import numpy as np
import os
import json
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import random
import cv2
from tqdm import tqdm
import copy



def one_model_wbf(json_results, score_level, max_class_id, th_nmsiou, th_score, weights=None):
    final_result = []
    levelup_id2annos = {}
    leveldown_id2annos = {}
    # 筛选
    levelup_result=[]
    leveldown_result = []
    model_weights = [1]
    for obj in json_results:
        levelup_result.append(obj)

    for box in levelup_result:
        img_id = box['image_id']
        if img_id not in levelup_id2annos:
            levelup_id2annos[img_id] = []
        levelup_id2annos[img_id].append(box)


    print("result bbox num: ",len(json_results))
    print("levelup_result bbox num: ",len(levelup_result))
    print("leveldown_result bbox num: ",len(leveldown_result))

    # wbf
    print("start wbf...")
    iou_thr = th_nmsiou
    skip_box_thr = th_score
    
    filter_num = 0
    for id, _ in levelup_id2annos.items():
        scores_list = []
        boxes_list = []
        labels_list = []
        for anno in levelup_id2annos[id]:
            box = anno['bbox']
            xmin = box[0]
            ymin = box[1]
            width = box[2]
            height = box[3]
            xmax = xmin + width
            ymax = ymin + height
            xmax = xmax / 2048
            xmin = xmin / 2048
            ymin = ymin / 2048
            ymax = ymax / 2048
            confidence = anno["score"]
            label_class = anno["category_id"]
            scores_list.append(confidence)
            boxes_list.append([xmin, ymin, xmax, ymax])
            labels_list.append(label_class)
        boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=model_weights,
                                                    iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='max')
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            score = float(scores[i])
            label = int(labels[i])
            x1 = round(float(x1*2048), 5)
            y1 = round(float(y1*2048), 5)
            x2 = round(float(x2*2048), 5)
            y2 = round(float(y2*2048), 5)
            if x2<7 or x1>(2048-7): # area refine
                filter_num+=1
                continue
            final_result.append({'image_id': id, "bbox": [x1, y1, x2-x1, y2-y1], "score": score, "category_id": label})
    print("wbf bbox num: ", len(final_result))
    print('filter num: ',filter_num)

    return final_result



def det_visualization(image_list, color, save_path, before_wbf_results=None, after_wbf_results=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for id, image in enumerate(tqdm(image_list)):
        img = cv2.imread(image)
        img2 = copy.copy(img)
        # print(img)
        if  before_wbf_results is not None:
            for bef_result in before_wbf_results:
                if bef_result['score']<0.05:continue
                image_id = bef_result['image_id']
                x1 = bef_result['bbox'][0]
                y1 = bef_result['bbox'][1]
                w = bef_result['bbox'][2]
                h = bef_result['bbox'][3]
                cat_id = bef_result['category_id']
                if id+1 ==image_id:
                    cv2.rectangle(img,(int(x1), int(y1)), (int(x1+w), int(y1+h)),color[int(cat_id)], 2)
                    cv2.putText(img, 'bef_wbf', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(save_path, f"bef_wbf_{id}.jpg"),img)
        if after_wbf_results is not None:
            for aft_result in after_wbf_results:
                if aft_result['score']<0.05:continue
                image_id = aft_result['image_id']
                x1 = aft_result['bbox'][0]
                y1 = aft_result['bbox'][1]
                w = aft_result['bbox'][2]
                h = aft_result['bbox'][3]
                cat_id = aft_result['category_id']
                if id+1 ==image_id:
                    cv2.rectangle(img2,(int(x1), int(y1)), (int(x1+w), int(y1+h)),color[::-1][int(cat_id)], 2)
                    cv2.putText(img2, 'aft_wbf', (int(x1+w), int(y1+h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 121, 91), 2)
            cv2.imwrite(os.path.join(save_path, f"aft_wbf_{id}.jpg"),img2)


if __name__ == "__main__":
    pred_json = 'outputs/tta/last_e60_noseg_det_mosaic_seg0.9_1_1.2_1.5_1.6_cls_0.93_1_1.2_1.4_det0.8_0.9_1_1.1_q100_flip3/pred_results.json'
    with open(pred_json, 'r') as f:
        datas = json.load(f)
    json_results = datas['dec']
    save_path = os.path.dirname(pred_json)
    pred_name = os.path.basename(pred_json)
    print(pred_name)
    save_json_path = os.path.join(save_path,f'wbf_0.55filter_{pred_name}')
    wbf_dec_results = one_model_wbf(json_results, score_level=0.01, max_class_id=45, th_nmsiou=0.55, th_score=0.0001, weights=None)
    submit_results = {}
    # submit_results['seg'] = datas['seg']
    submit_results['cls'] = datas['cls']
    submit_results['dec'] = wbf_dec_results
    with open(save_json_path, 'w') as f_w:
        json.dump(submit_results,f_w)
    print('wbf over, monster!')

    # Results Visualization

    # image_list = []
    # test_txt_path = 'datasets/track1_test_data/dec/test.txt'
    # test_img_path = 'datasets/track1_test_data/dec/test'
    # visual_img_save = 'det_wbf'
    # f_txt = open(test_txt_path, 'r')
    # datas = f_txt.readlines()
    # # get image path

    # r = lambda: random.randint(0,255)
    # color = [[r(),r(),r()] for _ in range(45)]
    # for data in datas[:20]:
    #     data = data.strip()
    #     img_path = os.path.join(test_img_path, data)
    #     image_list.append(img_path)
    # # det_visualization(image_list, color, visual_img_save, json_results)
    # det_visualization(image_list, color, visual_img_save, json_results, wbf_dec_results)
    # print('Visualization over, monster!')