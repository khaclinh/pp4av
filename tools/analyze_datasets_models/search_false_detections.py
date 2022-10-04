"""
Browse ground truth and prediction folders to search for fail detection cases.

"""
import argparse
import os
from collections import defaultdict
from typing import Dict

import cv2
from tqdm import tqdm

from utils import parse_annotation, draw_annotation


def parse_args():
    parser = argparse.ArgumentParser('Search for false detections')
    parser.add_argument('--img', required=True, type=str, help='Path to image folder')
    parser.add_argument('--gt', required=True, type=str, help='Path to ground truth folder')
    parser.add_argument('--pred', required=True, type=str, help='Path to prediction folder')
    return parser.parse_args()


def find_missing_preds(img_dir: str, gt_dir: str, pred_dir: str, cls: int=0, iou_thresh: float=0.5) -> Dict[str, float]:
    img_names = sorted(os.listdir(img_dir))
    img_names = list(filter(lambda n: n.endswith('.jpg') or n.endswith('.png'), img_names))

    missing_preds = defaultdict(float)

    for img_name in tqdm(img_names, desc=img_dir):
        label_name = img_name[:-4] + '.txt'
        gt_path, pred_path = os.path.join(gt_dir, label_name), os.path.join(pred_dir, label_name)
        gt, pred = parse_annotation(gt_path), parse_annotation(pred_path)
        for gt_x1, gt_y1, gt_x2, gt_y2 in gt[cls]:
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            match = False

            for pred_x1, pred_y1, pred_x2, pred_y2 in pred[cls]:
                pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                intersect = max(min(gt_x2, pred_x2) - max(gt_x1, pred_x1), 0) * max(min(gt_y2, pred_y2) - max(gt_y1, pred_y1), 0)
                union = gt_area + pred_area - intersect
                iou = intersect / union

                if iou >= iou_thresh:
                    match = True
                    break

            if not match:
                missing_preds[img_name] = max(missing_preds[img_name], gt_area)

    return missing_preds


def find_false_alarms(img_dir: str, gt_dir: str, pred_dir: str, cls: int=0, iou_thresh: float=0.5) -> Dict[str, float]:
    img_names = sorted(os.listdir(img_dir))
    img_names = list(filter(lambda n: n.endswith('.jpg') or n.endswith('.png'), img_names))

    false_alarms = defaultdict(float)

    for img_name in tqdm(img_names, desc=img_dir):
        label_name = img_name[:-4] + '.txt'
        gt_path, pred_path = os.path.join(gt_dir, label_name), os.path.join(pred_dir, label_name)
        gt, pred = parse_annotation(gt_path), parse_annotation(pred_path)
        for pred_x1, pred_y1, pred_x2, pred_y2 in pred[cls]:
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            match = False

            for gt_x1, gt_y1, gt_x2, gt_y2 in gt[cls]:
                gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                intersect = max(min(gt_x2, pred_x2) - max(gt_x1, pred_x1), 0) * max(min(gt_y2, pred_y2) - max(gt_y1, pred_y1), 0)
                union = gt_area + pred_area - intersect
                iou = intersect / union

                if iou >= iou_thresh:
                    match = True
                    break

            if not match:
                false_alarms[img_name] = max(false_alarms[img_name], pred_area)

    return false_alarms


def main():
    img_root = r'C:\Users\giangnd\Documents\bug\testset'
    gt_root = r'C:\Users\giangnd\Documents\bug\fixed\fixed_label'

    model_names = ['alpr', 'anonymizer', 'aws', 'azure', 'gg', 'nvidia', 'retina', 'yolov5face', 'our']
    for model_name in model_names:
        pred_root = os.path.join(r'C:\Users\giangnd\Documents\bug\pred', model_name)

        for cls in [0, 1]:
            subdirs = os.listdir(img_root)
            for dirname in subdirs:
                img_dir = os.path.join(img_root, dirname)
                gt_dir = os.path.join(gt_root, dirname)
                pred_dir = os.path.join(pred_root, dirname)

                missing_preds = find_missing_preds(img_dir, gt_dir, pred_dir, cls=cls, iou_thresh=0.2)
                missing_imgs, missing_area = missing_preds.keys(), missing_preds.values()
                top_misses = sorted(zip(missing_area, missing_imgs), reverse=True)[:20]

                output_dir = os.path.join('missing_preds_%s_%s' % (model_name, cls), dirname)
                os.makedirs(output_dir, exist_ok=True)
                for _, img_name in top_misses:
                    label_name = img_name[:-4] + '.txt'

                    img_path, label_path = os.path.join(img_dir, img_name), os.path.join(gt_dir, label_name)
                    img = draw_annotation(img_path, label_path)
                    cv2.imwrite(os.path.join(output_dir, img_name[:-4] + '_gt' + img_name[-4:]), img)

                    img_path, label_path = os.path.join(img_dir, img_name), os.path.join(pred_dir, label_name)
                    img = draw_annotation(img_path, label_path)
                    cv2.imwrite(os.path.join(output_dir, img_name[:-4] + '_pred' + img_name[-4:]), img)


                false_alarms = find_false_alarms(img_dir, gt_dir, pred_dir, cls=cls, iou_thresh=0.2)
                false_imgs, false_area = false_alarms.keys(), false_alarms.values()
                top_falses = sorted(zip(false_area, false_imgs), reverse=True)[:20]

                output_dir = os.path.join('false_alarms_%s_%s' % (model_name, cls), dirname)
                os.makedirs(output_dir, exist_ok=True)
                for _, img_name in top_falses:
                    label_name = img_name[:-4] + '.txt'

                    img_path, label_path = os.path.join(img_dir, img_name), os.path.join(gt_dir, label_name)
                    img = draw_annotation(img_path, label_path)
                    cv2.imwrite(os.path.join(output_dir, img_name[:-4] + '_gt' + img_name[-4:]), img)

                    img_path, label_path = os.path.join(img_dir, img_name), os.path.join(pred_dir, label_name)
                    img = draw_annotation(img_path, label_path)
                    cv2.imwrite(os.path.join(output_dir, img_name[:-4] + '_pred' + img_name[-4:]), img)


if __name__ == '__main__':
    main()
