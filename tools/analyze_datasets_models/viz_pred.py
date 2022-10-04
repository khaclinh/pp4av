"""
Visualize ground truth and prediction of models on a specific image for easy comparison.

"""
import os
import argparse

import cv2

from utils import draw_annotation


def parse_args():
    parser = argparse.ArgumentParser('Search for false detections')
    parser.add_argument('type', choices=['normal', 'woodscape'], type=str, help='Image type')
    parser.add_argument('model', nargs='+', type=str, help='Model names')
    parser.add_argument('image', type=str, help='Relative path to image from root')
    return parser.parse_args()


NORMAL_IMAGE_ROOT = r'C:\Users\giangnd\Documents\bug\testset'
NORMAL_GT_ROOT = r'C:\Users\giangnd\Documents\bug\fixed\fixed_label'
NORMAL_PRED_ROOT = r'C:\Users\giangnd\Documents\bug\pred'

WOODSCAPE_IMAGE_ROOT = r'C:\Users\giangnd\Documents\bug\woodscape_data\woodscape'
WOODSCAPE_GT_ROOT = r'C:\Users\giangnd\Documents\bug\woodscape_gt\woodscape'
WOODSCAPE_PRED_ROOT = r'C:\Users\giangnd\Documents\bug\woodscape_pred\woodscape'


if __name__ == '__main__':
    args = parse_args()

    if args.type == 'normal':
        img_path = os.path.join(NORMAL_IMAGE_ROOT, args.image)
        gt_path = os.path.join(NORMAL_GT_ROOT, args.image[:-4] + '.txt')

        gt_output_dir = os.path.join('viz', 'gt')
        gt_img = draw_annotation(img_path, gt_path)
        os.makedirs(os.path.dirname(os.path.join(gt_output_dir, args.image)), exist_ok=True)
        cv2.imwrite(os.path.join(gt_output_dir, args.image), gt_img)

        for model_name in args.model:
            pred_path = os.path.join(NORMAL_PRED_ROOT, model_name, args.image[:-4] + '.txt')
            pred_img = draw_annotation(img_path, pred_path)
            pred_output_dir = os.path.join('viz', model_name)
            os.makedirs(os.path.dirname(os.path.join(pred_output_dir, args.image)), exist_ok=True)
            cv2.imwrite(os.path.join(pred_output_dir, args.image), pred_img)

    elif args.type == 'woodscape':
        img_path = os.path.join(WOODSCAPE_IMAGE_ROOT, args.image)
        gt_path = os.path.join(WOODSCAPE_GT_ROOT, args.image[:-4] + '.txt')

        gt_output_dir = os.path.join('viz', 'gt', 'woodscape')
        gt_img = draw_annotation(img_path, gt_path)
        os.makedirs(os.path.dirname(os.path.join(gt_output_dir, args.image)), exist_ok=True)
        cv2.imwrite(os.path.join(gt_output_dir, args.image), gt_img)

        for model_name in args.model:
            pred_path = os.path.join(WOODSCAPE_PRED_ROOT, model_name, args.image[:-4] + '.txt')
            pred_img = draw_annotation(img_path, pred_path)
            pred_output_dir = os.path.join('viz', model_name, 'woodscape')
            os.makedirs(os.path.dirname(os.path.join(pred_output_dir, args.image)), exist_ok=True)
            cv2.imwrite(os.path.join(pred_output_dir, args.image), pred_img)
