"""
Crop some random faces and plates to survey the relationship between object size and clearliness.

"""
import random
import os

import cv2
from tqdm import tqdm
from PIL import Image

from utils import parse_annotation


FACE_WIDTH_THRESHOLDS = [8, 12, 15, 20, 25, 30]
PLATE_HEIGHT_THRESHOLDS = [6, 8, 10, 15, 20]

face_bins = [[] for _ in range(len(FACE_WIDTH_THRESHOLDS) + 1)]
plate_bins = [[] for _ in range(len(PLATE_HEIGHT_THRESHOLDS) + 1)]

img_root = r'C:\Users\giangnd\Documents\bug\testset'
gt_root = r'C:\Users\giangnd\Documents\bug\fixed\fixed_label'

subdirs = os.listdir(img_root)
for dirname in subdirs:
    img_dir = os.path.join(img_root, dirname)
    gt_dir = os.path.join(gt_root, dirname)

    img_names = sorted(os.listdir(img_dir))
    img_names = list(filter(lambda n: n.endswith('.jpg') or n.endswith('.png'), img_names))

    for img_name in tqdm(img_names, desc=img_dir):
        label_name = img_name[:-4] + '.txt'
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, label_name)
        gt = parse_annotation(gt_path)

        img = Image.open(img_path)
        img_h, img_w = img.height, img.width
        for x1, y1, x2, y2 in gt[0]:
            x1, y1, x2, y2 = round(x1 * img_w), round(y1 * img_h), round(x2 * img_w), round(y2 * img_h)
            bin_idx = 0
            for i in range(len(FACE_WIDTH_THRESHOLDS)):
                if x2 - x1 >= FACE_WIDTH_THRESHOLDS[i]:
                    bin_idx = i + 1
            face_bins[bin_idx].append((dirname, img_name, x1, y1, x2, y2))

        for x1, y1, x2, y2 in gt[1]:
            x1, y1, x2, y2 = round(x1 * img_w), round(y1 * img_h), round(x2 * img_w), round(y2 * img_h)
            bin_idx = 0
            for i in range(len(PLATE_HEIGHT_THRESHOLDS)):
                if y2 - y1 >= PLATE_HEIGHT_THRESHOLDS[i]:
                    bin_idx = i + 1
            plate_bins[bin_idx].append((dirname, img_name, x1, y1, x2, y2))

# Select 5 samples for each bin
face_samples = [random.sample(face_bin, k=10) for face_bin in face_bins]
plate_samples = [random.sample(plate_bin, k=10) for plate_bin in plate_bins]

for bin_idx in range(len(FACE_WIDTH_THRESHOLDS) + 1):
    os.makedirs(os.path.join('sample_faces', str(bin_idx)), exist_ok=True)
    for dirname, img_name, x1, y1, x2, y2 in face_samples[bin_idx]:
        img_path = os.path.join(img_root, dirname, img_name)
        img = cv2.imread(img_path)
        w, h = x2 - x1, y2 - y1
        face_img = img[max(y1 - h // 2, 0):y2 + h // 2, max(x1 - w // 2, 0):x2 + w // 2]
        cv2.imwrite(os.path.join('sample_faces', str(bin_idx), f'{dirname}_{img_name}_{x1}_{y1}_{x2}_{y2}.png'), face_img)

for bin_idx in range(len(PLATE_HEIGHT_THRESHOLDS) + 1):
    os.makedirs(os.path.join('sample_plates', str(bin_idx)), exist_ok=True)
    for dirname, img_name, x1, y1, x2, y2 in plate_samples[bin_idx]:
        img_path = os.path.join(img_root, dirname, img_name)
        img = cv2.imread(img_path)
        w, h = x2 - x1, y2 - y1
        plate_img = img[max(y1 - h // 2, 0):y2 + h // 2, max(x1 - w // 2, 0):x2 + w // 2]
        cv2.imwrite(os.path.join('sample_plates', str(bin_idx), f'{dirname}_{img_name}_{x1}_{y1}_{x2}_{y2}.png'), plate_img)
