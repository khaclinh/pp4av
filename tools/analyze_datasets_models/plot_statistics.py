import sys
import os
import re
import math
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

import numpy as np
import hub
import matplotlib.pyplot as plt
from matplotlib import rc
from PIL import Image
from tqdm import tqdm

from utils import parse_annotation


rc('axes', linewidth=2)
rc('font', weight='bold')


FACE_WIDTH_THRESHOLDS = list(range(5, 91, 5))
PLATE_HEIGHT_THRESHOLDS = list(range(5, 91, 5))


def parse_ccpd():
    plate_bins = [[] for _ in range(len(PLATE_HEIGHT_THRESHOLDS) + 1)]

    img_dir = '/home/giangnd/Documents/data/CCDP/CCPD2019/ccpd_base'
    img_names = sorted(os.listdir(img_dir))
    for img_name in tqdm(img_names):
        pattern = r'^\d+-\d+_\d+-(\d+)&(\d+)_(\d+)&(\d+)-\d+&\d+_\d+&\d+_\d+&\d+_\d+&\d+-\d+_\d+_\d+_\d+_\d+_\d+_\d+-\d+-\d+\.jpg$'
        assert re.match(pattern, img_name) is not None, 'Wrong image name format %s' % img_name

        x1, y1 = int(re.sub(pattern, r'\1', img_name)), int(re.sub(pattern, r'\2', img_name))
        x2, y2 = int(re.sub(pattern, r'\3', img_name)), int(re.sub(pattern, r'\4', img_name))
        w, h = x2 - x1, y2 - y1
        bin_idx = 0
        for i in range(len(PLATE_HEIGHT_THRESHOLDS)):
            if h >= PLATE_HEIGHT_THRESHOLDS[i]:
                bin_idx = i + 1
        plate_bins[bin_idx].append((w, h))

    return plate_bins


def parse_fddb():
    face_bins = [[] for _ in range(len(FACE_WIDTH_THRESHOLDS) + 1)]

    label_dir = '/home/giangnd/Documents/data/FDDB/FDDB-folds'
    filenames = sorted(os.listdir(label_dir))
    for filename in tqdm(filenames):
        if not re.match(r'^FDDB-fold-\d{2}-ellipseList\.txt$', filename):
            continue

        pattern = r'^([\d\.-]+) ([\d\.-]+) ([\d\.-]+) ([\d\.-]+) ([\d\.-]+)  1$'
        with open(os.path.join(label_dir, filename), 'r') as f:
            line = f.readline().strip()
            while line:
                if re.match(pattern, line):
                    a = float(re.sub(pattern, r'\1', line))
                    b = float(re.sub(pattern, r'\2', line))
                    angle = float(re.sub(pattern, r'\3', line))
                    w = 2 * max(abs(b * math.sin(angle)), abs(a * math.cos(angle)))
                    h = 2 * max(abs(a * math.sin(angle)), abs(b * math.cos(angle)))
                    bin_idx = 0
                    for i in range(len(FACE_WIDTH_THRESHOLDS)):
                        if w >= FACE_WIDTH_THRESHOLDS[i]:
                            bin_idx = i + 1
                    face_bins[bin_idx].append((w, h))

                line = f.readline().strip()

    return face_bins


def parse_lucian():
    plate_bins = [[] for _ in range(len(PLATE_HEIGHT_THRESHOLDS) + 1)]

    xml_dir = '/home/giangnd/Documents/data/Lucian/license-plate-dataset/dataset/train/annots'
    xml_paths = [os.path.join(xml_dir, x) for x in sorted(os.listdir(xml_dir))]
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for item in root.findall('./object/bndbox'):
            x1, y1 = float(item.find('xmin').text), float(item.find('ymin').text)
            x2, y2 = float(item.find('xmax').text), float(item.find('ymax').text)
            w, h = x2 - x1, y2 - y1
            bin_idx = 0
            for i in range(len(PLATE_HEIGHT_THRESHOLDS)):
                if h >= PLATE_HEIGHT_THRESHOLDS[i]:
                    bin_idx = i + 1
            plate_bins[bin_idx].append((w, h))

    return plate_bins


def parse_ufpr():
    plate_bins = [[] for _ in range(len(PLATE_HEIGHT_THRESHOLDS) + 1)]

    data_dir = '/home/giangnd/Documents/data/UFPR-ALPR dataset/training'
    track_dirs = [os.path.join(data_dir, track_name) for track_name in sorted(os.listdir(data_dir))]
    for track_dir in tqdm(track_dirs):
        for label_name in os.listdir(track_dir):
            if not label_name.endswith('.txt'):
                continue

            pattern = r'^position_plate: (\d+) (\d+) (\d+) (\d+)$'
            with open(os.path.join(track_dir, label_name), 'r') as f:
                line = f.readline().strip()
                while line:
                    if re.match(pattern, line):
                        x1, y1 = int(re.sub(pattern, r'\1', line)), int(re.sub(pattern, r'\2', line))
                        w, h = int(re.sub(pattern, r'\3', line)), int(re.sub(pattern, r'\4', line))
                        bin_idx = 0
                        for i in range(len(PLATE_HEIGHT_THRESHOLDS)):
                            if h >= PLATE_HEIGHT_THRESHOLDS[i]:
                                bin_idx = i + 1
                        plate_bins[bin_idx].append((w, h))

                    line = f.readline().strip()

    return plate_bins


def parse_ssig_segplate():
    plate_bins = [[] for _ in range(len(PLATE_HEIGHT_THRESHOLDS) + 1)]

    data_dir = '/home/giangnd/Documents/data/SSIG-SegPlate/training'
    track_dirs = [os.path.join(data_dir, track_name) for track_name in sorted(os.listdir(data_dir))]
    for track_dir in tqdm(track_dirs):
        for label_name in os.listdir(track_dir):
            if not label_name.endswith('.txt'):
                continue

            pattern = r'^position_plate: (\d+) (\d+) (\d+) (\d+)$'
            with open(os.path.join(track_dir, label_name), 'r') as f:
                line = f.readline().strip()
                while line:
                    if re.match(pattern, line):
                        x1, y1 = int(re.sub(pattern, r'\1', line)), int(re.sub(pattern, r'\2', line))
                        w, h = int(re.sub(pattern, r'\3', line)), int(re.sub(pattern, r'\4', line))
                        bin_idx = 0
                        for i in range(len(PLATE_HEIGHT_THRESHOLDS)):
                            if h >= PLATE_HEIGHT_THRESHOLDS[i]:
                                bin_idx = i + 1
                        plate_bins[bin_idx].append((w, h))

                    line = f.readline().strip()

    return plate_bins


def parse_wdf():
    face_bins = [[] for _ in range(len(FACE_WIDTH_THRESHOLDS) + 1)]

    with open('/home/giangnd/Documents/data/WIDERFACE/wider_face_split/wider_face_train_bbx_gt.txt', 'r') as f:
        line = f.readline().strip()
        while line:
            if re.match(r'^\d+--\w+', line):
                count = int(f.readline().strip())
                if count == 0:
                    f.readline()

            else:
                w, h = int(line.split()[2]), int(line.split()[3])
                bin_idx = 0
                for i in range(len(FACE_WIDTH_THRESHOLDS)):
                    if w >= FACE_WIDTH_THRESHOLDS[i]:
                        bin_idx = i + 1
                face_bins[bin_idx].append((w, h))

            line = f.readline().strip()

    return face_bins


def parse_vantix_face():
    face_bins = [[] for _ in range(len(FACE_WIDTH_THRESHOLDS) + 1)]

    img_root = '/home/giangnd/Documents/data/vantix/testset'
    gt_root = '/home/giangnd/Documents/data/vantix/fixed_label'

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
                x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
                bin_idx = 0
                for i in range(len(FACE_WIDTH_THRESHOLDS)):
                    if x2 - x1 >= FACE_WIDTH_THRESHOLDS[i]:
                        bin_idx = i + 1
                face_bins[bin_idx].append((dirname, img_name, x1, y1, x2, y2))

    return face_bins


def parse_vantix_plate():
    plate_bins = [[] for _ in range(len(PLATE_HEIGHT_THRESHOLDS) + 1)]

    img_root = '/home/giangnd/Documents/data/vantix/testset'
    gt_root = '/home/giangnd/Documents/data/vantix/fixed_label'

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
            for x1, y1, x2, y2 in gt[1]:
                x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
                bin_idx = 0
                for i in range(len(PLATE_HEIGHT_THRESHOLDS)):
                    if y2 - y1 >= PLATE_HEIGHT_THRESHOLDS[i]:
                        bin_idx = i + 1
                plate_bins[bin_idx].append((dirname, img_name, x1, y1, x2, y2))

    return plate_bins


def parse_woodscape_face():
    face_bins = [[] for _ in range(len(FACE_WIDTH_THRESHOLDS) + 1)]

    img_dir = '/home/giangnd/Documents/data/woodscape/images'
    gt_dir = '/home/giangnd/Documents/data/woodscape/labels'

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
            x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
            bin_idx = 0
            for i in range(len(FACE_WIDTH_THRESHOLDS)):
                if x2 - x1 >= FACE_WIDTH_THRESHOLDS[i]:
                    bin_idx = i + 1
            face_bins[bin_idx].append((img_name, x1, y1, x2, y2))

    return face_bins


def parse_woodscape_plate():
    plate_bins = [[] for _ in range(len(PLATE_HEIGHT_THRESHOLDS) + 1)]

    img_dir = '/home/giangnd/Documents/data/woodscape/images'
    gt_dir = '/home/giangnd/Documents/data/woodscape/labels'

    img_names = sorted(os.listdir(img_dir))
    img_names = list(filter(lambda n: n.endswith('.jpg') or n.endswith('.png'), img_names))

    img_names = sorted(os.listdir(img_dir))
    img_names = list(filter(lambda n: n.endswith('.jpg') or n.endswith('.png'), img_names))

    for img_name in tqdm(img_names, desc=img_dir):
        label_name = img_name[:-4] + '.txt'
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, label_name)
        gt = parse_annotation(gt_path)

        img = Image.open(img_path)
        img_h, img_w = img.height, img.width
        for x1, y1, x2, y2 in gt[1]:
            x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
            bin_idx = 0
            for i in range(len(PLATE_HEIGHT_THRESHOLDS)):
                if y2 - y1 >= PLATE_HEIGHT_THRESHOLDS[i]:
                    bin_idx = i + 1
            plate_bins[bin_idx].append((img_name, x1, y1, x2, y2))

    return plate_bins


def parse_ufdd():
    face_bins = [[] for _ in range(len(FACE_WIDTH_THRESHOLDS) + 1)]

    with open('/home/giangnd/Documents/data/UFDD-V1-05042018/UFDD-annotationfile/UFDD_split/UFDD_val_bbx_gt.txt', 'r') as f:
        line = f.readline().strip()
        while line:
            if re.match(r'.+\.(jpg)|(png)', line):
                count = int(f.readline().strip())

            else:
                w, h = int(line.split()[2]), int(line.split()[3])
                bin_idx = 0
                for i in range(len(FACE_WIDTH_THRESHOLDS)):
                    if w >= FACE_WIDTH_THRESHOLDS[i]:
                        bin_idx = i + 1
                face_bins[bin_idx].append((w, h))

            line = f.readline().strip()

    return face_bins


def parse_afw():
    face_bins = [[] for _ in range(len(FACE_WIDTH_THRESHOLDS) + 1)]
    ds = hub.load("hub://activeloop/AFW")
    for face in tqdm(ds):
        keypoints = face.keypoints.numpy()
        keypoints = np.concatenate((keypoints[::3], keypoints[1::3]), axis=1)
        x1, y1 = np.min(keypoints[:, 0]), np.min(keypoints[:, 1])
        x2, y2 = np.max(keypoints[:, 0]), np.max(keypoints[:, 1])
        w, h = x2 - x1, y2 - y1
        bin_idx = 0
        for i in range(len(FACE_WIDTH_THRESHOLDS)):
            if w >= FACE_WIDTH_THRESHOLDS[i]:
                bin_idx = i + 1
        face_bins[bin_idx].append((w, h))

    return face_bins


def plot_face(face_bins, save_path=os.path.join('statistics', 'statistics_face.png')):
    face_bin_counts = list(map(len, face_bins))
    face_bin_ratios = [x / sum(face_bin_counts) * 100 for x in face_bin_counts]
    labels = [f'0-{FACE_WIDTH_THRESHOLDS[0] - 1}'] + \
            [f'{FACE_WIDTH_THRESHOLDS[i]}-{FACE_WIDTH_THRESHOLDS[i + 1] - 1}' for i in range(len(FACE_WIDTH_THRESHOLDS) - 1)] + \
            [f'{FACE_WIDTH_THRESHOLDS[-1]}+']

    # creating the bar plot
    plt.figure(figsize=(4, 4))
    plt.bar(labels, face_bin_ratios, width=1)
    plt.xlim(left=-0.5, right=len(PLATE_HEIGHT_THRESHOLDS) + 0.5)
    plt.ylim(bottom=0, top=45)
    plt.xlabel('Face width (pixels)', weight='bold')
    plt.ylabel('Sample percentage (%)', weight='bold')
    plt.xticks(rotation=270)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)


def plot_plate(plate_bins, save_path=os.path.join('figures', 'statistics_plate.png')):
    plate_bin_counts = list(map(len, plate_bins))
    plate_bin_ratios = [x / sum(plate_bin_counts) * 100 for x in plate_bin_counts]
    labels = [f'0-{PLATE_HEIGHT_THRESHOLDS[0] - 1}'] + \
            [f'{PLATE_HEIGHT_THRESHOLDS[i]}-{PLATE_HEIGHT_THRESHOLDS[i + 1] - 1}' for i in range(len(PLATE_HEIGHT_THRESHOLDS) - 1)] + \
            [f'{PLATE_HEIGHT_THRESHOLDS[-1]}+']

    # creating the bar plot
    plt.figure(figsize=(4, 4))
    plt.bar(labels, plate_bin_ratios, width=1)
    plt.xlim(left=-0.5, right=len(PLATE_HEIGHT_THRESHOLDS) + 0.5)
    plt.ylim(bottom=0, top=55)
    plt.xlabel('Plate height (pixels)', weight='bold')
    plt.ylabel('Sample percentage (%)', weight='bold')
    plt.xticks(rotation=270)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--name', required=True, type=str,
                        choices=['ccpd', 'fddb', 'lucian', 'ufpr', 'wdf', 'vantix_face', 'vantix_plate',
                                 'woodscape_face', 'woodscape_plate', 'ufdd', 'afw', 'ssig_segplate'],
                        help='Statistic name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    this_module = sys.modules[__name__]
    func = getattr(this_module, f'parse_{args.name}')
    bins = func()
    
    if args.name in ['fddb', 'wdf', 'vantix_face', 'woodscape_face', 'ufdd', 'afw']:
        plot_face(bins, save_path=os.path.join('/home/giangnd/Documents/figures', f'statistics_{args.name}.png'))
    else:
        plot_plate(bins, save_path=os.path.join('/home/giangnd/Documents/figures', f'statistics_{args.name}.png'))
