import argparse
import os
import re

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Plot histogram of face width and height for custom dataset')
parser.add_argument('--image-root', '-i', required=True, type=str, help='Path to image root')
parser.add_argument('--label-root', '-l', required=True, type=str, help='Path to label root')
parser.add_argument('--output-root', '-o', required=True, type=str, help='Path to output root')
parser.add_argument('--class-id', '-c', type=int, default=0, help='Class id')
args = parser.parse_args()


def count_wh(image_dir, label_dir, class_id):
    face_heights, face_widths = [], []

    item_names = sorted(os.listdir(image_dir))
    for item_name in item_names:
        item_path = os.path.join(image_dir, item_name)

        if os.path.isdir(item_path):
            image_subdir, label_subdir = item_path, os.path.join(label_dir, item_name)
            wh_subdir = count_wh(image_subdir, label_subdir, class_id)
            face_widths.extend(wh_subdir[0])
            face_heights.extend(wh_subdir[1])

    img_names = list(filter(lambda x: x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg'), item_names))
    label_names = ['.'.join(img_name.split('.')[:-1] + ['txt']) for img_name in img_names]
    
    img_paths = [os.path.join(image_dir, img_name) for img_name in img_names]
    label_paths = [os.path.join(label_dir, label_name) for label_name in label_names]

    for img_path, label_path in tqdm(zip(img_paths, label_paths), desc=os.path.basename(image_dir)):
        img = Image.open(img_path)

        with open(label_path, 'r') as f:
            for line in f.readlines():
                label = line.strip().split()
                clazz = int(label[0])

                if clazz == class_id:
                    face_width_frac, face_height_frac = float(label[3]), float(label[4])

                    if face_width_frac < 0 or face_width_frac > 1 or face_height_frac < 0 or face_height_frac > 1:
                        print('Label error %s' % label_path)
                        continue

                    face_width, face_height = img.width * face_width_frac, img.height * face_height_frac
                    face_widths.append(face_width)
                    face_heights.append(face_height)

    return face_widths, face_heights

if __name__ == '__main__':
    STEP_SIZE = 5

    face_widths, face_heights = count_wh(args.image_root, args.label_root, args.class_id)
    os.makedirs(args.output_root, exist_ok=True)

    class_name = 'face' if args.class_id == 0 else 'lisence'

    plt.clf()
    plt.hist(face_widths, bins=range(0, int(max(face_widths + [50])) + STEP_SIZE, STEP_SIZE))
    plt.xlabel('%s width' % class_name)
    plt.ylabel('number of %ss' % class_name)
    plt.title('Distribution of %s widths' % class_name)
    plt.savefig(os.path.join(args.output_root, '%s_width_hist.png' % class_name))

    plt.clf()
    plt.hist(face_heights, bins=range(0, int(max(face_heights + [50])) + STEP_SIZE, STEP_SIZE))
    plt.xlabel('%s height' % class_name)
    plt.ylabel('number of %ss' % class_name)
    plt.title('Distribution of %s heights' % class_name)
    plt.savefig(os.path.join(args.output_root, '%s_height_hist.png' % class_name))