import re
from collections import defaultdict
from typing import List

import cv2
import matplotlib.pyplot as plt


def hex_to_rgb(hex):
    return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))


color_cycle = [hex_to_rgb(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]


def parse_annotation(annotation_path: str) -> List[float]:
    annotation = defaultdict(list)
    with open(annotation_path, 'r') as f:
        line = f.readline().strip()
        while line:
            assert re.match(r'^\d( [\d\.]+){4,5}$', line), 'Incorrect line: %s' % line
            cls, cx, cy, w, h = line.split()[:5]
            cls, cx, cy, w, h = int(cls), float(cx), float(cy), float(w), float(h)
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            annotation[cls].append([x1, y1, x2, y2])
            line = f.readline().strip()
    return annotation


def draw_annotation(img_path: str, annotation_path: str) -> None:
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    annotation = parse_annotation(annotation_path)
    for cls, bboxes in annotation.items():
        for x1, y1, x2, y2 in bboxes:
            x1, y1, x2, y2 = round(x1 * img_w), round(y1 * img_h), round(x2 * img_w), round(y2 * img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color_cycle[cls], thickness=2)
    return img
