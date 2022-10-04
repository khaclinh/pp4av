import os
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


if __name__ == '__main__':
    data = np.zeros((4, 4), dtype=int)

    with open('dataset/WIDERFACE/wider_face_split/wider_face_train_bbx_gt.txt', 'r') as f:
        img = None
        line = f.readline().strip()
        while line:
            if re.match(r'^\d+--\w+', line):
                img_path = os.path.join('dataset/WIDERFACE/WIDER_train/images', line)
                img = Image.open(img_path)

                count = int(f.readline().strip())
                if count == 0:
                    f.readline()

            else:
                x1, y1, w, h = tuple(map(int, line.split()[:4]))
                cx, cy = x1 + (w - 1) / 2, y1 + (h - 1) / 2

                bin_x, bin_y = min(int(cx / img.width * 4), 3), min(int(cy / img.height * 4), 3)
                data[bin_y, bin_x] += 1

            line = f.readline().strip()
    
    columns = ['0% - 25%', '25% - 50%', '50% - 75%', '75% - 100%']
    rows = ['0% - 25%', '25% - 50%', '50% - 75%', '75% - 100%']
    ax = sns.heatmap(data, cmap='YlGnBu', annot=True, xticklabels=columns, yticklabels=rows)
    ax.set_xlabel('image width')
    ax.set_ylabel('image height')
    plt.title('Distribution of face position')
    plt.savefig('wdf/face_position.png')
