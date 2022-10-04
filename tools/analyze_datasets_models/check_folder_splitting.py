"""
Check the correctness of splitting annotation into folders of minimum sizes.

"""
import os

from tqdm import tqdm
from PIL import Image

from utils import parse_annotation


thresholds = [10, 20]
img_root = r'C:\Users\giangnd\Documents\bug\testset'
small_root = r'C:\Users\giangnd\Documents\bug\fixed\fixed_label'

for thresh in thresholds:
    large_root = os.path.join(r'C:\Users\giangnd\Documents\bug\fixed', str(thresh))

    subdirs = os.listdir(img_root)
    for dirname in subdirs:
        img_dir = os.path.join(img_root, dirname)
        small_dir = os.path.join(small_root, dirname)
        large_dir = os.path.join(large_root, dirname)

        img_names = sorted(os.listdir(img_dir))
        img_names = list(filter(lambda n: n.endswith('.jpg') or n.endswith('.png'), img_names))

        for img_name in tqdm(img_names, desc=img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path)

            label_name = img_name[:-4] + '.txt'
            small_path = os.path.join(small_dir, label_name)
            large_path = os.path.join(large_dir, label_name)

            small = parse_annotation(small_path)
            large = parse_annotation(large_path)

            for lx1, ly1, lx2, ly2 in large[0]:
                lw = lx2 - lx1
                assert (lx2 - lx1) * img.width >= thresh, ( f'Error face too small for threshold {thresh}\n'
                                                            f'label: 0 {lx1} {ly1} {lx2} {ly2}\n'
                                                            f'face width: {(lx2 - lx1) * img.width}\n'
                                                            f'image width: {img.width}, image height: {img.height}\n'
                                                            f'image path all size: {small_path}\n'
                                                            f'image path width >= {thresh}: {large_path}')

                match = False
                for sx1, sy1, sx2, sy2 in small[0]:
                    large_area = (lx2 - lx1) * (ly2 - ly1)
                    sw = sx2 - sx1

                    small_area = (sx2 - sx1) * (sy2 - sy1)
                    intersect = max(min(sx2, lx2) - max(sx1, lx1), 0) * max(min(sy2, ly2) - max(sy1, ly1), 0)
                    union = small_area + large_area - intersect
                    iou = intersect / union
                    if iou > 0.8:
                        match = True

                assert match, ( f'Error face in {thresh} folder but not found in original folder\n'
                                f'label: 0 {lx1} {ly1} {lx2} {ly2}\n'
                                f'face width: {(lx2 - lx1) * img.width}\n'
                                f'image width: {img.width}, image height: {img.height}\n'
                                f'image path all size: {small_path}\n'
                                f'image path width >= {thresh}: {large_path}')


            for sx1, sy1, sx2, sy2 in small[0]:
                sw = sx2 - sx1

                small_area = (sx2 - sx1) * (sy2 - sy1)
                if sw >= thresh:
                    match = False
                    for lx1, ly1, lx2, ly2 in large[0]:
                        large_area = (lx2 - lx1) * (ly2 - ly1)
                        intersect = max(min(sx2, lx2) - max(sx1, lx1), 0) * max(min(sy2, ly2) - max(sy1, ly1), 0)
                        union = small_area + large_area - intersect
                        iou = intersect / union
                        if iou > 0.8:
                            match = True

                    assert match, ( f'Error face and in original folder and width >= {thresh} but not found in {thresh} folder\n'
                                    f'label: 0 {sx1} {sy1} {sx2} {sy2}\n'
                                    f'face width: {(sx2 - sx1) * img.width}\n'
                                    f'image width: {img.width}, image height: {img.height}\n'
                                    f'image path all size: {small_path}\n'
                                    f'image path width >= {thresh}: {large_path}')
