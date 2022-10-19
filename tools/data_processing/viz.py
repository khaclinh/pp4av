import os
import cv2
import glob as glob
from tqdm import tqdm

from dataset import CocoDataset

COLOR_PLATE_TRUE = (0,255,0) 
COLOR_PLATE_PRED = (0,0,255)
COLOR_FACE_TRUE = (255,0,0)
COLOR_FACE_PRED = (0, 165, 255)
IMG_EXT = ['jpg', 'jpeg', 'png']

def draw_bbox_yolo1(img_path, txt_path, save_path, threshold = 0.1, label_viz = True):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    os.makedirs(save_path, exist_ok=True)

    with open(txt_path, 'r') as f:
        data = [x for x in f.read().strip().split('\n') if x != '']
    
    if len(data) == 0:
        return

    for line in data:
        bbox = [x for x in line.split(' ') if x != '']
        conf = float(bbox[-1]) if len(bbox) == 6 else 1

        if conf < threshold:
            print(conf, threshold)
            continue

        x_center, y_center = float(bbox[1]) * width, float(bbox[2]) * height
        box_width, box_height = float(bbox[3]) * width, float(bbox[4]) * height
        
        x_min = int(x_center - box_width/2)
        y_min = int(y_center - box_height/2)
        x_max = int(x_center + box_width/2)
        y_max = int(y_center + box_height/2)

        if bbox[0] in ['plate', '0'] : #plate
            color = COLOR_PLATE_TRUE if label_viz else COLOR_PLATE_PRED
        elif bbox[0] in ['face', '1']: #face
            color = COLOR_FACE_TRUE if label_viz else COLOR_FACE_PRED
        else:
            raise ValueError(f'not class: {bbox[0]}')
        
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        # if not label_viz:
            # blur_bbox(img, (x_min, y_min), (x_max, y_max))
    
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)),img)

def draw_bbox_yolov1_folder(img_root, true_txt_root, pred_txt_root, save_path, threshold=0.05, double_viz = True):
    pred_txt_root = pred_txt_root if double_viz else true_txt_root
    img_paths = sorted([x for x in glob.glob(f'{img_root}/**/*', recursive=True) if x.split('.')[-1].lower() in IMG_EXT])
    true_paths = sorted([x for x in glob.glob(f'{true_txt_root}/**/*.txt', recursive=True)])
    pred_paths = sorted([x for x in glob.glob(f'{pred_txt_root}/**/*.txt', recursive=True)])
    
    for index, img_path in enumerate(tqdm(img_paths)):
        img_name = os.path.splitext(img_path.split(f'{img_root}/')[-1])[0]
        true_path = f'{true_txt_root}/{img_name}.txt'
        pred_path = f'{pred_txt_root}/{img_name}.txt'
        
        if true_path not in true_paths or pred_path not in pred_paths:
            print(f'{img_name} txt file not found in directory')
            continue

        draw_bbox_yolo1(img_path, true_path, save_path, threshold = threshold)
        if double_viz:
            draw_bbox_yolo1(f'{save_path}/{img_path.split(f"{img_root}/")[-1]}', pred_path, save_path, threshold = threshold, label_viz=False)
    
def blur_bbox(img, top_left, bottom_right):
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    bbox = img[y_min:y_max, x_min:x_max]
    bbox =  cv2.GaussianBlur(bbox,(51,51),cv2.BORDER_DEFAULT)

    img[y_min:y_max, x_min:x_max] = bbox

    return img

if __name__ == '__main__':
    import yaml

    with open("utils/configs/viz_config.yaml", 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    for index in range(len(config_dict['true_annotation_path'])):
        coco_dataset = CocoDataset(
            true_annotation_path = config_dict['true_annotation_path'][index], 
            pred_annotation_path = config_dict['pred_annotation_path'][index], 
            save_path = config_dict['save_path'][index], 
            src_img_root = config_dict['src_img_root'], 
            dest_img_root = config_dict['dest_img_root'],
            )

        ## Visualize both onto images
        coco_dataset.visualize_images(annotation_type = 'bbox', conf_threshold = 0.05, size_threshold=144)

    