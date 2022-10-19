import os
import json
import cv2
import datetime
import glob as glob
from tqdm import tqdm
from pathlib import Path

IMG_EXT = ['png', 'jpeg', 'jpg']

def anonymizer_json_to_yolo1(json_folder_path, image_folder_path, save_folder, labels = ['face', 'plate']):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    json_files = sorted(glob.glob(f'{json_folder_path}/*json'))
    
    image_files = sorted([x for x in glob.glob(f'{image_folder_path}/*') if os.path.basename(x).split('.')[-1] in IMG_EXT])
    
    for index, json_file in enumerate(json_files):
        image_file = image_files[index]
        
        print(os.path.basename(json_file).split('.')[0], os.path.basename(image_file).split('.')[0])
        assert os.path.basename(json_file).split('.')[0] == os.path.basename(image_file).split('.')[0]

        height, width, _ = cv2.imread(image_file).shape

        with open(json_file, 'r') as f:
            data = json.load(f)
        lines = []
        for box in data:
            x_min = int(box['x_min'])
            y_min = int(box['y_min'])
            x_max = int(box['x_max'])
            y_max = int(box['y_max'])

            object_class = str(labels.index(box['kind']))
            conf_score = str(box['score'])
            
            x = str((x_min + (x_max - x_min)/2) / width)
            y = str((y_min + (y_max - y_min)/2) / height)

            bbox_width = str(abs(x_min - x_max) /  width)
            bbox_height = str(abs(y_min - y_max) /  height)

            line = ' '.join([object_class, x, y, bbox_width, bbox_height, conf_score])
            lines.append(line) 
        
        save_txt_path = f'{save_folder}/{os.path.basename(json_file).split(".json")[0]}.txt'
        with open(save_txt_path, 'w') as f:
            f.write('\n'.join(lines))


def yolov1_to_coco(img_folder_paths, txt_folder_paths, save_path, labels = ['face', 'plate'], is_eval = False):
    '''
    img_folder_paths: List of paths of directories containing images 
    txt_folder_paths: List of paths of directories containing txt files
    save_path: string of json file to save the labels in 
    labels: list of string, labels to index
    is_eval: bool, if it is True, generate prediction result-format json file for evaluation, instead of data-format json
    '''
    categories = []
    for j,label in enumerate(labels):
        label = label.strip()
        categories.append({'id':j,'name':label,'supercategory': label})
        
    write_json_context = dict()
    write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '', 'date_created': f'{datetime.datetime.now()}'}
    write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    write_json_context['categories'] = categories
    write_json_context['images'] = []
    write_json_context['annotations'] = []

    img_paths = []
    txt_paths = []
    len_list = []
    
#     img_folder_paths.sort()
#     txt_folder_paths.sort()
    
    for index, folder_path in enumerate(img_folder_paths):
        print(folder_path)
        img_list = []
        for ext in IMG_EXT:
            img_list.extend(sorted(list(Path(img_folder_paths[index]).glob(f'**/*.{ext}'))))
        img_list = [str(x) for x in img_list if '.ipynb_checkpoints' not in str(x)]
        img_paths.extend(img_list)

        len_list.append(len(img_list))
        print('Number images', len(img_list))

        all_txt_list = sorted(list(Path(txt_folder_paths[index]).glob(f'**/*.txt')))
#         txt_list = all_txt_list
        txt_list = []
        for img in img_list:
            img_name = os.path.splitext(os.path.basename(img))[0]

            try:
                txt_list.append([str(x) for x in all_txt_list if img_name in str(x)][0])
            except IndexError:
                print('Missing txt:', img)
            
        txt_paths.extend(txt_list)
        print('Number txts', len(txt_paths))
            
    print('Total images:', len(img_paths))    
    assert len(img_paths) == len(txt_paths)
    
    
    #Spliting train val
    #   img_paths = img_paths[:int(0.8*len(img_paths))]
    #   txt_paths = txt_paths[:int(0.8*len(txt_paths))]
#     print(len(img_paths), len(txt_paths))

    file_number = 0
    num_bboxes = 0
    # count num boxes for each class
    num_face_boxes = 0
    num_plate_boxes = 0
    list_class_len = []
    img_fol_id = 1

    for index in tqdm(range(len(img_paths))):
                        
        file_number += 1
        img_path = img_paths[index]
        yolo_annotation_path  = txt_paths[index]
        # img_name = os.path.basename(img_path) # name of the file without the extension
        img_context = {}
        height,width = cv2.imread(img_path).shape[:2]
        img_context['file_name'] = img_path
        img_context['height'] = height
        img_context['width'] = width
        img_context['date_captured'] = f'{datetime.datetime.now()}'
        img_context['id'] = file_number # image id
        img_context['license'] = 1
        img_context['coco_url'] =''
        img_context['flickr_url'] = ''

        write_json_context['images'].append(img_context)
            
        if file_number == sum(len_list[:img_fol_id]) + 1:
            list_class_len.append([num_face_boxes, num_plate_boxes])
            img_fol_id += 1
            num_face_boxes = 0
            num_plate_boxes = 0
        
        with open(yolo_annotation_path,'r') as f2:
            lines2 = f2.readlines() 

        for i,line in enumerate(lines2): # for loop runs for number of annotations labelled in an image
            '''
                {
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [
                        466.484375,
                        76.4969711303711,
                        224.9879150390625,
                        94.5906753540039
                    ],
                    "score": 0.9025385975837708,
                    "segmentation": []
                },
            '''
            line = line.strip().split(' ')
            bbox_dict = {}
            class_name, x_yolo,y_yolo,width_yolo,height_yolo= line[0:5]
            x_yolo,y_yolo,width_yolo,height_yolo = float(x_yolo),float(y_yolo),float(width_yolo),float(height_yolo)
                        
            num_bboxes += 1
            
            if is_eval:
                bbox_dict['score'] = float(line[-1])
            
            if class_name.isdigit():
                class_id = int(class_name)
            elif class_name.replace('.', '', 1).isdigit():
                class_id = int(float(class_name))
            else:
                class_id = labels.index(class_name)
            if class_id == 0:
                num_face_boxes += 1
            else:
                num_plate_boxes += 1                
                
            bbox_dict['id'] = num_bboxes
            bbox_dict['image_id'] = file_number
            bbox_dict['category_id'] = class_id
            bbox_dict['iscrowd'] = 0 # There is an explanation before
            h,w = abs(height_yolo*height),abs(width_yolo*width)            
            bbox_dict['area']  = h * w
            x_coco = round(x_yolo*width -(w/2))
            y_coco = round(y_yolo*height -(h/2))
            
            x_coco, y_coco, w, h = correct_coor_box(width, height, [x_coco, y_coco, w, h])
            
            bbox_dict['bbox'] = [x_coco,y_coco,w,h]
            bbox_dict['segmentation'] = [[x_coco,y_coco,x_coco+w,y_coco, x_coco+w, y_coco+h, x_coco, y_coco+h]]
            write_json_context['annotations'].append(bbox_dict)
            
    list_class_len.append([num_face_boxes, num_plate_boxes])    

    with open(save_path,'w') as fw:
        if is_eval:
            json.dump(write_json_context['annotations'],fw, indent = 4)
        else:
            json.dump(write_json_context,fw, indent = 4)

    print(list_class_len)
    
    with open(os.path.splitext(save_path)[0] + '.txt', 'w') as tw:
        for path, num_img, num_box in zip(sorted(img_folder_paths), len_list, list_class_len):
            print(path, num_img, num_box)
            tw.write(path + '\n')
            tw.write('Num images: ' + str(num_img) + '\n')
            tw.write('Num face box: ' + str(num_box[0]) + '\n')
            tw.write('Num plate box: ' + str(num_box[1]) + '\n')


def correct_coor_box(width_img, height_img, bbox):
    """
        Correct coordinate of bounding box if coordinate is out of boundary
            bbox format: xmin, ymin, width, height
    """
    new_x, new_y, new_w, new_h = bbox
    if bbox[0] < 0:
        new_w = bbox[0] + bbox[2]
        new_x = 0
    if bbox[1] < 0:
        new_h = bbox[1] + bbox[3]
        new_y = 0
    if bbox[0] + bbox[2] >= width_img:
        new_w = width_img - 1
    if bbox[1] + bbox[3] >= height_img:
        new_h = height_img - 1
    
    return new_x, new_y, new_w, new_h
    
    

if __name__ == '__main__':
    import yaml

    with open("configs.yaml", 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    yolov1_to_coco(
        txt_folder_paths=config_dict['txt_folder_paths'],
        img_folder_paths=config_dict['img_folder_paths'],
        save_path= config_dict['save_path'],
        is_eval = config_dict['eval'],
        labels = config_dict['labels']
                        ) 
