import os
from re import L
import cv2
import json
import numpy as np
from tqdm import tqdm
import torch
from torchvision.ops import nms
from pathlib import Path

COLOR_PLATE_TRUE = (0,255,0) 
COLOR_PLATE_PRED = (0,0,255)
COLOR_FACE_TRUE = (255,0,0)
COLOR_FACE_PRED = (0, 165, 255)
IMG_EXT = ['jpg', 'jpeg', 'png']

class CocoDataset():
    def __init__(self, true_annotation_path, pred_annotation_path, save_path, src_img_root, dest_img_root):
        self.src_img_root = src_img_root if src_img_root[-1] != '/' else src_img_root[:-1]
        self.dest_img_root = dest_img_root if dest_img_root[-1] != '/' else dest_img_root[:-1]
        self.save_img_root = save_path

        self.true_annotation_path = true_annotation_path
        self.pred_annotation_path = pred_annotation_path
        
        self.root_src = "/data/"

        with open(true_annotation_path, 'r') as f:
            self.coco = json.load(f)
        
        self.process_info()
        self.process_licenses()
        self.process_categories() #Create a dict of {labels encode: dict of labels}

        self.process_images() #Create a dict of {image_id: images}
        self.true_annotations = self.process_annotations(self.coco['annotations']) #Create a dict of {image_id: list of annotations}

        if pred_annotation_path not in [None, '', true_annotation_path]:
            with open(pred_annotation_path, 'r') as f:
                self.coco_pred = json.load(f)
            self.pred_annotations = self.process_annotations(self.coco_pred)
        else: 
            self.pred_annotations = None

    def display_info(self):
        print('Dataset Info:')
        print('=============')
        for key, item in self.info.items():
            print('  {}: {}'.format(key, item))
        
        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', int],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print('ERROR: {} is missing'.format(req))
            elif type(self.info[req]) != req_type:
                print('ERROR: {} should be type {}'.format(req, str(req_type)))
        print('')
        
    def display_licenses(self):
        print('Licenses:')
        print('=========')
        
        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
        for license in self.licenses:
            for key, item in license.items():
                print('  {}: {}'.format(key, item))
            for req, req_type in requirements:
                if req not in license:
                    print('ERROR: {} is missing'.format(req))
                elif type(license[req]) != req_type:
                    print('ERROR: {} should be type {}'.format(req, str(req_type)))
            print('')
        print('')
        
    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print('    id {}: {}'.format(cat_id, self.categories[cat_id]['name']))
            print('')

    def display_dataset_stats(self, subdataset = False):
        print('Statistics:')
        print('=========')
        self.gen_stats_dataset(subdataset)
        print('=========')

    @staticmethod
    def draw_bbox(img, list_of_bboxes, labels, scores, conf_threshold, size_threshold, label_viz):
#         print('draw_box func')
        for index, bbox in enumerate(list_of_bboxes):
            x_min, y_min = int(bbox[0]), int(bbox[1])
            x_max, y_max = int(x_min + bbox[2]), int(y_min + bbox[3])
            conf = scores[index]
            label = labels[index]
#             print(bbox)
#             print(label, type(label))

            size = (x_max - x_min) * (y_max - y_min)
            # hot fix to visualize face 256, plate 225 
#             if conf < conf_threshold or (size < 256 and label == 0) or (size < 225 and label == 1):
            if conf < conf_threshold or size < size_threshold:
                continue

            if label in [1] : #plate
                color = COLOR_PLATE_TRUE if label_viz else COLOR_PLATE_PRED
            elif label in [0]: #face
                color = COLOR_FACE_TRUE if label_viz else COLOR_FACE_PRED
            else:
                raise ValueError(f'not class: {label}')
        
            if not label_viz:
                
                x_min, y_min, x_max, y_max = x_min + 1, y_min + 1, x_max - 1, y_max - 1
                # blur_bbox(img, (x_min, y_min), (x_max, y_max))
                conf = round(conf* 100,1)
                text = str(conf) + '%'
                txt_color = (255,255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX

                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

                txt_bk_color = color
                cv2.rectangle(
                    img,
                    (x_max - txt_size[0] + 1, y_max + 1),
                    (x_max  + 1, y_max + int(1.5*txt_size[1])),
                    txt_bk_color,
                    -1
                )
                cv2.putText(img, text, (x_max - txt_size[0] + 1, y_max + txt_size[1]), font, 0.4, txt_color, thickness=1)

            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        return img

    def visualize_image(self, image_id, annotation_type = 'bbox', conf_threshold = 0.05, size_threshold = 12**2):
        image_id = int(image_id)
        img_path = os.path.join(self.dest_img_root, self.images[image_id]['file_name'])
        #Read img path in destination 
        img = cv2.imread(img_path)

        
        true_bboxes = [x[annotation_type] for x in self.true_annotations[image_id]]
        true_labels = [x['category_id'] for x in self.true_annotations[image_id]]
        true_scores = [x['score'] for x in self.true_annotations[image_id]]

        if self.pred_annotations != None:
            pred_scores = [x['score'] for x in self.pred_annotations[image_id]]
            pred_bboxes = [x[annotation_type] for x in self.pred_annotations[image_id]]
            pred_labels = [x['category_id'] for x in self.pred_annotations[image_id]]

        # print(image_id, img_path, 'pred', len(pred_scores), 'true', len(true_scores))

        if annotation_type == 'bbox':
            #[top left x position, top left y position, width, height]
            img = CocoDataset.draw_bbox(img, true_bboxes, true_labels, true_scores, conf_threshold = conf_threshold, size_threshold = size_threshold,label_viz = True)
            
            if self.pred_annotations != None:
                img = CocoDataset.draw_bbox(img, pred_bboxes, pred_labels, pred_scores, conf_threshold = conf_threshold, size_threshold = size_threshold, label_viz = False)
            
            
            image_name = Path(self.images[image_id]['file_name'])
            relative_path = image_name.relative_to(self.root_src)            
            
            save_path = Path(self.save_img_root) / relative_path
#             save_path = os.path.join(self.save_img_root, self.images[image_id]['file_name'])    

            if save_path == self.images[image_id]['file_name']:
                raise ValueError('OVERWRITING THE SOURCE IMAGE WITH VISUALIZATION. TO DO THIS, DELTE THIS PART OF THE CODE')

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path,img)

        elif annotation_type == 'segmentation':
            print('---Unsupported segmentation---') #TODO
        else:
            raise ValueError(f"{annotation_type} is not a type of annotation")

    def visualize_images(self, annotation_type = 'bbox', conf_threshold = 0.1, size_threshold = 144):
        for image_id in tqdm(self.image_ids):
            self.visualize_image(image_id = image_id, annotation_type = annotation_type, conf_threshold = conf_threshold, size_threshold=size_threshold)
      
    def process_info(self):
        self.info = self.coco['info']
    
    def process_licenses(self):
        self.licenses = self.coco['licenses']
    
    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            
            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print("ERROR: Skipping duplicate category id: {}".format(category))

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id} # Create a new set with the category id
            else:
                self.super_categories[super_category] |= {cat_id} # Add category id to the set
                
    def process_images(self):
        self.images = {}
        for image in self.coco['images']:
            image_id = image['id']
            if image_id in self.images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                self.images[image_id] = image
                src_file_path = image['file_name']
                self.images[image_id]['file_name'] = src_file_path.split(f'{self.src_img_root}/')[-1]

        self.image_ids = sorted([x for x in self.images.keys()])

        self.sub_datasets = []
        img_paths = [self.images[image_id]['file_name'] for image_id in self.image_ids]
        for img_path in img_paths:
            if "OpenDataset" in img_path:
                self.sub_datasets.append(img_path.split('OpenDataset/')[-1].split('/')[0])
            else:
                self.sub_datasets.append(img_path.split('GDPR_dataset_split/')[-1].split('/')[0])
        self.sub_datasets = sorted(list(set(self.sub_datasets)))

    def process_annotations(self, annotations_from_json):
        annotations = {}
        for annotation in annotations_from_json:
            image_id = annotation['image_id'] 

            if image_id not in annotations:
                annotations[image_id] = []

            if 'score' not in annotation.keys():
                annotation['score'] = 1.0

            annotations[image_id].append(annotation)

        missing_annotation_key = [x for x in self.image_ids if x not in annotations.keys()]
        for key in missing_annotation_key:
            annotations[key] = []

        return annotations
    
    def gen_nms_image(self, annotation: list, nms_threshold: float, conf_threshold:float, size_threshold: float,  pred: bool,):
        labels = [x['category_id'] for x in annotation]
        unique_categories = np.unique(labels)

        new_ann_list = []

        for label in unique_categories:
            if 'id' not in annotation[0].keys() and pred == False:
                continue
        
            bboxes = []
            boxes_ids_within_img = []
            scores = []
            for index, ann in enumerate(annotation):

                if ann['category_id'] == label:
                    if (ann['score'] > conf_threshold) and (ann['bbox'][2] * ann['bbox'][3] > size_threshold): 
                        bboxes.append(ann['bbox'])
                        boxes_ids_within_img.append(index)
                        scores.append(ann['score'])
            
            if bboxes == []:
                continue
            
            bboxes = [torch.tensor([x[0], x[1], x[0]+ x[2], x[1]+x[3]]) for x in bboxes]
            scores = torch.tensor(scores).float()
            
            bboxes = torch.vstack(bboxes).float()
        
            keeps = nms(boxes = bboxes, scores = scores, iou_threshold=nms_threshold)            

            acceptable_ids = [boxes_ids_within_img[keep_id] for keep_id in keeps]
            new_ann_list.extend([annotation[id] for id in acceptable_ids])
        
        return new_ann_list

    def gen_nms_images(self, conf_threshold = 0.25, nms_threshold = 0.3, size_threshold = 12**2, pred = False, save = True, remove_empty_img = False):
        annotations = self.pred_annotations if pred else self.true_annotations

        for image_id in tqdm(self.image_ids):
            if image_id in annotations.keys():    
                image_ann_list = self.gen_nms_image(annotations[image_id], nms_threshold= nms_threshold,conf_threshold= conf_threshold ,size_threshold= size_threshold,pred= pred)
                annotations[image_id] = image_ann_list
            else:
                annotations[image_id] = []
            
        if pred:
            self.pred_annotations = annotations
        else:
            self.true_annotations = annotations

        if save: 
            annotation_path = self.pred_annotation_path if pred else self.true_annotation_path
            annotation_save_path = os.path.splitext(annotation_path)[0] + '_nms.json'
            self.save_annotation(annotation_save_path, pred, remove_empty_img)
    
    def save_annotation(self, annotation_save_path, pred, remove_empty_img):
        annotations = self.pred_annotations if pred else self.true_annotations
        
        ann_list = []
        img_list = []
        for image_id in self.image_ids:
            ann_list.extend(annotations[image_id])

            if remove_empty_img and len(annotations[image_id]) == 0:
                continue
            
            img_list.append(self.images[image_id])

        with open(annotation_save_path,'w') as fw:
            if pred:
                json.dump(ann_list, fw, indent=4)
            else:
                self.coco["annotations"] = ann_list  
                self.coco["images"] = img_list
                json.dump(self.coco, fw, indent=4)
                
        print(f'New file saved at {annotation_save_path}')
    
    def save_img_list(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        image_paths = [self.images[id]['file_name'] for id in self.image_ids]
        print(len(image_paths), len(self.image_ids))
        with open(save_path, 'w') as f:
            f.write('\n'.join(image_paths))

    @staticmethod
    def calculate_ap_ar(true_annotation, pred_annotation, overlapping_tp_threshold):
        tp = 0
        fn = 0
        for true_index, true_box in enumerate(true_annotation):
            true_xmin, true_ymin = max(true_box['bbox'][0], 0), max(true_box['bbox'][1], 0)
            true_xmax, true_ymax = max(true_box['bbox'][0] + true_box['bbox'][2], 0), max(true_box['bbox'][1] + true_box['bbox'][3], 0)
            true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)

            for pred_index, pred_box in enumerate(pred_annotation):
                pred_xmin, pred_ymin = max(pred_box['bbox'][0], 0), max(pred_box['bbox'][1], 0)
                pred_xmax, pred_ymax = max(pred_box['bbox'][0] + pred_box['bbox'][2], 0), max(pred_box['bbox'][1] + pred_box['bbox'][3], 0)
                pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)

                intersection_xmin, intersection_ymin = max(true_xmin, pred_xmin), max(true_ymin, pred_ymin)
                intersection_xmax, intersection_ymax = min(true_xmax, pred_xmax), min(true_ymax, pred_ymax)

                intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
                union_area = pred_area + true_area - intersection_area

                iou = intersection_area / union_area
                if iou > overlapping_tp_threshold:
                    # true box did find match, model predicts this box correctly
                    tp += 1
                    pred_annotation.pop(pred_index) #Remove the matched prediction
                    break
            else:
                # true box couldn't find match, model misses this box
                fn += 1
        
        fp = len(pred_annotation) # the rest of the prediction couldn't match to any ground truth box, thus are false positives

        return tp, fp, fn

    def gen_box_stats_images(self, true_conf_threshold, pred_conf_threshold, category, size_threshold = 12**2):  
        '''
        Outputing the number of true and pred bboxes
        '''      
        num_box_pred = [len([x for x in self.pred_annotations[img_id] if x['score'] > pred_conf_threshold and x['category_id'] == category and x['bbox'][2] * x['bbox'][3] > size_threshold]) for img_id in self.image_ids]
        num_box_true = [len([x for x in self.true_annotations[img_id] if x['score'] > true_conf_threshold and x['category_id'] == category and x['bbox'][2] * x['bbox'][3] > size_threshold]) for img_id in self.image_ids]
        return num_box_true, num_box_pred

    def gen_ap_stats_images(self, true_conf_threshold, pred_conf_threshold,  size_threshold, overlapping_tp_threshold = 0.3):
        '''
        Outputing the ap, ar and a result statistic dictionary calculated from each image of the annotations
        '''
        results = {}
        ap = []
        ar = []

        for category in self.categories:
            results[f'{category}_ap'] = []
            results[f'{category}_ar'] = []
        
        print('Generate ap stats')
        
        for img_id in tqdm(self.image_ids):
            #Filtering by size and conf threshold
            true_img_annotation = [ann for ann in self.true_annotations[img_id] if ann['score'] > true_conf_threshold and ann['bbox'][2] * ann['bbox'][3] > size_threshold]    
            pred_img_annotation = [ann for ann in self.pred_annotations[img_id] if ann['score'] > pred_conf_threshold and ann['bbox'][2] * ann['bbox'][3] > size_threshold]

            if len(pred_img_annotation) == 0:
                for category in self.categories:
                    results[f'{category}_ap'].append(float('nan'))
                    results[f'{category}_ar'].append(float('nan'))
                ap.append(float('nan'))
                ar.append(float('nan'))
                continue
            
            all_tp = 0
            all_fn = 0
            all_fp = 0
            for category in self.categories:
                true_annotation = [ann for ann in true_img_annotation if ann['category_id']==category]
                pred_annotation = [ann for ann in pred_img_annotation if ann['category_id']==category]
                
                if (type(overlapping_tp_threshold) == float or len(overlapping_tp_threshold) == 1):
                    # Input is a single float
                    overlapping_tp_threshold = overlapping_tp_threshold[0] if type(overlapping_tp_threshold) == list else overlapping_tp_threshold
                    tp, fp, fn = CocoDataset.calculate_ap_ar(true_annotation=true_annotation, pred_annotation=pred_annotation, overlapping_tp_threshold=overlapping_tp_threshold)
                
                elif(len(overlapping_tp_threshold) > 1):
                    # Input is a list of floats
                    tp, fp, fn = 0, 0, 0
                    for thre in overlapping_tp_threshold:
                        cur_tp, cur_fp, cur_fn = CocoDataset.calculate_ap_ar(true_annotation=true_annotation, pred_annotation=pred_annotation, overlapping_tp_threshold=thre)
                        tp += cur_tp
                        fp += cur_fp
                        fn += cur_fn
                    
                all_tp += tp
                all_fn += fn
                all_fp += fp

                precision = float('nan') if tp+fp == 0 else tp/(tp+fp)
                recall    = float('nan') if tp+fn == 0 else tp/(tp+fn)
                results[f'{category}_ap'].append(precision)
                results[f'{category}_ar'].append(recall)
            
            all_precision = float('nan') if all_tp + all_fp == 0 else all_tp/(all_tp + all_fp)
            all_recall = float('nan') if all_tp + all_fn == 0 else all_tp/(all_tp + all_fn)
            ap.append(all_precision)
            ar.append(all_recall)
        
        return ap, ar, results

    def gen_stats_images(self, true_conf_threshold = 0.3, pred_conf_threshold = 0.05, size_threshold = 12**2, overlapping_tp_threshold = 0.3):        
        image_paths = [self.images[id]['file_name'] for id in self.image_ids]
        categories = [x for x in self.categories.keys()]

        box_stats = {}        
        box_stats['ap'], box_stats['ar'], ap_results = self.gen_ap_stats_images(true_conf_threshold= true_conf_threshold,pred_conf_threshold= pred_conf_threshold, size_threshold= size_threshold,overlapping_tp_threshold= overlapping_tp_threshold)

        for category in categories:
            box_stats[f'{category}_box_true'], box_stats[f'{category}_box_pred'] = self.gen_box_stats_images(true_conf_threshold, pred_conf_threshold,  category, size_threshold)
            box_stats[f'{category}_diff'] = np.array(box_stats[f'{category}_box_true']) - np.array(box_stats[f'{category}_box_pred'])

            box_stats[f'{category}_ar'] = ap_results[f'{category}_ar']
            box_stats[f'{category}_ap'] = ap_results[f'{category}_ap']

        import pandas as pd
        df = pd.DataFrame()
        df['image'] = [os.path.join(self.src_img_root , x) for x in image_paths]
        df['id'] = list(np.array(self.image_ids))
        true = np.array([0 for x in range(len(df['image']))])
        pred = np.array([0 for x in range(len(df['image']))])

        for key in box_stats.keys():
            df[key]  = list(np.array(box_stats[key]))
            if 'true' in key:
                true += np.array(box_stats[key])
            elif 'pred' in key:
                pred += np.array(box_stats[key])

        df['total_true'] = list(true)
        df['total_pred'] = list(pred)
        df['total_diff'] = list(true - pred)

        save_path = os.path.join(self.save_img_root, 'box_stats.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(save_path)
        df.to_csv(save_path, index=False)

    def gen_stats_dataset(self, subdataset = False):
        if subdataset:
            os.system('echo ++++ Sub dataset')
            image_paths = [self.images[image_id]['file_name'] for image_id in self.image_ids]
            for sub_dataset in self.sub_datasets:
                os.system(f'echo {sub_dataset} imgs: {len([x for x in image_paths if sub_dataset in x])}')

        os.system(f'echo ++++ Ground Truth stats:')
        os.system(f'echo - num_images: {len(self.coco["images"])}')
        os.system(f'echo - num_bboxes: {len(self.coco["annotations"])}')
        for category in self.categories.keys():
            os.system(f'echo -- num_bboxes {self.categories[category]["name"]}: {len([x for x in self.coco["annotations"] if x["category_id"] ==  category])}')

        if (self.pred_annotations != None):
            os.system(f'echo ++++ Prediction stats:')
            os.system(f'echo - num_images: {self.coco_pred[-1]["image_id"]}')
            os.system(f'echo - num_bboxes: {len(self.coco_pred)}')
            for category in self.categories.keys():
                os.system(f'echo -- num_bboxes {self.categories[category]["name"]}: {len([x for x in self.coco_pred if x["category_id"] ==  category])}')

    def dataset_filter(
        self,
        chosen_img_list: list,
        coco_json_save_path: str,
        pred: False,
        ):
        '''
        Input:
        chosen_img_list: list of image paths to choose to extract out of the coco json label file
        coco_json_save_path: output path to coco json file saving annotation and/or image paths, chosen in chosen_img_list
        pred: if pred = true, coco output file will be in coco_pred format, else it will be coco_true
        Output:
        New coco json file with only the chosen images' annotations
        '''
        #Get image ids
        target_image_ids = [int(x.split(' ')[-1]) for x in chosen_img_list]
        
        #Get annotation
        src_annotations = self.true_annotations if pred != True else self.pred_annotations
        target_annotations = []
        removed_img_id = []
        for image_id in target_image_ids:
            try:
                target_annotations.extend(src_annotations[image_id])
            except KeyError: #Quick fix missed image_id due to eliminated imgs with no annotation
                removed_img_id.append(image_id)
                continue
        print('Eliminated id', len(removed_img_id))

        os.makedirs(os.path.dirname(coco_json_save_path), exist_ok=True)
        with open(coco_json_save_path,'w') as fw:
            if pred:
                json.dump(target_annotations, fw, indent=4)
            else:
                new_coco = dict(self.coco)
                new_coco['annotations'] = target_annotations

                #Get images
                target_images = []
                target_image_ids = [x for x in target_image_ids if x not in removed_img_id]
  
                for image_id in target_image_ids:                    
                    target_images.append(self.images[image_id])

                new_coco['images'] = target_images

                json.dump(new_coco, fw, indent=4)

        print(f'New file saved at {coco_json_save_path}')

    def gen_sub_dataset(self, pred = True, dataset_names = None):
        file_paths = [self.images[id]['file_name'] for id in self.image_ids]

        if dataset_names in ['', None, []]:
            dataset_names = self.sub_datasets
        
        save_paths = []
        for sub_name in dataset_names:
            new_imgs = []
            new_anns = []

            for index, image_name in enumerate(file_paths):                
                if sub_name in image_name:
                    cur_id = self.image_ids[index]                    
                    if pred:
                        new_anns.extend(self.pred_annotations[cur_id])
                    else:
                        new_imgs.append(self.images[cur_id])
                        new_anns.extend(self.true_annotations[cur_id])

            if len(new_anns) == 0:
                print(f'{sub_name} is not in the mother dataset')
                continue

            annotation_path = self.pred_annotation_path if pred else self.true_annotation_path
            save_path = os.path.splitext(annotation_path)[0] + f'_{sub_name}.json'
            
            with open(save_path, 'w') as fw:
                if pred:
                    json.dump(new_anns, fw, indent=4)
                else:
                    new_coco =  dict(self.coco)
                    new_coco["annotations"] = new_anns
                    new_coco["images"] = new_imgs
                    json.dump(new_coco, fw, indent=4)
                    print(f'New file saved at {save_path}')
            
            save_paths.append(save_path)
        
        return save_paths

if __name__ == '__main__':
    import yaml

    with open("utils/configs/dataset_config.yaml", 'r') as stream:
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

        ## Generate nms file for true file
        # coco_dataset.gen_nms_images(pred = False)

        ## Generate nms file for pred file
        # coco_dataset.gen_nms_images(pred = True)

        ## Visualize both onto images
        # coco_dataset.visualize_images(annotation_type = 'bbox', conf_threshold = 0.05)
        
        ## Saving img list
        # coco_dataset.save_img_list('./sample/tmp/selected_0511.txt')

        ## Generate num_box statistics for images
        # coco_dataset.gen_stats_images()

    pass