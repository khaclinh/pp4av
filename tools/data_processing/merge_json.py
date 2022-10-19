import json
import yaml
import datetime


def filter_merge_img_json(txt_paths: list, json_paths: list, save_path: str, full: bool=False):
    """
    Create new json file (coco_true format) with image_paths from correspondence txt file
    Input:
        txt_paths: list of txt path
        json_path: list of json path
        save path: path to save output json file
        full: if full is True, merge full json files, not get paths from txt_paths
    """
    
    print(f'Number txt paths: {len(txt_paths)}', '\n', f'Number json paths: {len(json_paths)}')
    assert len(txt_paths) == len(json_paths)
    
    write_json_context = dict()
    write_json_context['info'] = {
        'description': '', 'url': '', 'version': '', 'year': 2021,
        'contributor': '', 'date_created': f'{datetime.datetime.now()}'
    }
    write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    write_json_context['categories'] = [
        {'id': 0,'name': 'face','supercategory': 'face'},
        {'id': 1,'name': 'plate','supercategory': 'plate'}
    ]
    write_json_context['images'] = []
    write_json_context['annotations'] = []
    start_img_id = 0
    # debug
    num_imgs = 0
    num_bboxes = 0
    
    for txt_p, js_p in zip(txt_paths, json_paths):
        with open(txt_p) as txt_file:
            img_path_list = txt_file.read().strip().split('\n')
        
        with open(js_p) as json_file:
            image_data = json.load(json_file)
            
        old_img_ids = []
        new_img_ids = []
        for img in image_data['images']:
#             if img['file_name'] in img_path_list:
            if not full and img['file_name'] not in img_path_list:
                continue
            old_img_ids.append(img['id'])
            img['id'] = start_img_id + 1
            new_img_ids.append(img['id'])
            write_json_context['images'].append(img)
            start_img_id += 1
            num_imgs += 1

                
        for anno in image_data['annotations']:
#             if anno['image_id'] in old_img_ids:
            if not full and anno['image_id'] not in old_img_ids:
                continue
            anno['image_id'] = new_img_ids[old_img_ids.index(anno['image_id'])]
            write_json_context['annotations'].append(anno)
            num_bboxes += 1
        
    with open(save_path, 'w') as write_f:
        json.dump(write_json_context, write_f, indent=4)
    
    print('num_imgs:', num_imgs)
    print('num_bboxes', num_bboxes)

    
if __name__ == "__main__":

    with open("utils/configs/merge_json.yaml", 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    filter_merge_img_json(
        config_dict['txt_paths'],
        config_dict['json_paths'],
        config_dict['save_path'],
        config_dict['full']
    )        
    