import yaml
from .dataset import CocoDataset

if __name__ == '__main__':

    with open("utils/configs/nms_config.yaml", 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    for index in range(len(config_dict['true_annotation_path'])):
        coco_dataset = CocoDataset(
            true_annotation_path = config_dict['true_annotation_path'][index], 
            pred_annotation_path = config_dict['pred_annotation_path'][index], 
            save_path = 'sample', #Not relevant
            src_img_root = config_dict['src_img_root'], 
            dest_img_root = config_dict['dest_img_root'],
            )
        
        ## Generate nms file for true file
#         coco_dataset.gen_nms_images(pred = False, conf_threshold = config_dict['conf_threshold'], nms_threshold = config_dict['nms_threshold'])
        coco_dataset.gen_nms_images(pred = False, conf_threshold = 0.3, nms_threshold = 0.1, size_threshold=0)

        ## Generate nms file for pred file
        coco_dataset.gen_nms_images(pred = True, conf_threshold = config_dict['conf_threshold'], nms_threshold = config_dict['nms_threshold'], size_threshold=config_dict['size_threshold'])

    