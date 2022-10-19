import yaml
from dataset import CocoDataset

if __name__ == '__main__':

    with open("configs/analysis_stats_config.yaml", 'r') as stream:
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
        
        ## Generate num_box statistics for images
        coco_dataset.gen_stats_images(
            size_threshold=config_dict['size_threshold'], 
            true_conf_threshold=config_dict['true_conf_threshold'], 
            pred_conf_threshold=config_dict['pred_conf_threshold'],
            overlapping_tp_threshold=config_dict['overlapping_tp_threshold'])
    