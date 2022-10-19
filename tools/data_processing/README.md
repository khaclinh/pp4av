# PP4AV data manipulation

## Process data: Using repo ds_gdpr_eval 

- Split dataset into `train/val/test`: edit `image path`, `label path`, and `save directory` in `utils/config/dataset_split_config.yaml` and run
```
python dataset_split.py 
```

- Convert data to Coco format (output of anonymizer model saved `.txt` and need convert to `.json`): edit config in `utils/config/format_transform_config.yaml` and run:
```
python format_transform.py  
``` 

- Non-maximum suppression: edit config in `sample_configconfig/nms_config.yaml` and run:
``` 
python nms.py 
```

- Statistic number of boxes, AP, AR of json ground-truth and json prediction file 
    + input: json coco true file, json coco pred file corresponding 
    + output: csv statistic file (each image is a row with number boxes, AP, AR of each class)
    + Edit `config` in `analysis_stats_config.yaml`: `true_annotation_path` is list of path to json coco true file, `pred_annotation_path` is list pf path json coco pred file corresponding, `save_path` is list of path to save csv file, `nms_threshold`, `true_conf_threshold` is confidence score threhold of coco true file, `pred_conf_threshold` is confidence score threhold of coco pred file, `size_threshold` is area threshold box removed (if area is smaller than size_threshold, remove it) 
    ```
    python –m utils.analysis_stats 
    ```

