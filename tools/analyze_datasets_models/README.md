# Analyze some datasets and models for anonymization

## Plot evaluation metrics on normal dataset
Prepare evaluation results csv file with columns:
```
model name, size threshold, face precision, face recall, plate precision, plate recall
```
Example:
```
azure,0,0.0025,0.0019,0.0,0.0
azure,10,0.0,0.0,0.0,0.0
azure,20,0.0,0.0,0.0,0.0
azure,30,0.0,0.0,0.0,0.0
retina,0,0.6284,0.8879,0.0,0.0
retina,10,0.38249142161502186,0.602510460251046,0.0,0.0
retina,20,0.2012958648806057,0.41966173361522197,0.0,0.0
retina,30,0.14571007210800527,0.39375,0.0,0.0
aws,0,0.6964,0.7973,0.0,0.0
aws,10,0.4572632875814379,0.5439330543933054,0.0,0.0
aws,20,0.24713338185413547,0.3276955602536998,0.0,0.0
aws,30,0.2753770773108064,0.396875,0.0,0.0
```
Then run
```
python plot_single_metric.py --input <path/to/csv> --output <path/to/image/folder>
```

## Plot evaluation metrics on woodscape dataset
Prepare evaluation results csv file with columns:
```
model name, size threshold, face precision, face recall, plate precision, plate recall
```
Example:
```
azure,0,0.0025,0.0019,0.0,0.0
azure,10,0.0,0.0,0.0,0.0
azure,20,0.0,0.0,0.0,0.0
azure,30,0.0,0.0,0.0,0.0
retina,0,0.6284,0.8879,0.0,0.0
retina,10,0.38249142161502186,0.602510460251046,0.0,0.0
retina,20,0.2012958648806057,0.41966173361522197,0.0,0.0
retina,30,0.14571007210800527,0.39375,0.0,0.0
aws,0,0.6964,0.7973,0.0,0.0
aws,10,0.4572632875814379,0.5439330543933054,0.0,0.0
aws,20,0.24713338185413547,0.3276955602536998,0.0,0.0
aws,30,0.2753770773108064,0.396875,0.0,0.0
```
Then run
```
python plot_single_metric_woodscape.py --input <path/to/csv> --output <path/to/image/folder>
```

## Plot Number of Detection/Groundtruth Samples on normal dataset
Prepare csv file for number of detected boxes and number of ground truth boxes with columns:
```
ground truth/model name, size threshold, number of faces, number of plates
```
Example:
```
gt,0,3599,7787
gt,10,3242,3861
gt,20,946,1055
gt,30,320,372
anonymizer,0,8954,7259
anonymizer,10,6964,3156
anonymizer,20,2011,899
anonymizer,30,794,351
yolov5face,0,11672,0
yolov5face,10,3501,0
yolov5face,20,874,0
yolov5face,30,357,0
```
Then run
```
python plot_single_count.py --input <path/to/csv> --output <path/to/image/folder>
```

## Plot Number of Detection/Groundtruth Samples on woodscape dataset
Prepare csv file for number of detected boxes and number of ground truth boxes with columns:
```
ground truth/model name, size threshold, number of faces, number of plates
```
Example:
```
gt,0,3599,7787
gt,10,3242,3861
gt,20,946,1055
gt,30,320,372
anonymizer,0,8954,7259
anonymizer,10,6964,3156
anonymizer,20,2011,899
anonymizer,30,794,351
yolov5face,0,11672,0
yolov5face,10,3501,0
yolov5face,20,874,0
yolov5face,30,357,0
```
Then run
```
python plot_single_count_woodscape.py --input <path/to/csv> --output <path/to/image/folder>
```

## Search for false detection cases
Modify the paths to image folder, label folder and prediction folder if needed, then run
```
python search_false_detections.py
```

## Check folder splitting based on minimum size thresholds
Modify these constants and paths if needed
- `thresholds`: list of split thresholds to check.
- `img_root`: path to image root directory.
- `small_root`: path to labels for all object sizes.
- `large_root`: path to labels for objects with size larger than some threshold.

Then run
```
python check_folder_splitting.py
```

## Plot statistics of a dataset
Modify path to the dataset in function `parse_<dataset_name>` if needed,

Then run
```
python plot_statistics.py --name <dataset_name>
```

## Visualize ground truth and prediction of models on a specific image for easy comparison
Modify these paths if needed
- `NORMAL_IMAGE_ROOT`: path to image root directory of normal images dataset.
- `NORMAL_GT_ROOT`: path to ground truth labels root directory of normal images dataset.
- `NORMAL_PRED_ROOT`: path to prediction root directory for normal images.
- `WOODSCAPE_IMAGE_ROOT`: path to image root directory of fisheye images dataset.
- `WOODSCAPE_GT_ROOT`: path to ground truth labels root directory of fisheye images dataset.
- `WOODSCAPE_PRED_ROOT`: path to prediction root directory for fisheye images.

Then run
```
python viz_pred.py --type <normal/woodscape> --model <model_name> --image <relative/path/to/image>
```