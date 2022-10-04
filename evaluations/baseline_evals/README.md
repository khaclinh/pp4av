# Installation
Install [pycocotools](https://github.com/cocodataset/cocoapi) for redundancy.

```shell
pip3 install scikit-image
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
``` 

Install the pycocotools api inside the directory
```shell
cd utils/cocoapi/PythonAPI
make
``` 

# Introduction 
Compute the metrics given the prediction and ground truth of bounding box and return
- Recall
- Precision
- mAP score
- PR-curve

# Usage

## Transforming YOLO v1.1 format to COCO Json format
`.txt` file for each `.jpg` image file and put to file: `<object-class> <x> <y> <width> <height>`, maybe with `<conf_score>` at the end
- `<object-class>` - integer number of object from 0 to (classes-1)
- `<x> <y> <width> <height>` - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
    - `<x> = <absolute_x> / <image_width>`
    - `<y> = <absolute_y> / <image_height>`
    - `<width> = <absolute_width> / <image_width>`
    - `<height> = <absolute_height> / <image_height>`
- - `<conf-score>` - float objectiveness score from 0 to 1

Here are input sample lines in YOLO v1.1 format `txt` file

```python
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```
But for prediction `txt` label file, we need to add confidence for each line `<object-class> <x> <y> <width> <height> <confidence>`

For information about COCO format, there are to types:
- Data format: https://cocodataset.org/#format-data
- Result format: https://cocodataset.org/#format-results
+ For the purpose of building simple dataloader, I modified coco_data['image']['file_name'] to be the entire file path of the image, because of which, in viz and nms operations, I had the img_root switching function.

To transform yolov1 formats to coco format, fix the utils/format_transform_config.yaml files
    img_folder_paths: List of paths of directories containing images 
    txt_folder_paths: List of paths of directories containing txt files
    save_path: string of json file to save the labels in 
    labels: list of string, labels to index
    eval: boolean, if it is True, generate prediction json file for evaluation

Run 
```shell
python utils/format_transform.py
```

## Run visualization, nms or get num_box_statistics per image 
1. Edit the utils/configs/dataset_config.yaml
    true_annotation_path: List of json path by the coco data-format 
    pred_annotation_path: List of json path by the coco result-format 
    save_path: List output save folder path for visualization
    src_img_root: img_root at the source of the json 
    dest_img_root: img_root at the current local machine
** As aforementioned, the src_img_root and dest_img_root would be switch to imread images when visualize.
2. Run
To visualize:
```shell
python utils/viz.py
```

To produce nms json files:
```shell
python utils/nms.py
```

To produce box_stats csv file:
```shell
python utils/box_stats.py
```

## Split dataset into train test val
1. Edit the utils/configs/dataset_split_config.yaml
2. Run
```shell
python utils/dataset_split.py
```

## Run evaluation
Edit the eval.sh
Run:
```shell
sh eval.sh
```

## Merge json files
1. Edit the utils/configs/merge_json_config.yaml
2. Run
```shell
python utils/merge_json.py
```



