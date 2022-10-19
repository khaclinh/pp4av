# Evaluate and plot PR curve

Usage:
- Edit `pred_dir`, `gt_dir` `legend_name`, `iou_thresh` in `wider_eval.m`  
    + `pred_dir`:
    ```
    pred_dir/
        - sub_dir/
            - frame1.txt
            - frame2.txt
            - ........

    ``` 

    + `gt_dir`:
    ```
    gt_dir/
        - sub_dir/
            - frame1.txt
            - frame2.txt
            - ........

    ``` 

The format data of bouding box is following YOLO 1.1 standard. You can check the  [yolo](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#yolo) format.

- Run `wider_eval.m`
