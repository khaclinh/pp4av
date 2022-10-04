import numpy as np

try:
    from utils.cocoapi.PythonAPI.pycocotools.coco import COCO
    from utils.cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
except Exception as e:
    print(f'---{e} | importing installed modules------')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

def evaluate_predictions_on_coco(
    gt_path, pred_path, iou_type="bbox",show = True,
):
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(pred_path)
    
    coco_eval = COCOeval(cocoGt, cocoDt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    if show == False:
        coco_eval.summarize(show=show)
    else:
        coco_eval.summarize()

    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image

    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting

    precisions = coco_eval.eval["precision"]
    recalls = coco_eval.eval['recall'] # iou*class_num*Areas*Max_det TP/(TP+FN) right/gt
    class_num = precisions.shape[2]
    results_per_category = ''
    result = []
    for cls_idx in range(class_num):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, cls_idx, 0, -1]
        precision_50 = precisions[0, :, cls_idx, 0, -1]
        precision = precision[precision > -1]

        recall = recalls[ :, cls_idx, 0, -1]
        recall_50 = recalls[0, cls_idx, 0, -1]
        recall = recall[recall > -1]
        
        ap = np.mean(precision) if precision.size else float("nan")
        ap_50 = np.mean(precision_50) if precision.size else float("nan")
        rec = np.mean(recall) if precision.size else float("nan")
        rec_50 = np.mean(recall_50) if precision.size else float("nan")
        
        result.append(ap_50)
        result.append(rec_50)
        
        results_per_category += '{}:\tAP_0595:{:5.2f}, Recall_0595:{:5.2f}, AP_05:{:5.2f}, Recall_05:{:5.2f}.\n'.format(
            cls_idx, float(ap * 100), float(rec * 100), float(ap_50 * 100), float(rec_50 * 100)
        )

    ap50_95 = coco_eval.stats[0]
    ap50 = coco_eval.stats[1]

    return ap50_95, ap50, results_per_category, precisions, recalls, result

if __name__ == '__main__':
    val_ap50_95, val_ap50, results_per_category, precisions, recalls = evaluate_predictions_on_coco(
                                                                                gt_path='./sample/input/labels/sample_truth.json',
                                                                                pred_path='./sample/input/predictions/sample_pred.json',
                                                                            )

    print(results_per_category)                                                                            