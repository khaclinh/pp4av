import os
import matplotlib.pyplot as plt
import numpy as np
import datetime

def draw_pr_curve(aps, colors, name = 'sample/output/plot/foo.png', specify_threshold = [0.1, 0.3, 0.5, 0.7, 0.9]):
    iou_thresholds = [str(x)[:4] for x in np.arange(0.05, 1, 0.05)]
    assert len(iou_thresholds) == aps.shape[0]

    fig = plt.figure()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0,1,0.1))
    plt.yticks(np.arange(0,1,0.1))

    if specify_threshold == None:
        #Output all ious on the same graph
        fig.suptitle(f'{os.path.splitext(os.path.basename(name))[0]}_all_iou')
        for index, iou_threshold in enumerate(iou_thresholds):
            ap = aps[index, :]
            plt.plot(np.arange(0, 1.01, 0.01), ap, '-o',markerfacecolor = colors[index],  label = f'iou: {iou_thresholds[index]}')

        plt.legend(loc="upper right")

    elif specify_threshold == 'average':
        # Average of all iou threshold
        fig.suptitle(f'{os.path.splitext(os.path.basename(name))[0]}_05_95')
        map = np.mean(aps, axis = 0)        
        plt.plot(np.arange(0, 1.01, 0.01), map, '-o',markerfacecolor = colors[0])

    elif type(specify_threshold) == list:
        iou_thresholds = [float(x) for x in iou_thresholds]
    
        fig.suptitle(f'{os.path.splitext(os.path.basename(name))[0]}_{"_".join([str(x) for x in specify_threshold])}')
        
        for index, threshold in enumerate(specify_threshold):
            map = aps[iou_thresholds.index(threshold),:]
            plt.plot(np.arange(0, 1.01, 0.01), map, '-o',markerfacecolor = colors[index], label = f'iou: {specify_threshold[index]}')

        plt.legend(loc="upper right")
    
    else:            
        fig.suptitle(f'{os.path.splitext(os.path.basename(name))[0]}_iou:{specify_threshold}')
    
    plt.savefig(name)

    if specify_threshold != 'average':
        draw_pr_curve(aps, colors, name = f'{os.path.splitext(name)[0]}_average.png', specify_threshold = 'average')
    pass

def draw_pr_curves(precisions, output_folder = 'sample/output/plot', specify_threshold = [0.1, 0.3, 0.5, 0.7]):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image

    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    num_cls = precisions.shape[2]

    precision = precisions[:,:, :,0, -1] #Getting all dets and all areaRanges
    map = np.mean(precision, axis = -1)

    colors = []
    for i in range(precisions.shape[0]):
        colors.append(tuple((np.random.rand(3,))))
    
    time_now = str(datetime.datetime.now())
    output_folder = f'{output_folder}/{time_now}'
    os.makedirs(output_folder, exist_ok=True)
    draw_pr_curve(map, colors, name = f'{output_folder}/all_pr_curves.png', specify_threshold=specify_threshold)

    labels = ['plate','face']
    #Draw pr curve for each maps
    for cls in range(num_cls):
        aps = precision[:,:, cls]
        draw_pr_curve(aps, colors, name = f'{output_folder}/{labels[cls]}_pr_curves.png', specify_threshold=specify_threshold)