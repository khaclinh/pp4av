import argparse
from ast import parse

from matplotlib.pyplot import plot

from metrics.eval import evaluate_predictions_on_coco
from utils.plot import draw_pr_curves

def evaluate_n_plot(gt_path,pred_path, plot = True ):
    val_ap50_95, val_ap50, results_per_category, precisions, recalls, result = evaluate_predictions_on_coco(
                                                                            gt_path= gt_path,
                                                                            pred_path= pred_path,
                                                                        )
    if plot:
        draw_pr_curves(precisions)
        
    print(results_per_category)   
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a evaluation schema')
    parser.add_argument('--gt_path', type= str, default='./sample/input/labels/sample_truth.json',
                        help='path to json file of ground truth labels')
    parser.add_argument('--pred_path', type= str, default='./sample/input/predictions/sample_pred.json',
                        help='path to json file of predictions labels')
    parser.add_argument('--plot', default=True, help= "Visualize pr curves")        
    
    args = parser.parse_args()

    
    evaluate_n_plot(
        gt_path=args.gt_path, 
        pred_path=args.pred_path,
        plot=args.plot
        )


    pass