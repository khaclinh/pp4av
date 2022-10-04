clear;
close all;
addpath(genpath('./plot'));

% Please specify your prediction directory.
pred_dir = '/home/giangnd/Documents/data/test_wdf_plot/wdf_val/yolo5face';
gt_dir = '/home/giangnd/Documents/data/test_wdf_plot/wdf_val/labels';

% preprocessing
pred_list = read_pred(pred_dir, gt_dir, 0);
norm_pred_list = norm_score(pred_list);

% Please specify your algorithm name.
legend_name = 'YOLO5Face';
iou_thresh = 0.5;
evaluation(norm_pred_list, gt_dir, legend_name, iou_thresh);

% for iou_thresh = [0.9, 0.92, 0.94, 0.96, 0.98]
%     evaluation(norm_pred_list, gt_dir, sprintf('IoU %.2f', iou_thresh), iou_thresh);
% end

fprintf('Plot PR curve under overall setting.\n');
dateset_class = 'Val';

% scenario:
plot_dir = sprintf('./plot/baselines/%s', dateset_class);
wider_plot(plot_dir, dateset_class);
