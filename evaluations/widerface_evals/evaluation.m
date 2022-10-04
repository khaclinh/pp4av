function evaluation(norm_pred_list, gt_dir, legend_name, iou_thresh)

if ~exist(sprintf('./plot/baselines/Val/%s', legend_name), 'dir')
    mkdir(sprintf('./plot/baselines/Val/%s', legend_name));
end

thresh_num = 1000;
org_pr_curve = zeros(thresh_num, 2);
count_obj = 0;

label_paths = glob_labels(gt_dir);

for i = 1:length(label_paths)
    gt_path = append(gt_dir, '/', label_paths(i));
    fprintf('Current file %s\n', label_paths(i));

    gt_bbx = parse_label(gt_path, 0);
    pred_bbx = norm_pred_list{i};
    count_obj = count_obj + size(gt_bbx, 1);

    if isempty(gt_bbx) || isempty(pred_bbx)
        continue;
    end

    pred_recall = image_evaluation(pred_bbx, gt_bbx, iou_thresh);
    img_pr_info = image_pr_info(thresh_num, pred_bbx, pred_recall);
    if ~isempty(img_pr_info)
        org_pr_curve(:, 1) = org_pr_curve(:, 1) + img_pr_info(:, 1);
        org_pr_curve(:, 2) = org_pr_curve(:, 2) + img_pr_info(:, 2);
    end
end

pr_curve = dataset_pr_info(thresh_num, org_pr_curve, count_obj);
save(sprintf('./plot/baselines/Val/%s/wider_pr_info_%s.mat', legend_name, legend_name), 'pr_curve', 'legend_name');
end

function pred_recall = image_evaluation(pred_bbx, gt_bbx, iou_thresh)
    pred_recall = zeros(size(pred_bbx, 1), 1);
    recall_list = zeros(size(gt_bbx, 1), 1);
    for i = 1:size(pred_bbx, 1)
        overlap_list = boxoverlap(gt_bbx, pred_bbx(i, 1:4));
        [max_overlap, idx] = max(overlap_list);
        if max_overlap >= iou_thresh
            recall_list(idx) = 1;
        end
        pred_recall(i) = sum(recall_list);
    end
end

function img_pr_info = image_pr_info(thresh_num, pred_bbx, pred_recall)
    img_pr_info = zeros(thresh_num, 2);
    for t = 1:thresh_num
        thresh = 1 - t / thresh_num;
        r_index = find(pred_bbx(:, 5) >= thresh, 1, 'last');
        if (isempty(r_index))
            img_pr_info(t, 1) = 0;
            img_pr_info(t, 2) = 0;
        else
            img_pr_info(t, 1) = r_index;
            img_pr_info(t, 2) = pred_recall(r_index);
        end
    end
end

function pr_curve = dataset_pr_info(thresh_num, org_pr_curve, count_obj)
    pr_curve = zeros(thresh_num, 2);
    pr_curve(:, 1) = org_pr_curve(:, 2) ./ org_pr_curve(:, 1);
    pr_curve(:, 2) = org_pr_curve(:, 2) ./ count_obj;
end
