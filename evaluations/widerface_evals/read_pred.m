function pred_list = read_pred(pred_dir, gt_dir, class_id)

label_paths = glob_labels(gt_dir);
file_num = length(label_paths);
pred_list = cell(file_num, 1);

for i = 1:file_num
    fprintf('Read prediction: current file %s\n', label_paths(i));
    pred_path = append(pred_dir, '/', label_paths(i));
    pred = parse_label(pred_path, class_id);
    pred_list{i} = pred;
end
