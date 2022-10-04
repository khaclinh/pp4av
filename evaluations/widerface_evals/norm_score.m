function norm_pred_list = norm_score(org_pred_list)

file_num = size(org_pred_list, 1);
norm_pred_list = cell(file_num, 1);

max_score = realmin('single');
min_score = realmax('single');

for i = 1:file_num
    pred_list = org_pred_list{i};
    if(isempty(pred_list))
        continue;
    end
    score_list = pred_list(:, 5);
    max_score = max(max_score, max(score_list));
    min_score = min(min_score, min(score_list));
end

for i = 1:file_num
    pred_list = org_pred_list{i};
    if(isempty(pred_list))
        continue;
    end
    score_list = pred_list(:, 5);
    norm_score_list = (score_list - min_score) / (max_score - min_score);
    pred_list(:, 5) = norm_score_list;
    norm_pred_list{i} = pred_list;
end
end
