function label = parse_label(label_path, class_id)

fid = fopen(label_path, 'r');
data = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);

try
    data = data{1};
    bbx_num = length(data);
    label = ones(bbx_num, 6);

    for i = 1:bbx_num
        raw_info = str2num(data{i});
        label(i, 1) = raw_info(1);
        label(i, 2) = raw_info(2);
        label(i, 3) = raw_info(3);
        label(i, 4) = raw_info(4);
        label(i, 5) = raw_info(5);
        if length(raw_info) > 5
            label(i, 6) = raw_info(6);
        end
    end

    label = label(label(:, 1) == class_id, :);
    label = label(:, 2:6);
    [~, s_index] = sort(label(:, 5), 'descend');
    label = label(s_index,:);

    cx = label(:, 1);
    cy = label(:, 2);
    w = label(:, 3);
    h = label(:, 4);
    label(:, 1) = cx - w / 2;
    label(:, 2) = cy - h / 2;
    label(:, 3) = cx + w / 2;
    label(:, 4) = cy + h / 2;

catch
    fprintf('Invalid format %s\n', pred_path);
end

end