function file_paths = glob_labels(root_dir)
root_dir = get_abs_folder_path(root_dir);
file_info_list = dir(fullfile(root_dir, '**/*.txt'));
file_info_list = file_info_list(~[file_info_list.isdir]);

root_dir = append(root_dir, '/');
file_paths = strings(length(file_info_list), 1);
for i = 1:length(file_info_list)
    abs_path = append(file_info_list(i).folder, '/', file_info_list(i).name);
    rel_path = erase(abs_path, root_dir);
    file_paths(i) = rel_path;
end
end

function abs_path = get_abs_folder_path(dir_path)
info = dir(dir_path);
abs_path = info(1).folder;
end