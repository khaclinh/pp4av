import re


def summarize_file(filepath):
    count_face = 0
    count_plate = 0
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        while line:
            if re.match(r'^Num face box: (\d+)$', line):
                n = int(re.sub(r'^Num face box: (\d+)$', r'\1', line))
                count_face += n
            elif re.match(r'^Num plate box: (\d+)$', line):
                n = int(re.sub(r'^Num plate box: (\d+)$', r'\1', line))
                count_plate += n
            line = f.readline().strip()
    return count_face, count_plate

def summarize(gt_path, pred_path):
    return *summarize_file(gt_path), *summarize_file(pred_path)


if __name__ == '__main__':
    print(summarize('data/GT.txt', 'data/ALPR.txt'))
