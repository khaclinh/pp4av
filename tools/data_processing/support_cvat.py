import os 
import scipy.io
import glob
import numpy as np

import joblib

from pathlib import Path
from tqdm import tqdm


def load_txt_data(txt_path, w_original, h_original):
    lb = []
    for line in open(txt_path).read().strip().split('\n'):
        if not line:
            continue
        info = line.split()
        label_idx = int(float(info[0]))
        xc, yc, width, height, conf= map(float, info[1:])
        # add to list:
        if len(lb) == 0:
            lb = [[label_idx, xc, yc, width, height]]
        else:
            lb.append([label_idx, xc, yc, width, height])
                
    return lb


def process(in_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    assert Path(out_dir).is_dir(), 'Output path must be a directory'

    files = []
    for file_type in ['txt']:
        files.extend(list(Path(in_dir).glob(f'**/*.{file_type}')))

    for input_path in tqdm(files):
        data = load_txt_data(input_data)
        
        relative_path = input_path.relative_to(in_dir)
                
        out_file = (Path(out_dir) / relative_path
                    
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_file, 'w') as f:
            for i in range(len(data)):
                f.write(str(data[i][0])) 
                f.write(f' {data[i][1]:.6f} {data[i][2]:.6f} {data[i][3]:.6f} {data[i][4]:.6f}\n')

if __name__ == "__main__":
    in_idr =  "/annotations/data/raw_img/"
    out_dir = "/annotations/data/"

    process(in_dir, out_dir)
