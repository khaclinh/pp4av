import os
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc


rc('axes', linewidth=2)
rc('font', weight='bold')


def parse_args():
    parser = argparse.ArgumentParser('Plot box count on normal images')
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to metrics csv file')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to output folder')
    return parser.parse_args()


def main(args):
    metrics = ['Face count', 'Plate count']
    df = pd.read_csv(args.input, names=('model', 'threshold', *metrics))
    key2name = OrderedDict({'anonymizer': 'UAI Anonymizer',
                            'yolov5face': 'YOLO5Face',
                            'retina': 'RetinaFace',
                            'gg': 'Google API',
                            'azure': 'Azure API',
                            'aws': 'AWS API',
                            'alpr': 'ALPR',
                            'nvidia': 'NVIDIA LPDnet',
                            'our': 'Our'})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(key2name)]
    colors[3], colors[-1] = colors[-1], colors[3] # Our is red

    gt_df = df[df['model'] == 'gt']
    for _, row in gt_df.iterrows():
        for metric in metrics:
            df.loc[df['threshold'] == row['threshold'], metric] /= row[metric]

    for metric in metrics:
        plt.clf()
        fig, ax = plt.subplots(figsize=(4, 4))
        for model_key, color in zip(key2name.keys(), colors):
            model_df = df[df['model'] == model_key]
            if model_df[metric].sum() > 0:
                ax.plot(model_df['threshold'], model_df[metric], label=key2name[model_key], color=color)

        ax.set_xlim(left=0, right=55)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-2, top=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        ax.set_xticks(ticks=range(0, 55, 10), labels=[f'{i}+' for i in range(0, 55, 10)])
        ax.legend(loc='lower right', framealpha=0.4)
        if 'Face' in metric:
            ax.set_xlabel('Face width (pixels)', weight='bold')
        elif 'Plate' in metric:
            ax.set_xlabel('Plate height (pixels)', weight='bold')
        ax.set_ylabel('Number of Detection/Groundtruth Samples', weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, metric.lower().replace(' ', '_')), dpi=300, transparent=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
