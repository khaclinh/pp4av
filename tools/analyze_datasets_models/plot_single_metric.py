import os
import argparse
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


rc('axes', linewidth=2)
rc('font', weight='bold')


def parse_args():
    parser = argparse.ArgumentParser('Plot metrics on normal images')
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to box count csv file')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to output folder')
    return parser.parse_args()


def main(args):
    metrics = ['Face AP50', 'Face AR50', 'Plate AP50', 'Plate AR50']
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

    for metric in metrics:
        plt.clf()
        plt.figure(figsize=(4, 4))
        for model_key, color in zip(key2name.keys(), colors):
            model_df = df[df['model'] == model_key]
            if model_df[metric].sum() > 0:
                plt.plot(model_df['threshold'], model_df[metric], label=key2name[model_key], color=color)

        plt.xlim(left=0, right=55)
        plt.ylim(bottom=0, top=1)
        plt.xticks(ticks=range(0, 55, 10), labels=[f'{i}+' for i in range(0, 55, 10)])
        if 'Face' in metric:
            plt.legend(loc='upper right', framealpha=0.4)
            plt.xlabel('Face width (pixels)', weight='bold')
        elif 'Plate' in metric:
            plt.legend(loc='lower left', framealpha=0.4)
            plt.xlabel('Plate height (pixels)', weight='bold')
        if 'AP' in metric:
            plt.ylabel('Average Precision', weight='bold')
        elif 'AR' in metric:
            plt.ylabel('Average Recall', weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, metric.lower().replace(' ', '_')), dpi=300, transparent=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
