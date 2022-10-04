import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


metrics = ['Face AP50', 'Face AR50', 'Plate AP50', 'Plate AR50']
df = pd.read_csv('data/normal_result.csv', names=['model', 'threshold', *metrics])

model_names = ['UAI Anonymizer', 'YOLO5Face', 'RetinaFace', 'Google API',
               'Azure API', 'AWS API', 'ALPR', 'NVIDIA LPDnet', 'Our']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(model_names)]
colors[3], colors[-1] = colors[-1], colors[3] # Our is red

fig, axs = plt.subplots(1, len(metrics), figsize=(12, 3))
for metric, ax in zip(metrics, axs):
    for model_name, color in zip(model_names, colors):
        model_df = df[df['model'] == model_name]
        if model_df[metric].sum() > 0:
            ax.plot(model_df['threshold'], model_df[metric], label=model_name, color=color)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - box.height * 0.2, box.width, box.height])
    ax.set_xlim(left=df['threshold'].min(), right=df['threshold'].max())
    ax.set_ylim(bottom=0, top=1)
    ax.set_xticks(range(df['threshold'].min(), df['threshold'].max() + 1, 10))
    if 'Face' in metric:
        ax.set_xlabel('Face width (pixels)')
    elif 'Plate' in metric:
        ax.set_xlabel('Plate height (pixels)')
    if 'AP' in metric:
        ax.set_ylabel('Average Precision')
    elif 'AR' in metric:
        ax.set_ylabel('Average Recall')

handels = [mlines.Line2D([], [], color=color, label=model_name) 
           for model_name, color in zip(model_names, colors)]
fig.legend(handles=handels, ncol=5, loc='lower center')
plt.tight_layout()
plt.gcf().set_dpi(300)
plt.savefig('plot_metrics.png')
