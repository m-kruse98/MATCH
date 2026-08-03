import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


def compare_histogram(scores, classes, class_name, prefix, thresh=None, n_bins=64, log=False, name='', title=None):
    if log:
        scores = np.log(scores + 1e-8)

    if thresh is not None:
        if np.max(scores) < thresh:
            thresh = np.max(scores)
        scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes >= 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True,
             label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True,
             label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(np.min(scores), np.max(scores), 5)
    labels = ['{:.2f}'.format(i) for i in ticks[:-1]] + \
        ['>' + '{:.2f}'.format(np.max(scores))]
    plt.xticks(ticks, labels=labels)
    if title is not None:
        plt.title(title)
    plt.xlabel('Anomaly Score' if not log else 'Log Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y')
    # dir_to_save = Path("viz", "hists", class_name, prefix)
    dir_to_save = Path("viz", "histos", prefix, class_name)
    os.makedirs(dir_to_save, exist_ok=True)
    plt.savefig(Path(dir_to_save, f"{name}.png"))
    plt.close()


def viz_maps(img, gt, ano_map, prefix='', norm=True, class_name=None, vmin=0, vmax=1,
             filename="test.png", title="sample_title"):

    # gt[fg == 0] = np.nan
    ano_map = np.copy(ano_map)
    # ano_map[fg == 0] = np.nan
    if True or norm:
        img = np.moveaxis(img, 0, 2)
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(img)
    # axs[0, 1].imshow(depth)
    axs[1, 0].imshow(gt, vmin=0, vmax=1)
    axs[1, 1].imshow(ano_map, vmin=vmin, vmax=vmax)
    # axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    fig.suptitle(title)
    dir_to_save = Path("viz", "maps", class_name, prefix)
    os.makedirs(dir_to_save, exist_ok=True)
    plt.savefig(Path(dir_to_save, filename))
    plt.clf(), plt.cla()
    plt.close(fig)
