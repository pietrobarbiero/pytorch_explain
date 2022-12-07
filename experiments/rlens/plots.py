import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def main():
    ood = False
    ood = True
    results = pd.read_csv(f'./results/reasoner_results.csv', index_col=None)
    res_dir = f'./results/'
    if ood:
        results = pd.read_csv(f'./results/ood/reasoner_results.csv', index_col=None)
        res_dir = f'./results/ood/'

    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, 'result_accuracy.png')

    results['model'] = results['model'].str.replace('DCR', 'DCR (ours)')

    sns.set_style('whitegrid')
    sns.despine()
    plt.figure(figsize=[5, 3])
    plt.title('Task Accuracy')
    if ood:
        plt.title('OOD Task Accuracy')
    ax = sns.barplot(data=results, x="dataset", y="accuracy", hue='model')
    num_locations = len(results["dataset"].unique())
    hatches = itertools.cycle(['', '/', '/', ''])
    for i, bar in enumerate(ax.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)
    plt.ylim([0.5, 1.01])
    # if ood:
    #     plt.ylim([0., 1.01])
    plt.tight_layout()
    plt.savefig(out_file)
    plt.show()


if __name__ == '__main__':
    main()
