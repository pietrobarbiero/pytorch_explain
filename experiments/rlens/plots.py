import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os


def main():
    datasets = ['xor', 'trig', 'vec']
    folds = [i + 1 for i in range(5)]
    results = pd.DataFrame()
    for dataset in datasets:
        res = pd.read_csv(f'./results/{dataset}_activations_final_rerun/explanations/reasoner_results.csv', index_col=None)
        res['dataset'] = dataset
        if len(results) == 0:
            results = res
        else:
            results = pd.concat((results, res))

    res_dir = f'./results/'
    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, 'result_accuracy.png')

    sns.set_style('whitegrid')
    sns.despine()
    plt.figure(figsize=[5, 3])
    plt.title('R2N results')
    sns.barplot(data=results, x="dataset", y="accuracy", hue='model')
    plt.ylim([0.5, 1.01])
    plt.tight_layout()
    plt.savefig(out_file)
    plt.show()


if __name__ == '__main__':
    main()
