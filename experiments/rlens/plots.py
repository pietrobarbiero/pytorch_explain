import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os


def main():
    results = pd.read_csv(f'./results/reasoner_results.csv', index_col=None)

    res_dir = f'./results/'
    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, 'result_accuracy.png')

    sns.set_style('whitegrid')
    sns.despine()
    plt.figure(figsize=[5, 3])
    plt.title('Task Accuracy')
    sns.barplot(data=results, x="dataset", y="accuracy", hue='model')
    plt.ylim([0.8, 1.01])
    plt.tight_layout()
    plt.savefig(out_file)
    plt.show()


if __name__ == '__main__':
    main()
