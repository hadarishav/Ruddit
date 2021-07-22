import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

RANDOM_SEED = 12
np.random.seed(RANDOM_SEED)


def stratified_kfold_regression(dataset: csv, save_path: str, num_elements: int = 6000) -> None:
    """
    Creates 5 (can be modified) folds from the dataset while making the score distribution even in all folds.

    :param dataset: Path to csv file containing the dataset.
    :param save_path: Path to the folder to create the folds in.
    :param num_elements: Number of comments in the csv file.
    """
    # Sorted stratification: https://scottclowe.com/2016-03-19-stratified-regression-partitions/
    with open(dataset, 'r', encoding='utf-8') as f:
        main_reader = csv.reader(f)
        content = {}
        k_idx = [[], [], [], [], [], [], [], [], [], []]
        for i, line in enumerate(main_reader):
            if (i == 0):
                continue
            k_id = int(line[-1])
            content[k_id] = line
            index = k_id % 5
            k_idx[index].append(k_id)
        f.close()

    for i in range(1, 6):
        train_name = f'train{str(i)}.csv' if save_path is None else f'{save_path}/train{str(i)}.csv'
        test_name = f'test{str(i)}.csv' if save_path is None else f'{save_path}/test{str(i)}.csv'
        index = i - 1
        all_ids = [i for i in range(1, num_elements)]
        with open(train_name, 'w', encoding='utf-8') as t, open(test_name, 'w', encoding='utf-8') as e:
            train_w = csv.writer(t)
            test_w = csv.writer(e)
            train_w.writerow(['comment', 'Score', 'k_id'])
            test_w.writerow(['comment', 'Score', 'k_id'])
            for num, j in enumerate(all_ids):
                if(j in k_idx[index]):
                    test_w.writerow(content[j])
                else:
                    train_w.writerow(content[j])

        t.close()
        e.close()


def plot_all_folds(path_to_folds: str = None) -> None:
    """
    Plots the distribution of the train and test folds.

    :param path_to_folds: Path to the folder that contains the folds.
    """
    for i in range(1, 6):
        train_name = f'train{str(i)}.csv' if path_to_folds is None else f'{path_to_folds}/train{str(i)}.csv'
        test_name = f'test{str(i)}.csv' if path_to_folds is None else f'{path_to_folds}/test{str(i)}.csv'
        df_train = pd.read_csv(train_name)
        df_test = pd.read_csv(test_name)
        m1 = df_train.Score.to_numpy().mean()
        m2 = df_test.Score.to_numpy().mean()
        ax = sns.displot(data=df_train, x="Score", kde=True)
        # sns.displot(data = df_test, x = "Score", kde=True, ax = ax)
        ax2 = sns.displot(data=df_test, x="Score", kde=True)
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create folds for training and testing")
    parser.add_argument('--filename', help="Path to Ruddit (or its variant)", default="sample_input_file.csv", type=str)
    parser.add_argument('--plot', help="1 for plotting, 0 otherwise", default=0, type=int)
    parser.add_argument('--save_path', help="Path to the folder to create the folds in", type=str)
    parser.add_argument('--dataset_range',
                        help='1 for identity-agnostic or Ruddit, 2 for no-swearing, 3 for reduced-range', default=1,
                        type=int)
    args = parser.parse_args()

    # create folds
    if(args.dataset_range == 1):
        num_elements = 6001
    elif(args.dataset_range == 2):
        num_elements = 5133
    else:
        num_elements = 5152
    stratified_kfold_regression(args.filename, args.save_path, num_elements)
    if(args.plot):
        plot_all_folds()
