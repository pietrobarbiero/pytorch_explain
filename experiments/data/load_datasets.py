import itertools
import os
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from torchvision.datasets import MNIST
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor

from experiments.data.CUB200 import cub_loader


def load_mimic(base_dir: str = './data/'):
    data = pd.read_csv(f'{base_dir}/mimic-ii/full_cohort_data.csv')
    # data.drop('hgb_first')
    fs = [
        'aline_flg',
        'gender_num',
        # 'hosp_exp_flg',
        # 'icu_exp_flg',
        # 'day_28_flg',
        # 'censor_flg',
        'sepsis_flg', 'chf_flg', 'afib_flg',
        'renal_flg', 'liver_flg', 'copd_flg', 'cad_flg', 'stroke_flg',
        'mal_flg', 'resp_flg',
    ]
    features = fs
    data1 = data[fs].values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data1 = imp_mean.fit_transform(data1)

    f2 = fs.copy()
    f2.append('day_icu_intime')
    f2.append('service_unit')
    f2.append('day_28_flg')
    f2.append('hospital_los_day')
    f2.append('icu_exp_flg')
    f2.append('hosp_exp_flg')
    f2.append('censor_flg')
    f2.append('mort_day_censored')
    f2 = data.columns.difference(f2)
    data2 = data[f2].values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data2 = imp_mean.fit_transform(data2)
    scaler = MinMaxScaler((0, 1))
    data2 = scaler.fit_transform(data2)
    features = features + list(f2)
    est = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
    data2d = est.fit_transform(data2)
    f2d = []
    for feature in f2:
        # f2d.append(feature + '_VLOW')
        f2d.append(feature + '_LOW')
        f2d.append(feature + '_NORMAL')
        f2d.append(feature + '_HIGH')
        # f2d.append(feature + '_VHIGH')
    features = fs + f2d

    datax = np.hstack((data1, data2d))
    # datay = data['day_28_flg'].values
    # datay = (data['hospital_los_day']>6).values
    # datay = data['hosp_exp_flg'].values

    datay = (data['day_28_flg'].values + data['hosp_exp_flg'].values + data['icu_exp_flg'].values + (1-data['censor_flg'].values)) > 0
    # datay = data['day_28_flg'].values

    # model = DecisionTreeClassifier(max_depth=3)
    # # model = RandomForestClassifier()
    # scores = cross_val_score(model, datax, datay, cv=10)
    # print(scores.mean())

    # datax = np.vstack((datax, datax[datay==1], datax[datay==1]))
    # datay = np.hstack((datay, datay[datay==1], datay[datay==1]))

    x = torch.FloatTensor(datax)
    y = one_hot(torch.tensor(datay).to(torch.long)).to(torch.float)
    return x, y, features


def load_celldiff(base_dir='./data'):
    gene_expression_matrix = pd.read_csv(f'{base_dir}/celldiff/data_matrix.csv', index_col=0)
    clustering_labels = pd.read_csv(f'{base_dir}/celldiff/cluster_labels.csv', index_col=0)
    biomarkers = pd.read_csv(f'{base_dir}/celldiff/markers.csv', index_col=0)

    labels = clustering_labels.values.squeeze()

    scaler = MinMaxScaler((0, 1))
    scaler.fit(gene_expression_matrix.values)
    data_normalized = scaler.transform(gene_expression_matrix.values)

    x = torch.FloatTensor(data_normalized)
    y = torch.FloatTensor(labels).to(torch.long).squeeze()

    # model = DecisionTreeClassifier()
    # model = RandomForestClassifier()
    # scores = cross_val_score(model, data_normalized, labels, cv=10)
    # print(scores.mean())

    x = torch.FloatTensor(data_normalized)
    y = one_hot(torch.tensor(labels).to(torch.long)).to(torch.float)
    return x, y, gene_expression_matrix.columns


def load_vDem(base_dir='./data'):
    data = pd.read_csv(f'{base_dir}/vdem/V-Dem-CY-Core-v10.csv')
    data['country_name_year'] = data['country_name'] + '_' + data['year'].astype(str)
    data_2000 = data[data['year'] > 2000].iloc[:, 12:-1].dropna(axis=1)

    high_level_indicators = [
        'v2x_polyarchy',
        # 'v2x_libdem',
        # 'v2x_partipdem',
        'v2x_delibdem',
        'v2x_egaldem'
    ]
    mid_level_indicators = [
        'v2x_api',
        'v2x_mpi',
        'v2x_freexp_altinf',
        'v2x_frassoc_thick',
        'v2x_suffr',
        'v2xel_frefair',
        'v2x_elecoff',
        # 'v2x_liberal',
        'v2xcl_rol',
        # 'v2x_jucon',
        # 'v2xlg_legcon',
        # 'v2x_partip',
        'v2x_cspart',
        # 'v2xdd_dd',
        # 'v2xel_locelec',
        # 'v2xel_regelec',
        'v2xdl_delib',
        'v2x_egal',
        'v2xeg_eqprotec',
        'v2xeg_eqaccess',
        'v2xeg_eqdr',
    ]

    # drop_list = ['codelow', 'codehigh', 'sd', 'osp', 'nr', 'mean']
    low_level_indicators = []
    for f in data_2000.columns:
        if f.endswith('_ord') and f not in high_level_indicators and f not in mid_level_indicators:
            low_level_indicators.append(f)

    low_level_indicators_continuous = []
    for f in data_2000.columns:
        if f.endswith('_codehigh') or f.endswith('_codelow') and \
                f not in high_level_indicators and f not in mid_level_indicators:
            low_level_indicators_continuous.append(f)

    print(f'Main {len(high_level_indicators)} - Area {len(mid_level_indicators)} - Raw {len(low_level_indicators)}')

    data_low_continuous = data_2000[low_level_indicators_continuous]

    data_low_raw = data_2000[low_level_indicators]
    one_hots = []
    for indicator in low_level_indicators:
        c = data_low_raw[indicator].values
        n_bins = int(c.max())
        kbin = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='uniform')
        c1h = kbin.fit_transform(c.reshape(-1, 1))
        one_hots.append(c1h)

    new_indicator_names = []
    for clist, cname in zip(one_hots, low_level_indicators):
        if clist.shape[1] > 1:
            for i in range(clist.shape[1]):
                new_indicator_names.append(f'{cname}_{i}')
        else:
            new_indicator_names.append(f'{cname}')

    data_low = pd.DataFrame(np.hstack(one_hots), columns=new_indicator_names)
    data_mid = data_2000[mid_level_indicators] > 0.5
    data_high = data_2000[high_level_indicators].iloc[:, 0] > 0.5

    # data_mid = pd.DataFrame(np.hstack([data_low, data_mid]), columns=data_low.columns.append(data_mid.columns))

    # scores = cross_val_score(LogisticRegression(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())
    # scores = cross_val_score(DecisionTreeClassifier(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())
    # scores = cross_val_score(RandomForestClassifier(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())

    x = torch.FloatTensor(data_low.values)
    c = torch.FloatTensor(data_mid.values)
    y = one_hot(torch.tensor(data_high.values).to(torch.long)).to(torch.float)
    return x, c, y, data_mid.columns


def load_mnist2(base_dir='./data'):
    train_data = torch.load(os.path.join(base_dir, 'MNIST_X_to_C/c2y_training.pt'))
    val_data = torch.load(os.path.join(base_dir, 'MNIST_X_to_C/c2y_validation.pt'))
    test_data = torch.load(os.path.join(base_dir, 'MNIST_X_to_C/c2y_test.pt'))
    train_data.tensors = ((train_data.tensors[0]).to(torch.float),
                          (one_hot((train_data.tensors[1].argmax(dim=1) % 2 == 1).to(torch.long)).to(torch.float)))
    val_data.tensors = ((val_data.tensors[0]).to(torch.float),
                        (one_hot((val_data.tensors[1].argmax(dim=1) % 2 == 1).to(torch.long)).to(torch.float)))
    test_data.tensors = ((test_data.tensors[0]).to(torch.float),
                         (one_hot((test_data.tensors[1].argmax(dim=1) % 2 == 1).to(torch.long)).to(torch.float)))

    concept_names = ['isZero', 'isOne', 'isTwo', 'isThree', 'isFour', 'isFive', 'isSix', 'isSeven', 'isEight', 'isNine']
    return train_data, val_data, test_data, concept_names


def load_mnist(base_dir='./data'):
    train_data = pd.read_csv(os.path.join(base_dir, 'MNIST_C_to_Y/mnist.csv'))
    x = train_data.iloc[:, :-1].values
    y = train_data.iloc[:, -1].values
    concept_names = [f'feature{i:03}' for i in range(x.shape[1])]

    x = torch.FloatTensor(x)
    y = one_hot(torch.tensor(y).to(torch.long)).to(torch.float)
    return x, y, concept_names


def load_vector_mnist(base_dir='./data', to_one_hot: bool = True):
    train_ds = MNIST("../data/mnist", train=True, download=True, transform=ToTensor())
    test_ds = MNIST("../data/mnist", train=False, download=True, transform=ToTensor())
    concept_names = [f'is{i}' for i in range(10)]
    label_names = [f"is{i}" for i in range(20)]

    train_dataset = train_ds.train_data.unsqueeze(1).float()
    test_dataset = test_ds.test_data.unsqueeze(1).float()

    n_samples = len(train_ds.train_data) // 2
    x_train_img1 = train_dataset[:n_samples]
    x_train_img2 = train_dataset[n_samples:]
    c_train_img1 = train_ds.train_labels[:n_samples]
    c_train_img2 = train_ds.train_labels[n_samples:]
    y_train = train_ds.train_labels[:n_samples] + train_ds.train_labels[n_samples:]

    n_samples = len(test_ds.test_data) // 2
    x_test_img1 = test_dataset[:n_samples]
    x_test_img2 = test_dataset[n_samples:]
    c_test_img1 = test_ds.test_labels[:n_samples]
    c_test_img2 = test_ds.test_labels[n_samples:]
    y_test = test_ds.test_labels[:n_samples] + test_ds.test_labels[n_samples:]

    if to_one_hot:
        c_train_img1 = one_hot(c_train_img1).float()
        c_train_img2 = one_hot(c_train_img2).float()
        y_train = one_hot(y_train).float()
        c_test_img1 = one_hot(c_test_img1).float()
        c_test_img2 = one_hot(c_test_img2).float()
        y_test = one_hot(y_test).float()

    train_data = TensorDataset(x_train_img1, x_train_img2, c_train_img1, c_train_img2, y_train)
    test_data = TensorDataset(x_test_img1, x_test_img2, c_test_img1, c_test_img2, y_test)
    return train_data, test_data, concept_names, label_names

# def load_cub2(base_dir='./data'):
#     train_data = pd.read_csv(os.path.join(base_dir, 'CUB/cub200.csv'))
#     x = train_data.iloc[:, :-1].values
#     y = train_data.iloc[:, -1].values
#     concept_names = [f'feature{i:03}' for i in range(x.shape[1])]
#
#     x = torch.FloatTensor(x)
#     y = one_hot(torch.tensor(y).to(torch.long)).to(torch.float)
#     return x, y, concept_names


def load_cub(base_dir='./data'):
    train_data = pd.read_csv(os.path.join(base_dir, 'CUB/cub200.csv'))
    x = train_data.iloc[:, :-1].values
    y = train_data.iloc[:, -1].values
    concept_names = [f'feature{i:03}' for i in range(x.shape[1])]

    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    skf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for trainval_idx, test_idx in skf.split(x, y):
        x_trainval, y_trainval = x[trainval_idx], y[trainval_idx]
        for i, (train_idx, val_idx) in enumerate(skf2.split(x_trainval, y_trainval)):
            # if i <= 0:
            #     continue

            x_train, x_val, x_test = x_trainval[train_idx], x_trainval[val_idx], x[test_idx]
            y_train, y_val, y_test = y_trainval[train_idx], y_trainval[val_idx], y[test_idx]

            # print(np.unique(y_train, return_counts=True))
            # print(np.unique(y_val, return_counts=True))
            # print(np.unique(y_test, return_counts=True))
            # print(np.unique(y, return_counts=True))

            x_train = torch.FloatTensor(x_train)
            y_train = one_hot(torch.tensor(y_train).to(torch.long)).to(torch.float)
            x_val = torch.FloatTensor(x_val)
            y_val = one_hot(torch.tensor(y_val).to(torch.long)).to(torch.float)
            x_test = torch.FloatTensor(x_test)
            y_test = one_hot(torch.tensor(y_test).to(torch.long)).to(torch.float)

            # print(one_hot(torch.tensor(y[trainval_idx]).to(torch.long)).to(torch.float).shape)
            # # print(one_hot(torch.tensor(y[trainval_idx]).to(torch.long)).to(torch.float).sum(dim=0))
            # print(one_hot(torch.tensor(y).to(torch.long)).to(torch.float).sum(dim=0))
            # print(y_train.shape)
            # print(y_val.shape)
            # print(y_test.shape)
            # print(y_train.sum(dim=0))
            # print(y_val.sum(dim=0))
            # print(y_test.sum(dim=0))
            # print(np.unique(y, return_counts=True))

            train_data = TensorDataset(x_train, y_train)
            val_data = TensorDataset(x_val, y_val)
            test_data = TensorDataset(x_test, y_test)
            return train_data, val_data, test_data, concept_names



# Return the number of labels of each concept
def get_latent_sizes():
    return np.array([1, 3, 6, 40, 32, 32])


# See "primer on bases" above, to understand
def get_latent_bases():
    latents_sizes = get_latent_sizes()
    return np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                           np.array([1, ])))


# Convert a concept-based index (i.e. (color_id, shape_id, ..., pos_y_id)) into a single index
def latent_to_index(latents, latents_bases):
    return np.dot(latents, latents_bases).astype(int)



def load_dsprites(dataset_path='./data', c_filter_fn=lambda x: True, filtered_c_ids=np.arange(6), label_fn=lambda x: x[1],
                  train_test_split_flag=True, train_size=0.85):
    '''
    :param dataset_path:  path to the .npz dsprites file
    :param c_filter_fn: function returning True/False for whether to keep the concept combination or not
    :param filtered_c_ids: np array specifying which concepts to keep in the dataset
    :param label_fn: function taking in concept values, and returning an output task label
    :param train_test_split_flag: whether to perform a train-test split or not
    :param train_size: fraction of data to use for train
    :return:
    '''

    # Load dataset
    dataset_zip = np.load(os.path.join(dataset_path, 'dsprites/dsprites.npz'))
    imgs = dataset_zip['imgs']

    # Compute the index conversion scheme
    latent_sizes = get_latent_sizes()
    latents_bases = get_latent_bases()

    # Get all combinations of concept values
    latent_size_listss = [list(np.arange(i)) for i in latent_sizes]
    all_combs = np.array(list(itertools.product(*latent_size_listss)))

    # Compute which concept labels to filter out
    c_ids = np.array([c_filter_fn(i) for i in all_combs])
    c_ids = np.where(c_ids == True)[0]
    c_data = all_combs[c_ids]

    # Compute the class labels from concepts
    y_data = np.array([label_fn(i) for i in c_data])

    # Get corresponding ids of these combinations in the 'img' array
    img_indices = [latent_to_index(i, latents_bases) for i in c_data]

    # Select the corresponding imgs
    x_data = imgs[img_indices]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = x_data.astype(('float32'))

    # Filter out specified concepts, with their names
    names = ['color', 'shape', 'scale', 'rotation', 'x_pos', 'y_pos']
    c_names = [names[i] for i in filtered_c_ids]
    c_data = c_data[:, filtered_c_ids]

    # If no train/test split speficied - return data as-is
    if train_test_split_flag is False:
        return x_data, y_data, c_data, c_names

    random_state = 42
    x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(x_data, y_data, c_data, train_size=train_size,
                                                                         random_state=random_state)

    x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(x_train, y_train, c_train, test_size=0.33, random_state=42)
    x_train = np.transpose(x_train, [0, 3, 1, 2])
    x_val = np.transpose(x_val, [0, 3, 1, 2])
    x_test = np.transpose(x_test, [0, 3, 1, 2])
    # x_train = np.repeat(x_train, 3, axis=1)
    # x_val = np.repeat(x_val, 3, axis=1)
    # x_test = np.repeat(x_test, 3, axis=1)

    enc = OneHotEncoder(sparse=False)
    c_train_list, c_val_list, c_test_list = [], [], []
    for concept in range(c_train.shape[1]):
        c_train_list.append(enc.fit_transform(c_train[:, concept].reshape(-1, 1)))
        c_val_list.append(enc.fit_transform(c_val[:, concept].reshape(-1, 1)))
        c_test_list.append(enc.fit_transform(c_test[:, concept].reshape(-1, 1)))
    c_train = np.hstack(c_train_list)
    c_val = np.hstack(c_val_list)
    c_test = np.hstack(c_test_list)
    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_val = enc.fit_transform(y_val.reshape(-1, 1))
    y_test = enc.fit_transform(y_test.reshape(-1, 1))

    c_train = c_train[:, :9]
    c_val = c_val[:, :9]
    c_test = c_test[:, :9]

    print('x_train shape:', x_train.shape)
    print('c_train shape:', c_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_val', x_val.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(c_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(c_val), torch.FloatTensor(y_val))
    test_data = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(c_test), torch.FloatTensor(y_test))

    return train_data, val_data, test_data, c_names


def load_cub_full(root_dir='./CUB200'):
    concept_names = [str(s) for s in pd.read_csv(f'{root_dir}/attributes.txt', sep=' ', index_col=0, header=None).values.ravel()]
    class_names = [str(s) for s in pd.read_csv(f'{root_dir}/CUB_200_2011/classes.txt', sep=' ', index_col=0, header=None).values.ravel()]
    train_data = cub_loader.load_data(pkl_paths=[f'{root_dir}/train.pkl'], use_attr=True, no_img=False, batch_size=128, root_dir=root_dir)
    val_data = cub_loader.load_data(pkl_paths=[f'{root_dir}/val.pkl'], use_attr=True, no_img=False, batch_size=128, root_dir=root_dir)
    test_data = cub_loader.load_data(pkl_paths=[f'{root_dir}/test.pkl'], use_attr=True, no_img=False, batch_size=128, root_dir=root_dir)
    return train_data, val_data, test_data, concept_names, class_names


if __name__ == '__main__':
    train_data, val_data, test_data, concept_names, class_names = load_cub_full()
    # train_data, test_data, concept_names, label_names = load_vector_mnist('.')
    # train_data, val_data, test_data, c_names = load_dsprites('.')
    print('ok!')
    # x, y, c = load_celldiff('.')
