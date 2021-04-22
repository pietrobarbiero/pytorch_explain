import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset


def load_mimic():
    data = pd.read_csv('./data/mimic-ii/full_cohort_data.csv')
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
    # f2.append('hospital_los_day')
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
    datay = data['hosp_exp_flg'].values

    # model = DecisionTreeClassifier(max_depth=3)
    # # model = RandomForestClassifier()
    # scores = cross_val_score(model, datax, datay, cv=10)
    # print(scores.mean())

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

    # scores = cross_val_score(LogisticRegression(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())
    # scores = cross_val_score(DecisionTreeClassifier(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())
    # scores = cross_val_score(RandomForestClassifier(), data_mid.values, data_high.values, cv=10)
    # print(scores.mean())

    c0 = torch.FloatTensor(data_low.values)
    x = torch.FloatTensor(data_mid.values)
    y = one_hot(torch.tensor(data_high.values).to(torch.long)).to(torch.float)
    return x, y, data_mid.columns


if __name__ == '__main__':
    x, y, c = load_vDem('.')
    # x, y, c = load_celldiff('.')
