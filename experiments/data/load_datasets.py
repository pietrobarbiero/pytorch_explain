import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
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
    datay = data['day_28_flg'].values

    x = torch.FloatTensor(datax)
    y = torch.LongTensor(datay)
    return x, y, features
