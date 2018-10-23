# -*- coding: utf-8 -*-
"""Helper functions for loading real-world data"""
# Author: Xiangyu Guo     xiangyug[at]buffalo.edu
#         Shi Li          shil[at]buffalo.edu

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from scipy.io import loadmat
import os

DATA_DIR = os.path.join(os.path.expanduser('~'), 'exp_data/clustering')


def _get_parkinsons_data():
    PARKINSONS_DATA_DIR = 'parkinsons'

    full_data_path = os.path.join(DATA_DIR, PARKINSONS_DATA_DIR)
    file_name = 'Tsanas_TBME2010_data.mat'
    data = loadmat(os.path.join(full_data_path, file_name))

    return data['Features_only']


def _get_higgs_data():
    HIGGS_DATA_DIR = 'higgs'

    full_data_path = os.path.join(DATA_DIR, HIGGS_DATA_DIR)
    file_name = 'higgs.npy'
    data = np.load(os.path.join(full_data_path, file_name))

    return data


def _get_gas_data():
    GAS_DATA_DIR = 'gas'

    full_data_path = os.path.join(DATA_DIR, GAS_DATA_DIR)
    file_name = 'gas.npy'
    data = np.load(os.path.join(full_data_path, file_name))

    return data


def _get_power_data():
    POWER_DATA_DIR = 'power'

    full_data_path = os.path.join(DATA_DIR, POWER_DATA_DIR)
    file_name = 'household_power_consumption.npy'
    data = np.load(os.path.join(full_data_path, file_name))

    return data


def _get_covertype_data():
    covtype = fetch_covtype()
    # only use the ten quantitative features
    return covtype.data[:, 0:10]


def _get_skin_data():
    SKIN_DATA_DIR = 'skin'

    full_data_path = os.path.join(DATA_DIR, SKIN_DATA_DIR)
    file_name = 'skin.npz'
    data = np.load(os.path.join(full_data_path, file_name))

    return data['features']


def _get_npz_data(name):
    NPZ_DATA_DIR = 'npz_data'

    full_data_path = os.path.join(DATA_DIR, NPZ_DATA_DIR)
    data = np.load(os.path.join(full_data_path, name))
    return data['features']


DATASETS = {
    'spambase': lambda **kwargs: _get_npz_data(name='spambase.npz', **kwargs),
    'letter': lambda **kwargs: _get_npz_data(name='letter.npz', **kwargs),
    'pendigits': lambda **kwargs: _get_npz_data(name='pendigits.npz', **kwargs),
    'parkinsons': _get_parkinsons_data,
    'higgs': _get_higgs_data,
    'gas': _get_gas_data,
    'power': _get_power_data,
    'covertype': _get_covertype_data,
    'skin': _get_skin_data,
}


def get_realworld_data(dataset, **kwargs):
    """

    :param dataset:
    :param kwargs:
    :return:
    """
    if dataset in DATASETS:
        return DATASETS[dataset](**kwargs)
    else:
        raise NotImplementedError
