#!/usr/bin/env python
import os
import re
import sys
import h5py
import time
import random
import string
import contextlib
import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
from model import ae

identifier = '(([a-zA-Z]_)?([a-zA-Z0-9_]*))'
replacement_field = '{' + identifier + '}'


def reset():
    tf.reset_default_graph()
    random.seed(19)
    np.random.seed(19)
    tf.set_random_seed(19)


def load_phenotypes(pheno_path):
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']

    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v)-1)
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['MEAN_FD'] = pheno['func_mean_fd']
    pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)
    pheno["STRAT"] = pheno[["SITE_ID", "DX_GROUP"]].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID', 'MEAN_FD', 'SUB_IN_SMP', 'STRAT']]


def hdf5_handler(filename, mode="r"):
    #h5py.File(filename, "a").close()
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[1] = 0
    settings[2] = 0
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(filename, fapl=propfaid)) as fid:
        return h5py.File(fid, mode)


def load_fold(patients, experiment, fold):

    derivative = experiment.attrs["derivative"]

    X_train = []
    y_train = []
    temp=experiment[fold]["train"][:]
    train=[]
    for i in temp:
        train.append(str(str(i).split("'")[1]))
    for pid in train:
        X_train.append(np.array(patients[pid][derivative]))
        y_train.append(patients[pid].attrs["y"])

    X_valid = []
    y_valid = []
    temp=experiment[fold]["valid"][:]
    valid=[]
    for i in temp:
        valid.append(str(str(i).split("'")[1]))
    for pid in valid:
        X_valid.append(np.array(patients[pid][derivative]))
        y_valid.append(patients[pid].attrs["y"])

    X_test = []
    y_test = []
    temp=experiment[fold]["test"][:]
    test=[]
    for i in temp:
        test.append(str(str(i).split("'")[1]))
    for pid in test:
        X_test.append(np.array(patients[pid][derivative]))
        y_test.append(patients[pid].attrs["y"])

    return np.array(X_train), np.array(y_train), \
           np.array(X_valid), np.array(y_valid), \
           np.array(X_test), np.array(y_test)


def sparsity_penalty(x, p, coeff):
    p_hat = tf.reduce_mean(tf.abs(x), 0)
    kl = p * tf.log(p / p_hat) + \
        (1 - p) * tf.log((1 - p) / (1 - p_hat))
    return coeff * tf.reduce_sum(kl)
