#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import sys

import pandas as pd
from itertools import product
from sklearn.preprocessing import scale
import numpy as np
from keras.utils import to_categorical

# 加载数据
def get_data(filename):
    sequence = []
    arrsequence = []

    ids = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            elif '>' in lines:
                id = lines.split(" ")[0]
                ids.append(id)
                seq = file_to_read.readline()
                sequence.append(seq.replace(' ', '').replace('\n', '').replace('\r', '').upper())
                arrsequence.append([seq.replace(' ', '').replace('\n', '').replace('\r', '').upper()])

            else:
                print(lines)
    return ids, sequence, arrsequence

def get_Mono_mer(seqs):
    X1 = np.empty((len(seqs), len(seqs[0])))
    alphabet = 'ATGC'
    for i in range(len(seqs)):
        for j in range(len(seqs[0])):
            X1[i, j] = next((k for k, letter in enumerate(alphabet) if letter == seqs[i][j]))
    X1 = to_categorical(X1)
    return X1

def get_Tri_mer(seqs):
    lookup_table = []
    for p in product('ATGC', repeat=3):
        w = ''.join(p)
        lookup_table.append(w)
    X2 = np.empty((len(seqs), len(seqs[0]) - 2))
    for i in range(len(seqs)):
        for j in range(len(seqs[0]) - 2):
            w = seqs[i][j:j + 3]
            X2[i, j] = lookup_table.index(w)
    X2 = to_categorical(X2)
    return X2

def get_SP_Di_Nucleotide(seqs):
    di_prop = pd.read_csv('./dataset/DNA_Di_Prop.txt')
    di_prop = di_prop.iloc[:, 1:]
    scaled_di_prop = scale(di_prop, axis=1)
    di_cols = di_prop.columns.tolist()
    di_prop = pd.DataFrame(scaled_di_prop, columns=di_cols)
    pp_di = {}
    for i in range(16):
        key = di_prop.columns[i]
        items = di_prop.iloc[:, i].tolist()
        pp_di[key] = items

    X3 = np.empty([len(seqs), 80, 90], dtype=float)
    for i in range(len(seqs)):
        for j in range(80):
            word = seqs[i][j:j + 2]
            value = pp_di[word]
            for k in range(90):
                X3[i, j, k] = value[k]
    return X3

def get_SP_Tri_nucleotide(seqs):
    tri_prop = pd.read_csv('./dataset/DNA_Tri_Prop.txt')
    tri_prop = tri_prop.iloc[:, 1:]
    scaled_tri_prop = scale(tri_prop, axis=1)  # Standardization
    tri_cols = tri_prop.columns.tolist()
    tri_prop = pd.DataFrame(scaled_tri_prop, columns=tri_cols)
    pp_tri = {}
    for i in range(64):
        key = tri_prop.columns[i]
        items = tri_prop.iloc[:, i].tolist()
        pp_tri[key] = items

    X4 = np.empty([len(seqs), 79, 12], dtype=float)
    for i in range(len(seqs)):
        for j in range(79):
            word = seqs[i][j:j + 3]
            value = pp_tri[word]
            for k in range(12):
                X4[i, j, k] = value[k]
    return X4