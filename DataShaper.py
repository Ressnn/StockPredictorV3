# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:11:44 2018

@author: Pranav Devarinti
"""

import sklearn as sk
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd

def Reshape1(dataset,target_size):
    dataset = np.array(dataset)
    target_list = []
    for i in range(0,target_size):
        target_list.append(target_size)
    dataset = np.array(dataset)
    nl = []
    for i in dataset:
        nl.append(i)
        nl.append(i)
    dataset = np.array(nl)
    dataset = np.array(dataset)
    while np.array(dataset).shape[0] < target_size:
        nl = []
        for i in dataset:
            nl.append(i)
            nl.append(i)
        dataset = np.array(nl)
        dataset = np.array(dataset)
        print(dataset.shape)
    dataset = nl
    n2dl = []
    target_size = np.array(target_size)
    n2d = (np.array(dataset).shape[0]-target_size)
    scale_fac = np.array(dataset).shape[0]/(np.array(dataset).shape[0]-target_size)
    for i in range(0,n2d):
        n2dl.append(int(i*scale_fac))
    print(np.array(n2dl).shape)
    dl = []
    for i in range(0,np.array(dataset).shape[0]):
        if i not in n2dl:
            dl.append(dataset[i])

    return np.array(dl)


def Reshape2(dataset,targetsize):
    new_list = []
    for i in dataset:
        new_list.append(Reshape1(i,target_size=targetsize))
        print('One Done')
    return new_list

def ProduceDiff(final):
    diffedfinal = []
    for i in final:
        diffedfinal.append(pd.DataFrame(i).transpose().diff().transpose().fillna(0))
    return diffedfinal

def Makebatches(dataset,number_of_splits):
    new_dataset = []
    for i in dataset:
        i = pd.DataFrame(i).transpose()
        new_dataset.append(np.vsplit(i,number_of_splits))
    return new_dataset
