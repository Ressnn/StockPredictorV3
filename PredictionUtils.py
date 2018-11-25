# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:28:13 2018

@author: ASUS
"""


def UndoDiff(nl):
    lval = 0
    return_to_user = []
    for i in nl:
        lval = lval+i
        return_to_user.append(lval)
    return return_to_user
