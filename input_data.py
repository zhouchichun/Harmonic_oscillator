# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:45:46 2019

@author: 15250
"""
import random
import math
import numpy as np
data_set_t=[i*1.0/50 for i in range(0,500)]
data_set_x=[i*1.0/50 for i in range(-500,500)]
data_t_x=[]
for t in data_set_t:
    for x in data_set_x:
        data_t_x.append([t,x])
def generate(size=500):
    #data_t_x=[[t,x] for t,x in zip(random.sample(data_set_t,size),random.sample(data_set_x,size))]
    data_t_x_r=random.sample(data_t_x,size)
    data_x=[[x] for t,x in data_t_x_r]
    
    #y_data=[[math.sin(3.1415926*x[0]**3+3.1415*x[1]+0.3)] for x in x_data]
    return data_t_x_r,data_x
