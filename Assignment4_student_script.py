# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pf
#from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import scipy as sp
from sklearn.metrics.pairwise import rbf_kernel as RBF
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


#Task1
#1
#Data Generate Functions 
def create_toy_data(f, sample_size, noise_var):
    x = np.linspace(0, 1, sample_size)
    t = f(x) + np.random.normal(scale=noise_var, size=x.shape)
    return x, t

def f(x):
    return np.sin(2 * np.pi * x)

#generate 10 points for training, 200 points for testing.


#2
#Apply polynomial basis function (order M=9)


#3train model in parametric way and report test MSE




#4
#Get prediction in non-parametric way
#define your gram matrix and k vector
K=
k=
#predict and report test MSE


#5 
#
#

#Task2
#1 Gram matrix
gamma=5
K=

#2 Covariance matrix beta=10 
C=
#check whether C is invertable or not.


#3 



#Task 3
#1

#2


#2.1 

#2.2 How many support vectors in total

#2.3Check whether the 18-th data sample is a support vector.

#2.4How many support vectors from class 2?

#3


