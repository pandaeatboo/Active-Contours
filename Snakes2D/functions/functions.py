# Script to hold all the helper functions that will be used for the 2D Snake

import numpy as np
import scipy
import pandas as pd

def computeA(n,alpha,beta):
    d1 = -2*np.ones(n)
    d2 = np.ones(n-1)
    A2 = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)
    A2[0,n-1] = 1
    A2[n-1,0] = 1
    
    A4 = A2 @ A2
    
    return -alpha*A2 + beta*A4


def distPoints(V):
    
    V = np.append(V,V[0,:][np.newaxis],axis=0)
    
    difference_squared =  np.square(V[0:-1,:] - V[1:,:])
    distance = np.sqrt(np.sum(difference_squared,axis=1))[np.newaxis]
    
    return distance.T

def interpSnake2(G,V):
    pass