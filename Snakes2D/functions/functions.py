# Script to hold all the helper functions that will be used for the 2D Snake

import numpy as np
import scipy
import pandas as pd

def computeA(n,alpha,beta):
    d1 = -2*np.ones(n)
    d2 = np.ones(n-1)
    A2 = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)
    A2[1,n] = 1
    A2[n,1] = 1
    
    A4 = A2 ** 2
    
    return -alpha*A2 + beta*A4
    