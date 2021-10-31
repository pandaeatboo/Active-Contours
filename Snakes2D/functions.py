# Script to hold all the helper functions that will be used for the 2D Snake

import numpy as np
from scipy.interpolate import interp2d

# compute the A matrix 
def computeA(n,alpha,beta):
    d1 = -2*np.ones(n)
    d2 = np.ones(n-1)
    A2 = np.diag(d1) + np.diag(d2,1) + np.diag(d2,-1)
    A2[0,n-1] = 1
    A2[n-1,0] = 1
    
    A4 = A2 @ A2
    
    return -alpha*A2 + beta*A4


# calculate euclidean distance between points
def distPoints(V):
    
    V = np.append(V,V[0,:][np.newaxis],axis=0)
    
    difference_squared =  np.square(V[0:-1,:] - V[1:,:])
    distance = np.sqrt(np.sum(difference_squared,axis=1))[np.newaxis]
    
    return distance.T

# interpolate the snake V, on the vector field G
# have to interpret matlab's interp2 using scipy interp2d
    
def interpSnake2(G,V):
    
    x_range = np.arange(1,G.shape[1]+1)
    y_range = np.arange(1,G.shape[0]+1)
    
    
    vert = interp2d(x_range, y_range, G[:,:,0])
    vertical = vert(V[:,0],V[:,1])
    
    horiz = interp2d(x_range, y_range,G[:,:,1])
    horizontal = horiz(V[:,0], V[:,1])
    print(vertical.shape)
    print(horizontal.shape)
    
    return 

def balloonForce():
    pass

def displayImageAndSnake():
    pass

def gradientCentered():
    pass

def polygonParity():
    pass

def selectPoints():
    pass

def smoothForces():
    pass

def splinesInterpolation2D():
    pass

def subdivision():
    pass

