# Script to hold all the helper functions that will be used for the 2D Snake

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates 

from scipy.io import loadmat

#compute the A matrix
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
    

np.random.seed(10)
#V = np.random.rand(224,2)
#G = np.random.rand(423,417,2)
data = loadmat(r"C:\Users\EricQ\ECE588 Project\snakes\snakes\release2D\interp2 checker data.mat")

G = data["nabla_P"]
V = data["V"]
S = data["GV"]
    
def interpSnake2(G,V): 
    
    x_range = np.arange(1,G.shape[1]+1)
    y_range = np.arange(1,G.shape[0]+1)
   
    #[X,Y] = np.meshgrid(y_range,x_range) 
    #vertical = griddata((Y.ravel(),X.ravel()), G[:,:,0].ravel(),(V[:,0], V[:,1]), method="linear", fill_value=np.nan)
    #horizontal = griddata((Y.ravel(),X.ravel()), G[:,:,1].ravel(), (V[:,0], V[:,1]), method="linear", fill_value=np.nan)
    
    #vert = interp2d(x_range,y_range,G[:,:,0],fill_value=np.NaN)
    #vertical = vert(V[:,0],V[:,1])
    #horiz = interp2d(x_range,y_range,G[:,:,1],fill_value=np.NaN)
    #horizontal = horiz(V[:,0],V[:,1])
    #print(vertical.shape)
    #print(horizontal.shape)
    
    ## RECT BIVARIATE SPLINE DOES NOT WORK 
    vert = RectBivariateSpline(y_range,x_range,G[:,:,0])
    vertical = vert(V[:,0],V[:,1],grid=False)[np.newaxis].T
    horiz = RectBivariateSpline(y_range,x_range,G[:,:,1])
    horizontal = horiz(V[:,0],V[:,1],grid=False)[np.newaxis].T
    
    # using the map_coordinates -NOT linear interpolation though, it's cubic spline 
    #vertical = map_coordinates(G[:,:,0], [V[:,0].ravel(), V[:,1].ravel()], order=1, mode='constant')[np.newaxis]
    #horizontal = map_coordinates(G[:,:,1], [V[:,0].ravel(), V[:,1].ravel()], order=1, mode='constant')[np.newaxis]
    #combined = np.concatenate((vertical.T,horizontal.T),axis=1)
    
    combined = np.concatenate((horizontal,vertical),axis=1)
    
    return combined

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

