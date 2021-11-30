# Script to hold all the helper functions that will be used for the 2D Snake
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp2d, griddata
from scipy.ndimage import map_coordinates
from scipy.sparse import spdiags
import math
from scipy.io import loadmat

import matlab.engine
eng = matlab.engine.start_matlab()
#%%
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
    V = np.vstack([V,V[0,:]])
    
    difference_squared =  np.square(V[0:-1,:] - V[1:,:])
    distance = np.sqrt(np.sum(difference_squared,axis=1)).reshape(-1,1)
    
    return distance

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
   
    #x_range = np.arange(1,G.shape[0]+1)
    #y_range = np.arange(1,G.shape[1]+1)
    
    #[X,Y] = np.meshgrid(x_range,y_range) 
    #vertical = griddata((X.ravel(),Y.ravel()), G[:,:,0].ravel(),(V[:,0], V[:,1]), method="linear", fill_value=np.nan)
    #horizontal = griddata((X.ravel(),Y.ravel()), G[:,:,1].ravel(), (V[:,0], V[:,1]), method="linear", fill_value=np.nan)
    #vertical = vertical.reshape(-1,1)
    #horizontal = horizontal.reshape(-1,1)
    
    #vert = interp2d(x_range,y_range,G[:,:,0],kind="linear",fill_value=np.NaN)
    #vertical = np.matrix(vert(V[:,0],V[:,1]))
    #vertical = np.diag(vertical).reshape(-1,1)# trying to see if taking diagonals work
    #horiz = interp2d(x_range,y_range,G[:,:,1],kind="linear",fill_value=np.NaN)
    #horizontal = np.matrix(horiz(V[:,0],V[:,1]))
    #horizontal = np.diag(horizontal).reshape(-1,1)
    
    
    ## RECT BIVARIATE SPLINE DOES NOT WORK 
    vert = RectBivariateSpline(y_range,x_range,G[:,:,0])
    vertical = vert(V[:,0],V[:,1],grid=False).reshape(-1,1)
    horiz = RectBivariateSpline(y_range,x_range,G[:,:,1])
    horizontal = horiz(V[:,0],V[:,1],grid=False).reshape(-1,1)
    
    #using the map_coordinates -NOT linear interpolation though, it's cubic spline 
    #vertical = map_coordinates(G[:,:,0], [V[:,0].ravel(), V[:,1].ravel()], order=1, mode='constant').reshape(-1,1)
    #horizontal = map_coordinates(G[:,:,1], [V[:,0].ravel(), V[:,1].ravel()], order=1, mode='constant').reshape(-1,1)
    #combined = np.concatenate((horizontal,vertical),axis=1)
    
    #print("This is the horizontal and vertical shape of interpSnake2")
    #print(horizontal.shape)
    #print(vertical.shape)
    #print(combined.shape)
    #combined = np.concatenate((horizontal.reshape(-1,1),vertical.reshape(-1,1)),axis=1)
    combined = np.concatenate((horizontal,vertical),axis=1)
    
    return combined # only return np.matrix(combined) if it's not already a matrix


## CURRENT WORKAROUND IS TO JUST CALL THE MATLAB FUNCTION
'''
def interpSnake2(G,V):
    mat_G = matlab.double(G.tolist())
    mat_V = matlab.double(V.tolist())
    s = eng.interpSnake2(mat_G,mat_V,nargout=1)
    return np.array(s)
'''   
# Cathy's 
# CROSS CHECKED -- THIS IS CORRECT!
#balloon_data = loadmat(r"C:\Users\EricQ\ECE588 Project\snakes\snakes\release2D\balloonForce checker data.mat")
#B = balloon_data["B"]
#V_balloon = balloon_data["V"]
def balloonForce(V):
  n=np.shape(V)[0]

  # Normal operator for polygon.
  A = np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1);
  A[n-1, 0] = 1
  A[0, n-1] = -1
  
  # operator application. 
  B = A @ V @ np.matrix('0 -1; 1 0')
  
  #  Normalization of normal vectors. 
  normb = np.sqrt(np.sum(np.power(B,2),axis=1));
  B = B / (normb @ np.matrix('1 1')) # Equation (22) in the paper
  return B

# Matthew's
def displayImageAndSnake(I, V):
  
    # displays the image and the polygon the user drew

    # expects an image that is 2D, map to grayscale
    plt.figure(figsize=(15,15))
    plt.imshow(np.uint8(I), cmap='gray')
    # we expect V is a polygon with first column containing x, second column containing y
    # x values of the snake: add the first coordinate to the end to close the snake
    # note column is extracted as a row vector so we use hstack
    xvals = np.hstack((V[:,0], V[0,0]))
    # y values are similar
    yvals = np.hstack((V[:,1], V[0,1]))
    plt.axis(False)
    plt.plot(xvals, yvals, 'k-', linewidth=2)

# Matthew's
def gradientCentered(I):

  # Computes the gradient of I with a centered scheme and symmetrical boundary conditions
  
  # Initialize
  # creates array of zeros to hold the gradient with dimensions corresponding to image width x height x 2
  G = np.zeros((np.size(I, 0), np.size(I, 1), 2))
  # not sure what two lines below do
  G[1:-1,:,0] = 0.5*(I[2:,:] - I[0:-2,:])
  G[:,1:-1,1] = 0.5*(I[:,2:] - I[:,0:-2])

  # equation 25 from paper
  G[0,:,0] = 0.5*(I[1,:] - I[0,:])
  G[-1,:,0] = 0.5*(I[-1,:] - I[-2,:])

  # equation 26 from paper
  G[:,0,1] = 0.5*(I[:,1] - I[:,1])
  G[:,-1,1] = 0.5*(I[:,-1] - I[:,-2])
  
  return G

# Matthew's
def polygonParity(V):
    
    # Let V be a polygon.  polygonParity computes an integer which value is 
    # equal to 1 if the interior of the polygon is right-handed when the points
    # are browsed in the regular order. 
    # -1 if the interior of the polygon is left-handed when the points are
    # browsed in the regular order. 
    
    # basically: returns 1 if the points are ordered ccw, -1 if they are ordered cw
    
    vn1 = np.hstack(((V[-1,0] - V[0,0]), (V[-1,1] - V[0,1]))) / \
        (np.sqrt((V[-1,0] - V[0,0])**2 + (V[-1,1] - V[0,1])**2))
        
    vn2 =  np.hstack(((V[0,0] - V[1,0]), (V[0,1] - V[1,1]))) / \
        (np.sqrt((V[0,0] - V[1,0])**2 + (V[0,1] - V[1,1])**2))
    
    vb = np.hstack(((vn1[0] + vn2[0]), (vn1[1] + vn2[1])))
    
    a_eb = vb[0]
    b_eb = vb[1]
    c_eb = (vb[0] * V[0,0]) + (vb[1] * V[0, 1])
    
    c = 0
    
    for i in range(1, np.size(V, 1)-1):
        denom = a_eb * V[i,0] - a_eb * V[i+1,0] + b_eb * V[i,1] - b_eb * V[i+1,1]
        if denom != 0:
            t = (c_eb - a_eb * V[i+1,0] - b_eb * V[i+1,1]) / denom
            if t >= 0 and t < 1:
                vt = np.hstack(((t * V[i,0] + (1-t) * V[i+1,0] - V[0,0]), (t * V[i,1] + (1-t) * V[i+1,1] - V[0,1])))
                nt = np.hstack((vt[1], -1*vt[0]))
                scal = nt @ vb.conj().T
                if scal > 0:
                    c += 1
    
    parity = 0
    if (c % 2) == 0:
        parity = 1
    else: 
        parity = -1
            
    return parity

# Matthew's 
def selectPoints(I): 

    # select initialization points to create the polygon
    # returns the polygon
    
    # [x,y] are the coordinates the user selects manually on the image
    # left click to choose a point
    # right click to stop choosing points
    x = []
    y = []
    np.disp('Left click to select a position')
    np.disp('Right click to stop')
    plt.clf()
    button = 1
    
    while button == 1:
        # image must be 2-D (use cv2.imread() in greyscale mode)
        plt.imshow(np.uint8(I), cmap='gray')
        plt.axis(False)
        # times out after 5 seconds - once the user right clicks and waits 5s, it times out
        # coords will be an empty list if it times out
        # if the user clicks, coords is a list with the tuple of the coordinates clicked
        coords = plt.ginput(1, timeout=5)
        # check if the user added a point (the coord list has entries)
        if bool(coords):
            # add the point's coordinates to the polygon
            x.append(coords[0][0])
            y.append(coords[0][1])
        # plot the points the user has selected so far
        plt.plot(x, y, 'r-', linewidth=3)
        plt.plot(x, y, 'b+', linewidth=3)
        # if the user didn't left click, no coordinates, done picking
        if not bool(coords):
            break
    plt.close()
    
    # create the polygon: first column is the x coords, second column is the y coords
    x = np.array(x)[np.newaxis].T
    y = np.array(y)[np.newaxis].T
    V = np.hstack((x,y))
    
    # if the user didn't create a right-handed polygon, switch the order of the points to make it right handed
    if polygonParity(V) != 1:
        V = np.flipud(V)
    return V

# Cathy's
smooth_data = loadmat(r"C:\Users\EricQ\ECE588 Project\snakes\snakes\release2D\smoothForce checkerdata.mat")
f = smooth_data["B1"]
sig = smooth_data["sb"]
smooth_B = smooth_data["B"]
def smoothForces(f, sig):
  n=np.shape(f)[0]
  l2 = np.maximum(1,int(np.floor(n/2)))
  t = [i for i in range(l2)]
  t.extend([i for i in range(-(n-l2),0)])
  P = np.exp(np.divide(-1*np.power(t,2),abs(sig)+1e-6))
  P = np.divide(P, np.sum(P))
  fc1 = np.real(np.fft.ifft( np.multiply(np.fft.fft(f[:,0].T),np.fft.fft(P[:]))))
  fc2 = np.real(np.fft.ifft( np.multiply(np.fft.fft(f[:,1].T),np.fft.fft(P[:]))))
  fc = np.column_stack((fc1.T, fc2.T))
  return fc
'''
def smoothForces(f,sig):
    mat_f = matlab.double(f.tolist())
    mat_sig = sig
    result = eng.smoothForces(mat_f,mat_sig,nargout=1)
    return np.array(result)
'''

# Cathy's
def splinesInterpolation2D(V, NbPoints):
  N2 = NbPoints  
  N=np.shape(V)[0]
  W = spdiags([np.ones((N,)), -2*np.ones((N,)), np.ones((N,))],
                np.array([-1,0,1]),
                N,
                N)

  W = np.matrix(W.toarray())
  W[0,N-1] = 1
  W[N-1,0] = 1
  VV = (W*V).T
  d = np.power(N,2)*VV
  d = d.T

  M = spdiags([2/3*np.ones((N,)), 1/6*np.ones((N,)), 1/6*np.ones((N,)), 1/6*np.ones((N,)), 1/6*np.ones((N,))],
                np.array([0, 1, -1, N - 1, -(N - 1)]),
                N,
                N)


  x = np.linalg.lstsq(M.todense(),d[:,0],rcond=None)[0] # Added rcond argument
  y = np.linalg.lstsq(M.todense(),d[:,1],rcond=None)[0]
  X = np.column_stack((x,y))
  L = np.zeros(N,)
  NT = 100*N

  for i in range(N):
    A = np.matrix(np.linspace(0, 1 ,100))
    A = A.T
    B = 1 - A
    der  =  N*(V[np.mod(i+1,N),:] - V[i,:]) + (-(3*np.multiply(A,A)-1)* X[i,:] + (3*np.multiply(B,B)-1)* X[np.mod(i+1,N),:])/(6*N)
    L[i] = np.sum(np.sqrt(np.sum(np.multiply(der,der),1))/NT)


  PT = np.zeros((N2,2))
  a = np.matrix(np.linspace(0, 1, N2)).T
  L = np.cumsum(L).T
  L = L/L[-1]
  L = np.insert(L, 0, 0, axis=0)

  for i in range(N):
    condi = np.logical_or(np.logical_and(a >= L[i], a < L[i+1]) , np.logical_and(i == N-1, a >= L[N-1]) , np.logical_and(i == 0, a < L[i]))
    A = (np.multiply(a,condi) - L[i+1]) / (L[i] - L[i+1])
    A = (a[condi].T - L[i+1]) / (L[i] - L[i+1])
    B = 1 - A
    C = 1/6 * (np.power(A,3) - A) * (i/N - (i-1)/N)**2 # check from here on out
    D = 1/6 * (np.power(B,3) - B) * (i/N - (i-1)/N)**2
    xy = A*V[i,:] + B*V[np.mod(i+1,N),:] + C*X[i,:] + D*X[np.mod(i+1,N),:]

    xy_index = 0
    for i in range(N2):
      if condi[i] == True:
        PT[i, :] = xy[xy_index]
        xy_index=xy_index+1

  return PT

# Cathy's 
def subdivision(V0,k):
  x=V0[:,0]
  y=V0[:,1]
  N=len(x); 
  p=1
  xi = []
  yi = []
  for i in range(N-1):
    # subdivision between two vertices of the snake in x and y.
    long_num=float(np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2))
    nbre=max(1, (round(long_num/k)))
    h_x= float((x[i+1]-x[i])/nbre)
    h_y=float((y[i+1]-y[i])/nbre)

    for j in range(nbre):
      j2 = j+1
      xi.append(float(x[i]+j2*h_x))
      yi.append(float(y[i]+j2*h_y))

  #  subdivision between the first and the last vertex of the snake.
  long=float(np.sqrt((x[N-1]-x[0])**2+(y[N-1]-y[0])**2))
  nbre=math.floor(long/k);
  h_x=float(( x[0]-x[N-1] )/nbre)
  h_y=float(( y[0]-y[N-1] )/nbre)

  for j in range(nbre):
    j2 = j+1
    xi.append(float(x[N-1]+j2*h_x))
    yi.append(float(y[N-1]+j2*h_y))

  V = np.column_stack((xi[:],yi[:]))
  return V

