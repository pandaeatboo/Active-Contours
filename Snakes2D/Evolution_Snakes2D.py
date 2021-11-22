# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 00:01:30 2021

@author: EricQ
"""

import time
import math
from PIL import Image
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.sparse import eye
import cv2
from functions import *

# Instead of the I_init and V_init used in the MATLAB version. Here we just ask the 
# user to input the original image, as well as how many subdivisions they want.

class SnakesEvolution():
    def __init__(self,I_init, V_init, Alpha=1750, Beta=500, Sigma=1, Gamma=5e-5, BalloonCoefficient=-90,\
                 BalloonSmoothing=1, MaxIteration=1e4, CubicSplineRefinement=False, Verbose=False,\
                 MaxDistance=2, RefinementFactor=1.1, TimeOut=math.inf):
        
        self.I_init = I_init
        self.V_init = V_init 
        self.Alpha = Alpha
        self.Beta = Beta
        self.Sigma = Sigma
        self.Gamma = Gamma
        self.BalloonCoefficient = BalloonCoefficient
        self.BalloonSmoothing = BalloonSmoothing
        self.MaxIteration = MaxIteration
        self.CubicSplineRefinement = CubicSplineRefinement
        self.Verbose = Verbose
        self.MaxDistance = MaxDistance
        self.RefinementFactor = RefinementFactor 
        self.TimeOut = TimeOut
        
    def forward(self):
        # timing stopwatch
        t = time.time()
        
        verbose = self.Verbose == "on" or self.Verbose == True
        cubic_spline_refinement = self.CubicSplineRefinement == "on" or self.CubicSplineRefinement == True
        
        if verbose:
            plt.figure()
        
        # there is no GNU OCTAVE compatability for Python, so we skip the conditional
        I = cv2.GaussianBlur(self.I_init, ksize=(0,0), sigmaX=self.Sigma, borderType=cv2.BORDER_REPLICATE ) 
        V = self.V_init
        N = V.shape[0]
        
        A = computeA(N, self.Alpha, self.Beta)
        A = self.Gamma * A + eye(N)
        igpi = inv(A) # Equation (24) in the paper (first part)
        
        # gradient computation
        Grad = gradientCentered(I)
        Potential = np.sum(np.square(Grad),2)
        nabla_P = gradientCentered(Potential)
        
        # stopping criteria
        n = 1
        eps = 1
        
        condi = (np.ones((N,1)) == 1).astype(int) # This might be an issue for us 
        
        elapsedtime = time.time() - t
        iteration = 0
        while ((eps>9e-2) and (n <= self.MaxIteration) and (elapsedtime < self.TimeOut )):
            iteration+=1
            #print(f"This is iteration number:{iteration}\n\n\n")
            
            B = balloonForce(V)
            B = smoothForces(B,self.BalloonSmoothing)
            B = np.matrix(B)
            #print(B.shape)
            GV = interpSnake2(nabla_P, np.asarray(V))
            #print(GV.shape)
            
            # stop the balloon force
            #print("Going through all the different shapes")
            #print(type(GV))
            #print(type(B))
            
            #print((np.square(GV)).shape)
            #print((np.square(B)).shape)
            #print((np.sum(np.square(GV),1)).shape)
            #print((np.sum(np.square(B),1)).shape)
            
            
            nrmi = np.sum(np.square(GV),1)
            nrmb = np.sum(np.square(B),1)

            #print(nrmb.shape)
            #print(nrmi.shape)
            #print(condi.shape)
            #print((nrmi < (self.BalloonCoefficient**2*nrmb)).shape)
            #print((np.matrix((nrmi < (self.BalloonCoefficient**2*nrmb))).T).shape)
            condi = condi & ((nrmi < (self.BalloonCoefficient**2*nrmb)))
            #print(condi.shape)
            #print(B.shape)
            #print(GV.shape)
            #print(V.shape)
            
            '''
            if iteration > 1:
                print("This is the shape of V1 and then V")
                print(V1.shape)
                print(V.shape)
            '''
            # iteration
            #print(igpi.shape)
            #print((V + (GV+self.BalloonCoefficient*np.multiply(condi*np.array([1,1]),B)*self.Gamma)).shape)
            
            #print((condi*np.array([1,1])).shape)
            V1 = np.matmul(igpi,(V + (GV+self.BalloonCoefficient*np.multiply(condi*np.array([1,1]),B)\
                                        *self.Gamma)))
            #print("This is the shape of V1 and then V")
            #print(V1.shape)
            #print(V.shape)
            
            # display
            if n % 20 == 0:
                if (verbose):
                    displayImageAndSnake(I_init, V)
                    
            if (n % 100 == 0) and cubic_spline_refinement:
                dv = distPoints(V1)
                if (np.max(dv) > self.MaxDistance):
                    V1 = splinesInterpolation2D(V1, math.floor(V1.shape[0]*self.RefinementFactor))
                    #print("This is the shape of V1 after splinesInterpolation2D")
                    #print(V1.shape)
                    N2 = V1.shape[0]
                    # computation of the regularization matrix
                    A = computeA(N2,self.Alpha,self.Beta)
                    A = self.Gamma * A + eye(N2)
                    igpi = inv(A)
                    # computation of the potential
                    nabla_P = gradientCentered(Potential)
                    condi = np.ones((N2,1)) == 1


            # computation of the stopping criteria
            n = n + 1
            if (V.shape[0] == V1.shape[0]):
                #print(V.flatten()[np.newaxis].T.shape)
                #print(V1.flatten().T.shape)
                eps = np.sum(np.power((V.flatten()-V1.flatten().squeeze()),2))
                #print(f"This is eps: {eps}")
            else:
                eps = 1e5
            
            #print("This is the shape of V1 and then V")
            #print(V1.shape)
            #print(V.shape)
            V = V1
        
        elapsedtime = time.time() - t
        if (n == self.MaxIteration+1): 
            print(f"Maximum Iterations Number ({self.MaxIteration}) reached. Time(s): {elapsedtime}s. Algorithm may have not converged.")
        elif (elapsedtime < self.TimeOut + 0.5):
            print(f"Stopping Criterion reached at iteration {n}. Time(s): {elapsedtime}")
        else:
            print(f"TIMEOUT ({elapsedtime})s. Iteration: {n}. Algorithm may have not converged.")

        return V 
       
        
        
            
        
        
    
    
    
    
    