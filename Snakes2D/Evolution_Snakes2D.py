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
        
        self.I_init = I_init # TODO: Need to change I_init so that it is transferred into gray value image
        self.V_init = V_init # TODO: Need to change V_init so that it is split into subdivisions
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
        
        # there is no GNU OCTAVE compatability, so we skip the conditional
        I = cv2.GaussianBlur(self.I_init, ksize=(0,0), sigmaX=self.Sigma, borderType=cv2.BORDER_REPLICATE ) 
        V = V_init
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
        
        condi = np.ones((N,1)) == 1
        
        elapsedtime = time.time() - t
        while ((eps>9e-2) and (n <= self.MaxIteration) and (elapsedtime < self.TimeOut )):
            
            B = balloonForce(V)
            B = smoothForces(B,self.BalloonSmoothing)
            GV = interpSnake2(nabla_P, V)
            
            # stop the balloon force
            nrmi = np.sum(np.square(GV),2)
            nrmb = np.sum(np.square(B),2)
            condi = condi and (nrmi < (self.BalloonCoefficient**2*nrmb))
            
            # iteration
            V1 = igpi * (V + (GV+self.BalloonCoefficient*np.multiply(condi*np.array([1,1]),B)*self.Gamma))
            
            # display
            if n % 20 == 0:
                if (verbose):
                    displayImageAndSnake(I_init, V)
                    
            if (n % 100 == 0) and cubic_spline_refinement:
                dv = distPoints(V1)
                if (np.max(dv) > self.MaxDistance):
                    V1 = splinesInterpolation2D(V1, math.floor(V1.shape[0]*self.RefinementFactor))
                    N2 = V1.shape[0]
                    # computation of the regularization matrix
                    
                    
            
            
            # computation of the stopping criteria
            n = n + 1
            
            
            
            
            elapsedtime = time.time() - t
            pass
        
        
        
            
        
        
    
    
    
    
    