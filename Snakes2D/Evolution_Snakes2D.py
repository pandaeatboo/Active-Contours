# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 00:01:30 2021

@author: EricQ
"""

from math import inf
from functions import *

class SnakesEvolution():
    def __init__(self,I_init, V_init, Alpha=1750, Beta=500, Sigma=1, Gamma=5e-5, BalloonCoefficient=-90,\
                 BalloonSmoothing=1, MaxIteration=1e4, CubicSplineRefinement=False, Verbose=False,\
                 MaxDistance=2, RefinementFactor=1.1, TimeOut=inf):
        
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
        
        
        
    
    
    
    
    