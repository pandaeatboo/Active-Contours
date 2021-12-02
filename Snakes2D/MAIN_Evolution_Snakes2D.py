import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import cv2
from functions import *

from Evolution_Snakes2D import SnakesEvolution

import IPython
shell = IPython.get_ipython()
shell.enable_matplotlib(gui='qt') # change backend 


def run(image_path, k=3, Alpha=1750, Beta=500, Sigma=1, Gamma=5e-5, BalloonCoefficient=-90,\
                 BalloonSmoothing=1, MaxIteration=1e4, CubicSplineRefinement=True, Verbose=False,\
                 MaxDistance=2, RefinementFactor=1.1, TimeOut=math.inf):
                 
    # assuming that shamrock is in a subdirectory called images for the current folder
    #image = "./images/shamrock.png"
    image = image_path
    I_init = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    
    k = 3
    V_init = subdivision(selectPoints(I_init),k)

    Active_Contour = SnakesEvolution(I_init, V_init, Alpha, Beta, Sigma, Gamma, BalloonCoefficient,\
                                        BalloonSmoothing, MaxIteration, CubicSplineRefinement, Verbose,\
                                        MaxDistance, RefinementFactor, TimeOut)
    
    V_final = Active_Contour.forward()
    displayImageAndSnake(I_init, V_final)
    plt.title("Snake Segmentation")
    
    return V_final

    
    

if __name__ == "__main__":
    final_V = run("./images/shield.png",Alpha=1750,Beta=700,Sigma=1.4,Gamma=0.0015,\
                          BalloonCoefficient=-100,BalloonSmoothing=100,MaxIteration=300)
    