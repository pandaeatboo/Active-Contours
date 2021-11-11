import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import cv2
from functions import *

from Evolution_Snakes2D import SnakesEvolution


def run(image_path, k=3, Alpha=1750, Beta=500, Sigma=1, Gamma=5e-5, BalloonCoefficient=-90,\
                 BalloonSmoothing=1, MaxIteration=1e4, CubicSplineRefinement=False, Verbose=False,\
                 MaxDistance=2, RefinementFactor=1.1, TimeOut=math.inf):
                 
    # assuming that shamrock is in a subdirectory called images for the current folder
    #image = "./images/shamrock.png"
    image = image_path
    I_init = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    
    k = 3
    V_init = subdivision(selectPoints(I_init),k)
    
    Active_Contour = Evolution_Snakes2D(I_init, V_init, Alpha, Beta, Sigma, Gamma, BalloonCoefficient,\
                                        BalloonSmoothing, MaxIteration, CubicSplineRefinement, Verbose,\
                                        MaxDistance, RefinementFactor, TimeOut)
    
    V_final = Active_Contour.forward()
    
    plt.figure()
    displayImageAndSnake(I_init, V_final)
    plt.title("Snake Segmentation")
    
    

if __name__ == "__main__":
    run("./images/shamrock.png")
    