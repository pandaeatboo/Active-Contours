import matplotlib.pyplot as plt
import numpy as np

def display(image:np.ndarray,V):
    plt.imshow(np.uint8(image))
    return