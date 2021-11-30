# Active-Contours
Implementation of Active Contours Segmentation in Python 

Python implementation of the paper Segmentation with Active Contours by Fabien Pierre, Mathieu Amendola, Clémence Bigeard, Timothé Ruel, and Pierre-Frédéric Villard. 
Originally published in IPOL 2021 and implemented in MATLAB. 


For more information about the paper and the implementation, please refer to:

Fabien Pierre, Mathieu Amendola, Clémence Bigeard, Timothé Ruel, and Pierre-Frédéric Villard, Segmentation with Active Contours, Image Processing On Line, 11 (2021), pp. 120–141. https://doi.org/10.5201/ipol.2021.298

Due to the discrepancies between how the MATLAB backend implements the interp2d function and how scipy implements different types of 2D interpolation, we are unable to completely port the code over into Python. The Interp2_discrepancies file is present to showcase these differences on a toy example. 
