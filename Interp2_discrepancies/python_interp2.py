import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import loadmat

rng = np.random.seed(42)
x_scattered = -100+(100+100)*np.random.rand(60,60).T
y_scattered = -100+(100+100)*np.random.rand(60,60).T

x_range = np.linspace(-100,100,60)
y_range = np.linspace(-100,100,60)
X,Y = np.meshgrid(x_range,y_range)
def function(x,y):
    return np.multiply(np.cos(np.pi*x), np.sin(np.pi*y))

Z = function(X,Y)

interpolator = interp.interp2d(x_range,y_range,Z,kind="linear")
results = []
for i in range(len(x_scattered)):
    intermediate = []
    for j in range(len(y_scattered)):
        intermediate.append(interpolator(x_scattered[i,j],y_scattered[i,j]))
    results.append(np.array(intermediate).squeeze())

results = np.array(results)
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(X,Y,results, cmap=cm.coolwarm,linewidth=0, antialiased=False)

#%%

data = loadmat(r"C:\Users\EricQ\ECE588 Project\snakes\snakes\release2D\interp2 checker data.mat")

G = data["nabla_P"]
V = data["V"]
S = data["GV"]
def interpSnake2(G,V): 
    
    x_range = np.arange(1,G.shape[1]+1)
    y_range = np.arange(1,G.shape[0]+1)
    
    vert = interp.interp2d(x_range,y_range,G[:,:,0],kind="linear",fill_value=np.NaN)
    vertical_values = []
    for i in range(len(V[:,0])):
        vertical_values.append(vert(V[i,0],V[i,1]))
    vertical_values = np.array(vertical_values)
    print(vertical_values.shape)

    horiz = interp.interp2d(x_range,y_range,G[:,:,1],kind="linear",fill_value=np.NaN)
    horizontal_values = []
    for i in range(len(V[:,0])):
        horizontal_values.append(horiz(V[i,0],V[i,1]))
    horizontal_values = np.array(horizontal_values)
    print(horizontal_values.shape)

    combined = np.concatenate((horizontal_values,vertical_values),axis=1)
    
    return combined # only return np.matrix(combined) if it's not already a matrix