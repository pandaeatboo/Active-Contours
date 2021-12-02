%% scattered
% X, Y, and Z are the grid points we will use to interpolate with

rng(42,"twister") % for reproducibility
x_scattered = -100+(100+100).*rand(60,60);
y_scattered = -100+(100+100).*rand(60,60);

smooth_function = @(X,Y) cos(pi*X) .* sin(pi*Y);
x_range = linspace(-100,100,60);
y_range = linspace(-100,100,60);
[X,Y] = meshgrid(x_range,y_range);
Z = smooth_function(X,Y);

results = interp2(X,Y,Z,x_scattered,y_scattered);
figure(1);
title("Results of interpolating on random data points")
surf(X,Y,results)
