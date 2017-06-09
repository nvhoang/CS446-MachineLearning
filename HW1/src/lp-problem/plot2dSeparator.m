% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)
xdata = -10:0.01:10;
y = -w(1)/w(2) * xdata - theta/w(2);
plot(xdata, y, 'b');
end
