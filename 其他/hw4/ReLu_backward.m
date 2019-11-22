function [dLdx] = ReLu_backward(dLdy, x, y)
% dLdy 1 x 10;  dLdx 1x10

dydx = x;
dydx(x > 0) = 1;
dydx(x <= 0) = 0;
dLdx = dLdy .* dydx;

end