function [L, dLdy] = Loss_euclidean(y_tilde, y)
L = sum( (y_tilde-y).^2);
dLdy = 2* (y_tilde - y);
end