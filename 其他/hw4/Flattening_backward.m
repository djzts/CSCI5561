function [dLdx] = Flattening_backward(dLdy, x, y)
    [m,n,c_in] = size(x);
    dLdx = reshape(dLdy, [m, n, c_in]);
end