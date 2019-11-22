function [dLdx,dLdw,dLdb] = FC_backward(dLdy, x, w, b, y)
    [m,n] = size(w); %10x196
    dLdx = w' * dLdy;  %  196 x 10  10 x 1
    dLdw = reshape(dLdy * x', [1, n*m]); % 10 x 1  1 x 196
    dLdb = dLdy;

end