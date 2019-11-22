function [L, dLdy] = Loss_cross_entropy_softmax(x, y)
    x_tilde = exp(x);
    x_sum = sum(sum(x_tilde));
    y_tilde = x_tilde ./ x_sum;
    
    L = - sum( sum(y .* log(y_tilde) ));
    dLdy = y_tilde - y;
end