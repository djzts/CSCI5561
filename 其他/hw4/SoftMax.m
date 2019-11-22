function y = SoftMax(x)
    x_tilde = exp(x);
    x_sum = sum(sum(x_tilde));
    y = x_tilde ./ x_sum;
end