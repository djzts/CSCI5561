function [y] = Flattening(x)
    [H, W, C1] = size(x);
    y = reshape(x, [H*W*C1, 1]);
end