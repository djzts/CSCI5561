function y = FC(x, w, b)
% w 10x196, x 1x 196,  b 1x10
m = max(size(x));
temp = reshape(x, [m, 1]);
y =  w * temp + b;
% y 1x10
end