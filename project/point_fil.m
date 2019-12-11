function [ value ] = point_fil( I, h, hp_size, y, x )

value = sum(sum(I(y - hp_size : y + hp_size, x - hp_size : x + hp_size) .*  h));

enda

