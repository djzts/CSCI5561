close all; % closes all figures
clear;
clc
%img = imread('./pic/new.png');
files = dir('./test/imgs');

for i = 3:length(files)
dirname = files(i).name
base_str = "./test/imgs/";
img = imread(base_str + dirname);
%img_ori = imread('base_str + dirname');

im = im2single(img);
%In=figure('position', [0, 0, 1300, 800]);
%imagesc(img_ori);
[row, col, channel] = size(im);
%patch_size =14;  
patch_size =ceil(max(row, col)/41.5);  
mask = zeros(row,col);
for i=1:1:row
    for j=1:1:col
        if img(i,j,:) == [255,255,255]
            mask(i,j) = 1;
        end
    end
end
    mask = logical(mask);
    output = go(im, mask, patch_size, 0.01);
    imwrite(output,"./output/" + dirname);
end
