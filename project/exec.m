 function exec()
    close all; % closes all figures
    clear;
    clc
    img = imread("./pic/chi.png");
    img_ori = imread("./pic/chi_ori.png");
    im = im2single(img);
    In=figure('position', [0, 0, 1300, 800]);
    imagesc(img_ori);
    [row, col, ~] = size(im);
    patch_size =20;   
    mask = zeros(row,col);
    for i=1:1:row
        for j=1:1:col
           if img(i,j,:) == [255,255,255]
              mask(i,j) = 1;
           end
        end
    end
    mask = logical(mask);
    
    go(im, mask, patch_size, 0.01);
 end
