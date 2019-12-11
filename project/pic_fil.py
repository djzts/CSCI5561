#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy import signal


# In[ ]:


def ssd_patch_2d( I, T ):
    mask = np.where(T >= 0,1,0); # -1 represents empty

    ssd_map = signal.convolve2d(mask, np.rot90(I * I), mode='valid') + np.sum(T * T) - 2 *signal.convolve2d(mask * T, np.rot90(I), mode='valid')

    return ssd_map 


# In[1]:


def  ssd_patch(I, T):

    ssd_map_r = ssd_patch_2d(I[:, :, 0], T[:, :, 0])
    ssd_map_g = ssd_patch_2d(I[:, :, 1], T[:, :, 1])
    ssd_map_b = ssd_patch_2d(I[:, :, 2], T[:, :, 2])

    ssd_map = ssd_map_r + ssd_map_g + ssd_map_b

    ssd_map = normalize_2d_matrix(ssd_map)

    return ssd_map


# In[ ]:


def set_forbid_region( ssd_map, target_mask, patch_size ):
    LARGE_CONST = 100

    hp_size = np.floor(patch_size / 2)

    forbid_area = imdilate(target_mask, np.ones((patch_size, patch_size)))
    
    ssd_map = ssd_map + (forbid_area[hp_size + 1 : len(target_mask) - hp_size+1,  hp_size + 1 : target_mask.shape()[1] - hp_size+1] * LARGE_CONST)

    return ssd_map


# In[ ]:


function p_norm = point_norm(map, y, x ):
   rect = map[y - 1 : y + 1, x - 1 : x + 1]



   #dx=diff(rect(:,ceil(end/2)),2)
   #dy=diff(rect(ceil(end/2),:),2)

   dx = np.diff(rect[:,np.ceil(rect.shape[1]/2],axis=0,n=2)
   dy = np.diff(rect[np.ceil(rect.shape[0]/2,:],axis=0,n=2)

   #vec=[dx;dy]
   vec = np.concatenate((dx,dy))
   
   if dx == 0 and dy == 0:
       p_norm =np.ones((2,1))/ np.linalg.norm(np.ones((2,1))) 
   else:
       p_norm = vec / np.linalg.norm(vec)
   
    return p_norm


# In[ ]:


def point_fil( I, h, hp_size, y, x ):

    value = np.sum(I[y - hp_size : y + hp_size+1, x - hp_size : x + hp_size+1] *  h)

    return value 


# In[ ]:


def normalize_2d_matrix( m ):
    norm_m = (m - np.min(m)) / (np.max(m) - np.min(m))
    return norm_m


# In[ ]:
def pol2cart(rho, phi):
    x = rho * math.cos(math.radians(phi))
    y = rho * math.sin(math.radians(phi))
    return x, y

def isophote(im, y, x):
    window = im[y - 1 : y + 2, x - 1 : x + 2]
    center_value = window[1, 1]
    np.where(window==-1,center_value,window)  #window(window == -1) = center_value
    fx = window[1, 2] - window[1, 0]
    fy = window[2, 1] - window[0, 1]
    if fx == 0 and fy == 0：
       isoV = np.ones((2,1))
    else：
        I = np.sqrt(fx**2 + fy**2)
        theta = math.atan(fx/fy) #theta = acot(fy / fx)

        isoV_x, isoV_y = pol2cart(theta, I)
        isoV = np.array([[isoV_x], [isoV_y]])
    return isoV


# In[ ]:


def go( I, mask, patch_size, tol):
    #I = repmat((~mask), 1, 1, 3) .* I;
    mask_not = np.where(mask>0,0,1)
    temp_1 = mask_not.reshape((mask_not.shape[0],mask_not.shape[1],1))
    I = np.tile(temp_1,(1,1,3))


    syn_im = I;
    syn_im(syn_im == 0) = -1;
    hp_size = floor(patch_size / 2);
    confidence_map = double(~mask);
    i = 1;
    while any(mask(:) == 1)
        [t_candi_x, t_candi_y] = fill_front(mask);
        [template, y, x, confidence] = choose_template_criminisi(syn_im, t_candi_y, t_candi_x, mask, confidence_map, patch_size);
        ssd_map = ssd_patch(syn_im, template);
        ssd_map = set_forbid_region( ssd_map, mask, patch_size );
        patch = choose_sample(ssd_map, tol, syn_im, patch_size, 0.0001); 
        tplt_mask = template >= 0;
        patch = tplt_mask .* template + ~tplt_mask .* patch;
        syn_im(y - hp_size : y + hp_size, x - hp_size : x + hp_size, :) = patch;
        figure(9)
        set(gcf,'Position',[700,0,1300,800]);
        imagesc(syn_im);
        get_ipython().run_line_magic('pause', '(0.1);')
        
        mask(y - hp_size : y + hp_size, x - hp_size : x + hp_size) = 0;
        confidence_map(y - hp_size : y + hp_size, x - hp_size : x + hp_size) =...
            confidence_map(y - hp_size : y + hp_size, x - hp_size : x + hp_size)...
            + ((~tplt_mask(:, :, 1)) * confidence);
        i = i + 1;
    return syn_im



# In[ ]:


function [ err_patch ] = find_err_patch_2D( T, patch, overlap)
    diff = T(1 : overlap, :) - patch(1 : overlap, :);
    err_patch = diff .* diff;


end


# In[ ]:


function [ err_patch ] = find_err_patch( T, patch, overlap)
    err_patch_r = find_err_patch_2D( T(:, :, 1), patch(:, :, 1), overlap);
    err_patch_g = find_err_patch_2D( T(:, :, 2), patch(:, :, 2), overlap);
    err_patch_b = find_err_patch_2D( T(:, :, 3), patch(:, :, 3), overlap);
    err_patch = err_patch_r + err_patch_g + err_patch_b;
   
end


# In[ ]:


function [ mask ] = find_cut_mask(template, patch, overlap)
    t_size = size(template, 1);
    mask = zeros(t_size);
    mask_up = zeros(overlap, t_size);
    mask_left = zeros(t_size, overlap);
    is_up = nnz(template(1 : overlap, ceil(t_size / 2), 1) >= 0);
    is_left = nnz(template(ceil(t_size / 2), 1 : overlap, 1) >= 0);
    if is_up > 0
        err_patch = find_err_patch(template, patch, overlap);
        mask_up = cut_dp(err_patch);
    end
    if is_left > 0
        err_patch = find_err_patch(permute(template, [2 1 3]), permute(patch, [2 1 3]), overlap);
        mask_left = cut_dp(err_patch)';
    end
    mask(1 : overlap, :) = mask(1 : overlap, :) | mask_up;
    mask(:, 1 : overlap) = mask(:, 1 : overlap) | mask_left;
    mask;

end


# In[ ]:


function [ front_x, front_y ] = fill_front( target_mask )
    front = imdilate(target_mask, ones(3,3)) & ~target_mask;
    [front_y, front_x] = find(front);
end


# In[ ]:


function [ template, y, x, conf] = choose_template_criminisi(I, t_candi_y, t_candi_x, target_mask, confidence_map,  patch_size)
   
    data = zeros(size(t_candi_y));
    confidence = zeros(size(t_candi_y));
    hp_size = floor(patch_size / 2);
    
    for i = 1 : size(t_candi_y, 1)
        yy = t_candi_y(i); xx = t_candi_x(i);
        p_norm = point_norm(target_mask, yy, xx);
        iso_v = isophote(I(:, :, 1), yy, xx);
        get_ipython().run_line_magic('confidence', '(i) = point_fil(confidence_map, ones(size(patch_size)), hp_size, yy, xx) / (patch_size^2);')
        h=ones(size(patch_size));
        confidence(i) = sum(sum(confidence_map(yy - hp_size : yy + hp_size, xx - hp_size : xx + hp_size) .*  h)) / (patch_size^2);
        data(i) = abs(dot(iso_v, p_norm(:, 1)));
    end
    priority = confidence + data;
    [~, sorted_id] = sort(priority, 'descend');
    t_candi_y = t_candi_y(sorted_id);
    t_candi_x = t_candi_x(sorted_id);
    confidence = confidence(sorted_id);
    data = data(sorted_id);
    y = t_candi_y(1); x = t_candi_x(1);
    conf = confidence(1);
    template = I(y - hp_size : y + hp_size, x - hp_size : x + hp_size, : );


end


# In[ ]:


function [patch] = choose_sample( ssd_map, tol, I, patch_size, small_cost_value)
    min_c = min(min(ssd_map));
    min_c = max(min_c,small_cost_value);
    [y, x] = find(ssd_map <= min_c * (1 + tol));
    index = randi(size(y, 1));
    hp_size = floor(patch_size / 2);
    y = y(index) + hp_size; % transfrom to I coordinate
    x = x(index) + hp_size;
    patch = I((y - hp_size) : (y + hp_size), (x - hp_size) : (x + hp_size), :);
end


# In[ ]:


close all; % closes all figures
clear;
clc
img = imread('chi.png');
img_ori = imread('chi_ori.png');
im = im2single(img);
In=figure('position', [0, 0, 1300, 800]);
imagesc(img_ori);
[row, col, channel] = size(im);
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

