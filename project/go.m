function [ syn_im ] = go( I, mask, patch_size, tol)
    I = repmat((~mask), 1, 1, 3) .* I;
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
        %figure(9)
        %set(gcf,'Position',[700,0,1300,800]);
        %imagesc(syn_im);
        %pause(0.1);
        
        mask(y - hp_size : y + hp_size, x - hp_size : x + hp_size) = 0;
        confidence_map(y - hp_size : y + hp_size, x - hp_size : x + hp_size) =...
            confidence_map(y - hp_size : y + hp_size, x - hp_size : x + hp_size)...
            + ((~tplt_mask(:, :, 1)) * confidence);
        i = i + 1;
    end
end

