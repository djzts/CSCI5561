function [ template, y, x, conf] = choose_template_criminisi(I, t_candi_y, t_candi_x, target_mask, confidence_map,  patch_size)
   
    data = zeros(size(t_candi_y));
    confidence = zeros(size(t_candi_y));
    hp_size = floor(patch_size / 2);
    
    for i = 1 : size(t_candi_y, 1)
        yy = t_candi_y(i); xx = t_candi_x(i);
        p_norm = point_norm(target_mask, yy, xx);
        iso_v = isophote(I(:, :, 1), yy, xx);
        %confidence(i) = point_fil(confidence_map, ones(size(patch_size)), hp_size, yy, xx) / (patch_size^2);
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

