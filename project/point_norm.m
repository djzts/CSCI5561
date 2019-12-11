function p_norm = point_norm(map, y, x )
    rect = map(y - 1 : y + 1, x - 1 : x + 1);
    dx=diff(rect(:,ceil(end/2)),2);
    dy=diff(rect(ceil(end/2),:),2);
    vec=[dx;dy];
    
    if dx == 0 && dy == 0
        p_norm = [1; 1] / norm([1; 1]); 
    else
        p_norm = vec / norm(vec);
    end
end

