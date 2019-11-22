function [dLdx] = Pool2x2_backward(dLdy, x, y)
    [H, W, C1] = size(x);

    Hpad = mod(H, 2);
    Wpad = mod(W, 2);
    
    Hp = ceil(H/2);
    Wp = ceil(W/2);
    dLdx = zeros(2*Hp, 2*Wp, C1);
    
    for k = 1:C1
        x_pad = padarray(x(:,:,k),[Hpad Wpad],0,'post');
        for i = 1:Hp
            for j = 1:Wp
                maximum = max(max(x_pad(2*(i-1)+1:2*i, 2*(j-1)+1:2*j)));
                [idc, idr] = find( (x_pad(2*(i-1)+1:2*i, 2*(j-1)+1:2*j)) == maximum);
                dLdx(2*(i-1)+idc, 2*(j-1)+idr, k) = dLdy(i, j, k);
            end
        end
        dLdx = dLdx(1:H, 1:W, :);
    end
end