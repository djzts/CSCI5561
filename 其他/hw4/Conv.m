function [y] = Conv(x, w_conv, b_conv)

    [H, W, C1] = size(x);
    [Hw, Ww, ~, C2] = size(w_conv);
    y = zeros(H,W,C2);
    w_fliped = flip(flip(w_conv,1),2);
    
    for jj = 1:C2
        for ii = 1:C1
            x_pad = padarray(x(:,:,ii), [1 1], 'both');
            X = im2col(x_pad, [3, 3], 'sliding' );
            w = reshape(w_fliped(:,:,ii,jj), [1, Hw*Ww]);
            b = b_conv(jj);
            y(:,:,jj) = y(:,:,jj) + reshape(w * X - b, [H W]);
        end
    end

end