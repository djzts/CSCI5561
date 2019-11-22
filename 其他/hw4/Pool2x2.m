function [y] = Pool2x2(x)
    [H, W, C1] = size(x);
    
    Hpad = mod(H, 2);
    Wpad = mod(W, 2);

    Hp = ceil(H/2);
    Wp = ceil(W/2);
    y = zeros(Hp, Wp, C1);
    
    for ii = 1:C1
        x_pad = padarray(x(:,:,ii),[Hpad Wpad],0,'post');
        X = im2col(x_pad, [2, 2], 'distinct' );
        y_temp = max(X);
        y(:, :, ii) = reshape(y_temp, [Hp, Wp]);
    end
    
end