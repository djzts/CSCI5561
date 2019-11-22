function [dLdw, dLdb] = Conv_backward(dLdy, x, w_conv, b_conv, y)
    [H, W, ~] = size(x);
    [kernel_m, kernel_n,C1,C2] = size(w_conv);
    % m = n = 3
    dLdw = zeros(kernel_m,kernel_n,C1,C2);
    dLdb = zeros(1, C2);
    % b is a column vector of size(1, c_out)
    
    for jj = 1:C2
        l = dLdy(:,:,jj);
        L = reshape(l, [1 H*W]);
        for ii = 1:C1
            x_pad = padarray(x(:,:,ii), [1 1], 'both');
            X = im2col(x_pad, [H, W], 'sliding' );
              
            dldw = reshape(L*X, [kernel_m, kernel_n]);
            dLdw(:,:,ii,jj) = dldw;
        end
        dLdb(1, jj) = sum(sum(dLdy(:,:,jj)));
    end
end