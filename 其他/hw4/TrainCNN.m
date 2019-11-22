function [w_conv, b_conv, w_fc, b_fc, Loss] = TrainCNN(mini_batch_x, mini_batch_y)

[m, batchSize] = size(mini_batch_x{1});
[n, ~] = size(mini_batch_y{1});
[nBatch, ~] = size(mini_batch_y);

lr = 1e-5;
lambda = 0.8;

w_conv_size = [3 3 1 3];
b_conv_size = [1 3];
w_fc_size = [10 147];
b_fc_size = [10 1];

w_conv = random('normal', 0, 1, w_conv_size);
b_conv = random('normal', 0, 1, b_conv_size);
w_fc = random('normal', 0, 1, w_fc_size);
b_fc = random('normal', 0, 1, b_fc_size);

mini_x = zeros(nBatch, m, batchSize);
mini_y = zeros(nBatch, n, batchSize);
for i= 1: nBatch
    mini_x(i,:, :) = mini_batch_x{i,1};
    mini_y(i, :, :) = mini_batch_y{i,1};
end


iteration = 20;
k = 1;
Loss = zeros(1,iteration);

for iter = 1:iteration
    disp(['CNN training iteration #',num2str(iter)]);
    tic;     

    if mod(iter, 100) == 0
        lr = lr * lambda;
    end

    dLdw_conv = zeros(w_conv_size);
    dLdb_conv = zeros(b_conv_size);
    dLdw_fc = zeros(w_fc_size);
    dLdb_fc = zeros(b_fc_size);

    for i = 1:nBatch
        for j = 1:batchSize
            x = reshape(mini_x(i,:,j), [14 14]); % input

            a_1 = Conv(x, w_conv, b_conv);      % Conv
            f_1 = ReLu(a_1);                    % Relu
            f_2 = Pool2x2(f_1);                 % Pooling
            f_3 = Flattening(f_2);              % Flatten

            a_2 = FC(f_3, w_fc, b_fc);          % FC 	     

            y = reshape(mini_y(i, :, j), [10, 1]);
            % loss
            [loss, dlda_2] = Loss_cross_entropy_softmax(a_2, y);  %dldy -- n*1               
            [dldf_3, dldw_fc, dldb_fc] = FC_backward(dlda_2, f_3, w_fc, b_fc, a_2);
            dLdw_fc = dLdw_fc + reshape(dldw_fc, w_fc_size);
            dLdb_fc = dLdb_fc + dldb_fc;

            [dldf_2] = Flattening_backward(dldf_3, f_2, f_3);
            [dldf_1] = Pool2x2_backward(dldf_2, f_1, f_2);
            [dlda_1] = ReLu_backward(dldf_1, a_1, f_1);
            [dldw_conv, dldb_conv] = Conv_backward(dlda_1, x, w_conv, b_conv, a_1);
            dLdw_conv = dLdw_conv + dldw_conv;
            dLdb_conv = dLdb_conv + dldb_conv;

        end
        w_conv = w_conv - lr/batchSize*dLdw_conv;
        b_conv = b_conv - lr/batchSize*dLdb_conv;
        w_fc = w_fc - lr/batchSize*dLdw_fc;
        b_fc = b_fc - lr/batchSize*dLdb_fc;
    end
    Loss(iter) = loss;
    disp([' Time: ',num2str(toc),'seconds']);
end

end