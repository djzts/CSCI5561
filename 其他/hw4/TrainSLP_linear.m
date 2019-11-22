function [w, b] = TrainSLP_linear(mini_batch_x, mini_batch_y)
% learning rate
gamma = 0.000001;
% decay rate
lambda = 0.985;

% init weight
w = normrnd(0,1,[10,196]);
b = normrnd(0,1,[10, 1]);

% number of iteration
nIter = 2000;

% number of minibatch
nBatch = size(mini_batch_x, 1);
batchSize = 30;
k = 1;

% cell to matrix
mini_x = zeros(nBatch, 196, batchSize);
mini_y = zeros(nBatch, 10, batchSize);
for i= 1: nBatch
    mini_x(i,:, :) = mini_batch_x{i,1};
    mini_y(i, :, :) = mini_batch_y{i,1};
end

loss_curve = [];
figure();

for idx = 1:nIter
    if mod(idx, 10) == 0
        gamma = gamma * lambda;
    end
    dLdw = 0;
    dLdb = 0;
    for i = 1:nBatch
        
        L = 0;
        for j = 1:batchSize
            x = reshape(mini_x(i,:,j), [196, 1]);
            y_tilde = FC(x, w, b);
            y = reshape(mini_y(i, :, j), [10, 1]);
            % loss
            [loss, dldy] = Loss_euclidean(y_tilde, y);  
            %dldy -- n*1
            [dldx, dldw, dldb] = FC_backward(dldy, x, w, b, y);
            dLdw = dLdw + reshape(dldw, [10, 196]);
            dLdb = dLdb + dldb;
            L = L + loss;
        end
        w = w - gamma/batchSize*dLdw;
        b = b - gamma/batchSize*dLdb;
        L = L / batchSize;
        if mod(i, 100) == 0
            loss_curve = [loss_curve , L];
            plot(loss_curve, 'r-');
            hold on
        end
    end
end

end