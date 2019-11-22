function [w1, b1, w2, b2] = TrainMLP(mini_batch_x, mini_batch_y)

    % Loss = [];
    [m, batchSize] = size(mini_batch_x{1});
    [n, ~] = size(mini_batch_y{1});
    [nBatch, ~] = size(mini_batch_x);

    lr = 1e-5;
    lambda = 0.2;

    w1_size = [30, m];
    w2_size = [n, 30];
    b1_size = [30, 1];
    b2_size = [n, 1];
    w1 = random('normal', 0, 1, w1_size);
    w2 = random('normal', 0, 1, w2_size);
    b1 = random('normal', 0, 1, b1_size);
    b2 = random('normal', 0, 1, b2_size);

    nIter = 50;
    k = 1;
    
    mini_x = zeros(nBatch, m, batchSize);
    mini_y = zeros(nBatch, n, batchSize);
    for i= 1: nBatch
        mini_x(i,:, :) = mini_batch_x{i,1};
        mini_y(i, :, :) = mini_batch_y{i,1};
    end

    loss_curve = [];
    figure();
    
    %Training Loop
    for idx = 1:nIter
        if mod(k, 1000) == 0
            lr = lr * lambda;
        end
        dLdw1 = 0;
        dLdw2 = 0;
        dLdb1 = 0;
        dLdb2 = 0;
        
        for i = 1:nBatch
            L = 0;
            for j = 1:batchSize
                x = reshape(mini_x(i,:,j), [196, 1]);
                a_1 = FC(x, w1, b1);      % a_1 = w * x + b
                f_1 = ReLu(a_1);          
                a_2 = FC(f_1, w2, b2);    % a_2 = w * f_1 + b
                
                y = reshape(mini_y(i, :, j), [10, 1]);
                % loss
                [loss, dlda_2] = Loss_cross_entropy_softmax(a_2, y);  %dldy -- n*1
                [dldf_1, dldw2, dldb2] = FC_backward(dlda_2, f_1, w2, b2, y);
                dlda_1 = ReLu_backward(dldf_1, a_1, f_1);
                [dldx, dldw1, dldb1] = FC_backward(dlda_1, x, w1, b1, a_1);
                % To be continue
                
                dLdw1 = dLdw1 + dldw1;
                dLdb1 = dLdb1 + dldb1;
                dLdw2 = dLdw2 + dldw2;
                dLdb2 = dLdb2 + dldb2;
                L = L + loss;
                
            end

            w1 = w1 - lr/batchSize*reshape(dLdw1,w1_size) ;
            b1 = b1 - lr/batchSize*dLdb1;
            w2 = w2 - lr/batchSize*reshape(dLdw2,w2_size);
            b2 = b2 - lr/batchSize*dLdb2;
        end
        
        L = L / batchSize;
        if mod(i, 100) == 0
            loss_curve = [loss_curve , L];
            plot(loss_curve, 'r-');
            hold on
        end
        % Loss = [Loss, loss];
    end

end