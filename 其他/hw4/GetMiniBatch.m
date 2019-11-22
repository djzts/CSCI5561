function [mini_batch_x, mini_batch_y] = GetMiniBatch(im_train,label_train, batch_size)
n_train = size(im_train, 2);
im_size = size(im_train, 1);
p = randperm(n_train);
n_batch = floor(n_train/batch_size);

mini_batch_x = cell(n_batch,1);
mini_batch_y = cell(n_batch, 1);

for i=1:n_batch
    idx = p((i-1)*batch_size + 1: i*batch_size);
    mini_batch_x{i, 1} = im_train(:, idx);
    temp = label_train(1, idx) + 1;
    I = eye(10);
    mini_batch_y{i, 1} = I(temp,:)';

end

end