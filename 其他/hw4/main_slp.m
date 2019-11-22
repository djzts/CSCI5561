
load('mnist_train.mat');
load('mnist_test.mat');

% Data preprocessing
batch_size = 30;
im_train = im_train/255;
im_test = im_test/255;
[mini_batch_x, mini_batch_y] = GetMiniBatch(im_train, label_train, batch_size);

% input->fc(10)->softmax->cross_entropy
[w, b] = TrainSLP(mini_batch_x, mini_batch_y);

% Test
acc = 0;
confusion = zeros(10,10);
for iTest = 1 : size(im_test,2)
    x = [im_test(:,iTest)];
    
    pred1 = FC(x, w, b);
    y = SoftMax(pred1);
    [~,l] = max(y);
    confusion(label_test(iTest)+1, l) = confusion(label_test(iTest)+1, l) + 1;
    
    if l == label_test(iTest)+1
        acc = acc + 1;
    end    
end
accuracy = acc / size(im_test,2);
for i = 1 : size(confusion,1)
    confusion(i,:) = confusion(i,:)/sum(confusion(i,:));
end

categories = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

fig_handle = figure();
clf;
imagesc(confusion, [0, 1]);
set(fig_handle, 'Color', [.988, .988, .988])
axis_handle = get(fig_handle, 'CurrentAxes');
set(axis_handle, 'XTick', 1:10)
set(axis_handle, 'XTickLabel', categories)
set(axis_handle, 'YTick', 1:10)
set(axis_handle, 'YTickLabel', categories)
xlabel(sprintf('Accuracy: %f', accuracy));
