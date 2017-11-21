clear;close all;clc

load('digit.mat'); % training data stored in arrays X, y
m = size(X, 1);
num_labels = 10; 

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

X_train = X(rand_indices(1:4500), :);
X_test = X(rand_indices(4501:5000), :);
y_train = y(rand_indices(1:4500), :);
y_test = y(rand_indices(4501:5000), :);


for i = 1:60
    lambda = 0.1;
    [all_theta] = oneVsAll(X_train, y_train, num_labels, lambda, i);

    pred_train = predictOneVsAll(all_theta, X_train);
    pred_test = predictOneVsAll(all_theta, X_test);
    
    Accu_train(i) = mean(double(pred_train == y_train)) * 100;
    Accu_test(i) = mean(double(pred_test == y_test)) * 100;

    fprintf('\nTrain Set Accuracy: %f\n', Accu_train(i)); 
    fprintf('\nTest Set Accuracy: %f\n', Accu_test(i)); 
end

iter = 1:1:60;
plot(iter, Accu_train, iter, Accu_test);

title(sprintf('Learning Curve (lambda = %f)', lambda));
xlabel('Number of iterations')
ylabel('Accuracy')
axis([1 60 60 100])
legend('Train', 'Test')