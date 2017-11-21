clear ; close all; clc

%% =========== Loading and Visualizing Data =============
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   


fprintf('Loading and Visualizing Data ...\n')

load('digit.mat');
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(size(X, 1));
sel = rand_indices(1:100);

displayData(X(sel, :));
X_train = X(rand_indices(1:ceil(m*9/10)), :);
X_test = X(rand_indices(ceil(m*9/10)+1:m), :);
y_train = y(rand_indices(1:ceil(m*9/10)), :);
y_test = y(rand_indices(ceil(m*9/10)+1:m), :);


%% ================ Initializing Pameters ================
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

for i = 10:60
%i = 1;
%% =================== Training NN ===================
    fprintf('\nTraining Neural Network... \n')
    options = optimset('MaxIter', i);


    lambda = 1;
    costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             

%% ================= Visualize Weights =================
    fprintf('\nVisualizing Neural Network... \n')
    displayData(Theta1(:, 2:end));

%% ================= Implement Predict =================
    pred_train = predict(Theta1, Theta2, X_train);
    pred_test = predict(Theta1, Theta2, X_test);
    Accu_train(i-9) = mean(double(pred_train == y_train)) * 100;
    Accu_test(i-9) = mean(double(pred_test == y_test)) * 100;
    
    fprintf('\nTrain Set Accuracy: %f\n', Accu_train(i-9)); 
    fprintf('\nTest Set Accuracy: %f\n', Accu_test(i-9)); 
end
iter = 10:1:60;
plot(iter, Accu_train, iter, Accu_test);

title(sprintf('Learning Curve (lambda = %f)', lambda));
xlabel('Number of iterations')
ylabel('Accuracy')
axis([10 60 70 100])
legend('Train', 'Test')