clear ; close all; clc

%% =========== Loading and Visualizing Data =============
input_layer_size  = 28;  
hidden_layer_size = 2;   
num_labels = 2;       


fprintf('Loading and Visualizing Data ...\n')

load('chip.txt');
X = chip(:, 1:2);
y = chip(:, 3);
y = y + 1;
m = size(X, 1);

rand_indices = randperm(size(X, 1));

X_train = X(rand_indices(1:90), :);
X_test = X(rand_indices(91:118), :);
y_train = y(rand_indices(1:90), :);
y_test = y(rand_indices(91:118), :);

plotData(X, y - 1);


hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')


hold off;

% Add Polynomial Features
X = mapFeature(X(:,1), X(:,2));
X_train = mapFeature(X_train(:,1), X_train(:,2));
X_test = mapFeature(X_test(:,1), X_test(:,2));



%% ================ Initializing Pameters ================
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 30);


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
             


             
% Plot Boundary
plotDecisionBoundary(Theta1, Theta2, X, y);
hold on;
title(sprintf('lambda = %g', lambda))


% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
%% ================= Implement Predict =================

pred_test = predict(Theta1, Theta2, X_test);
pred_train = predict(Theta1, Theta2, X_train);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);             
fprintf('\nTrain Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);
             
