clear;close all;clc;

data = load('chip.txt');
X = data(:, [1, 2]); y = data(:, 3);
m = size(X, 1);

rand_indices = randperm(m);


X_train = X(rand_indices(1:90), :);
X_test = X(rand_indices(91:118), :);
y_train = y(rand_indices(1:90), :);
y_test = y(rand_indices(91:118), :);

plotData(X, y);

hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

% Add Polynomial Features
X = mapFeature(X(:,1), X(:,2));
X_train = mapFeature(X_train(:,1), X_train(:,2));
X_test = mapFeature(X_test(:,1), X_test(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X_train, 2), 1);
% Set regularization parameter lambda to 1
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunction(t, X_train, y_train, lambda)), initial_theta, options);


% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))


% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p_test = predict(theta, X_test);
p_train = predict(theta, X_train);
fprintf('Test Accuracy: %f\n', mean(double(p_test == y_test)) * 100);
fprintf('Train Accuracy: %f\n', mean(double(p_train == y_train)) * 100);