function [J, grad] = costFunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost and gradient for logistic regression with regularization

m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));


J = 1/m * (-y' * log(sigmoid(X*theta)) - (1 - y') * log(1 - sigmoid(X * theta))) + lambda/2/m*sum(theta(2:end).^2);

grad(1,:) = 1/m * (X(:, 1)' * (sigmoid(X*theta) - y));
grad(2:size(theta), :) = 1/m * (X(:, 2:size(theta))' * (sigmoid(X*theta) - y))... 
        + lambda/m*theta(2:size(theta), :);

end
