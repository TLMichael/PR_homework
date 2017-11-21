function plotDecisionBoundary(Theta1, Theta2, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta

% Plot Data
plotData(X(:,2:3), y - 1);
hold on


u = linspace(-1, 1.5, 50);
v = linspace(-1, 1.5, 50);
z = zeros(length(u), length(v));
    
m = size(X, 1);


    for i = 1:length(u)
        for j = 1:length(v)
            
            temp = sigmoid([1 sigmoid([1 mapFeature(u(i), v(j))] * Theta1')] * Theta2');
            if(temp(1) - temp(2) > 0.001)
                z(i,j) = 0;
            else
                z(i,j) = 1;
            end
        end
    end
    z = z'; % important to transpose z before calling contour

    contour(u, v, z, [0.5, 0.5], 'LineWidth', 2)

    
hold off

end
