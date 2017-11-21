clear;close all;clc

load('digit.mat');

K = 10;
[centSet,clusterAssment] = kMeans(X,K);

y = y + 1;
y(1:500) = 1;

for i = 1:10
    
    Accu(i) = mean(double(clusterAssment(500*(i-1)+1 : 500*i, 1) == y(500*(i-1)+1 : 500*i))) * 100;
   
    fprintf('\nTrain Set Accuracy of digit %d : %f\n', i-1, Accu(i)); 
    
end