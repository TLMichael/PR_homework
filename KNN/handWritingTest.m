function handWritingTest
%%
clc
clear
close all
%%

load('digit.mat');
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(size(X, 1));
sel = rand_indices(1:100);

displayData(X(sel, :));
y(1:500) = 0;
X = X(rand_indices, :);
y = y(rand_indices, :);

len = size(X, 1);
k = 8;

% 测试数据比例
Ratio = 0.1;
numTest = Ratio * len;


for k = 1:20
    error = 0;
% 测试
    for i = 1:numTest
        classifyresult = KNN(X(i,:),X(numTest:len,:),y(numTest:len,:),k);
        fprintf('测试结果为：%d  真实结果为：%d\n',[classifyresult y(i)])
        if(classifyresult~=y(i))
           error = error+1;
        end
    end
    
    Accu(k) = 1-error/(numTest);
    fprintf('准确率为：%f\n',Accu(k))
end

count = 1:20;
plot(count, Accu);

title('Accuracy change with k');
xlabel('k')
ylabel('Accuracy')



end