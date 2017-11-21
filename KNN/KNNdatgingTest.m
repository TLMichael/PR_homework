function KNNdatgingTest
%%
clc
clear
close all
%%
data = load('datingTestSet2.txt');
dataMat = data(:,1:3);
labels = data(:,4);
len = size(dataMat,1);
k = 7;
error = 0;
% 测试数据比例
Ratio = 0.1;
numTest = Ratio * len;
% 归一化处理
maxV = max(dataMat);
minV = min(dataMat);
range = maxV-minV;
newdataMat = (dataMat-repmat(minV,[len,1]))./(repmat(range,[len,1]));

for k = 1:20
    error = 0;

% 测试
    for i = 1:numTest
        classifyresult = KNN(newdataMat(i,:),newdataMat(numTest:len,:),labels(numTest:len,:),k);
         fprintf('测试结果为：%d  真实结果为：%d\n',[classifyresult labels(i)])
         if(classifyresult~=labels(i))
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
