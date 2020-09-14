%% 基本绘制
test_0_rate = test_rate(:,1);
test_0_acc = test_rate(:,2);
test_1_acc = test_rate(:,3);
plot(test_0_rate,test_0_acc,'r*--',test_0_rate,test_1_acc,'bo--')
%% 其他添加
size_test_0_rate = size(test_0_rate);
horizon = ones(size_test_0_rate).*0.5;
hold on
plot(test_0_rate,horizon,'g-');
xlabel("test-set rate(label=0)");
ylabel("Accuracy");
title("The proportion of normal samples used for training");