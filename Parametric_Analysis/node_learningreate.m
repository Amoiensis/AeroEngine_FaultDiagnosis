%% 数据处理
A = node_learningrate.node;
B = node_learningrate.learning_rate;
Parameters_A = unique(A);  % 用节点数作为参数A
Parameters_B = unique(B); % 用学习率作为参数B
size_Parameters_A  = size(Parameters_A);
size_Parameters_B  = size(Parameters_B);
size_data = size(node_learningrate);
%% 生成矩阵
% Index = node_learningrate.accuracy; % 使用accuracy作为指标
rate = 0;
Index = (node_learningrate.accuracy./100).*(1-rate)+(node_learningrate.CMr11).*rate;
M_node_rate = zeros(size_Parameters_A (1),size_Parameters_B (1));
for i=1:size_data(1)
    for j=1:size_Parameters_A (1)
        if node_learningrate.node(i) == Parameters_A(j)
            for k=1:size_Parameters_B (1)
                if node_learningrate.learning_rate(i) == Parameters_B(k)
                    M_node_rate(j,k) = Index(i);
                end
            end
        end
    end
end
%% 绘图
% 直接绘图-散点图
plot3(A,B,Index,'*')
xlabel('Node');ylabel('Learning Rate');zlabel('Accuracy');title('Node-Learning Rate-Accuracy');
% 散点图拟合曲面
figure();
% 从曲面图中选择部分点来做插值，数据越多插值效果越好
A = [A,B,Index];
x=A(:,1);y=A(:,2);z=A(:,3);
% scatter3(x,y,z)%散点图
% figure
[X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'linear');%插值
% pcolor(X,Y,Z);shading interp%伪彩色图
% figure,contourf(X,Y,Z) %等高线图
% figure,surf(X,Y,Z);%三维曲面
% figure,meshc(X,Y,Z)%剖面图
% view(0,0); 
figure,meshc(X,Y,Z);%s三维曲面（浅色）+等高线
xlabel('Node');ylabel('Learning Rate');zlabel('Accuracy');title('Node-Learning Rate-Accuracy');
hidden off;
figure,surf(X,Y,Z);
xlabel('Node');ylabel('Learning Rate');zlabel('Accuracy');title('Node-Learning Rate-Accuracy');