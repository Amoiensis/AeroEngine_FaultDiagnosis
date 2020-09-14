%% ���ݴ���
A = node_learningrate.node;
B = node_learningrate.learning_rate;
Parameters_A = unique(A);  % �ýڵ�����Ϊ����A
Parameters_B = unique(B); % ��ѧϰ����Ϊ����B
size_Parameters_A  = size(Parameters_A);
size_Parameters_B  = size(Parameters_B);
size_data = size(node_learningrate);
%% ���ɾ���
% Index = node_learningrate.accuracy; % ʹ��accuracy��Ϊָ��
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
%% ��ͼ
% ֱ�ӻ�ͼ-ɢ��ͼ
plot3(A,B,Index,'*')
xlabel('Node');ylabel('Learning Rate');zlabel('Accuracy');title('Node-Learning Rate-Accuracy');
% ɢ��ͼ�������
figure();
% ������ͼ��ѡ�񲿷ֵ�������ֵ������Խ���ֵЧ��Խ��
A = [A,B,Index];
x=A(:,1);y=A(:,2);z=A(:,3);
% scatter3(x,y,z)%ɢ��ͼ
% figure
[X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'linear');%��ֵ
% pcolor(X,Y,Z);shading interp%α��ɫͼ
% figure,contourf(X,Y,Z) %�ȸ���ͼ
% figure,surf(X,Y,Z);%��ά����
% figure,meshc(X,Y,Z)%����ͼ
% view(0,0); 
figure,meshc(X,Y,Z);%s��ά���棨ǳɫ��+�ȸ���
xlabel('Node');ylabel('Learning Rate');zlabel('Accuracy');title('Node-Learning Rate-Accuracy');
hidden off;
figure,surf(X,Y,Z);
xlabel('Node');ylabel('Learning Rate');zlabel('Accuracy');title('Node-Learning Rate-Accuracy');