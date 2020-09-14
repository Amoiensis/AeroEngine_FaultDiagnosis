import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, RocCurveDisplay, f1_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

filename = "data/left_3835_titlechanged.csv"
data = pd.read_csv(filename)
data_shape = data.shape
# 数据区分
data_0 = data[data['Faulttype']==0]
data_1 = data[data['Faulttype']==1]
# 选取样本构成测试集和训练集
data_0_shape = data_0.shape
data_1_shape = data_1.shape

choice = range(data_0_shape[0])
num_ = data_0_shape[0]
random_seed = 38
random.seed(random_seed)
random_num = random.sample(choice,num_)

data_0_X = data_0.values[random_num, 1:]
data_0_y = data_0.values[random_num, 0]
data_1_X = data_1.values[:, 1:]
data_1_y = data_1.values[:, 0]

data_0_shape_test_rate = 0.3
data_1_shape_test_rate = 0.19
data_0_X_train,data_0_X_test,data_0_y_train,data_0_y_test = model_selection.train_test_split(data_0_X,data_0_y,test_size = data_0_shape_test_rate, random_state = random_seed)
data_1_X_train,data_1_X_test,data_1_y_train,data_1_y_test = model_selection.train_test_split(data_1_X,data_1_y,test_size = data_1_shape_test_rate, random_state = random_seed)

X_train = np.vstack((data_0_X_train, data_1_X_train))
y_train = np.hstack((data_0_y_train, data_1_y_train))
X_test = np.vstack((data_0_X_test, data_1_X_test))
y_test = np.hstack((data_0_y_test, data_1_y_test))

X = np.vstack((X_train,X_test))
y = np.hstack((y_train,y_test))
# 划分数据集后，对于训练集和进行重采样
over_samples = SMOTE(random_state=0)
over_samples_X,over_samples_y = over_samples.fit_sample(X_train, y_train)
# 重抽样前的类别比例
print('重抽样前的类别比例')
print(pd.Series(y_train).value_counts()/len(y_train))
# print(y_train.__len__()/len(y_train))
# 重抽样后的类别比例
print('重抽样后的类别比例')
print(pd.Series(over_samples_y).value_counts()/len(over_samples_y))
# 替更新原始训练数据
X_old = X_train
y_old = y_train
X_train = over_samples_X
y_train= over_samples_y
########################################
# 主成分分析
X_train_shape = X_train.shape
pca = PCA(n_components=X_train_shape[1]-1)
pca.fit(X_train)
PCA(copy=True, n_components=X_train_shape[1]-1, whiten=False)
print(pca.explained_variance_ratio_)

node_num = range(10, 30)
learning_rate_num = [0.1,0.08,0.06,0.04,0.02,0.01,0.008,0.005,0.001,0.0008,0.0005,0.0003,0.0001,0.00001]

record_file = open("record.txt", 'w')
record_file.writelines("node\t"+"learning_rate\t"+"accuracy\t"+"score\t"+"tpr\t"+"fpr\t"+"AUC\t"+"CM\n")
record_file.flush()

for node in node_num:
    for learning_rate_t_ in learning_rate_num:
        print("@@node is ==", node)
        print("@@learning_rate is ==", learning_rate_t_)
        # 建立MLP模型
        model = MLPClassifier(hidden_layer_sizes=(node,),
                              solver='adam',
                              activation="relu",
                              validation_fraction=0.3,
                              verbose = 0,
                              learning_rate = "constant",
                              # power_t = learning_rate_t_, 使用invscaling开启
                              random_state=random_seed,
                              learning_rate_init= learning_rate_t_)  # BP神经网络回归模型
        model.fit(X_train, y_train)  # 训练模型
        # y_pred = model.predict(X_test).tolist()
        y_pred = model.predict_proba(X_test)[:, 1].tolist()
        # predictions = [round(value) for value in y_pred] # 模型预测
        predictions =model.predict(X_test).tolist()


        #
        # np.abs(X_test.iloc[:,2]-predictions).mean()  # 模型评价


        #########################################
        # 模型评估
        #准确率输出
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # Get the score
        score = average_precision_score(y_test, y_pred)
        print("AP_Score: %.6f" % score)

        # Get the f1score
        print("f1score : {}" .format(f1_score(y_test, predictions, average='binary')))

        # Get the fpr, tpr, auc
        fpr, tpr, thresholds = roc_curve(y_test, predictions, pos_label=1)
        print("fpr: {}".format(fpr))
        print("tpr: {}".format(tpr))
        roc_auc = auc(fpr, tpr)
        print("roc_auc: {}".format(roc_auc))

        # ROC 曲线
        # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC Curve')
        # display.plot()
        # plt.show()

        # Plot the recall-precision curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        # plt.plot(recall, precision)
        # plt.axis("Square")
        # plt.xlim(-0.05, 1.05)
        # plt.ylim(-0.05, 1.05)
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.title("Average Precision = %.2f" % (score))
        # plt.show()

        CM = confusion_matrix(y_test, predictions)
        print(CM)
        CM_rate_1 = CM
        B = sum(CM_rate_1[1, :])
        C = CM_rate_1[1, :] / B
        CM_rate_1 = C
        CM_rate_0 = CM
        B = sum(CM_rate_0[0, :])
        C = CM_rate_0[0, :] / B
        CM_rate_0 = C
        CM_rate = [CM_rate_0,CM_rate_1]
        # CM_rate[0, :] = CM[0, :] / sum(CM[0, :])
        record_file.writelines(str(node)+"\t"+str(learning_rate_t_)+"\t"+str(accuracy*100.0)+"\t"+str(score)+"\t"+str(tpr)+"\t"+str(fpr)+"\t"+
                               str(roc_auc)+"\t"+
                               str(CM[0,0])+"\t"+str(CM[0,1])+"\t"+str(CM[1,0])+"\t"+str(CM[1,1])+"\t"+
                               str(CM_rate_0[0])+"\t"+str(CM_rate_0[1])+"\t"+str(CM_rate_1[0])+"\t"+str(CM_rate_1[1])+"\n")
        record_file.flush()
record_file.close()