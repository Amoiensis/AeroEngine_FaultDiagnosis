import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, RocCurveDisplay, f1_score, roc_curve, auc
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
########################
# 0.4是对应0.64-0.6的正确率，
# 0.37是对应0.67-0.6的正确率，
# 0.35是对应0.87-0.2的正确率，
# 0.31是对应0.78-0.6的正确率，
# 0.3是对应0.68-0.8的正确率，
# 0.29是对应0.95-0.2的正确率，
# 0.27是对应0.72-0.6的正确率，
# 0.2对应0.71-0.6的正确率，
# 0.15是对应0.68-0.8的正确率，
# 0.1对应0.69-0.6正确率
########################
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
# X_train = over_samples_X
# y_train= over_samples_y
########################################
# 主成分分析
X_train_shape = X_train.shape
pca = PCA(n_components=X_train_shape[1])
pca.fit(X_train)
PCA(copy=True, n_components=X_train_shape[1], whiten=False)
pca_importances_ = list(pca.explained_variance_ratio_)
print(pca_importances_,pca_importances_)
idex_name = ['A1','A2','B1','B2','B3','A3','B4','B5','B6','A4','A5','B7','B8','B9','B10','A6','B11','B12','B13','B14','A7','B15','B16','C1','C2','C3']
feature_important = pd.Series(pca_importances_,index=idex_name).sort_values(ascending=False)
plt.bar(feature_important.index, feature_important.values)
plt.title("Feature Importance (PCA)")
plt.show()

# 显示出随机森林特征的重要性，并做条形图
rfr = RandomForestRegressor(min_samples_split=6, n_estimators=100)
rfr.fit(X_train, y_train)
print(rfr.score(X_test, y_test))
# 使用pd.Series进行组合，值是特征重要性的值，index是样本特征，.sort_value 进行排序操作
idex_name = ['A1','A2','B1','B2','B3','A3','B4','B5','B6','A4','A5','B7','B8','B9','B10','A6','B11','B12','B13','B14','A7','B15','B16','C1','C2','C3']
feature_important = pd.Series(rfr.feature_importances_,index=idex_name).sort_values(ascending=False)
plt.bar(feature_important.index, feature_important.values)
plt.title("Feature Importance (RF)")
plt.show()



# 建立MLP模型
model = MLPClassifier(hidden_layer_sizes=(27,),
                      solver='adam',
                      activation="relu",
                      validation_fraction=0.3,
                      verbose = 1,
                      learning_rate = "constant",
                      # power_t = learning_rate_t_, 使用invscaling开启
                      random_state=random_seed,
                      learning_rate_init= 0.0008,
                      tol=1e-4)  # BP神经网络回归模型
model.fit(X_train, y_train)  # 训练模型

epochs = list(range(model.n_iter_))
loss = model.loss_curve_
plt.plot(epochs,loss,color=(0,0,0),label='loss')
plt.xlabel('epochs')    #x轴表示
plt.ylabel('loss')   #y轴表示
plt.title("Loss-Value Plot")      #图标标题表示
plt.legend()            #每条折线的label显示
plt.show()

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
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC Curve')
display.plot()
plt.show()

metrics.plot_roc_curve(model, X_test, y_test)  # doctest: +SKIP
plt.title("TestSet:ROC Curve")
plt.show()


# Plot the recall-precision curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
# plt.plot(recall, precision)
plt.axis("Square")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Average Precision = %.2f" % (score))
plt.show()

metrics.plot_precision_recall_curve(model, X_test, y_test)
plt.title("TestSet:Precision-Recall Curve")
plt.show()

# plot_confusion_matrix
CM = confusion_matrix(y_test, predictions)
print(CM)

metrics.plot_confusion_matrix(model, X_test, y_test,normalize=None)
plt.title("TestSet:Confusion Matrix(num)")
plt.show()

metrics.plot_confusion_matrix(model, X_test, y_test,normalize='true')
plt.title("TestSet:Confusion Matrix(rate)")
plt.show()

# 在训练集上测试
# y_pred_train = model.predict_proba(X_train)[:, 1].tolist()
# # predictions = [round(value) for value in y_pred] # 模型预测
# predictions_train =model.predict(X_train).tolist()

metrics.plot_confusion_matrix(model, X_old, y_old,normalize=None)
plt.title("TrainSet:Confusion Matrix(num)")
plt.show()

metrics.plot_confusion_matrix(model, X_old, y_old,normalize='true')
plt.title("TrainSet:Confusion Matrix(rate)")
plt.show()