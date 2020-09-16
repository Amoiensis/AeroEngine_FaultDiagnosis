数据挖掘方法在航空发动机故障诊断中的应用研究(ANN)
======================================================================
AeroEngine_Fault-Diagnosis(ANN)
---------------------------------------

Author:Xiping.Yu

Email:Amoiensis@outlook.com

Data:2020.09.16
***************************************
更多相关内容：

https://github.com/Amoiensis/AeroEngine_FaultDiagnosis


[CONTENT](https://github.com/Amoiensis/AeroEngine_FaultDiagnosis/README.md)
---------------------------------------
1. File

	[概述_过程介绍-OVERVIEW](https://github.com/Amoiensis/AeroEngine_FaultDiagnosis/)

2. Floder

	[神经网络--ANN_Method](https://github.com/Amoiensis/AeroEngine_FaultDiagnosis/tree/master/ANN_Method)

	[参数分析--Parametric_Analysis](https://github.com/Amoiensis/AeroEngine_FaultDiagnosis/tree/master/Parametric_Analysis)

	[结果分析--Result_Analysis&Record](https://github.com/Amoiensis/AeroEngine_FaultDiagnosis/tree/master/Result_Analysis%26Record)

DETAILS
---------------------------------------

项目主要使用Python进行相关分析，在神经网络超参数的寻优过程和图形绘制中使用Matlab进行辅助分析，具体而言：

1.	主要使用sklearn进行神经网络的搭建和训练；

2.	使用imblearn进行非平衡数据的处理，具体使用SMOTE算法的过采样；

3.	在测试集与训练集的划分中，为自行实现，保持两个集合中都有一定数目的负类样本；

4.	在特征分析中，使用的是sklearn的RandomForestRegressor的重要度分析；

5.	在超参数的最优化与图形绘制中，使用Matlab进行辅助分析与图形绘制；


ATTENTION
---------------------------------------

Please feel free to contact with me for any questions, thank you!

Don't spread the files without permission!

未经允许，请勿转载！

本项目所有文件仅供学习交流使用！
***************************************
