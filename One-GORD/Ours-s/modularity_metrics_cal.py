"""
this file is used for modularity accuracy with representation variants
"""
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

unitlLength=1
data_npz_path='./ValidateEncodedImgs/codes_CIFAR3_codesModularityMetricsCal.npz'
# data_npz_path='./codes_CIFAR3_codesModularityMetricsCal.npz'
classes=10
single_num=100
code_reprent=np.load(data_npz_path)['codes']
print(code_reprent.shape)
code_reprent=code_reprent[:,0:50]
mean_total=0
for i in range(classes):
    data = code_reprent[0 + i * 100:100 + i * 100]
    # print(data.shape)
    sum_mean = data.sum(axis=0) / data.size
    # print(sum_mean.shape)
    # print("it:{} sum_mean: {}".format(i,sum_mean))
    sub_ = abs(data - sum_mean)
    # print(sub_.shape)
    sum_sub_ = sub_.sum(axis=0) / sub_.shape[0]
    # print('it:{} sum_sub_: {} '.format(i,sum_sub_))
    mean_total = mean_total + sum_sub_
mean_total=mean_total/10
# print("Total sum_sub_mean for 2: {}".format(mean_total))
print('Final mean: {}'.format(mean_total.mean()))