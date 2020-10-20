"""
this file is used for prediect the class, for class accuracy with representation variants
"""
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

unitlLength=1
data_npz_path='./ValidateEncodedImgs/codes_CIFAR3_codesAndImgForMetricsCal.npz'
label_npz_path='../../npz_datas/SVHN10_img_N3200x32x32x3_testWithLabel_forMetrics.npz'
data=np.load(data_npz_path)['codes']
# print(data[0:5])
label=np.load(label_npz_path)['labelsGT']
print("data shape:{}".format(data.shape))
print("label sahpe:{}".format(label.shape))
# print(label[400])
# print(data.shape)
spot=np.where(np.isnan(data))[0]
data=np.delete(data,spot,axis=0)
# print(data.shape)
data=data[:,0:50] # use objecy latent feature, need to test two times: first part and second part because we don't know which part is object and which is background
# print(data.shape)
label=np.delete(label,spot,axis=0)
print("data shape:{}".format(data.shape))
print("label sahpe:{}".format(label.shape))
assert len(data)>=3000,"len data is <3000"
assert len(label)>=3000, "len label is <3000"
assert len(data)==len(label),'len data != len label!'
# label=np.array([np.argmax(d)+1 for d in label])
# print(label[639])
# print(label)
state=np.random.get_state()
np.random.shuffle(data)

np.random.set_state(state)
np.random.shuffle(label)

train_num=1500
test_num=1500
data_train=data[0:train_num,]
label_train=label[0:train_num,]

data_test=data[train_num:train_num+test_num,]
label_test=label[train_num:train_num+test_num,]


## multi classification
model_0 =OneVsRestClassifier(SVC(kernel='linear', probability=True,gamma='scale'))

model_0.fit(data_train, label_train)
pre_0 = model_0.predict_proba(data_test)

max_ind=np.argmax(pre_0,axis=1)
# print(max_ind)
pre=np.zeros_like(pre_0)
for i in range(pre.shape[0]):
    pre[i,max_ind[i]]=1
# print(pre)
pre_train0=model_0.predict_proba(data_train)
max_ind_train=np.argmax(pre_train0,axis=1)
# print(max_ind)
pre_train=np.zeros_like(pre_0)
for i in range(max_ind_train.shape[0]):
    pre_train[i,max_ind_train[i]]=1

print(metrics.accuracy_score(label_train,pre_train))
print(metrics.accuracy_score(label_test,pre))

print(model_0.score(data_train,label_train))
print(model_0.score(data_test,label_test))
