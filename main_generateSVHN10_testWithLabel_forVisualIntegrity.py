import numpy as np
from PIL import Image

Single_Num=320 # number of each digits in SVHN 320*10=3200=64*50
batch_size=64
batch_num=50
width=32
height=32
channel=3
classes=10
Total_Num=Single_Num*(classes)
SVHNArrayTemp=np.empty(shape=(Total_Num, width, height, channel)) # random one digits: e.g. 0
SVHN_LabelArrayTemp=np.zeros(shape=(Total_Num, classes))
OnedigitArray=np.empty(shape=(10, width, height, channel)) # save one digit
OnedigitArrayMasks=np.empty(shape=(10, width, height, 1)) # save one digit

# adding
one_cnt=0
cnt=0
for imgFolderName in range(1,classes+1,1):
    for k in range(2000,2000+Single_Num):
        temp = Image.open('SVHNimages/{}/{:06d}.png'.format(imgFolderName,k+1))
        temp_arr=np.array(temp,dtype=np.float)
        SVHNArrayTemp[cnt] = temp_arr
        if imgFolderName==10:
            # OnedigitArray[one_cnt] = temp_arr
            # one_cnt=one_cnt+1
            SVHN_LabelArrayTemp[cnt, 0] = 1
        else:
            # SVHNArrayTemp[cnt] = temp_arr
            SVHN_LabelArrayTemp[cnt, imgFolderName]=1
        cnt=cnt+1
print(cnt) # debug end
# print(one_cnt)
# SVHN_LabelArray1[:, 0] = 1
# print(SVHN_LabelArray1[0:3])
# print(SVHN_LabelArray1[307:322])
# print(SVHN_LabelArray1[3195:3200])
# exit

state1=np.random.get_state()
np.random.shuffle(SVHNArrayTemp) # don't shuffle with test
np.random.set_state(state1)
np.random.shuffle(SVHN_LabelArrayTemp)
"""select digits"""
temp = Image.open('onesample_VI_data/004065_0.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[0] = temp_arr
temp = Image.open('onesample_VI_data/013671_1.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[1] = temp_arr
temp = Image.open('onesample_VI_data/010555_2.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[2] = temp_arr
temp = Image.open('onesample_VI_data/007640_3.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[3] = temp_arr
temp = Image.open('onesample_VI_data/006708_4.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[4] = temp_arr
temp = Image.open('onesample_VI_data/006687_5.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[5] = temp_arr
temp = Image.open('onesample_VI_data/005710_6.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[6] = temp_arr
temp = Image.open('onesample_VI_data/005178_7.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[7] = temp_arr
temp = Image.open('onesample_VI_data/005045_8.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[8] = temp_arr
temp = Image.open('onesample_VI_data/003504_9.png')
temp_arr=np.array(temp,dtype=np.float)
OnedigitArray[9] = temp_arr

temp = Image.open('onesample_VI_data/004065_0_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[0] = temp_arr
temp = Image.open('onesample_VI_data/013671_1_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[1] = temp_arr
temp = Image.open('onesample_VI_data/010555_2_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[2] = temp_arr
temp = Image.open('onesample_VI_data/007640_3_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[3] = temp_arr
temp = Image.open('onesample_VI_data/006708_4_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[4] = temp_arr
temp = Image.open('onesample_VI_data/006687_5_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[5] = temp_arr
temp = Image.open('onesample_VI_data/005710_6_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[6] = temp_arr
temp = Image.open('onesample_VI_data/005178_7_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[7] = temp_arr
temp = Image.open('onesample_VI_data/005045_8_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[8] = temp_arr
temp = Image.open('onesample_VI_data/003504_9_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[9] = temp_arr

use_num=1000
imgs_arr1=np.empty(shape=(use_num,width,height,channel),dtype=np.float) # 0-9 digits, each digits 100
imgs_arr2=np.empty(shape=(use_num,width,height,channel),dtype=np.float) # random img
imgs_arr1_labels=np.empty(shape=(use_num,classes),dtype=np.int)
imgs_arr2_labels=np.empty(shape=(use_num,classes),dtype=np.int)


SVHN_maskArray=np.empty(shape=(use_num,width,height,channel)) # it is no use, but have to make

for i in range(use_num):
    imgs_arr1[i]=OnedigitArray[i//100]
    SVHN_maskArray[i]=OnedigitArrayMasks[i//100]
    imgs_arr1_labels[i]=i//100
    imgs_arr2[i]=SVHNArrayTemp[i]
    imgs_arr2_labels[i]=SVHN_LabelArrayTemp[i]
print(i)


imgs_arr1=imgs_arr1/255.
imgs_arr2=imgs_arr2/255.
SVHN_maskArray=SVHN_maskArray/255.

save_dir='npz_datas/'
np.savez(save_dir +'SVHN10_img_N' + str(use_num) +'x' + str(width) +'x' + str(height) +'x' + str(channel) +'_testWithLabel_forVI1.npz', images=imgs_arr1, masks=SVHN_maskArray, labelsGT=imgs_arr1_labels)
np.savez(save_dir +'SVHN10_img_N' + str(use_num) +'x' + str(width) +'x' + str(height) +'x' + str(channel) +'_testWithLabel_forVI2.npz', images=imgs_arr2, masks=SVHN_maskArray, labelsGT=imgs_arr2_labels)
