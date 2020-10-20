import numpy as np
from PIL import Image
import cv2

Single_Num=100
classes=10
Num=Single_Num*classes
width, height=32,32
channel=3

SVHNArray=np.empty((Num,32,32,3))
OnedigitArray=np.empty((classes,32,32,3)) # for saving classes SVHN digits
OnedigitArrayMasks=np.empty((classes,32,32,1)) # for saving classes SVHN digit's masks

# auxiliary dataset
# imgs
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

for imgFolderName in range(1,classes+1,1):
    for k in range(Single_Num):
        temp = Image.open('SVHNimages/{}/{:06d}.png'.format(imgFolderName,k+1))
        temp_arr=np.array(temp,dtype=np.float)
        SVHNArray[(imgFolderName-1)*Single_Num+k] = temp_arr
np.random.shuffle(SVHNArray) # shuffle

M_Array=np.empty((Num,32,32,3))
BgRndomIndexAux1=np.arange(Num)
np.random.shuffle(BgRndomIndexAux1)
# Num=1000
for i in range(classes):
    mask = np.concatenate([OnedigitArrayMasks[i], OnedigitArrayMasks[i], OnedigitArrayMasks[i]],axis=2)
    temp = OnedigitArray[i]
    mask=mask/255.
    for j in range(Single_Num):
        bg = SVHNArray[BgRndomIndexAux1[i*Single_Num+j]]
        bg = bg[0:7, 0:7, :]
        bg_img = Image.fromarray(np.uint8(bg))
        bg_img = bg_img.resize((width, height))
        bg1 = np.array(bg_img)
        # synthesis
        img=temp*mask+bg1*(1-mask)
        # cv2.imshow('tt',img/255)
        # cv2.waitKey()
        M_Array[i*Single_Num+j]=img
# exit()
SVHN_maskArray=np.empty(shape=(Num,classes)) # it is no use, but have to make for function
for i in range(Num):
    SVHN_maskArray[i]=np.zeros((classes))

M_Array=M_Array/255.

save_dir='npz_datas/'
np.savez(save_dir +'SVHN10_img_N' + str(Num) +'x' + str(width) +'x' + str(height) +'x' + str(channel) +'_testForModularity.npz', images=M_Array, masks=SVHN_maskArray)
print('done ')