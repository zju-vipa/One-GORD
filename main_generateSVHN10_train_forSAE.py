import numpy as np
from PIL import Image
import os

Single_Num=2000 # number of each digits
width=32
height=32
channel=3
classes=10
Total_Num=Single_Num*classes
SVHNArray=np.empty(shape=(Total_Num,width,height,channel))
SVHN_LabelArray=np.zeros(shape=(Total_Num,classes+1))
# adding
for imgFolderName in range(1,classes+1,1):
    for k in range(2000,2000+Single_Num):
        temp = Image.open('./SVHNimages/{}/{:06d}.png'.format(imgFolderName,k+1))
        temp_arr=np.array(temp,dtype=np.float)
        SVHNArray[(imgFolderName-1)*Single_Num+k-2000] = temp_arr
        if imgFolderName==10:
            SVHN_LabelArray[(imgFolderName - 1) * Single_Num + k - 2000, 0] = 1
        else:
            SVHN_LabelArray[(imgFolderName-1)*Single_Num+k-2000,imgFolderName]=1
# print((imgFolderName-1)*Single_Num+k-2000) # debug end
# print(SVHN_LabelArray[0:3])
# print(SVHN_LabelArray[307:322])
# print(SVHN_LabelArray[3195:3200])
# exit
# np.random.shuffle(SVHNArray) # don't shuffle with test
state=np.random.get_state()
np.random.shuffle(SVHNArray)

np.random.set_state(state)
np.random.shuffle(SVHN_LabelArray)

SVHN_maskArray=np.empty(shape=(Total_Num,classes)) # it is no use, but have to make
for i in range(Total_Num):
    SVHN_maskArray[i]=np.zeros((classes))

# normalize to 0-1
SVHNArray0_1=SVHNArray/255.
# SVHN_maskArray0_1=SVHN_maskArray/255.

save_dir='./npz_datas/'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)
np.savez(save_dir+'SVHN10_img_N'+str(Total_Num)+'x'+str(width)+'x'+str(height)+'x'+str(channel)+'_TrainWithLabel_forSAE.npz',images=SVHNArray0_1,masks=SVHN_maskArray,gts=SVHN_LabelArray)