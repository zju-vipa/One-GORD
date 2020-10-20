import numpy as np
from PIL import Image

Single_Num=320 # number of each digits in SVHN 320*10=3200=64*50
width=32
height=32
channel=3
classes=10
Total_Num=Single_Num*classes 
SVHNArray1=np.empty(shape=(Total_Num, width, height, channel))
SVHN_LabelArray1=np.zeros(shape=(Total_Num, classes))
SVHNArray2=np.empty(shape=(Total_Num, width, height, channel))
SVHN_LabelArray2=np.zeros(shape=(Total_Num, classes))
# adding
for imgFolderName in range(1,classes+1,1):
    for k in range(2000,2000+Single_Num):
        temp = Image.open('SVHNimages/{}/{:06d}.png'.format(imgFolderName,k+1))
        temp_arr=np.array(temp,dtype=np.float)
        SVHNArray1[(imgFolderName - 1) * Single_Num + k - 2000] = temp_arr
        SVHNArray2[(imgFolderName - 1) * Single_Num + k - 2000] = temp_arr
        if imgFolderName==10:
            SVHN_LabelArray1[(imgFolderName - 1) * Single_Num + k - 2000, 0] = 1
            SVHN_LabelArray2[(imgFolderName - 1) * Single_Num + k - 2000, 0] = 1
        else:
            SVHN_LabelArray1[(imgFolderName - 1) * Single_Num + k - 2000, imgFolderName]=1
            SVHN_LabelArray2[(imgFolderName - 1) * Single_Num + k - 2000, imgFolderName] = 1
# print((imgFolderName-1)*Single_Num+k-2000) # debug end
# print(SVHN_LabelArray1[0:3])
# print(SVHN_LabelArray1[307:322])
# print(SVHN_LabelArray1[3195:3200])
# exit
state1=np.random.get_state()
np.random.shuffle(SVHNArray1) # don't shuffle with test
np.random.set_state(state1)
np.random.shuffle(SVHN_LabelArray1)

state2=np.random.get_state()
np.random.shuffle(SVHNArray2) # don't shuffle with test
np.random.set_state(state2)
np.random.shuffle(SVHN_LabelArray2)


SVHN_maskArray=np.empty(shape=(Total_Num,classes)) # it is no use, but have to make
for i in range(Total_Num):
    SVHN_maskArray[i]=np.zeros((classes))

# normalize to 0-1
SVHNArray1= SVHNArray1 / 255.
SVHN_maskArray=SVHN_maskArray/255.

SVHNArray2= SVHNArray2 / 255.
# SVHNArray2=SVHN_maskArray/255.

save_dir='npz_datas/'
np.savez(save_dir +'SVHN10_img_N' + str(Total_Num) +'x' + str(width) +'x' + str(height) +'x' + str(channel) +'_testWithLabel_forVisual1.npz', images=SVHNArray1, masks=SVHN_maskArray, labelsGT=SVHN_LabelArray1)
np.savez(save_dir +'SVHN10_img_N' + str(Total_Num) +'x' + str(width) +'x' + str(height) +'x' + str(channel) +'_testWithLabel_forVisual2.npz', images=SVHNArray2, masks=SVHN_maskArray, labelsGT=SVHN_LabelArray2)
