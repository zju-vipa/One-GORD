
import numpy as np
# import os
# import cv2
from PIL import Image

width,height=32,32
OnedigitArrayMasks=np.empty(shape=(10, width, height, 1)) # save one digit
temp = Image.open('../onesample_VI_data/004065_0_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[0] = temp_arr
temp = Image.open('../onesample_VI_data/013671_1_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[1] = temp_arr
temp = Image.open('../onesample_VI_data/010555_2_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[2] = temp_arr
temp = Image.open('../onesample_VI_data/007640_3_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[3] = temp_arr
temp = Image.open('../onesample_VI_data/006708_4_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[4] = temp_arr
temp = Image.open('../onesample_VI_data/006687_5_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[5] = temp_arr
temp = Image.open('../onesample_VI_data/005710_6_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[6] = temp_arr
temp = Image.open('../onesample_VI_data/005178_7_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[7] = temp_arr
temp = Image.open('../onesample_VI_data/005045_8_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[8] = temp_arr
temp = Image.open('../onesample_VI_data/003504_9_mask.png')
temp_arr=np.array(temp,dtype=np.float).reshape((32,32,1))
OnedigitArrayMasks[9] = temp_arr

img_root='./VisualIntegrityImgsResults/'
sub_arr=np.zeros(shape=(1000,),dtype=np.float)
mask=np.zeros(shape=(width, height, 3),dtype=np.float)

print('calculating...')
for i in range(1000):
    origin_img=img_root+'iter{}_origX1_img.png'.format(i)
    swapped_img=img_root+'iter{}x1fg_x2bg_out_img.png'.format(i)
    ori_img_arr = Image.open(origin_img)
    swap_img_arr = Image.open(swapped_img)
    ori_img_arr=np.array(ori_img_arr)
    swap_img_arr=np.array(swap_img_arr)
    # ori_img_arr=cv2.imread(origin_img)
    # swap_img_arr=cv2.imread(swapped_img)
    # ori_img_arr=cv2.cvtColor(ori_img_arr,cv2.COLOR_BGR2RGB)
    # swap_img_arr=cv2.cvtColor(swap_img_arr,cv2.COLOR_BGR2RGB)
    # mask
    mask_t=OnedigitArrayMasks[i//100]/255.
    mask=np.concatenate([mask_t,mask_t,mask_t],axis=2)
    sub_arr[i]=abs((mask*ori_img_arr-mask*swap_img_arr).mean())
    # print(sub_arr[i])
print('done')
print('Avg:{}'.format(sub_arr.mean()))

