import random
import numpy as np
import os
from PIL import Image
import cv2

Single_Num=2000
classes=10
Num=Single_Num*classes

SVHNArray=np.empty((Num,32,32,3))
SVHN_img_Aux1_arr=np.empty((classes,32,32,3)) # for saving classes SVHN digits
SVHN_masks_Aux1_arr=np.empty((classes,32,32,1)) # for saving classes SVHN digit's masks
# class label
SVHN_class_Label_arr = np.zeros((classes, classes + 1))

# auxiliary dataset
# imgs
SVHN1_img=Image.open('./onesample_data/003633_1.png')
SVHN1_array=np.array(SVHN1_img)
SVHN2_img=Image.open('./onesample_data/001534.png')
SVHN2_array=np.array(SVHN2_img)
SVHN3_img=Image.open('./onesample_data/002517_3.png')
SVHN3_array=np.array(SVHN3_img)
SVHN4_img=Image.open('./onesample_data/000029_4.png')
SVHN4_array=np.array(SVHN4_img)
SVHN5_img=Image.open('./onesample_data/000976_5.png')
SVHN5_array=np.array(SVHN5_img)
SVHN6_img=Image.open('./onesample_data/000069_6.png')
SVHN6_array=np.array(SVHN6_img)
SVHN7_img=Image.open('./onesample_data/000168_7.png')
SVHN7_array=np.array(SVHN7_img)
SVHN8_img=Image.open('./onesample_data/000015_8.png')
SVHN8_array=np.array(SVHN8_img)
SVHN9_img=Image.open('./onesample_data/000618_9.png')
SVHN9_array=np.array(SVHN9_img)
SVHN10_img=Image.open('./onesample_data/001807_10.png')
SVHN10_array=np.array(SVHN10_img)
SVHN_img_Aux1_arr[0]=SVHN10_array
SVHN_img_Aux1_arr[1]=SVHN1_array
SVHN_img_Aux1_arr[2]=SVHN2_array
SVHN_img_Aux1_arr[3]=SVHN3_array
SVHN_img_Aux1_arr[4]=SVHN4_array
SVHN_img_Aux1_arr[5]=SVHN5_array
SVHN_img_Aux1_arr[6]=SVHN6_array
SVHN_img_Aux1_arr[7]=SVHN7_array
SVHN_img_Aux1_arr[8]=SVHN8_array
SVHN_img_Aux1_arr[9]=SVHN9_array
# labels
for i in range(classes): # [0,9)
    SVHN_class_Label_arr[i][i] = 1  # for i
# print(SVHN_class_Label_arr)
# exit()

# masks
SVHN1_mask_img=Image.open('./onesample_data/003633_1_mask.png')
SVHN1_mask_array=np.array(SVHN1_mask_img).reshape((32,32,1))
SVHN2_mask_img=Image.open('./onesample_data/001534_mask.png')
SVHN2_mask_array=np.array(SVHN2_mask_img).reshape((32,32,1))
SVHN3_mask_img=Image.open('./onesample_data/002517_3_mask.png')
SVHN3_mask_array=np.array(SVHN3_mask_img).reshape((32,32,1))
SVHN4_mask_img=Image.open('./onesample_data/000029_4_mask.png')
SVHN4_mask_array=np.array(SVHN4_mask_img).reshape((32,32,1))
SVHN5_mask_img=Image.open('./onesample_data/000976_5_mask.png')
SVHN5_mask_array=np.array(SVHN5_mask_img).reshape((32,32,1))
SVHN6_mask_img=Image.open('./onesample_data/000069_6_mask.png')
SVHN6_mask_array=np.array(SVHN6_mask_img).reshape((32,32,1))
SVHN7_mask_img=Image.open('./onesample_data/000168_7_mask.png')
SVHN7_mask_array=np.array(SVHN7_mask_img).reshape((32,32,1))
SVHN8_mask_img=Image.open('./onesample_data/000015_8_mask.png')
SVHN8_mask_array=np.array(SVHN8_mask_img).reshape((32,32,1))
SVHN9_mask_img=Image.open('./onesample_data/000618_9_mask.png')
SVHN9_mask_array=np.array(SVHN9_mask_img).reshape((32,32,1))
SVHN10_mask_img=Image.open('./onesample_data/001807_10_mask.png')
SVHN10_mask_array=np.array(SVHN10_mask_img).reshape((32,32,1))
SVHN_masks_Aux1_arr[0]=SVHN10_mask_array
SVHN_masks_Aux1_arr[1]=SVHN1_mask_array
SVHN_masks_Aux1_arr[2]=SVHN2_mask_array
SVHN_masks_Aux1_arr[3]=SVHN3_mask_array
SVHN_masks_Aux1_arr[4]=SVHN4_mask_array
SVHN_masks_Aux1_arr[5]=SVHN5_mask_array
SVHN_masks_Aux1_arr[6]=SVHN6_mask_array
SVHN_masks_Aux1_arr[7]=SVHN7_mask_array
SVHN_masks_Aux1_arr[8]=SVHN8_mask_array
SVHN_masks_Aux1_arr[9]=SVHN9_mask_array

# print('read object images finished')
processNumber = Num
width = 32
height = 32
ch=3

# aux1 imgs and gts
Aux1_SVHN2Array=np.ones((processNumber,width,height,ch))*255
Aux1_SVHN2Array_GT=np.ones((processNumber,width,height,ch))*255
# aux1 class label
Aux1_SVHN2Array_label = np.empty((processNumber, classes + 1))  # aux1 and aux2 label is same

# aux2 imgs ans gts
Aux2_SVHN2Array=np.ones((processNumber,width,height,ch))*255
Aux2_SVHN2Array_GT=np.ones((processNumber,width,height,ch))*255

BgRndomIndexAux1=np.arange(processNumber)
np.random.shuffle(BgRndomIndexAux1)

BgRndomIndexAux2=np.arange(processNumber)
np.random.shuffle(BgRndomIndexAux2)
# print(BgRndomIndex)

# unlabel dataset
for imgFolderName in range(1,classes+1,1):
    for k in range(Single_Num):
        temp = Image.open('SVHNimages/{}/{:06d}.png'.format(imgFolderName,k+1))
        temp_arr=np.array(temp,dtype=np.float)
        SVHNArray[(imgFolderName-1)*Single_Num+k] = temp_arr
np.random.shuffle(SVHNArray) # shuffle

# masks
masksForAux1Array=np.ones((processNumber,width,height,ch))*255

# selective color for digit 0 (label 10)
selectiveColor0=np.array([[127,137,104],[215,221,237],[239,255,254],[45,18,12],[33,48,119],[53,43,52],[78,110,152],[78,37,41],[40,50,109]]) # R G B
SVHN2RandonColor0_Aux1=np.random.randint(0,len(selectiveColor0),(processNumber))
SVHN2RandonColor0_Aux2=np.random.randint(0,len(selectiveColor0),(processNumber))
# selective color for digit 1
selectiveColor1= np.array([[206,214,228],[251,255,255],[44,54,115],[34,25,41],[16,21,44],[135,57,71],[76,83,102],[67,83,74],[115,33,40],[23,70,50],[136,126,91]]) # R G B
SVHN2RandonColor1_Aux1=np.random.randint(0,len(selectiveColor1),(processNumber))
SVHN2RandonColor1_Aux2=np.random.randint(0,len(selectiveColor1),(processNumber))
# selective color for digit 2
selectiveColor2=np.array([[245,255,255],[29,37,54],[223,224,184],[100,108,111],[118,56,71],[102,105,102],[133,122,91],[94,78,81],[126,108,80],[159,157,144],[148,160,162],
                         [53,58,10],[223,178,12],[131,71,99],[192,112,159],[21,8,24],[53,27,67],[186,223,64],[21,32,90]]) # R G B
SVHN2RandonColor2_Aux1=np.random.randint(0,len(selectiveColor2),(processNumber))
SVHN2RandonColor2_Aux2=np.random.randint(0,len(selectiveColor2),(processNumber))
# selective color for digit 3
selectiveColor3=np.array([[244,255,255],[122,124,167],[78,104,150],[33,19,16],[127,135,118],[105,149,170],[150,129,146],[127,135,147],[116,25,58],[52,228,245]]) # R G B
SVHN2RandonColor3_Aux1=np.random.randint(0,len(selectiveColor3),(processNumber))
SVHN2RandonColor3_Aux2=np.random.randint(0,len(selectiveColor3),(processNumber))
# selective color for digit 4
selectiveColor4=np.array([[162,161,151],[144,140,163],[87,106,134],[241,249,243],[56,56,58],[172,160,128],[58,44,72],[140,51,50],[57,76,199],[132,93,24]]) # R G B
SVHN2RandonColor4_Aux1=np.random.randint(0,len(selectiveColor4),(processNumber))
SVHN2RandonColor4_Aux2=np.random.randint(0,len(selectiveColor4),(processNumber))
# selective color for digit 5
selectiveColor5=np.array([[239,255,252],[123,126,141],[86,75,89],[16,9,42],[100,77,50],[69,42,60],[131,69,77],[59,84,106],[62,63,84],[94,80,57],[139,143,143]]) # R G B
SVHN2RandonColor5_Aux1=np.random.randint(0,len(selectiveColor5),(processNumber))
SVHN2RandonColor5_Aux2=np.random.randint(0,len(selectiveColor5),(processNumber))
# selective color for digit 6
selectiveColor6=np.array([[65,44,108],[156,170,203],[59,38,55],[68,66,67],[127,79,63],[115,57,79],[76,49,60],[248,248,249],[17,16,119]]) # R G B
SVHN2RandonColor6_Aux1=np.random.randint(0,len(selectiveColor6),(processNumber))
SVHN2RandonColor6_Aux2=np.random.randint(0,len(selectiveColor6),(processNumber))
# selective color for digit 7
selectiveColor7=np.array([[131,137,158],[30,21,83],[45,29,40],[100,81,54],[92,20,36],[213,213,215],[51,74,60],[87,79,69],[14,14,22]]) # R G B
SVHN2RandonColor7_Aux1=np.random.randint(0,len(selectiveColor7),(processNumber))
SVHN2RandonColor7_Aux2=np.random.randint(0,len(selectiveColor7),(processNumber))
# selective color for digit 8
selectiveColor8=np.array([[78,69,72],[181,182,185],[86,94,112],[38,18,86],[68,31,20],[253,251,253],[72,36,34],[204,100,124],[19,97,194]]) # R G B
SVHN2RandonColor8_Aux1=np.random.randint(0,len(selectiveColor8),(processNumber))
SVHN2RandonColor8_Aux2=np.random.randint(0,len(selectiveColor8),(processNumber))
# selective color for digit 9
selectiveColor9=np.array([[35,28,66],[135,124,124],[55,51,50],[40,40,81],[61,50,58],[246,249,253],[58,43,39],[124,66,41],[94,29,22]]) # R G B
SVHN2RandonColor9_Aux1=np.random.randint(0,len(selectiveColor9),(processNumber))
SVHN2RandonColor9_Aux2=np.random.randint(0,len(selectiveColor9),(processNumber))

#=================generate paired samples============================
for i  in range(processNumber):
    """
    for supervised auxiliary dataset
    """
    # select one digit of given
    ind=np.random.randint(0,classes)
    aux1_SVHN2_array=SVHN_img_Aux1_arr[ind]
    Aux1_SVHN2Array_label[i] = SVHN_class_Label_arr[ind] # label
    masksForAux1Array[i,:,:,0]=np.squeeze(SVHN_masks_Aux1_arr[ind])
    masksForAux1Array[i, :, :, 1] = np.squeeze(SVHN_masks_Aux1_arr[ind])
    masksForAux1Array[i, :, :, 2] = np.squeeze(SVHN_masks_Aux1_arr[ind])
    if ind==0:
        selectiveColor=selectiveColor0
        SVHN2RandonColor_Aux1=SVHN2RandonColor0_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor0_Aux2
    elif ind==1:
        selectiveColor=selectiveColor1
        SVHN2RandonColor_Aux1=SVHN2RandonColor1_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor1_Aux2
    elif ind ==2:
        selectiveColor = selectiveColor2
        SVHN2RandonColor_Aux1=SVHN2RandonColor2_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor2_Aux2
    elif ind ==3:
        selectiveColor = selectiveColor3
        SVHN2RandonColor_Aux1=SVHN2RandonColor3_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor3_Aux2
    elif ind ==4:
        selectiveColor = selectiveColor4
        SVHN2RandonColor_Aux1=SVHN2RandonColor4_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor4_Aux2
    elif ind ==5:
        selectiveColor = selectiveColor5
        SVHN2RandonColor_Aux1=SVHN2RandonColor5_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor5_Aux2
    elif ind == 6:
        selectiveColor = selectiveColor6
        SVHN2RandonColor_Aux1 = SVHN2RandonColor6_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor6_Aux2
    elif ind == 7:
        selectiveColor = selectiveColor7
        SVHN2RandonColor_Aux1 = SVHN2RandonColor7_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor7_Aux2
    elif ind == 8:
        selectiveColor = selectiveColor8
        SVHN2RandonColor_Aux1 = SVHN2RandonColor8_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor8_Aux2
    elif ind ==9:
        selectiveColor = selectiveColor9
        SVHN2RandonColor_Aux1 = SVHN2RandonColor9_Aux1
        SVHN2RandonColor_Aux2 = SVHN2RandonColor9_Aux2
    # ============== SVHN2_aux1 ===================
    # mask 32x32
    mask= SVHN_masks_Aux1_arr[ind]
    # cv2.imshow('mask',mask/255.) # debug
    # cv2.waitKey()
    bg_mask=np.ones_like(mask)*255-mask
    ran=np.random.randint(0,101)
    if ran<10:
        SVHN2_temp = aux1_SVHN2_array
    else:
        r = selectiveColor[SVHN2RandonColor_Aux1[i]][0]
        g = selectiveColor[SVHN2RandonColor_Aux1[i]][1]
        b = selectiveColor[SVHN2RandonColor_Aux1[i]][2]
        SVHN2_temp =np.concatenate([mask/255.0*r,mask/255.0*g,mask/255.0*b],axis=2)
        # cv2.imshow('ii',SVHN2_temp)
        # cv2.waitKey()
    # generate noise data
    uniform_noise=np.random.uniform(-1,1,size=(32,32,3))
    SVHN2_temp=SVHN2_temp+uniform_noise*30 # adding noise
    SVHN2_temp1=np.clip(SVHN2_temp,0,255) # constrain to 0-255
    # cv2.imshow('svhn2',SVHN2_temp1/255.)
    # cv2.waitKey()

    bg=SVHNArray[BgRndomIndexAux1[i]]
    bg=bg[0:7,0:7,:]
    bg_img=Image.fromarray(np.uint8(bg))
    bg_img=bg_img.resize((width,height))
    bg1=np.array(bg_img)
    # cv2.imshow('bg',bg1/255.)
    # cv2.waitKey()
    mask1=mask/255.
    bg_mask1=bg_mask/255.
    temp=SVHN2_temp1*mask1+bg1*bg_mask1
    # cv2.imshow('temp1',temp/255.)
    # cv2.waitKey()
    Aux1_SVHN2Array[i]=temp
    # ============== SVHN2_aux2 ===================
    ran = np.random.randint(0, 101)
    if ran<10:
        SVHN2_temp2 = aux1_SVHN2_array
    else:
        r = selectiveColor[SVHN2RandonColor_Aux2[i]][0]
        g = selectiveColor[SVHN2RandonColor_Aux2[i]][1]
        b = selectiveColor[SVHN2RandonColor_Aux2[i]][2]
        SVHN2_temp2 = np.concatenate([mask / 255.0 * r, mask / 255.0 * g, mask / 255.0 * b], axis=2)

    # generate noise data
    uniform_noise = np.random.uniform(-1, 1, size=(32, 32, 3))

    SVHN2_temp2 = SVHN2_temp2 + uniform_noise * 30  # adding noise
    SVHN2_temp2 = np.clip(SVHN2_temp2, 0, 255)  # constrain to 0-255
    # cv2.imshow('flow', SVHN2_temp2 / 255.)
    # cv2.waitKey()

    bg = SVHNArray[BgRndomIndexAux2[i]]
    bg = bg[0:7, 0:7, :]
    bg_img = Image.fromarray(np.uint8(bg))
    bg_img = bg_img.resize((width, height))
    bg2 = np.array(bg_img)
    # cv2.imshow('bg', bg2 / 255.)
    # cv2.waitKey()
    mask1 = mask / 255.
    bg_mask1 = bg_mask / 255.
    temp = SVHN2_temp2 * mask1 + bg2 * bg_mask1
    # cv2.imshow('temp2', temp / 255.)
    # cv2.waitKey()
    Aux2_SVHN2Array[i] = temp
    # ========== gts ===============
    tmp = SVHN2_temp2 * mask1 + bg1 * bg_mask1
    # cv2.imshow('gt1', tmp / 255.)
    # cv2.waitKey()
    Aux1_SVHN2Array_GT[i] = tmp
    tmp = SVHN2_temp1 * mask1 + bg2 * bg_mask1
    # cv2.imshow('gt2', tmp / 255.)
    # cv2.waitKey()
    Aux2_SVHN2Array_GT[i] = tmp
    # break

# normlize to 0~1
SVHNArray=SVHNArray/255.0
masksForAux1Array=masksForAux1Array/255.0
Aux1_SVHN2Array=Aux1_SVHN2Array/255.0
Aux1_SVHN2Array_GT=Aux1_SVHN2Array_GT/255.0
Aux2_SVHN2Array=Aux2_SVHN2Array/255.0
Aux2_SVHN2Array_GT=Aux2_SVHN2Array_GT/255.0

print('done')
save_dir='./npz_datas/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
np.savez(save_dir+'SVHN10WithBg_img1_oneguided_N'+str(processNumber)+'x'+str(width)+'x'+str(height)+'x'+str(ch)+'_train.npz',images=SVHNArray,masks=masksForAux1Array)
np.savez(save_dir+'SVHN10WithBg_mask1_oneguided_N'+str(processNumber)+'x'+str(width)+'x'+str(height)+'x'+str(ch)+'_train.npz',images=Aux1_SVHN2Array_label,masks=masksForAux1Array)
np.savez(save_dir+'SVHN10WithBg_aux1_GT1_oneguided_N'+str(processNumber)+'x'+str(width)+'x'+str(height)+'x'+str(ch)+'_train.npz',images=Aux1_SVHN2Array,masks=Aux1_SVHN2Array_GT)
np.savez(save_dir+'SVHN10WithBg_aux2_GT2_oneguided_N'+str(processNumber)+'x'+str(width)+'x'+str(height)+'x'+str(ch)+'_train.npz',images=Aux2_SVHN2Array,masks=Aux2_SVHN2Array_GT)


