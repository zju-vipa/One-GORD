import numpy as np
import sys
import scipy.misc
import time
import os
#from lib.models.data_providers import TeapotsDataProvider, FlexibleImageDataProvider
from lib.models.data_providers_diae import TeapotsDataProvider, FlexibleImageDataProvider
from lib.zero_shot import get_gap_ids

class DataManager(object):
    def __init__(self, data_dir, dataset_name1,dataset_name2, batch_size, image_shape, 
                 shuffle=False,file_ext='.npz', train_fract=0.8, 
                 dev_fract=None, inf=True, supervised=False):
        
        self.data_dir = data_dir
        self.dataset_name1 = dataset_name1
        self.dataset_name2 = dataset_name2
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.shuffle = shuffle
        self.file_ext = file_ext.strip()
        self.train_fract = train_fract
        self.dev_fract = dev_fract
        self.inf = inf
        self.supervised = supervised
        
        #self.file_ext == '.npz':
        self._data_provider = TeapotsDataProvider
        self.__create_data_provider = self.__create_data_provider_npz
        imgs1 = np.load(os.path.join(self.data_dir, self.dataset_name1 + ".npz"))['images']
        imgs2 = np.load(os.path.join(self.data_dir, self.dataset_name2 + ".npz"))['images']
        masks1= np.load(os.path.join(self.data_dir, self.dataset_name1 + ".npz"))['masks']
        masks2= np.load(os.path.join(self.data_dir, self.dataset_name2 + ".npz"))['masks']
        #print imgs1.shape
        self.n_samples = len(imgs1)


        self.__set_data_splits()
        imgs, masks, gts = self.__get_datasets(imgs1,imgs2,masks1,masks2)
        self.__create_data_provider(imgs,masks, gts)
    
    def __set_data_splits(self):
        if self.dev_fract is None:
            self.dev_fract = round((1. - self.train_fract) / 2., 3)
        self.n_train = int(self.n_samples * self.train_fract)
        self.n_dev = int(self.n_samples * self.dev_fract)
        self.n_test = self.n_samples - (self.n_train + self.n_dev)
        print("Train set: {0}\nDev set: {1}\nTest set: {2}".format(
              self.n_train, self.n_dev, self.n_test))
                                     
    def __split_data(self, data, start_idx, end_idx):
        return data[start_idx:end_idx]

    def __get_datasets(self, imgs1,imgs2,masks1,masks2):
        # dataset1          
        train_imgs1 = self.__split_data(imgs1, 0, self.n_train)
        dev_imgs1   = self.__split_data(imgs1, self.n_train, self.n_train + self.n_dev)
        test_imgs1  = self.__split_data(imgs1, self.n_train + self.n_dev,self.n_train + self.n_dev + self.n_test)
        # dataset2        
        train_imgs2 = self.__split_data(imgs2, 0, self.n_train)
        dev_imgs2   = self.__split_data(imgs2, self.n_train, self.n_train + self.n_dev)
        test_imgs2  = self.__split_data(imgs2, self.n_train + self.n_dev,self.n_train + self.n_dev + self.n_test)
        # mask1
        train_masks1 = self.__split_data(masks1, 0, self.n_train)
        dev_masks1   = self.__split_data(masks1, self.n_train, self.n_train + self.n_dev)
        test_masks1  = self.__split_data(masks1, self.n_train + self.n_dev,self.n_train + self.n_dev + self.n_test)
        # mask2 
        train_masks2 = self.__split_data(masks2, 0, self.n_train)
        dev_masks2   = self.__split_data(masks2, self.n_train, self.n_train + self.n_dev)
        test_masks2  = self.__split_data(masks2, self.n_train + self.n_dev,self.n_train + self.n_dev + self.n_test)
          
        if self.supervised:
            gts1 = np.load(os.path.join(self.data_dir, self.dataset_name1 + ".npz"))['gts'] #targets
            train_gts1  = self.__split_data(self.gts1, 0,self.n_train)
            dev_gts1    = self.__split_data(self.gts1, self.n_train,self.n_train + self.n_dev)
            test_gts1   = self.__split_data(self.gts1, self.n_train + self.n_dev,self.n_train + self.n_dev + self.n_test)

            gts2 = np.load(os.path.join(self.data_dir, self.dataset_name2 + ".npz"))['gts'] #targets
            train_gts2  = self.__split_data(self.gts2, 0,self.n_train)
            dev_gts2    = self.__split_data(self.gts2, self.n_train,self.n_train + self.n_dev)
            test_gts2   = self.__split_data(self.gts2, self.n_train + self.n_dev,self.n_train + self.n_dev + self.n_test)
        else:
            train_gts1, dev_gts1, test_gts1,train_gts2, dev_gts2, test_gts2  = None, None, None, None, None, None
        return (train_imgs1, dev_imgs1, test_imgs1,train_imgs2, dev_imgs2, test_imgs2), (train_masks1, dev_masks1, test_masks1,train_masks2, dev_masks2, test_masks2), (train_gts1, dev_gts1, test_gts1,train_gts2, dev_gts2, test_gts2)
   

    def __create_data_provider_npz(self, imgs,masks, gts):
        train_imgs1, dev_imgs1, test_imgs1,train_imgs2, dev_imgs2, test_imgs2 = imgs
        train_masks1, dev_masks1, test_masks1, train_masks2, dev_masks2, test_masks2=masks
        train_gts1, dev_gts1, test_gts1,train_gts2, dev_gts2, test_gts2 = gts
        #               
        self.train1 = self._data_provider(train_imgs1,train_masks1, train_gts1, self.batch_size, 
                                          inf=self.inf, shuffle_order=self.shuffle)
        self.dev1   = self._data_provider(dev_imgs1,dev_masks1, dev_gts1, self.batch_size,
                                          inf=self.inf, shuffle_order=self.shuffle)
        self.test1  = self._data_provider(test_imgs1,test_masks1, test_gts1, self.batch_size,
                                          inf=self.inf, shuffle_order=self.shuffle)
        #
        self.train2 = self._data_provider(train_imgs2, train_masks2, train_gts2, self.batch_size, 
                                          inf=self.inf, shuffle_order=self.shuffle)
        self.dev2   = self._data_provider(dev_imgs2,dev_masks2, dev_gts2, self.batch_size,
                                          inf=self.inf, shuffle_order=self.shuffle)
        self.test2  = self._data_provider(test_imgs2,test_masks2, test_gts2, self.batch_size,
                                          inf=self.inf, shuffle_order=self.shuffle)                                      
    def get_iterators(self):
        return self.train1, self.dev1, self.test1,self.train2, self.dev2, self.test2
    
    def set_divisor_batch_size(self):
        '''Ensure batch size evenly divides into n_samples.'''
        while self.n_samples % self.batch_size != 0:
            self.batch_size -= 1

            
class TeapotsDataManager(DataManager):
    def __init__(self, data_dir,data1Name ,data2Name,  batch_size, image_shape, shuffle=False, 
                 file_ext='.npz', train_fract=0.8, 
                 dev_fract=None, inf=True, supervised=False):
        #data1Name="geometry1_30000"
        #data2Name="geometry2_30000"
        print('get data From data1 and data2'+data1Name+'|'+data2Name)
        super(TeapotsDataManager, self).__init__(data_dir,data1Name ,data2Name, 
              batch_size, image_shape, shuffle, file_ext,
              train_fract, dev_fract, inf, supervised)
        
        if self.file_ext == '.npz':
            self._data_provider = TeapotsDataProvider #transpose image batch in provider