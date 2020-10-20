import os, sys
import time
import re
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import lib.models as lib
from lib.models import params_with_name
from lib.models.save_images import save_images
from lib.models.distributions import Bernoulli, Gaussian, Product
from lib.models.nets_32x32_small import NetsRetreiver, NetsRetreiverWithClassifier

TINY = 1e-8
SEED = 123
ch=1
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter

class DIAE(object):
    def __init__(self, session, arch,lr,alpha,beta,latent_dim,latent_num,class_net_unit_num,output_dim, batch_size, image_shape, exp_name, dirs,
        vis_reconst):
        """
        :type output_dist: Distribution
        :type z_dist: Gaussian
        """
        self.session = session
        self.arch = arch
        self.lr=lr
        self.alpha=alpha
        self.beta=beta
        self.latent_dim=latent_dim
        self.latent_num=latent_num
        self.class_net_unit_num=class_net_unit_num
        self.output_dim=output_dim
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.exp_name = exp_name
        self.dirs = dirs
        self.vis_reconst = vis_reconst
        
        self.__build_graph()

    def __build_graph(self):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        self.is_training = tf.placeholder(tf.bool)
        self.x1 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.aux1_mask = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        # auxilary dataset
        self.aux1 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.aux2 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.aux_GT1 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.aux_GT2 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.class_gt0 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+9]))
        self.class_gt1 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+9]))
        self.class_gt2 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+9]))
        self.class_gt3 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+9]))
        self.class_gt4 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num + 9]))
        self.class_gt5 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num + 9]))
        self.class_gt6 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num + 9]))
        self.class_gt7 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num + 9]))
        self.class_gt8 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num + 9]))
        self.class_gt9 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num + 9]))
        self.class_gt10 = tf.placeholder(tf.float32, shape=[None] + list([self.latent_num + 9]))
        # onesample labels
        self.aux_class_gt=tf.placeholder(tf.float32,shape=[None]+list([self.latent_num+9]))

        # Normalize + reshape 'real' input data
        norm_x1 = 2*(tf.cast(self.x1, tf.float32)-.5)

        norm_aux1 = 2*(tf.cast(self.aux1, tf.float32)-.5)
        norm_aux2 = 2*(tf.cast(self.aux2, tf.float32)-.5)
        norm_aux_GT1 = 2*(tf.cast(self.aux_GT1, tf.float32)-.5)
        norm_aux_GT2 = 2*(tf.cast(self.aux_GT2, tf.float32)-.5)
        # Set Encoder and Decoder archs
        self.Encoder, self.Decoder,self.Classifier,self.gan_discriminator = NetsRetreiverWithClassifier(self.arch) 
    
        # Encode and decode
        self.z1 = self.__Enc(norm_x1)
        self.x1_out = self.__Dec(self.z1)
        # aux data
        self.aux_z1 = self.__Enc(norm_aux1)
        self.aux1_out = self.__Dec(self.aux_z1)
        self.aux_z2 = self.__Enc(norm_aux2)
        self.aux2_out = self.__Dec(self.aux_z2)

        aux1_head,aux1_bg=tf.split(self.aux_z1,2,axis=1)
        aux2_head,aux2_bg=tf.split(self.aux_z2,2,axis=1)

        GT1_z=tf.concat([aux2_head,aux1_bg],axis=1)
        GT2_z=tf.concat([aux1_head,aux2_bg],axis=1)
        self.GT1_out = self.__Dec(GT1_z)
        self.GT2_out = self.__Dec(GT2_z)
        #dual swap
        x1_head,x1_bg=tf.split(self.z1,2,axis=1)
        self.mix_head_out=self.__Dec(tf.concat([aux1_head,x1_bg],axis=1))
        mix_head,mix_bg=tf.split(self.__Enc(self.mix_head_out),2,axis=1)
        x1_dual_out=self.__Dec(tf.concat([x1_head,mix_bg],axis=1))

        self.aux1_mix_head_out=self.__Dec(tf.concat([x1_head,aux1_bg],axis=1))

        # classification loss
        ## for x1
        r_part1, r_part2 = tf.split(self.z1, 2, axis=1)
        c_p0 = self.__Classifier(r_part1)
        c_p1 = self.__Classifier(r_part2)
        ## for aux1
        aux1_r_part1, aux1_r_part2 = tf.split(self.aux_z1, 2, axis=1)
        aux1_c_p0 = self.__Classifier(aux1_r_part1)
        aux1_c_p1 = self.__Classifier(aux1_r_part2)
        ## for aux2
        aux2_r_part1, aux2_r_part2 = tf.split(self.aux_z2, 2, axis=1)
        aux2_c_p0 = self.__Classifier(aux2_r_part1)
        aux2_c_p1 = self.__Classifier(aux2_r_part2)

        # Loss and optimizer
        self.__prep_loss_optimizer(norm_x1,norm_aux1,norm_aux2,norm_aux_GT1,norm_aux_GT2,x1_dual_out,c_p0,c_p1,aux1_c_p0,aux1_c_p1,aux2_c_p0,aux2_c_p1)

    def __Enc(self, x):
        #resnet_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
        z= self.Encoder('Encoder', x, self.image_shape[0], self.latent_dim,self.is_training)
        return z
    
    def __Dec(self, z):
        x_out_logit = self.Decoder('Decoder', z, self.image_shape[0], self.is_training)
        x_out = tf.tanh(x_out_logit)
        return x_out
    
    def __Classifier(self,z):
        x_out= self.Classifier('Classifier', z, self.class_net_unit_num,self.latent_num+9, self.is_training)
        x_out = tf.nn.softmax(x_out)
        return x_out
    
    def __prep_loss_optimizer(self,norm_x1,norm_aux1,norm_aux2,norm_aux_GT1,norm_aux_GT2,x1_dual_out,c_p0,c_p1,aux1_c_p0,aux1_c_p1,aux2_c_p0,aux2_c_p1):
        
        norm_x1= tf.reshape(norm_x1, [-1, self.output_dim])
        norm_aux1= tf.reshape(norm_aux1, [-1, self.output_dim])
        norm_aux2= tf.reshape(norm_aux2, [-1, self.output_dim])
        norm_aux_GT1= tf.reshape(norm_aux_GT1, [-1, self.output_dim])
        norm_aux_GT2= tf.reshape(norm_aux_GT2, [-1, self.output_dim])

        #[Loss1]dual unsupervised img reconstruction loss
        self.rec_img_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(norm_x1 -self.x1_out), axis=1))
        self.rec_aux1_loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(norm_aux1 -self.aux1_out), axis=1))
        self.rec_aux2_loss3 = tf.reduce_mean(tf.reduce_sum(tf.square(norm_aux2 -self.aux2_out), axis=1))
        #swap loss
        self.rec_aux1_swap_loss4 = tf.reduce_mean(tf.reduce_sum(tf.square(norm_aux_GT1 -self.GT1_out), axis=1))
        self.rec_aux2_swap_loss5 = tf.reduce_mean(tf.reduce_sum(tf.square(norm_aux_GT2 -self.GT2_out), axis=1))
        # dual swap loss
        self.rec_dual_loss6 =  tf.reduce_mean(tf.reduce_sum(tf.square(x1_dual_out -self.x1_out), axis=1))
        # head loss
        # segment head and do head loss with mask
        x1_out_img = tf.reshape(self.mix_head_out, shape=[-1] + self.image_shape)  # to img tensor
        x1_out_head = tf.multiply(x1_out_img, self.aux1_mask)
        norm_aux1_img = tf.reshape(norm_aux1, shape=[-1] + self.image_shape)  # to img tensor
        aux1_head = tf.multiply(norm_aux1_img, self.aux1_mask)
        self.head_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x1_out_head - aux1_head), axis=1))
        # classification loss
        # temp_1=self.vec_gt-tf.reduce_sum((self.class_gt1-self.class_gt1*c_p1),1)*tf.reduce_sum((self.class_gt2-self.class_gt2*c_p1),1)
        # self.class1_loss=-tf.reduce_mean(tf.log(temp_1))
        temp=1-tf.reduce_sum((self.class_gt0-self.class_gt0*c_p0),1)*tf.reduce_sum((self.class_gt1-self.class_gt1*c_p0),1)*tf.reduce_sum((self.class_gt2-self.class_gt2*c_p0),1)*tf.reduce_sum((self.class_gt3-self.class_gt3*c_p0),1)*\
             tf.reduce_sum((self.class_gt4-self.class_gt4*c_p0),1)*tf.reduce_sum((self.class_gt5-self.class_gt5*c_p0),1)*tf.reduce_sum((self.class_gt6-self.class_gt6*c_p0),1)*tf.reduce_sum((self.class_gt7-self.class_gt7*c_p0),1)*tf.reduce_sum((self.class_gt8-self.class_gt8*c_p0),1)*\
             tf.reduce_sum((self.class_gt9-self.class_gt9*c_p0),1)
        self.fuzzy_class_loss = -tf.reduce_mean(tf.log(temp))
        self.fuzzy_bg_class_loss= -tf.reduce_mean(self.class_gt10 * tf.log(c_p1))
        # class_loss1 = -tf.reduce_mean(self.class_gt1 * tf.log(c_p1))
        self.aux1_class_loss = -tf.reduce_mean(self.aux_class_gt * tf.log(aux1_c_p0))
        self.aux1_bg_class_loss = -tf.reduce_mean(self.class_gt10 * tf.log(aux1_c_p1))  # [0,0,0,1]
        self.aux2_class_loss = -tf.reduce_mean(self.aux_class_gt * tf.log(aux2_c_p0))
        self.aux2_bg_class_loss = - tf.reduce_mean(self.class_gt10 * tf.log(aux2_c_p1))
        self.class_loss = self.fuzzy_class_loss+self.fuzzy_bg_class_loss+self.aux1_class_loss+  self.aux1_bg_class_loss+self.aux2_class_loss+ self.aux2_bg_class_loss

        self.loss=2*self.rec_img_loss1+self.rec_aux1_loss2+self.rec_aux2_loss3+2*self.rec_aux1_swap_loss4+2*self.rec_aux2_swap_loss5+self.rec_dual_loss6+50*self.head_loss+5*self.class_loss
        lr=self.lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0., beta2=0.9).minimize(self.loss) 
    
        print('Learning rate=')
        print(lr)
        
    def load(self):
        #self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(max_to_keep=3760)
        ckpt = tf.train.get_checkpoint_state(self.dirs['ckpt'])
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = ckpt.model_checkpoint_path
            self.saver.restore(self.session, ckpt_name)
            print("Checkpoint restored: {0}".format(ckpt_name))
            prev_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print('prev_step=')
            print(prev_step)
        
        else:
            print("Failed to find checkpoint.")
            prev_step = 0
        sys.stdout.flush()
        return prev_step + 1

    def load_fixedNum(self,inter_num):
        #self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(max_to_keep=3760)
        ckpt = tf.train.get_checkpoint_state(self.dirs['ckpt'])
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = ckpt.model_checkpoint_path
            ckpt_name_prefix=ckpt_name.split('-')[0]
            ckpt_name_new=ckpt_name_prefix+'-'+str(inter_num)
            self.saver.restore(self.session, ckpt_name_new)
            print("Checkpoint restored: {0}".format(ckpt_name_new))
            prev_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name_new)).group(0))
            print('prev_step=')
            print(prev_step)
        else:
            print("Failed to find checkpoint.")
            prev_step = 0
        sys.stdout.flush()
        return prev_step + 1
    
    def train(self, n_iters, n_iters_per_epoch, stats_iters, ckpt_interval):
        # for save loss  
        count=0      
        self.session.run(tf.global_variables_initializer())
        
        # Fixed GT samples - save
        fixed_x1,fixed_mask_1,_= next(self.train_iter1)
        fixed_x2,fixed_mask_2,_= next(self.train_iter2)
        fixed_aux1,fixed_GT1,_= next(self.train_iter3)
        fixed_aux2,fixed_GT2,_= next(self.train_iter4)

        # print(fixed_x1.shape)
        # print(fixed_mask_1.shape)
        # print(fixed_x2.shape)
        # print(fixed_mask_2.shape)
        
        # print(fixed_aux1.shape)
        # print(fixed_GT1.shape)
        

        fixed_x1= self.session.run(tf.constant(fixed_x1))
        fixed_mask_1= self.session.run(tf.constant(fixed_mask_1))
        # print("fixed_mask_1.shape")
        # print(fixed_mask_1.shape)
        #fixed_x1_2 =fixed_x1.reshape([64,32,32,3])
        #fixed_x1 = ((fixed_x1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(fixed_x1, os.path.join(self.dirs['samples'], 'samples_1_groundtruth.png'))
        ##save_images(fixed_mask_1, os.path.join(self.dirs['samples'], 'mask_1_groundtruth.png'))
        
        fixed_aux1= self.session.run(tf.constant(fixed_aux1))
        fixed_aux2= self.session.run(tf.constant(fixed_aux2))
        #fixed_aux1 = ((fixed_aux1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        #fixed_aux2 = ((fixed_aux2+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(fixed_aux1, os.path.join(self.dirs['samples'], 'aux_1_groundtruth.png'))
        save_images(fixed_aux2, os.path.join(self.dirs['samples'], 'aux_2_groundtruth.png'))
        #
        start_iter = self.load()
        running_cost = 0.
        class_gt0, class_gt1,class_gt2,class_gt3,class_gt4,class_gt5, class_gt6, class_gt7, class_gt8,class_gt9,class_gt10 = self.generateClassificationLabel(self.batch_size)
        _gan_data=fixed_x1
        logs=open('loss_records.txt','w')
        for iteration in range(start_iter, n_iters):
            start_time = time.time()

            
            _data1,_mask1, _ = next(self.train_iter1)
            _aux_label, _, _ = next(self.train_iter2)
            _aux1,_gt1, _ = next(self.train_iter3)
            _aux2,_gt2, _ = next(self.train_iter4)

            _, cost = self.session.run((self.optimizer, self.loss),feed_dict={self.x1:_data1,self.aux_class_gt:_aux_label,self.aux1_mask:_mask1,self.aux1:_aux1,self.aux2:_aux2,self.aux_GT1:_gt1,self.aux_GT2:_gt2,self.is_training:True,self.class_gt0:class_gt0,self.class_gt1:class_gt1,self.class_gt2:class_gt2,self.class_gt3:class_gt3,
                                                                              self.class_gt4:class_gt4, self.class_gt5:class_gt5, self.class_gt6:class_gt6,self.class_gt7:class_gt7, self.class_gt8:class_gt8, self.class_gt9:class_gt9, self.class_gt10:class_gt10})
            running_cost += cost
            
            if iteration % n_iters_per_epoch == 1:
                print("Epoch: {0}".format(iteration // n_iters_per_epoch))
            
            # Print avg stats and dev set stats
            if (iteration < start_iter + 4) or iteration % stats_iters == 0:
                t = time.time()
                dev_data1,dev_mask1, _ = next(self.dev_iter1)
                dev_aux1,dev_gt1, _ = next(self.dev_iter3)
                dev_aux2,dev_gt2, _ = next(self.dev_iter4)
                
                dev_loss,dev_rec_img_loss1,dev_rec_aux1_loss2,dev_rec_aux2_loss3,dev_rec_aux1_swap_loss4,dev_rec_aux2_swap_loss5,dev_rec_dual_loss6,head_loss,class_loss,fuzzy_class,aux1_class_loss,aux2_class_loss,fuzzy_bg_class_loss,aux1_bg_class_loss,aux2_bg_class_loss= \
                    self.session.run([self.loss,self.rec_img_loss1,self.rec_aux1_loss2,self.rec_aux2_loss3,self.rec_aux1_swap_loss4,self.rec_aux2_swap_loss5,self.rec_dual_loss6,self.head_loss,self.class_loss,self.fuzzy_class_loss,self.aux1_class_loss,self.aux2_class_loss,self.fuzzy_bg_class_loss,self.aux1_bg_class_loss,self.aux2_bg_class_loss],
                                     feed_dict={self.x1:dev_data1,self.aux1_mask:dev_mask1,self.aux1:dev_aux1,self.aux2:dev_aux2,self.aux_GT1:dev_gt1,self.aux_GT2:dev_gt2,self.is_training:False,
                                                self.class_gt0: class_gt0, self.class_gt1: class_gt1,self.class_gt2: class_gt2,self.class_gt3:class_gt3,self.aux_class_gt:_aux_label,self.class_gt4:class_gt4, self.class_gt5:class_gt5, self.class_gt6:class_gt6,self.class_gt7:class_gt7, self.class_gt8:class_gt8, self.class_gt9:class_gt9, self.class_gt10:class_gt10})
                
                n_samples = 1. if (iteration < start_iter + 4) else float(stats_iters)
                avg_cost = running_cost / n_samples
                running_cost = 0.
                print("Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}(img1_loss:{3:.1f},aux1_loss2:{4:.1f},aux2_loss3:{5:.1f},aux1_swap_loss:{6:.1f},aux2_swap_loss:{7:.1f},dual_swap_loss:{8:.1f},head loss:{9:.1f},class loss:{10:.1f}(fuzzy_class:{11:.1f},aux1_class_loss:{12:.1f},aux2_class_loss:{13:.1f},fuzzy_bg_class_loss:{14:.1f},aux1_bg_class_loss:{15:.1f},aux2_bg_class_loss:{16:.1f}))".
                      format(iteration, avg_cost, dev_loss,dev_rec_img_loss1,dev_rec_aux1_loss2,dev_rec_aux2_loss3,dev_rec_aux1_swap_loss4,dev_rec_aux2_swap_loss5,dev_rec_dual_loss6,head_loss,class_loss,fuzzy_class,aux1_class_loss,aux2_class_loss,fuzzy_bg_class_loss,aux1_bg_class_loss,aux2_bg_class_loss))
                logs.writelines("Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}(img1_loss:{3:.1f},aux1_loss2:{4:.1f},aux2_loss3:{5:.1f},aux1_swap_loss:{6:.1f},aux2_swap_loss:{7:.1f},dual_swap_loss:{8:.1f},head loss:{9:.1f},class loss:{10:.1f}(fuzzy_class:{11:.1f},aux1_class_loss:{12:.1f},aux2_class_loss:{13:.1f},fuzzy_bg_class_loss:{14:.1f},aux1_bg_class_loss:{15:.1f},aux2_bg_class_loss:{16:.1f}))\n".
                      format(iteration, avg_cost, dev_loss,dev_rec_img_loss1,dev_rec_aux1_loss2,dev_rec_aux2_loss3,dev_rec_aux1_swap_loss4,dev_rec_aux2_swap_loss5,dev_rec_dual_loss6,head_loss,class_loss,fuzzy_class,aux1_class_loss,aux2_class_loss,fuzzy_bg_class_loss,aux1_bg_class_loss,aux2_bg_class_loss))

   
                count=count+1 
                if self.vis_reconst:
                    self.visualise_reconstruction(fixed_x1,fixed_aux1,fixed_aux2,iteration)
                    #self.visualise_reconstruction(img_zero,fixed_mk1,iteration)
                      
                if np.any(np.isnan(avg_cost)):
                    raise ValueError("NaN detected!")            
            # save checkpoint
            if (iteration > start_iter) and iteration % (ckpt_interval) == 0:
                self.saver.save(self.session, os.path.join(self.dirs['ckpt'], self.exp_name), global_step=iteration)  
            _gan_data=_data1
        logs.close()
        # for save loss
        #np.save('logArray.npy',logArray) 

    def reconstruct(self, X1, aux1,aux2, is_training=False):
        """ Reconstruct data. """
        return self.session.run([self.x1_out,self.mix_head_out,self.aux1_mix_head_out], 
                                feed_dict={self.x1: X1,self.aux1:aux1,self.aux2:aux2,self.is_training: is_training})
    

    def visualise_reconstruction(self, X1, aux1,aux2,iteration):
        X_out1,mix_head_out,aux1_mix_head_out= self.reconstruct(X1, aux1,aux2)
        # print(X_out1.shape)
        X1 = ((X_out1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        mix_head_out = ((mix_head_out+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        aux1_mix_head_out = ((aux1_mix_head_out+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X1, os.path.join(self.dirs['samples'], str(iteration)+'samples_1_rec.png'))
        save_images(mix_head_out, os.path.join(self.dirs['samples'], str(iteration)+'X1bg_aux1head.png'))
        save_images(aux1_mix_head_out, os.path.join(self.dirs['samples'], str(iteration)+'X1head_aux1bg.png'))

    def generateClassificationLabel(self, batch_size):
        # ==============get mask==============
        class_num = 11
        class_gt1 = np.zeros((batch_size, class_num))
        class_gt2 = np.zeros((batch_size, class_num))
        class_gt3 = np.zeros((batch_size, class_num))
        class_gt4 = np.zeros((batch_size, class_num))
        class_gt5 = np.zeros((batch_size, class_num))
        class_gt6 = np.zeros((batch_size, class_num))
        class_gt7 = np.zeros((batch_size, class_num))
        class_gt8 = np.zeros((batch_size, class_num))
        class_gt9 = np.zeros((batch_size, class_num))
        class_gt10 = np.zeros((batch_size, class_num))
        class_gt11 = np.zeros((batch_size, class_num))

        for i in range(batch_size):
            class_gt1[i, 0] = 1
            class_gt2[i, 1] = 1
            class_gt3[i, 2] = 1
            class_gt4[i, 3] = 1
            class_gt5[i, 4] = 1
            class_gt6[i, 5] = 1
            class_gt7[i, 6] = 1
            class_gt8[i, 7] = 1
            class_gt9[i, 8] = 1
            class_gt10[i, 9] = 1
            class_gt11[i, 10] = 1


        return class_gt1, class_gt2, class_gt3, class_gt4,class_gt5, class_gt6, class_gt7, class_gt8,class_gt9, class_gt10,class_gt11
    def getCodesAndImgs(self, pathForSave, X1, k, is_training=False):
        z1, X_r0 = self.session.run([self.z1, self.x1_out],
                                    feed_dict={self.x1: X1, self.is_training: is_training})
        ImageNorm0_1 = ((X_r0 + 1.) * (1.00 / 2)).astype('double').reshape(
            [-1, self.image_shape[1], self.image_shape[2], self.image_shape[0]])
        # for visual the first result to valide it effectiveness
        if k == 1:
            X_save = ((X_r0 + 1.) * (255.99 / 2)).astype('int32').reshape([-1] + self.image_shape)
            save_images(X_save, os.path.join(pathForSave, 'iter' + str(k) + '_samples_reconstructed.png'))
        return z1, ImageNorm0_1
