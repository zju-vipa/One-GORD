import os, sys
import time
import re
import numpy as np
import tensorflow as tf

from lib.models.save_images import save_images
from lib.models.distributions import Gaussian
from lib.models.nets_32x32_small import NetsRetreiver

TINY = 1e-8
SEED = 123


class DIAE(object):
    def __init__(self, session, arch,lr,alpha,beta,latent_dim,output_dim, batch_size, image_shape, exp_name, dirs,
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
        self.x2 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.gt1= tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.gt2= tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        # label
        # self.label=tf.placeholder(tf.float32, shape=[None] + list([11]))
        #unpaired
        self.x3 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.x4 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))

        # Normalize + reshape 'real' input data
        norm_x1 = 2*((tf.cast(self.x1, tf.float32)/255.)-.5)
        norm_x2 = 2*((tf.cast(self.x2, tf.float32)/255.)-.5)
        #unpaired
        norm_x3 = 2*((tf.cast(self.x3, tf.float32)/255.)-.5)
        norm_x4 = 2*((tf.cast(self.x4, tf.float32)/255.)-.5)

        norm_gt1=2*((tf.cast(self.gt1, tf.float32)/255.)-.5)
        norm_gt2 = 2 * ((tf.cast(self.gt2, tf.float32) / 255.) - .5)
        # Set Encoder and Decoder archs
        self.Encoder, self.Decoder = NetsRetreiver(self.arch)        
        #------------------------ paired inputs--------------------------
        # `````````primary stage```````````
        # Encode
        self.z1 = self.__Enc(norm_x1)
        self.z2 = self.__Enc(norm_x2)
        # original stage
        # Decode
        self.x_out1 = self.__Dec(self.z1)
        self.x_out2 = self.__Dec(self.z2)

        x1_part1,x1_part2=tf.split(self.z1,2,axis=1)
        x2_part1, x2_part2 = tf.split(self.z2, 2, axis=1)

        self.R1=tf.concat([x2_part1,x1_part2],axis=1)
        self.R2=tf.concat([x1_part1,x2_part2],axis=1)
        # Decode
        self.x_out11 = self.__Dec(self.R1)
        self.x_out22 = self.__Dec(self.R2)
        #------------------------unpaired inputs---------------------------
        # `````````primary stage```````````
        # Encode
        self.z3 = self.__Enc(norm_x3)
        self.z4 = self.__Enc(norm_x4)
        # original stage
        # Decode
        self.x_out3 = self.__Dec(self.z3)
        self.x_out4 = self.__Dec(self.z4)

        x3_part1, x3_part2 = tf.split(self.z3,2,axis=1)
        x4_part1, x4_part2 = tf.split(self.z4, 2, axis=1)
        # swap
        self.R3=tf.concat([x4_part1,x3_part2],axis=1)
        self.R4=tf.concat([x3_part1,x4_part2],axis=1)
        # Decode
        self.x_hybrid3 = self.__Dec(self.R3)
        self.x_hybrid4 = self.__Dec(self.R4)
        # `````````dual stage```````````   
        # Encode
        self.z33 = self.__Enc(self.x_hybrid3)
        self.z44 = self.__Enc(self.x_hybrid4)

        hbr3_part1,hbr3_part2=tf.split(self.z33,2,axis=1)
        hbr4_part1, hbr4_part2 = tf.split(self.z44, 2, axis=1)
        # swap
        self.R33=tf.concat([hbr4_part1,hbr3_part2],axis=1)
        self.R44=tf.concat([hbr3_part1,hbr4_part2],axis=1)
        # Decode
        self.x_out33 = self.__Dec(self.R33)
        self.x_out44 = self.__Dec(self.R44)
        # Loss and optimizer
        self.__prep_loss_optimizer(norm_x1,norm_x2,norm_x3,norm_x4,norm_gt1,norm_gt2)

    
    def __Enc(self, x):
        #resnet_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
        z= self.Encoder('Encoder', x, self.image_shape[0], self.latent_dim,self.is_training)
        return z
    
    def __Dec(self, z):
        x_out_logit = self.Decoder('Decoder', z, self.image_shape[0], self.is_training)
        x_out = tf.tanh(x_out_logit)
        return x_out

    
    def __prep_loss_optimizer(self, norm_x1,norm_x2,norm_x3,norm_x4,norm_gt1,norm_gt2):
        #paired
        norm_x1= tf.reshape(norm_x1, [-1, self.output_dim])
        norm_x2= tf.reshape(norm_x2, [-1, self.output_dim])
        # unpaired
        norm_x3= tf.reshape(norm_x3, [-1, self.output_dim])
        norm_x4= tf.reshape(norm_x4, [-1, self.output_dim])
        gt1=tf.reshape(norm_gt1, [-1, self.output_dim])
        gt2 = tf.reshape(norm_gt2, [-1, self.output_dim])
        #----------------paired loss-----------------------------
        # original loss
        reconstr_loss1 =  tf.reduce_sum(tf.square(norm_x1 -self.x_out1), axis=1) 
        reconstr_loss2 =  tf.reduce_sum(tf.square(norm_x2 -self.x_out2), axis=1)   
        # interchanging loss
        reconstr_loss11 =  tf.reduce_sum(tf.square(gt1 -self.x_out11), axis=1)
        reconstr_loss22 =  tf.reduce_sum(tf.square(gt2 -self.x_out22), axis=1)
        #----------------unpaired loss-----------------------------
        # original loss
        reconstr_loss3 =  tf.reduce_sum(tf.square(norm_x3 -self.x_out3), axis=1) 
        reconstr_loss4 =  tf.reduce_sum(tf.square(norm_x4 -self.x_out4), axis=1)   
        # interchanging loss
        reconstr_loss33 =  tf.reduce_sum(tf.square(norm_x3 -self.x_out33), axis=1) 
        reconstr_loss44 =  tf.reduce_sum(tf.square(norm_x4 -self.x_out44), axis=1)    

        # average over batch
        self.loss = tf.reduce_mean(reconstr_loss1+reconstr_loss2+self.alpha*(reconstr_loss11+reconstr_loss22)+reconstr_loss3+reconstr_loss4+self.beta*(reconstr_loss33+reconstr_loss44))
        lr=self.lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0., beta2=0.9).minimize(self.loss) 
    
        print('Learning rate=')
        print(lr)

    def load(self):
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
        count=0        
        self.session.run(tf.global_variables_initializer())
        
        # Fixed GT samples - save
        fixed_x1, fixed_gt1 , _ = next(self.train_iter1)

        fixed_x1 = self.session.run(tf.constant(fixed_x1))
        save_images(fixed_x1, os.path.join(self.dirs['samples'], 'samples_1_groundtruth.png'))
        #
        fixed_x2, fixed_gt2 , _ = next(self.train_iter2)
        fixed_x2 = self.session.run(tf.constant(fixed_x2))
        save_images(fixed_x2, os.path.join(self.dirs['samples'], 'samples_2_groundtruth.png'))

        fixed_x3, fixed_mk3 , _ = next(self.train_iter3)
        fixed_x3 = self.session.run(tf.constant(fixed_x3))
        save_images(fixed_x3, os.path.join(self.dirs['samples'], 'samples_3_groundtruth.png'))
        start_iter = self.load()
        running_cost = 0.

        fixed_x4, fixed_label , _ = next(self.train_iter4)
        fixed_x4 = self.session.run(tf.constant(fixed_x4))
        save_images(fixed_x4, os.path.join(self.dirs['samples'], 'samples_4_groundtruth.png'))
        logs=open('record_loss.txt','w')
        for iteration in range(start_iter, n_iters):

            _data1, _gt1 , _ = next(self.train_iter1)
            _data2, _gt2 , _ = next(self.train_iter2)
            _data3, _mask1 , _ = next(self.train_iter3)
            _data4, _label, _ = next(self.train_iter4)

            _, cost = self.session.run((self.optimizer, self.loss),feed_dict={self.x1: _data1,self.x2: _data2,self.x3: _data3,self.x4: _data4, self.gt1:_gt1,self.gt2:_gt2, self.is_training:True})
            running_cost += cost


            
            if iteration % n_iters_per_epoch == 1:
                print("Epoch: {0}".format(iteration // n_iters_per_epoch))
            
            # Print avg stats and dev set stats
            if (iteration < start_iter + 4) or iteration % stats_iters == 0:

                dev_data1,dev_gt1, _= next(self.dev_iter1)
                dev_data2,dev_gt2, _ = next(self.dev_iter2)
                dev_data3, dev_mask1, _= next(self.dev_iter3)
                dev_data4, dev_label, _ = next(self.dev_iter4)
                
                dev_cost = self.session.run(self.loss,feed_dict={self.x1: dev_data1,self.x2: dev_data2,self.x3: dev_data3,self.x4: dev_data4, self.gt1:dev_gt1,self.gt2:dev_gt2, self.is_training:False})
                
                n_samples = 1. if (iteration < start_iter + 4) else float(stats_iters)
                avg_cost = running_cost / n_samples
                running_cost = 0.                
                print("Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}".format(iteration, avg_cost, dev_cost))
                logs.writelines(
                    "Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}\n".format(iteration, avg_cost, dev_cost))
                count=count+1 
                if self.vis_reconst:
                    self.visualise_reconstruction(fixed_x1,fixed_x2,fixed_gt1,fixed_gt2,fixed_x3,fixed_x4,iteration)
          
                if np.any(np.isnan(avg_cost)):
                    raise ValueError("NaN detected!")            
            # save checkpoint
            if (iteration > start_iter) and iteration % (ckpt_interval) == 0:
                self.saver.save(self.session, os.path.join(self.dirs['ckpt'], self.exp_name), global_step=iteration)  
        # for save loss
        logs.close()

    def encode(self, X, is_training=False):
        """Encode data, i.e. map it into latent space."""
        code = self.session.run(self.z1, feed_dict={self.x1: X, self.is_training: is_training})
        return code


    def reconstruct(self, X1, X2,gt1,gt2,X3, X4, is_training=False):
        """ Reconstruct data. """
        return self.session.run([self.x_out1,self.x_out2,self.x_out11,self.x_out22,self.x_out33,self.x_out44], 
                                feed_dict={self.x1: X1,self.x2: X2,self.x3: X3,self.x4: X4,self.gt1:gt1,self.gt2:gt2,self.is_training: is_training})
        
    def getCodesAndImgs(self,pathForSave,X1,k, is_training=False):
        z1,X_r0=self.session.run([self.z1,self.x_out1],feed_dict={self.x1: X1,self.is_training: is_training})
        ImageNorm0_1 = ((X_r0+1.)*(1.00/2)).astype('double').reshape([-1,self.image_shape[1],self.image_shape[2],self.image_shape[0]])
        # for visual the first result to valide it effectiveness
        if k==1:
            X_save = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
            save_images(X_save, os.path.join(pathForSave, 'iter'+str(k)+'_samples_reconstructed.png'))
        return z1,ImageNorm0_1

    def getVisualImgs(self,pathForSave,X1,mk1,X2,mk2,X3,mk3,X4,mk4,k,is_training=False):
        X_out1,X_out2,X_r0,X_swap=self.session.run([self.x_out1,self.x_out2,self.x_out11,self.x_hybrid3], 
                                feed_dict={self.x1: X1,self.x2: X2,self.mk1: mk1,self.mk2: mk2,self.x3: X3,self.x4: X4,self.mk3: mk3,self.mk4: mk4, self.is_training: is_training})
        X1_save = ((X_out1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X2_save = ((X_out2+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_orig1_save = (X1*255.99).astype('int32').reshape([-1] + self.image_shape)
        X_orig2_save  = (X2*255.99).astype('int32').reshape([-1] + self.image_shape)
        X_orig3_save = (X3*255.99).astype('int32').reshape([-1] + self.image_shape)
        X_orig4_save  = (X4*255.99).astype('int32').reshape([-1] + self.image_shape)
        X_reset0_save = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_Swap_save = ((X_swap+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_orig1_save, os.path.join(pathForSave, 'iter'+str(k)+'_orig_img1.png'))
        #save_images(X_orig2_save, os.path.join(pathForSave, 'iter'+str(k)+'_orig_img2.png'))
        #save_images(X_orig3_save, os.path.join(pathForSave, 'iter'+str(k)+'_orig_img3.png'))
        save_images(X_orig4_save, os.path.join(pathForSave, 'iter'+str(k)+'_orig_img4.png'))
        #save_images(X1_save, os.path.join(pathForSave, 'iter'+str(k)+'_reconst_img1.png'))
        #save_images(X2_save, os.path.join(pathForSave, 'iter'+str(k)+'_reconst_img2.png'))
        save_images(X_reset0_save, os.path.join(pathForSave, 'iter'+str(k)+'_reset0_img.png'))
        save_images(X_Swap_save, os.path.join(pathForSave, 'iter'+str(k)+'_swap_img.png'))
        


    def visualise_reconstruction(self, X1,X2,gt1,gt2,X3,X4,iteration):
        X_r1,X_r2,X_r11,X_r22,X_r33,X_r44  = self.reconstruct(X1,X2,gt1,gt2,X3, X4)
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r2 = ((X_r2+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r11 = ((X_r11+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r22 = ((X_r22+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r33 = ((X_r33+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r44 = ((X_r44+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(self.dirs['samples'], str(iteration)+'samples_1_reconstructed.png'))
        save_images(X_r2, os.path.join(self.dirs['samples'], str(iteration)+'samples_2_reconstructed.png'))
        save_images(X_r11, os.path.join(self.dirs['samples'], str(iteration)+'samples_11_reconstructed.png'))
        save_images(X_r22, os.path.join(self.dirs['samples'], str(iteration)+'samples_22_reconstructed.png'))
        save_images(X_r33, os.path.join(self.dirs['samples'], str(iteration)+'samples_33_reconstructed.png'))
        save_images(X_r44, os.path.join(self.dirs['samples'], str(iteration)+'samples_44_reconstructed.png'))
        
    def encodeImg(self,pathForSave,ratio, X1, mk1,X2, mk2,k, is_training=False): 
        
        X_r1,X_r2,X_r11,X_r22,X_r1twice,X_r2twice=self.session.run([self.x_out3,self.x_out4,self.x_hybrid3,self.x_hybrid4,self.x_out33,self.x_out44],feed_dict={self.x3: X1,self.x4: X2,self.mk3: mk1,self.mk4: mk2, self.is_training: is_training})
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r2 = ((X_r2+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r11 = ((X_r11+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r22 = ((X_r22+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(pathForSave, 'ratio='+str(ratio)+'_iter'+str(k)+'_samples_3_reconstructed.png'))
        save_images(X_r2, os.path.join(pathForSave, 'ratio='+str(ratio)+'_iter'+str(k)+'_samples_4_reconstructed.png'))
        save_images(X_r11, os.path.join(pathForSave, 'ratio='+str(ratio)+'_iter'+str(k)+'_samples_hrd3_reconstructed.png'))
        save_images(X_r22, os.path.join(pathForSave, 'ratio='+str(ratio)+'_iter'+str(k)+'_samples_hrd4_reconstructed.png')) 


        X_r1twice = ((X_r1twice+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r2twice = ((X_r2twice+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1twice, os.path.join(pathForSave, 'ratio='+str(ratio)+'_iter'+str(k)+'_samples_3_twice_reconstructed.png'))
        save_images(X_r2twice, os.path.join(pathForSave, 'ratio='+str(ratio)+'_iter'+str(k)+'_samples_4_twice_reconstructed.png')) 
