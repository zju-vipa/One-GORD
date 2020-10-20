import os, sys
import time
import re
import numpy as np
import tensorflow as tf

from lib.models.save_images import save_images
from lib.models.distributions import Bernoulli, Gaussian, Product
from lib.models.nets_32x32_small import NetsRetreiver, NetsRetreiverWithClassifier

TINY = 1e-8
SEED = 123
ch=3
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter

class AE2input(object):
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
        self.x2 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))

        # Normalize + reshape 'real' input data
        norm_x1 = 2*(tf.cast(self.x1, tf.float32)-.5)
        norm_x2 = 2 * (tf.cast(self.x2, tf.float32) - .5)
        # norm_img_black=2*(tf.cast(self.img_black, tf.float32)-.5)
        # Set Encoder and Decoder archs
        self.Encoder, self.Decoder,self.Classifier,self.gan_discriminator = NetsRetreiverWithClassifier(self.arch) 
    
        # Encode
        self.z1 = self.__Enc(norm_x1)
        # original stage
        # Decode
        self.x_out1 = self.__Dec(self.z1)

        self.z2=self.__Enc(norm_x2)

        z1_part1,z1_part2=tf.split(self.z1,2,axis=1)
        z2_part1, z2_part2 = tf.split(self.z2, 2, axis=1)
        x2fg_x1bg=tf.concat([z2_part1,z1_part2],axis=1)
        x1fg_x2bg=tf.concat([z1_part1,z2_part2],axis=1)

        self.x2fg_x1bg_out=self.__Dec(x2fg_x1bg)
        self.x1fg_x2bg_out = self.__Dec(x1fg_x2bg)

        # Loss and optimizer
        self.__prep_loss_optimizer(norm_x1)

    def __Enc(self, x):
        #resnet_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
        z= self.Encoder('Encoder', x, self.image_shape[0], self.latent_dim,self.is_training)
        return z
    
    def __Dec(self, z):
        x_out_logit = self.Decoder('Decoder', z, self.image_shape[0], self.is_training)
        x_out = tf.tanh(x_out_logit)
        return x_out
    
    def __Classifier(self,z):
        x_out= self.Classifier('Classifier', z, self.class_net_unit_num,self.latent_num+1, self.is_training)
        x_out = tf.nn.softmax(x_out)
        return x_out
    
    def __prep_loss_optimizer(self,norm_x1):
 
        norm_x1= tf.reshape(norm_x1, [-1, self.output_dim])
        #[Loss1]img reconstruction loss
        reconstr_img_loss =  tf.reduce_sum(tf.square(norm_x1 -self.x_out1), axis=1)
        #
        # # average over batch
        self.rec_loss=1.0*tf.reduce_mean(reconstr_img_loss)

        self.loss=self.rec_loss
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
        fixed_x1, fixed_mk1 ,_ = next(self.train_iter1)
        print("fixed_mk1=")
        print(fixed_mk1[0:4])
        # print(fixed_label[0:4])
        # replace mask
        #unitLength=3  #(need to changed when has larger unitLength)
        unitLength=int(self.latent_dim/self.latent_num)
        # generate zero representation and black image and gts0
        img_zero,fixed_zero_mk,fixed_gts0=self.generateMaskZeroAndGts(self.batch_size,unitLength)
        #
        fixed_x1= self.session.run(tf.constant(fixed_x1))
        save_images(fixed_x1, os.path.join(self.dirs['samples'], 'samples_1_groundtruth.png'))
        #
        start_iter = self.load()
        running_cost = 0.
        
        _gan_data=fixed_x1
        logs=open('loss_records.txt','w')
        for iteration in range(start_iter, n_iters):
            start_time = time.time()

            _data1, _mask1,_ = next(self.train_iter1)

            _, cost = self.session.run((self.optimizer, self.loss),feed_dict={self.x1: _data1,self.is_training:True})
            running_cost += cost
            
            if iteration % n_iters_per_epoch == 1:
                print("Epoch: {0}".format(iteration // n_iters_per_epoch))
            
            # Print avg stats and dev set stats
            if (iteration < start_iter + 4) or iteration % stats_iters == 0:
                t = time.time()
                dev_data1, dev_mask1, dev_label1= next(self.dev_iter1)
                
                #dev_cost,dev_rec_loss,dev_reset0_loss,vector_loss,rec_zero_loss,class_loss= self.session.run([self.loss,self.rec_loss,self.reset0_loss,self.vector_loss,self.rec_zero_loss,self.class_loss],feed_dict={self.x1: dev_data1, self.mask: dev_mask1, self.vec_one:vector_one,self.img_black:img_zero,self.mask_zero:fixed_zero_mk,self.class_gt1:class_gt1,self.class_gt2:class_gt2,self.class_gt3:class_gt3,self.class_gt4:class_gt4,self.class_gt4:class_gt4,self.is_training:False})
                dev_cost,dev_rec_loss= self.session.run([self.loss,self.rec_loss],feed_dict={self.x1: dev_data1,self.is_training:False})
                
                n_samples = 1. if (iteration < start_iter + 4) else float(stats_iters)
                avg_cost = running_cost / n_samples
                running_cost = 0.

                print("Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}(reconstr_loss:{3:.1f})".format(iteration, avg_cost, dev_cost,dev_rec_loss))
                logs.writelines("Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}(reconstr_loss:{3:.1f})\n".format(
                    iteration, avg_cost, dev_cost, dev_rec_loss))
                count=count+1 
                if self.vis_reconst:
                    self.visulize_rec(fixed_x1,iteration)
                    #self.visualise_reconstruction(img_zero,fixed_mk1,iteration)
                      
                if np.any(np.isnan(avg_cost)):
                    raise ValueError("NaN detected!")            
            # save checkpoint
            if (iteration > start_iter) and iteration % (ckpt_interval) == 0:
                self.saver.save(self.session, os.path.join(self.dirs['ckpt'], self.exp_name), global_step=iteration)  
            _gan_data=_data1
        logs.close()

    def reconstruct(self, X1, mk1, is_training=False):
        """ Reconstruct data. """
        return self.session.run([self.x_out1 ],
                                feed_dict={self.x1: X1,self.mask: mk1, self.is_training: is_training})
    

    def visualise_reconstruction(self, X1,mk1,iteration):
        X_r1,X_r0= self.reconstruct(X1,mk1)
        #print(X_r0[3])
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r0 = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(self.dirs['samples'], str(iteration)+'samples_reconstructed.png'))
        save_images(X_r0, os.path.join(self.dirs['samples'], str(iteration)+'reset0_reconstructed.png'))

    def visulize_rec(self,X1,iteration):
        X_r1=self.session.run(self.x_out1 ,feed_dict={self.x1: X1,self.is_training: False})
        #print(X_r0[3])
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(self.dirs['samples'], str(iteration)+'samples_reconstructed.png'))

    def visulize_rec_Origin(self,pathForSave,X1,X2,iteration):
        X_r1=self.session.run(self.x_out1 ,feed_dict={self.x1: X1,self.is_training: False})
        #print(X_r0[3])
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(pathForSave, 'iter'+str(iteration)+'_samplesOrigin_reconstructed_X1.png'))

        X_r2=self.session.run(self.x_out1 ,feed_dict={self.x1: X2,self.is_training: False})
        #print(X_r0[3])
        X_r2 = ((X_r2+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r2, os.path.join(pathForSave, 'iter'+str(iteration)+'_samplesOrigin_reconstructed_X2.png'))

    def encodeImg(self,pathForSave,X1, mk1,k, is_training=False): 
        
        X_r1,X_r0=self.session.run([self.x_out1,self.x_out_r0],feed_dict={self.x1: X1,self.mask: mk1, self.is_training: is_training})
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r0 = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(pathForSave, 'iter'+str(k)+'_samples_reconstructed.png'))
        save_images(X_r0, os.path.join(pathForSave, 'iter'+str(k)+'_reset0_reconstructed.png'))
        
    def encode(self, X, is_training=False):
        """Encode data, i.e. map it into latent space."""
        code = self.session.run(self.z1, feed_dict={self.x1: X, self.is_training: is_training})
        return code

    def generateMaskZeroAndGts(self,batch_size,unitLength):
        #==============get mask==============
        partNum=3
        w=32
        h=32
        maskArray=np.empty((batch_size,unitLength*partNum))
        labelArray=np.empty((batch_size,partNum+1))
        mask=np.zeros((unitLength*partNum))
        label0=np.zeros((partNum+1))
        label0[0]=1
        # reset value 0~64
        for i in range(0,batch_size):
            maskArray[i]=mask
            labelArray[i]=label0

        imgArray= np.zeros((batch_size,ch,w,h))*0.0

        return imgArray,maskArray,labelArray
    def getCodesAndImgs(self,pathForSave,X1,k, is_training=False):
        z1,X_r0=self.session.run([self.z1,self.x_out1],feed_dict={self.x1: X1,self.is_training: is_training})
        ImageNorm0_1 = ((X_r0+1.)*(1.00/2)).astype('double').reshape([-1,self.image_shape[1],self.image_shape[2],self.image_shape[0]])
        # for visual the first result to valide it effectiveness
        if k==1:
            X_save = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
            save_images(X_save, os.path.join(pathForSave, 'iter'+str(k)+'_samples_reconstructed.png'))
        return z1,ImageNorm0_1

    def getVisualImgs(self, pathForSave, X1,X2,k, is_training=False):
        x2fg_x1bg_out,x1fg_x2bg_out = self.session.run([self.x2fg_x1bg_out,self.x1fg_x2bg_out],feed_dict={self.x1: X1,self.x2: X2, self.is_training: is_training})

        X_orig1_save = (X1 * 255.99).astype('int32').reshape([-1] + self.image_shape)
        X_orig2_save = (X2 * 255.99).astype('int32').reshape([-1] + self.image_shape)
        x2fg_x1bg_out_save = ((x2fg_x1bg_out + 1.) * (255.99 / 2)).astype('int32').reshape([-1] + self.image_shape)
        x1fg_x2bg_out_save = ((x1fg_x2bg_out + 1.) * (255.99 / 2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_orig1_save, os.path.join(pathForSave, 'iter' + str(k) + '_origX1_img.png'))
        save_images(X_orig2_save, os.path.join(pathForSave, 'iter' + str(k) + '_origX2_img.png'))
        save_images(x2fg_x1bg_out_save, os.path.join(pathForSave, 'iter' + str(k) + 'x2fg_x1bg_out_img.png'))
        save_images(x1fg_x2bg_out_save, os.path.join(pathForSave, 'iter' + str(k) + 'x1fg_x2bg_out_img.png'))
