import os
import errno
import numpy as np
import tensorflow as tf
from operator import mul
from functools import reduce
import sys
sys.path.append("../../")


from model import DIAE
from lib.models.distributions import Gaussian
from lib.utils import init_directories, create_directories
#from lib.models.data_managers_diae import TeapotsDataManager
from lib.models.data_managers_dualdiae import TeapotsDataManager


flags = tf.app.flags
flags.DEFINE_integer("No",19, "The No of the test")
flags.DEFINE_integer("epochs",100, "Number of epochs to train [25]")
flags.DEFINE_integer("stats_interval", 1, "Print/log stats every [stats_interval] epochs. [1.0]")
flags.DEFINE_integer("ckpt_interval", 1, "Save checkpoint every [ckpt_interval] epochs. [10]")
flags.DEFINE_integer("latent_dim", 2*50, "Number of latent variables [10]")
flags.DEFINE_integer("latent_num", 2, "Number of latent part")
flags.DEFINE_integer("class_net_unit_num", 9, "Number of neuro cell")
#@flags.DEFINE_float("beta", 1., "D_KL term weighting [1.]")
flags.DEFINE_integer("batch_size", 64, "The size of training batches [64]")
flags.DEFINE_string("image_shape", "(3,32,32)", "Shape of inputs images [(1,32,32)]")
flags.DEFINE_integer("image_wh", 32, "Shape of inputs images 64*64")
flags.DEFINE_string("exp_name", None, "The name of experiment [None]")
flags.DEFINE_string("arch", "resnet", "The desired arch: low_cap, high_cap, resnet. [resnet]")
flags.DEFINE_integer("alpha", 0, "alpha value  vector base")
flags.DEFINE_float("beta",100, "beta value reset 000")
flags.DEFINE_float("ratio", 0.4, "ratio value")
flags.DEFINE_float("lr", 0.0005, "ratio value")
flags.DEFINE_string("output_dir", "./", "Output directory for checkpoints, samples, etc. [.]")
flags.DEFINE_string("data_dir", None, "Data directory [None]")
flags.DEFINE_string("file_ext", ".npz", "Image filename extension [.jpeg]")
flags.DEFINE_boolean("gaps", False, "Create gaps in data to faciliate zero-shot inference [False]")
flags.DEFINE_boolean("train", True, "Train [True]")
flags.DEFINE_boolean("save_codes", True, "Save latent representation or code for all data samples [False]")
flags.DEFINE_boolean("visualize_reconstruct", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("visualize_disentangle", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("n_disentangle_samples", 10, "The number of evenly spaced samples in latent space \
                     over the interval [-3, 3] [64]")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.exp_name is None:
        FLAGS.exp_name = "reconstrued_results_unitLength"+str(int(FLAGS.latent_dim/FLAGS.latent_num))
    image_shape = [int(i) for i in FLAGS.image_shape.strip('()[]{}').split(',')]
    dirs = init_directories(FLAGS.exp_name, FLAGS.output_dir)
    dirs['data'] = '../../npz_datas' if FLAGS.data_dir is None else FLAGS.data_dir
    dirs['codes'] = os.path.join(dirs['data'], 'codes/')
    create_directories(dirs, FLAGS.train, FLAGS.save_codes)
    
    output_dim  = reduce(mul, image_shape, 1)
    
    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth=True
    run_config.gpu_options.per_process_gpu_memory_fraction=0.9
    sess = tf.Session(config=run_config)

    diae = DIAE(
        session=sess,
        arch=FLAGS.arch,
        lr=FLAGS.lr,
        alpha=FLAGS.alpha,
        beta=FLAGS.beta,
        latent_dim=FLAGS.latent_dim,
        latent_num=FLAGS.latent_num,
        class_net_unit_num=FLAGS.class_net_unit_num,
        output_dim=output_dim,
        batch_size=FLAGS.batch_size,
        image_shape=image_shape,
        exp_name=FLAGS.exp_name,
        dirs=dirs,
        vis_reconst=FLAGS.visualize_reconstruct,
    )

  
    if FLAGS.train:

        data1Name='SVHN10WithBg_img1_oneguided_N20000x32x32x3_train'
        data2Name='SVHN10WithBg_mask1_oneguided_N20000x32x32x3_train'
        data3Name='SVHN10WithBg_aux1_GT1_oneguided_N20000x32x32x3_train'
        data4Name='SVHN10WithBg_aux2_GT2_oneguided_N20000x32x32x3_train'


        data_manager = TeapotsDataManager(dirs['data'],
                        data1Name,data2Name,data3Name,data4Name, FLAGS.batch_size, 
                        image_shape, shuffle=False,file_ext=FLAGS.file_ext, train_fract=0.8,inf=True)
        diae.train_iter1, diae.dev_iter1, diae.test_iter1,diae.train_iter2, diae.dev_iter2, diae.test_iter2,diae.train_iter3, diae.dev_iter3, diae.test_iter3,diae.train_iter4, diae.dev_iter4, diae.test_iter4= data_manager.get_iterators()
        
        n_iters_per_epoch = data_manager.n_train // data_manager.batch_size
        FLAGS.stats_interval = int(FLAGS.stats_interval * n_iters_per_epoch)
        FLAGS.ckpt_interval = int(FLAGS.ckpt_interval * n_iters_per_epoch)
        n_iters = int(FLAGS.epochs * n_iters_per_epoch)
        
        diae.train(n_iters, n_iters_per_epoch, FLAGS.stats_interval, FLAGS.ckpt_interval)

    


if __name__ == '__main__':
    tf.app.run()
