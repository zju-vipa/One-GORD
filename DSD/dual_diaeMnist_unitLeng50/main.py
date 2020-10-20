import os
import numpy as np
import tensorflow as tf
from operator import mul
from functools import reduce
import sys
sys.path.append("../")


from model import DIAE
from lib.utils import init_directories, create_directories
from lib.models.data_managers_dualdiae import TeapotsDataManager

flags = tf.app.flags
flags.DEFINE_integer("epochs",16, "Number of epochs to train [25]")
flags.DEFINE_integer("stats_interval", 1, "Print/log stats every [stats_interval] epochs. [1.0]")
flags.DEFINE_integer("ckpt_interval", 1, "Save checkpoint every [ckpt_interval] epochs. [10]")
flags.DEFINE_integer("latent_dim", 2*50, "Number of latent variables [10]")
flags.DEFINE_integer("batch_size", 64, "The size of training batches [64]")
flags.DEFINE_string("image_shape", "(3,32,32)", "Shape of inputs images [(3,32,32)]")
flags.DEFINE_integer("image_wh", 32, "Shape of inputs images 64*64")
flags.DEFINE_string("exp_name", None, "The name of experiment [None]")
flags.DEFINE_string("arch", "resnet", "The desired arch: low_cap, high_cap, resnet. [resnet]")
flags.DEFINE_integer("alpha", 5, "alpha value")
flags.DEFINE_float("beta", 0.2, "beta value")
flags.DEFINE_float("lr", 0.0005, "ratio value")
flags.DEFINE_string("output_dir", "./", "Output directory for checkpoints, samples, etc. [.]")
flags.DEFINE_string("data_dir", None, "Data directory [None]")
flags.DEFINE_string("file_ext", ".npz", "Image filename extension [.jpeg]")
flags.DEFINE_boolean("train", True, "Train [True]")
flags.DEFINE_boolean("save_codes", True, "Save latent representation or code for all data samples [False]")
flags.DEFINE_boolean("visualize_reconstruct", True, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.exp_name is None:
        FLAGS.exp_name = "dual_diaeMnist_unitLength_{0}".format(np.int32(FLAGS.latent_dim/3))
    image_shape = [int(i) for i in FLAGS.image_shape.strip('()[]{}').split(',')]
    dirs = init_directories(FLAGS.exp_name, FLAGS.output_dir)
    dirs['data'] = '../../npz_datas' if FLAGS.data_dir is None else FLAGS.data_dir
    dirs['codes'] = os.path.join('', 'codes/')
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
        output_dim=output_dim,
        batch_size=FLAGS.batch_size,
        image_shape=image_shape,
        exp_name=FLAGS.exp_name,
        dirs=dirs,
        vis_reconst=FLAGS.visualize_reconstruct,
    )
 
    if FLAGS.train:

        data1Name='SVHN10WithBg_aux1_GT1_255oneguided_N20000x32x32x3_train_forDSD'
        data2Name='SVHN10WithBg_aux2_GT2_255oneguided_N20000x32x32x3_train_forDSD'
        data3Name='SVHN10WithBg_img3_255oneguided_N20000x32x32x3_train_forDSD'
        data4Name='SVHN10WithBg_img4_255oneguided_N20000x32x32x3_train_forDSD'
    

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
