import os
import errno
import tensorflow as tf
from operator import mul
from functools import reduce
import sys
sys.path.append("../../")


#from model import AE
from model2input import AE2input
from lib.utils import init_directories, create_directories
#from lib.models.data_managers_ae import ShapesDataManager
from lib.models.data_managers_diae import TeapotsDataManager

flags = tf.app.flags
flags.DEFINE_integer("No",19, "The No of the test")
flags.DEFINE_integer("epochs",50, "Number of epochs to train [25]")
flags.DEFINE_integer("stats_interval", 1, "Print/log stats every [stats_interval] epochs. [1.0]")
flags.DEFINE_integer("ckpt_interval", 1, "Save checkpoint every [ckpt_interval] epochs. [10]")
flags.DEFINE_integer("latent_dim", 2*50, "Number of latent variables [10]")
flags.DEFINE_integer("latent_num", 2, "Number of latent part")
flags.DEFINE_integer("class_net_unit_num", 9, "Number of neuro cell")
flags.DEFINE_integer("batch_size", 1, "The size of training batches [64]")
flags.DEFINE_string("image_shape", "(3,32,32)", "Shape of inputs images [(1,32,32)]")
flags.DEFINE_integer("image_wh", 32, "Shape of inputs images 64*64")
flags.DEFINE_string("exp_name", None, "The name of experiment [None]")
flags.DEFINE_string("arch", "resnet", "The desired arch: low_cap, high_cap, resnet. [resnet]")
flags.DEFINE_integer("alpha", 0, "alpha value  vector base")
flags.DEFINE_float("beta",100, "beta value reset 000")
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
        #FLAGS.exp_name = "reconstrued_results"
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

    ae2input = AE2input(
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


    if FLAGS.visualize_reconstruct:
        sampleNum =1000 # 20x64 large batch, forward prop only
        dataVisualName1='SVHN10_img_N1000x32x32x3_testWithLabel_forVI1'
        dataVisualName2='SVHN10_img_N1000x32x32x3_testWithLabel_forVI2'
        
        data_manager = TeapotsDataManager(dirs['data'],
                        dataVisualName1,dataVisualName2, FLAGS.batch_size, 
                        image_shape, shuffle=False,file_ext=FLAGS.file_ext, train_fract=1, 
                        inf=True,supervised=False)  
        #data_manager.set_divisor_batch_size()

        #ae.train_iter, ae.dev_iter, ae.test_iter= data_manager.get_iterators()
        ae2input.train_iter1, ae2input.dev_iter1, ae2input.test_iter1,ae2input.train_iter2, ae2input.dev_iter2, ae2input.test_iter2= data_manager.get_iterators()
        
        ae2input.session.run(tf.global_variables_initializer())
        #saved_step = ae.load()
        saved_step = ae2input.load_fixedNum(500)
        assert saved_step > 1, "A trained model is needed to encode the data!"
        
        pathForSave='VisualIntegrityImgsResults'
        try:
            os.makedirs(pathForSave)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(pathForSave):
                pass
            else:
                raise

        
        for batch_num in range(int(sampleNum/FLAGS.batch_size)):
            img_batch1, _mask1, _ = next(ae2input.train_iter1)
            img_batch2, _mask2, _ = next(ae2input.train_iter2)
            #code = ae.encode(img_batch) #[batch_size, reg_latent_dim]
            ae2input.getVisualImgs(pathForSave,img_batch1,img_batch2, batch_num)
            if batch_num < 5 or batch_num % 100 == 0:
                print(("Batch number {0}".format(batch_num)))
           
        print("Swapped images saved to Folder: VisualIntegrityImgsResults_")


if __name__ == '__main__':
    tf.app.run()
