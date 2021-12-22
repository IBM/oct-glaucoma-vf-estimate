#!/usr/bin/struture_function1 python
# File: train_tp.py
# Author: Yasmeen George

import tensorflow as tf
#from tensorflow import keras
import argparse
from tensorpack.tfutils.summary import *
from thickness_dataflow_tp import *
import tensorflow.contrib.slim as slim
from keras import backend as K
from contextlib import contextmanager
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
from tensorpack.utils.viz import stack_patches
from vft_utils import perf_measures
def get_features(image,scope):
    with tf.compat.v1.variable_scope(scope):
        l = tf.layers.conv2d(image, 32, 3, padding='SAME',name='conv0')# input_shape=input_shape)
        l=tf.nn.relu(l,name='relu0')
        l = tf.layers.conv2d( l, 16, 3, padding='SAME',name = 'conv1')
        l = tf.nn.relu(l,name='relu1')
        i = 2
        name =""
        for nbchannel in nfilters:
            l =  tf.layers.conv2d(l, nbchannel, 3, padding='SAME',name='conv'+str(i))
            l = tf.layers.batch_normalization(l,axis=-1, momentum=0.8) # input_shape=(input_shape[0], input_shape[1], input_shape[2], nbchannel)
            l = tf.nn.relu(l,name='relu'+str(i))
            name = l.name
            l = tf.layers.max_pooling2d(l,2,2,name = 'maxpool3d'+str(i))
            i +=1

    return l,name


def get_keras_model(l):
    l = tf.layers.conv2d(l, 32, 3, padding='valid',name='conv0')  # input_shape=input_shape)
    l = tf.nn.relu(l,name='relu0')
    i=1
    name = ""
    for nbchannel in nfilters_merged:
        l =  tf.layers.conv2d(l, nbchannel, 3,padding='valid',name='conv'+str(i))
        l = tf.layers.batch_normalization(l,axis=-1, momentum=0.8)# input_shape=(input_shape[0],input_shape[1],input_shape[2],nbchannel),
        l = tf.nn.relu(l,name='relu'+str(i))
        name = l.name
        i+=1

    if CNN_OUT_GarWayHeathmap: # PREDICT GARWAY HEATHMAP AVERAGE REGIONS
        l = tf.reduce_mean(l, axis=[1,2])
        l=tf.layers.dense(l, 64, tf.nn.relu)
        l = tf.layers.dropout(l,rate=0.5)
        l = tf.layers.dense(l, out_num)

    else: # predict VFT THRESHOLD VALUES
        l = tf.layers.conv2d(l,1,2,padding='valid')
        l = tf.reduce_mean(l,axis=(3))
        #l = tf.layers.conv2d(l,10,2,padding='valid')
        #l = tf.nn.softmax(l,name='pred')
        #l = tf.math.sigmoid(l,name='pred')



    return l,tf.math.sigmoid(l,name='pred'),name


@contextmanager
def guided_relu():
        """
        Returns:
            A context where the gradient of :meth:`tf.nn.relu` is replaced by
            guided back-propagation, as described in the paper:
            `Striving for Simplicity: The All Convolutional Net
            <https://arxiv.org/abs/1412.6806>`_
        """
        from tensorflow.python.ops import gen_nn_ops  # noqa

        @tf.RegisterGradient("GuidedReLU")
        def GuidedReluGrad(op, grad):
            return tf.where(0. < grad,
                            gen_nn_ops.relu_grad(grad, op.outputs[0]),
                            tf.zeros(grad.get_shape()))

        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': 'GuidedReLU'}):
            yield

def saliency_map(output, input, name="saliency_map"):
        """
        Produce a saliency map as described in the paper:
        `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
        <https://arxiv.org/abs/1312.6034>`_.
        The saliency map is the gradient of the max element in output w.r.t input.

        Returns:
            tf.Tensor: the saliency map. Has the same shape as input.
        """
        max_outp = tf.reduce_max(output, 1)
        saliency_op = tf.gradients(max_outp, input)[:][0]
        return tf.identity(saliency_op, name=name)



class Model(ModelDesc):
    def __init__(self,thickness_name= 'nodisc_rnfl_gcipl'):
        if thickness_name == 'nodisc_rnfl_gcipl':
            self.THICKNESS = 0
        elif thickness_name == 'nodisc_rnfl':
            self.THICKNESS = 1
        if thickness_name == 'nodisc_gcipl':
            self.THICKNESS = 2

    def inputs(self):
        return [tf.TensorSpec((None,)+(SHAPE, SHAPE),tf.uint8, 'input1'),
                tf.TensorSpec((None,) + (SHAPE, SHAPE), tf.uint8, 'input2'),
            tf.TensorSpec((None,out_num), tf.float32, 'label'),
            tf.TensorSpec((None,) + vft_shape, tf.float32, 'vft_threshold'),
            tf.TensorSpec((None,), tf.string, 'uid')]
    
    def build_graph(self, image1,image2 ,label,vft_threshold,uid):

        image1 = tf.expand_dims(tf.cast(image1, tf.float32), axis=-1) / 128.0 - 1
        image2 = tf.expand_dims(tf.cast(image2, tf.float32), axis=-1) / 128.0 - 1
        if self.THICKNESS == 0:
            image = tf.concat([image1, image2], axis=-1) #--> RNFL and GCIPL
        elif self.THICKNESS == 1:
            image = image1 # --> RNFL
        elif self.THICKNESS == 2:
            image = image2 # --> GCIPL

        print(image1, image2, image,uid)

        f1,n1=get_features(image, 'pathway1')
        pred,sig_pred,n = get_keras_model(f1)

        model_summary()
        print(f1)
        print(pred)

        '''
        
        with guided_relu():
            saliency_map(pred, tf.get_default_graph().get_tensor_by_name(n1), name="saliency_p1")
            saliency_map(pred, tf.get_default_graph().get_tensor_by_name(n), name="saliency_p5")
        '''
        def dice_coef_loss(y_true, y_pred):
            def dice_coef(y_true, y_pred, smooth=1):
                """
                Dice = (2*|X & Y|)/ (|X|+ |Y|)
                     =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
                ref: https://arxiv.org/pdf/1606.04797v1.pdf
                """
                intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
                return (2. * intersection + smooth) / (
                            K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

            return 1 - dice_coef(y_true, y_pred)

        if CNN_OUT_GarWayHeathmap:
            y_true, y_pred = label, sig_pred
        else:
            y_true, y_pred = vft_threshold, sig_pred  # vft_threshold[..., 1:], pred[..., 1:]
            print(y_true, y_pred)

            # dice_loss = dice_coef_loss(y_true, y_pred)
            # ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            # dice_cost = tf.reduce_mean(dice_loss, name='dice_loss')
            # ce_cost = tf.reduce_mean(ce_loss, name='cross_entropy_loss')

        sce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=pred)
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        mae_loss = tf.keras.losses.MAE(y_true, y_pred)


        mse_cost = tf.reduce_mean(mse_loss, name='mean_squared_error')
        mae_cost = tf.reduce_mean(mae_loss, name='mean_absolute_error')
        sce_cost = tf.reduce_mean(sce_loss, name='sigmoid_cross_entropy')

        print(sce_loss, mse_loss, mae_loss)
        print("READUCED_MEAN")
        print(sce_cost, mse_cost, mae_cost)

        # weight decay on all W

        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(sce_cost, mse_cost, mae_cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))  # monitor W
        self.cost = tf.add_n([sce_cost, wd_cost], name='cost')
        print(self.cost)
        return self.cost
    def optimizer(self):
        lr = tf.compat.v1.get_variable('learning_rate', initializer=0.01, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.compat.v1.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def test(ds, model_path='',csv_save_path = 'test_results.csv',vft_type= 'THRESHOLD',thickness_name = ''):

    in_names = ['input1','input2', 'label', 'vft_threshold', 'uid']

    pred = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(thickness_name),
        input_names=in_names,
        output_names=['uid', 'vft_threshold', 'pred', 'logistic_loss', 'Mean_1', 'Mean_2'])

    df_result = perf_measures(ds, pred=pred, oct_type='onh',vft_type= vft_type)
    #df_result = perf_measures_thickness_maps(ds, pred=pred)

    df_result.to_csv(csv_save_path)

    return df_result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=os.getcwd() , help='output dir name')# metavar='out_dir'
    parser.add_argument('--data_dir', default=None, help='data dir name') # ,metavar='data_dir'
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')  # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1', default=70, help='Epoch to drop learning rate to 0.01.')#150  # nargs='*' in multi mode
    parser.add_argument('--drop_2', default=120, help='Epoch to drop learning rate to 0.001')#225
    parser.add_argument('--depth', default=20, help='The depth of densenet')# 40
    parser.add_argument('--max_epoch', default=150, help='max epoch') #300
    parser.add_argument('--task', help='task to perform: "train" or "test" or "all" ',
                        choices=['all', 'test', 'train'], default='train')

    parser.add_argument('--thickness_name', help='task to perform: "disc_rnfl_gcipl" or "nodisc_rnfl_gcipl" or "nodisc_rnfl" or "nodisc_gcipl" ',
                        choices=['disc_rnfl_gcipl','nodisc_rnfl_gcipl','nodisc_rnfl','nodisc_gcipl'], default='disc_rnfl_gcipl')
    parser.add_argument('--fold', help='fold number ',
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], default='1')
    parser.add_argument('--pred', help='Prediction map',
                        choices=['THRESHOLD', 'PATTERN'], default='THRESHOLD')
    parser.add_argument('--load_model', help='load model directory '
                        , default=None)
    parser.add_argument('--model_name', help='model name e.g. model-150000 '
                        , default=None)

    args = parser.parse_args()

    # prepare dataset
    if args.data_dir is None:

        mapsdatadir,vftdatadir='/Users/gyasmeen/Downloads/', '/Users/gyasmeen/Desktop/Results/nyu_vft_xml/'
        #base_dir = '/dccstor/aurmmaret1/Datasets/NYU/'
        #octdatadir,vftdatadir = base_dir+'MAC_ONH_1pairPerVisit/MAC_ONH_1pairPerVisit/',base_dir +'nyu_vft_xml/'
    else:
        mapsdatadir,vftdatadir = args.data_dir +'/', args.data_dir+'/'

    # oct-vft_si_exp10
    # log_dir = '/mnt/results/structure-function-results/training-ybXjrevMg/train_log/'

    # oct-vft_si_exp11_linear
    # log_dir = '/mnt/results/structure-function-results/training-1HwkNRdMg/train_log/'

    # oct-vft_si_exp13_global
    #log_dir = '/mnt/results/structure-function-results/training-h2xVagdMg/train_log/'

    if args.load_model is None:
        log_dir = args.out_dir + "/train_log"
    else:
        log_dir = args.load_model

    if args.thickness_name == "disc_rnfl_gcipl":
        thickness_layers = ['GCIPL_wd', 'RNFL_wd']
    elif args.thickness_name == 'nodisc_rnfl_gcipl':
        thickness_layers = ['RNFL', 'GCIPL']
    elif args.thickness_name == 'nodisc_rnfl':
        thickness_layers = ['RNFL', 'GCIPL'] #### CHANGE BUILD GRAPH
    elif args.thickness_name == 'nodisc_gcipl':
        thickness_layers = ['RNFL', 'GCIPL'] #### HANGE BUILD GRAPH
    else:
        thickness_layers = ['RNFL', 'GCIPL']  # ['GCIPL_wd', 'RNFL_wd']#['RNFL', 'GCIPL', 'GCIPL_wd', 'RNFL_wd']

    if args.task != 'train':
        '''
        '# oct-vft_si_exp11_linear
        if args.thickness_name == "disc_rnfl_gcipl":
            thickness_layers = ['GCIPL_wd', 'RNFL_wd']
            model_path = '/mnt/results/structure-function-results/training-wIEeiitGg/train_log/'
            model_name = 'model-337200'
        elif args.thickness_name == 'nodisc_rnfl_gcipl':
            thickness_layers = ['RNFL', 'GCIPL']
            model_path = '/mnt/results/structure-function-results/training-XRi2ZmtGg/train_log/'
            model_name = 'model-339448'

        elif args.thickness_name == 'nodisc_rnfl':
            thickness_layers = ['RNFL' ,'GCIPL'] ### CHANGE BUILD GRAPH
            model_path = '/mnt/results/structure-function-results/training-qzFeZmtGR/train_log/'
            model_name = 'model-334952'
        elif args.thickness_name == 'nodisc_gcipl':
            thickness_layers = ['RNFL', 'GCIPL'] ### CHANGE BUILD GRAPH
            model_path = '/mnt/results/structure-function-results/training-quYgGmtMR/train_log/'
            model_name = 'model-330456'
        '''
        if args.load_model is None:
            print('You must enter model path directory')
            exit()
        if args.model_name is None:
            print('You must enter model name')
            exit()
        model_path = args.load_model
        model_name = args.model_name

        dataset_test, te_batch_num = get_data(mapsdatadir, vftdatadir, SHAPE=SHAPE, BATCH=BATCH, task= args.task,Thicknesses=thickness_layers,fold = args.fold,vft_type = args.pred)
        df_result = test(dataset_test,model_path=model_path+model_name,csv_save_path=model_path+'perf_measures_thickness-'+args.thickness_name+'-f'+str(args.fold)+'_input.csv',vft_type = args.pred, thickness_name =  args.thickness_name)
        print('Test is finished for {} samples', len(df_result))
    elif args.task =='train':

        if args.out_dir is None:
            logger.auto_set_dir()
        else:
            logger_dir = os.path.join(log_dir)
            logger.set_logger_dir(logger_dir,action='k')

        dataset_train,batch_num = get_data(mapsdatadir,vftdatadir, SHAPE=SHAPE,BATCH=BATCH ,task=args.task,Thicknesses=thickness_layers,fold = args.fold,vft_type = args.pred)

        steps_per_epoch = batch_num
        dataset_val,v_batch_num = get_data(mapsdatadir,vftdatadir, SHAPE=SHAPE,BATCH=BATCH ,task='val',Thicknesses=thickness_layers,fold = args.fold,vft_type = args.pred)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.Session(config=config)

        cfg = AutoResumeTrainConfig(
            model=Model(args.thickness_name),
            dataflow=dataset_train,
            callbacks=[
                PeriodicTrigger(ModelSaver(max_to_keep=10,keep_checkpoint_every_n_hours=1), every_k_epochs=5),
                InferenceRunner(
                    dataset_val,
                    ScalarStats(['sigmoid_cross_entropy', 'mean_squared_error', 'mean_absolute_error'])),
                    #ScalarStats(['dice_loss','cross_entropy_loss','mean_squared_error','mean_absolute_error'])),
                # record GPU utilization during training
                GPUUtilizationTracker(),
                ScheduledHyperParamSetter('learning_rate',
                                          [(args.drop_1, 0.001), (args.drop_2, 0.0001)]),

            ],
            steps_per_epoch=steps_per_epoch,
            max_epoch=args.max_epoch,sess=session
        )

        if get_num_gpu() <= 1:
            # single GPU:
            launch_train_with_config(cfg, SimpleTrainer())
        else:
            # multi GPU:
            launch_train_with_config(cfg, SyncMultiGPUTrainerParameterServer(get_num_gpu()))
            # "Replicated" multi-gpu trainer is not supported for Keras model
            # since Keras does not respect variable scopes.
