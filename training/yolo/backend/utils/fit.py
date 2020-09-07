# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import csv
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime

# define your custom callback for prediction
# class PredictionCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         pass

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.init_lr = 0

    def on_train_begin(self, logs={}):
        self.lr = []

    def on_epoch_begin(self, epoch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        print('epoch:', epoch)
        if epoch == 0 :
            self.init_lr = lr
        print('lr:', lr)
        print('decay:', decay)
        print('init_lr:', self.init_lr)
        lr = self.init_lr / ( 2 ** (epoch // 20))
        print('lr:',lr)
        # lr = lr * (1. / (1 + decay * epoch))
        K.set_value(self.model.optimizer.lr, lr)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)

'''        
class LRTensorBoard(TensorBoard):
    def on_epoch_begin(self, epoch, logs=None):
        # get values
        lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        print('lr:', lr)
        print('decay:',decay)
        # computer lr
        lr = lr * (1. / (1 + decay * epoch))
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)
'''

class CheckpointPB(keras.callbacks.Callback):

    def __init__(self, filepath, date, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CheckpointPB, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.date = date
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.loss = []
        self.val_loss = []
        self.lr = []

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        self.lr.append(logs.get("lr"))
        if epoch>1:
            list = [self.loss, self.val_loss, self.lr]
            print(list)
            df = pd.DataFrame(list)
            print(df)
            df.to_csv(os.path.join(self.filepath, self.date +'.csv'), index=False, header=False)


        plot(self.loss,self.val_loss,self.filepath,self.date)
        plot_lr(self.lr,self.filepath,self.date)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(os.path.join(self.filepath, self.date + '.h5'), overwrite=True)
                            save_tflite(self.model, self.filepath, self.date)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)



def train(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate = 1e-4,
         nb_epoch = 300,
         saved_weights_name = 'best_weights.h5'):
    """A function that performs training on a general keras model.

    # Args
        model : keras.models.Model instance
        loss_func : function
            refer to https://keras.io/losses/

        train_batch_gen : keras.utils.Sequence instance
        valid_batch_gen : keras.utils.Sequence instance
        learning_rate : float
        saved_weights_name : str
    """
    # 1. create optimizer
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-2)
    # optimizer = Adam(lr=learning_rate)

    #optimizer = SGD(lr=learning_rate, nesterov=True)
    # 2. create loss function
    model.compile(loss=loss_func, optimizer=optimizer)

    # 4. training
    train_start = time.time()
    train_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = ""
    for item in saved_weights_name.split('/')[:-1]: name = os.path.join(name,item)
    path = os.path.join(name, train_date)
    os.mkdir(path)
    saved_weights_name = os.path.join(path, train_date + '.h5')
    try:
        history = model.fit_generator(generator = train_batch_gen,
                        steps_per_epoch  = len(train_batch_gen), 
                        epochs           = nb_epoch,
                        validation_data  = valid_batch_gen,
                        validation_steps = len(valid_batch_gen),
                        callbacks        = _create_callbacks(path, train_date),                        
                        verbose          = 1,
                        workers          = 2,
                        max_queue_size   = 4)
    except KeyboardInterrupt:
        save_tflite(model,path,(train_date+"_ctrlc_"))
        raise

    _print_time(time.time()-train_start)
    save_tflite(model,path,(train_date+"_end_"))

def _print_time(process_time):
    if process_time < 60:
        print("{:d}-seconds to train".format(int(process_time)))
    else:
        print("{:d}-mins to train".format(int(process_time/60)))

def save_tflite(model, path, train_date):
        output_node_names = [node.op.name for node in model.outputs]
        input_node_names = [node.op.name for node in model.inputs]
        output_layer = model.layers[-2].name+'/BiasAdd'
        sess = K.get_session()
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
        graph_io.write_graph(constant_graph, "" , os.path.join (path, train_date + '.pb'), as_text=False)
        model.save("tmp.h5",include_optimizer=False)
        converter = tf.lite.TFLiteConverter.from_keras_model_file("tmp.h5",output_arrays=[output_layer])
        tflite_model = converter.convert()
        open(os.path.join (path, train_date + '.tflite'), "wb").write(tflite_model)

def plot(acc,val_acc,path,train_date):
    plt.figure(1)
    plt.plot(acc,'b',label='Train')
    plt.plot(val_acc,'r',label='Test')
    print('acc:',acc)
    print('val_acc:',val_acc)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, train_date +'.png'))

def plot_lr(lr,path,train_date):
    plt.figure(2)
    plt.plot(lr)
    print('lr:',lr)
    plt.title('learning rate curve')
    plt.ylabel('learning rate')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(path, train_date +'_lr.png'))


def _create_callbacks(saved_weights_name, train_date):
    # Make a few callbacks
    logging = TensorBoard(log_dir=saved_weights_name)

    early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0,
                       patience=8,
                       mode='min', 
                       verbose=1,
                       restore_best_weights=True)

    checkpoint = CheckpointPB(saved_weights_name,train_date, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
    #                           patience=2, verbose=1)

    lr_hist = LossHistory()

    callbacks = [lr_hist, logging, checkpoint, early_stop]
    return callbacks
