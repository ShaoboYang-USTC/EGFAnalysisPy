# -*- coding: utf-8 -*-
"""
 @Author: Shaobo Yang
 @Time:5/26/2021 15:51 PM
 @Email: yang0123@mail.ustc.edu.cn
"""

import math
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import tensorflow as tf
import time
from config import Config
from plot.plot_train import plot_train
from reader.reader import Reader
from tflib import layers

class CNN(object):
    """ Build a CNN.

    Attributes:
        config: the config in config.py.
        train_writer: the path saving the trained CNN weights.
        result_path: the path saving the training, validation and test results.
    """

    def __init__(self):
        self.config = Config()
        self.batch_size = self.config.batch_size
        self.layer = self.setup_layer(self.batch_size)
        self.loss = self.setup_loss()
        self.valid_loss = self.setup_valid_loss()
        self.reader = Reader()
        self.training_data_path = self.config.training_data_path
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.config.root + '/DisperPicker/summary/logs')
        self.result_path = self.config.result_path

    def setup_layer(self, batch_size):
        layer = dict()
        size = self.config.input_size
        result_from_contract_layer = dict()
        layer['label'] = tf.placeholder(tf.float32,
                                        shape=[None, size[0], size[1], size[2]], name='label')
        layer['input'] = tf.placeholder(tf.float32,
                                        shape=[None, size[0], size[1], size[2]], name='input')
        # normalization
        # layer['input_norm'] = layers.input_norm(layer['input'], name = 'input_norm')
        layer['conv1'] = layers.cnn_layer(layer['input'],
                                          filter_size = [5, 3, 2, 8],
                                          strides = [1, 1, 1, 1],
                                          padding = 'SAME',
                                          damping = self.config.damping,
                                          bias = 0.0,
                                          name = 'conv1',
                                          norm = True)
        result_from_contract_layer['1'] = layer['conv1']
        layer['pooling1'] = layers.pool(layer['conv1'],
                                        ksize=[1, 4, 2, 1],
                                        strides=[1, 4, 2, 1],
                                        padding='SAME',
                                        pool_function=tf.nn.max_pool,
                                        name='pooling1')
        size1 = [math.ceil(size[0]/4), math.ceil(size[1]/2)]

        layer['conv2'] = layers.cnn_layer(layer['pooling1'],
                                          filter_size=[5, 3, 8, 16],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          damping=self.config.damping,
                                          bias=0.0,
                                          name='conv2',
                                          norm=False)
        result_from_contract_layer['2'] = layer['conv2']
        layer['pooling2'] = layers.pool(layer['conv2'],
                                        ksize=[1, 4, 2, 1],
                                        strides=[1, 4, 2, 1],
                                        padding='SAME',
                                        pool_function=tf.nn.max_pool,
                                        name='pooling2')
        size2 = [math.ceil(size1[0]/4), math.ceil(size1[1]/2)]

        layer['conv3'] = layers.cnn_layer(layer['pooling2'],
                                          filter_size=[5, 3, 16, 32],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          damping=self.config.damping,
                                          bias=0.0,
                                          name='conv3',
                                          norm=False)
        result_from_contract_layer['3'] = layer['conv3']
        layer['pooling3'] = layers.pool(layer['conv3'],
                                        ksize=[1, 4, 2, 1],
                                        strides=[1, 4, 2, 1],
                                        padding='SAME',
                                        pool_function=tf.nn.max_pool,
                                        name='pooling3')
        # size3 = [math.ceil(size2[0]/4), math.ceil(size2[1]/2)]

        layer['conv4'] = layers.cnn_layer(layer['pooling3'],
                                          filter_size=[5, 3, 32, 64],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          damping=self.config.damping,
                                          bias=0.0,
                                          name='conv4',
                                          norm=False)

        layer['trans_conv4'] = layers.trans_cnn_layer(layer['conv4'],
                                                      output_size=[batch_size, size2[0], size2[1], 32],
                                                      filter_size=[4, 2, 32, 64],
                                                      strides=[1, 4, 2, 1],
                                                      padding='SAME',
                                                      damping=self.config.damping,
                                                      bias=0.0,
                                                      name='trans_conv4',
                                                      norm=False)

        merge = layers.copy_and_crop_and_merge(
            result_from_contract_layer=result_from_contract_layer['3'],
            result_from_upsampling=layer['trans_conv4'])
        layer['conv5'] = layers.cnn_layer(merge,
                                          filter_size=[5, 3, 64, 32],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          damping=self.config.damping,
                                          bias=0.0,
                                          name='conv5',
                                          norm=False)
        layer['trans_conv5'] = layers.trans_cnn_layer(layer['conv5'],
                                                      output_size=[batch_size, size1[0], size1[1], 16],
                                                      filter_size=[4, 2, 16, 32],
                                                      strides=[1, 4, 2, 1],
                                                      padding='SAME',
                                                      damping=self.config.damping,
                                                      bias=0.0,
                                                      name='trans_conv5',
                                                      norm=False)

        merge = layers.copy_and_crop_and_merge(
            result_from_contract_layer=result_from_contract_layer['2'],
            result_from_upsampling=layer['trans_conv5'])
        layer['conv6'] = layers.cnn_layer(merge,
                                          filter_size=[5, 3, 32, 16],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          damping=self.config.damping,
                                          bias=0.0,
                                          name='conv6',
                                          norm=False)

        layer['trans_conv6'] = layers.trans_cnn_layer(layer['conv6'],
                                                      output_size=[batch_size, size[0], size[1], 8],
                                                      filter_size=[4, 2, 8, 16],
                                                      strides=[1, 4, 2, 1],
                                                      padding='SAME',
                                                      damping=self.config.damping,
                                                      bias=0.0,
                                                      name='trans_conv6',
                                                      norm=False)

        merge = layers.copy_and_crop_and_merge(
            result_from_contract_layer=result_from_contract_layer['1'],
            result_from_upsampling=layer['trans_conv6'])
        layer['conv7'] = layers.cnn_layer(merge,
                                          filter_size=[5, 3, 16, 8],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          damping=self.config.damping,
                                          bias=0.0,
                                          name='conv7',
                                          norm=True)

        layer['prediction'] = layers.cnn_layer(layer['conv7'],
                                          filter_size=[5, 3, 8, 2],
                                          strides=[1, 1, 1, 1],
                                          padding='SAME',
                                          acti_func=tf.nn.sigmoid,
                                          damping=self.config.damping,
                                          bias=0.0,
                                          name='prediction',
                                          norm=True)

        return layer

    def setup_loss(self):
        with tf.name_scope('loss'):
            raw_loss = tf.reduce_sum(tf.square(self.layer['prediction'] - self.layer['label']),
                                      reduction_indices=1)
            raw_loss = tf.reduce_mean(raw_loss)

            tf.summary.scalar('raw_loss', raw_loss)
            tf.add_to_collection('losses', raw_loss)
            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.summary.scalar('total_loss', loss)

        return loss, raw_loss

    def setup_valid_loss(self):
        with tf.name_scope('valid_loss'):
            valid_raw_loss = tf.reduce_sum(tf.square(self.layer['prediction'] - self.layer['label']),
                                           reduction_indices=1)
            valid_raw_loss = tf.reduce_mean(valid_raw_loss)

            tf.summary.scalar('valid_raw_loss', valid_raw_loss)
            tf.add_to_collection('valid_losses', valid_raw_loss)
            valid_loss = tf.add_n(tf.get_collection('valid_losses'), name='valid_total_loss')
            tf.summary.scalar('valid_total_loss', valid_loss)

        return valid_loss, valid_raw_loss


    def train(self, passes):
        b_time = time.time()
        epoch = 1
        start_point = 1
        all_valid_loss = []
        all_train_loss = []
        filename_train = self.reader.get_all_filename(self.config.training_data_path + '/group_velocity')
        filename_validation = self.reader.get_all_filename(self.config.validation_data_path + '/group_velocity')
        file_num_train = len(os.listdir(self.training_data_path + '/group_velocity'))
        
        # conf = tf.ConfigProto()
        # conf.gpu_options.per_process_gpu_memory_fraction = 0.80 
        with tf.Session() as sess:
            loss, raw_loss = self.loss
            training = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)

            saver = tf.train.Saver(max_to_keep=1000)
            global_step = 0
            
            # continue train
            with open(self.config.root + '/DisperPicker/saver/checkpoint') as file:
                line = file.readline()
                ckpt = line.split('"')[1]
                ckpt = ckpt.split('/')[-1]
                global_step = int(ckpt.split('-')[1])
            print('Restored from checkpoint ' + ckpt)
            saver.restore(sess, self.config.root + '/DisperPicker/saver/' + ckpt)
            
            print('Start training')

            ## new train
            # init = tf.global_variables_initializer()
            # sess.run(init)
            # self.train_writer.add_graph(sess.graph)

            for step in range(1 + global_step, 1 + global_step + passes):
            # for step in range(global_step, global_step + passes):
                if start_point > file_num_train - self.batch_size:
                    epoch += 1
                    start_point = 1
                input, label = self.reader.get_batch_data(start_point=start_point, 
                                                          seed=epoch, 
                                                          file_list=filename_train)
                
                summary, _, = sess.run([self.merged, training],
                                       feed_dict={self.layer['input']: input,
                                                  self.layer['label']: label})
                self.train_writer.add_summary(summary, step)

                if step % 10 == 0:
                    loss, raw_loss = sess.run(self.loss, feed_dict={self.layer['input']: input,
                                                                    self.layer['label']: label})
                    print('global step:',step)
                    print('training_loss {}'.format(loss))
                    print('raw_loss {}'.format(raw_loss), '\n')
                    all_train_loss.append(raw_loss)

                if step % 100 == 0:
                    validation_input, validation_label, validation_filename = \
                        self.reader.get_validation_data(file_list=filename_validation)

                    validation_output = sess.run(self.layer['prediction'],
                                                 feed_dict={self.layer['input']:
                                                                validation_input})
                    valid_loss, valid_raw_loss = sess.run(self.valid_loss,
                                                          feed_dict={self.layer['input']: validation_input,
                                                                     self.layer['label']: validation_label})

                    all_valid_loss.append(valid_raw_loss)

                if step % 100 == 0:
                    saver.save(sess, self.config.root + '/DisperPicker/saver/', global_step=step)
                    print('validation_loss {}'.format(valid_loss))
                    print('validation_raw_loss {}'.format(valid_raw_loss), '\n')
                    print('all_train_loss:', all_train_loss)
                    print('all_valid_loss:', all_valid_loss, '\n')
                    print(validation_filename, '\n')

                    target = self.result_path + '/validation_result/' + str(step) + '/'
                    if not os.path.exists(target):
                        os.makedirs(target)

                    plot_size = len(validation_output)

                    # save validation result
                    for i in range(plot_size):
                        input = validation_input[i].transpose((2, 0, 1))
                        curve1 = validation_output[i].transpose((2, 0, 1))
                        curve2 = validation_label[i].transpose((2, 0, 1))

                        curve1 = list(curve1)
                        curve2 = list(curve2)

                        name = validation_filename[i].split('.')
                        new_name = name[0] + '_' + name[1]
                        plot_train(fig=input, curve1=curve1, curve2=curve2, 
                                   name=target+new_name)

                if step % 3 == 0:
                    start_point += self.batch_size
        e_time = time.time()
        print('Time consuming:', e_time-b_time)
        

    def predict(self, sess, input):
        prediction = sess.run(self.layer['prediction'],
                          feed_dict = {self.layer['input']: input})

        return prediction

if __name__ == '__main__':
    cnn = CNN()
    cnn.train(passes = Config().training_step)
