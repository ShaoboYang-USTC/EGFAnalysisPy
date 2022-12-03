#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @Author: Shaobo Yang
 @Time:11/17/18 10:25 AM 2018
 @Email: yang0123@mail.ustc.edu.cn
"""
import numpy as np
import tensorflow as tf
from config import Config

def cnn_layer(input, filter_size, strides, padding, acti_func=tf.nn.relu,
              damping=None, bias=None, name=None, norm=False):

    with tf.name_scope(name):
        weights = tf.Variable(initial_value=tf.truncated_normal(filter_size, 
                              stddev=0.1), name='weight')
        tf.summary.histogram('weights', weights)
        if damping is not None:
            loss_w = tf.multiply(tf.nn.l2_loss(weights), damping, name='weight_loss')
            tf.add_to_collection('losses', loss_w)

        if bias is not None:
            bias = tf.Variable(tf.constant(bias, shape=[filter_size[-1]]), name='bias')
            tf.summary.histogram('bias', bias)

        convolution = tf.nn.conv2d(input, weights, strides=strides, padding=padding)
        convolution = tf.add(convolution, bias)

        if norm:
            # convolution = batch_norm(convolution)
            convolution = bn_layer(convolution, Config().train)
            

        output = acti_func(convolution, name='output')
        tf.summary.histogram('output', output)

    return output


def copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling):
    result_from_contract_layer_crop = result_from_contract_layer
    return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=-1)


def trans_cnn_layer(input, output_size, filter_size, strides, padding, acti_func=tf.nn.relu,
         damping=None, bias=None, name=None, norm=False):

    with tf.name_scope(name):
        weights = tf.Variable(initial_value=tf.truncated_normal(filter_size, stddev=0.1), 
                              name='weight')
        tf.summary.histogram('weights', weights)
        if damping is not None:
            loss_w = tf.multiply(tf.nn.l2_loss(weights), damping, name='weight_loss')
            tf.add_to_collection('losses', loss_w)

        if bias is not None:
            bias = tf.Variable(tf.constant(bias, shape=[filter_size[-1]]), name='bias')
            tf.summary.histogram('bias', bias)

        convolution = tf.nn.conv2d_transpose(value=input, filter=weights,
                                             output_shape=output_size, strides=strides, 
                                             padding=padding)
        if norm:
            convolution = bn_layer(convolution, True)

        output = acti_func(convolution, name='output')
        tf.summary.histogram('output', output)
        # print(output.get_shape())

    return output


def fc_layer(input, output_size, input_shape=None,
             acti_func=tf.nn.relu, damping=None, bias=None, name=None, norm=False):

    with tf.name_scope(name):
        if input_shape is None:
            batch_size, input_shape = input.get_shape()
            input_shape = input_shape.value

        weight_shape = [input_shape, output_size]
        #print(weight_shape)

        weights = tf.Variable(initial_value=tf.truncated_normal(weight_shape, stddev=0.1), 
                              name='weights')
        tf.summary.histogram('weights', weights)

        if damping is not None:
            loss_w = tf.multiply(tf.nn.l2_loss(weights), damping, name='weight_loss')
            tf.add_to_collection('losses', loss_w)

        if bias is not None:
            bias = tf.Variable(tf.constant(bias, shape=[1, output_size]), name='bias')
            tf.summary.histogram('bias', bias)

        output = tf.add(tf.matmul(input, weights), bias)

        if norm:
            output = batch_norm(output)

        if acti_func is not None:
            output = acti_func(output, name='output')
            tf.summary.histogram('output', output)

    return output


def pool(input, ksize=[1, 1, 3, 1], strides=[1, 1, 3,1], padding='SAME',
         pool_function=tf.nn.max_pool, name=None):

    with tf.name_scope(name):
        output = pool_function(input, ksize, strides, padding, name=name)
        tf.summary.histogram('output', output)
    return output


def unfold(input, name=None):
    batch_size = Config().batch_size
    batch_size, height, width, num_channels = input.get_shape()
    with tf.name_scope(name):
        output = tf.reshape(input, [-1, height*width*num_channels], name=name)
    tf.summary.histogram(name, output)

    return output


def fully_connected_layer(input, input_shape=None, output_shape=Config().input_size,
                          acti_func=tf.nn.relu, bias=None, damping=None,
                          name=None):

    with tf.name_scope(name):
        if input_shape is None:
            batch_size, input_shape = input.get_shape()
            input_shape = input_shape.value

        weights = tf.Variable(initial_value=
                              tf.truncated_normal(shape=[input_shape, output_shape],
                                                  stddev=0.1), name='weights')
        tf.summary.histogram('weights', weights)

        if damping is not None:
            loss_w = tf.multiply(tf.nn.l2_loss(weights), damping, name='weight_loss')
            tf.add_to_collection('losses', loss_w)

        if bias is not None:
            bias = tf.Variable(tf.constant(bias, shape=[output_shape]), name='bias')
            tf.summary.histogram('bias', bias)

        output = tf.matmul(input, weights) + bias
        #output = acti_func(output, name='output')
        tf.summary.histogram('output', output)

        return output


def input_norm(input, name, cnn=True):
    input_size = list(np.array(input).shape)
    if cnn:
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
    else:
        mean, variance = tf.nn.moments(input, axes=[0])
    with tf.name_scope(name):
        scale = tf.Variable(tf.ones(input_size), name = 'scale')
        shift = tf.Variable(tf.zeros(input_size), name = 'shift')
        epsilon = 0.001
        with tf.control_dependencies([mean, variance]):
            input_norm = tf.nn.batch_normalization(input, mean, variance, 
                                                   shift, scale, epsilon)

        #input_norm = input_norm * 100
        tf.summary.histogram('input_norm', input_norm)
        tf.summary.histogram('scale', scale)
        tf.summary.histogram('shift', shift)
    return input_norm

'''
def batch_norm(input):
    input_size = list(np.array(input).shape)
    mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
    scale = tf.Variable(tf.ones(input_size), name='scale')
    shift = tf.Variable(tf.zeros(input_size), name='shift')
    epsilon = 0.001
    with tf.control_dependencies([mean, variance]):
        norm_output = tf.nn.batch_normalization(input, mean, variance, shift, 
                                                scale, epsilon)
    #norm_output = norm_output * 100
    tf.summary.histogram('norm_output', norm_output)
    tf.summary.histogram('scale', scale)
    tf.summary.histogram('shift', shift)

    return norm_output
'''

def layer_norm_compute_python(x, epsilon, scale, bias):
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias
 
def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def bn_layer(x,is_training,moving_decay=0.9,eps=1e-5,scope=None,reuse=None):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2,4]

    param_shape = shape[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        scale = tf.get_variable('layer_norm_scale',param_shape,initializer=tf.constant_initializer(1))
        shift  = tf.get_variable('layer_norm_bias', param_shape,initializer=tf.constant_initializer(0))

        # 计算当前整个batch的均值与方差
        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training, True),mean_var_with_update,
                lambda:(ema.average(batch_mean),ema.average(batch_var)))

        # 最后执行batch normalization
        return tf.nn.batch_normalization(x,mean,var,shift,scale,eps)

