#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @Author: Shaobo Yang
 @Time:05/26/2021 15:39 PM
 @Email: yang0123@mail.ustc.edu.cn
"""

import os
import math
import numpy as np
import random
from config import Config

class Reader(object):
    def __init__(self):
        self.config = Config()
        self.root = self.config.root
        self.training_data_path = self.config.training_data_path
        self.validation_data_path = self.config.validation_data_path
        self.test_data_path = self.config.test_data_path
        self.batch_size = self.config.batch_size
        self.radius = self.config.radius
        self.data_size = self.config.input_size
        self.data_range_V = [self.config.range_V[0], self.config.range_V[1], self.config.dV]
        self.data_range_T = [self.config.range_T[0], self.config.range_V[1], self.config.dT]

    def get_all_filename(self, file_path):
        filename_list = []
        for file in os.listdir(file_path):
            filename = os.path.splitext(file)[0]
            filename_list.append(filename)
        #filename_list = np.array(filename_list)
        #filename_list.sort()
        return filename_list

    def get_batch_data(self, start_point, seed, file_list):

        np.random.seed(seed)
        np.random.shuffle(file_list)

        train_batch_file = file_list[start_point:start_point + self.batch_size]
        # print('Train File: \n')
        # print(train_batch_file)
        batch_data = []
        batch_label = []
        VRange = self.data_range_V
        TRange = self.data_range_T
        row = self.data_size[0]
        col = self.data_size[1]

        for each_train_file in train_batch_file:
            each_data1 = np.loadtxt(self.training_data_path + '/' + 
                                    '/group_image/' + each_train_file + '.dat')
            each_label1 = np.loadtxt(self.training_data_path + '/' + 
                                     '/group_velocity/' + each_train_file + '.dat')
            each_data1 = each_data1[:, :col]
            each_label1 = each_label1[:col, 1]
            if len(each_label1) < col:
                zero = np.zeros(col - len(each_label1))
                each_label1 = np.concatenate((each_label1, zero), axis=0)
            else:
                each_label1 = each_label1[:col]
            len_each_data1 = len(each_data1)
            if row > len_each_data1:
                zero = np.zeros((row - len_each_data1, col))
                each_data1 = np.concatenate((zero, each_data1), axis=0)
            else:
                each_data1 = each_data1[len_each_data1-row:]
            num = 0
            matrix1 = np.zeros((row, col))
            r = self.radius
            for i in each_label1:
                if i != 0:
                    y_index = int((i - VRange[0])/VRange[2])
                    for j in range(len(matrix1)):
                        matrix1[j, num] = np.exp(-((y_index-j)**2)/(2*r**2))
                num += 1

            each_data2 = np.loadtxt(self.training_data_path + '/' + 
                                    '/phase_image/' + each_train_file + '.dat')
            each_label2 = np.loadtxt(self.training_data_path + '/' + 
                                     '/phase_velocity/' + each_train_file + '.dat')
            each_data2 = each_data2[:, :col]
            each_label2 = each_label2[:col, 1]
            
            if len(each_label2) < col:
                zero = np.zeros(col - len(each_label2))
                each_label2 = np.concatenate((each_label2, zero), axis=0)
            else:
                each_label2 = each_label2[:col]
            len_each_data2 = len(each_data2)
            if row > len_each_data2:
                zero = np.zeros((row - len_each_data2, col))
                each_data2 = np.concatenate((zero, each_data2), axis=0)
            else:
                each_data2 = each_data2[len_each_data2-row:]
            
            num = 0
            matrix2 = np.zeros((row, col))
            r = self.radius
            for i in each_label2:
                if i != 0:
                    y_index = int((i - VRange[0])/VRange[2])
                    for j in range(len(matrix2)):
                        matrix2[j, num] = np.exp(-((y_index-j)**2)/(2*r**2))
                num += 1

            each_data = np.array([each_data1, each_data2])
            each_label = np.array([matrix1, matrix2])
            each_data = each_data[:, :self.data_size[0], :self.data_size[1]]
            each_label = each_label[:, :self.data_size[0], :self.data_size[1]]

            batch_data.append(each_data)
            batch_label.append(each_label)

        batch_data = np.array(batch_data)
        batch_data = batch_data.transpose((0, 2, 3, 1))
        batch_label = np.array(batch_label)
        batch_label = batch_label.transpose((0, 2, 3, 1))

        return batch_data, batch_label

    def get_validation_data(self, file_list):
        # random.seed(0)
        file_list = random.sample(file_list, self.config.batch_size)
        validation_data = []
        validation_label = []
        VRange = self.data_range_V
        TRange = self.data_range_T
        row = self.data_size[0]
        col = self.data_size[1]

        for each_valid_file in file_list:
            each_data1 = np.loadtxt(self.validation_data_path + '/' + 
                                    '/group_image/' + each_valid_file  + '.dat')
            each_label1 = np.loadtxt(self.validation_data_path + '/' + 
                                    '/group_velocity/' + each_valid_file  + '.dat')
            each_data1 = each_data1[:, :col]
            each_label1 = each_label1[:col, 1]
            if len(each_label1) < col:
                zero = np.zeros(col - len(each_label1))
                each_label1 = np.concatenate((each_label1, zero), axis=0)
            else:
                each_label1 = each_label1[:col]
            len_each_data1 = len(each_data1)
            if row > len_each_data1:
                zero = np.zeros((row - len_each_data1, col))
                each_data1 = np.concatenate((zero, each_data1), axis=0)
            else:
                each_data1 = each_data1[len_each_data1-row:]

            num = 0
            matrix1 = np.zeros((row,col))
            r = self.radius
            for i in each_label1:
                if i != 0:
                    y_index = int((i - VRange[0])/VRange[2])
                    for j in range(len(matrix1)):
                        matrix1[j, num] = np.exp(-((y_index-j)**2)/(2*r**2))
                num = num + 1

            each_data2 = np.loadtxt(self.validation_data_path + '/' + 
                                    '/phase_image/' + each_valid_file + '.dat')
            each_label2 = np.loadtxt(self.validation_data_path + '/' + 
                                    '/phase_velocity/' + each_valid_file + '.dat')
            each_data2 = each_data2[:, :col]
            each_label2 = each_label2[:col, 1]

            if len(each_label2) < col:
                zero = np.zeros(col - len(each_label2))
                each_label2 = np.concatenate((each_label2, zero), axis=0)
            else:
                each_label2 = each_label2[:col]
            len_each_data2 = len(each_data2)
            if row > len_each_data2:
                zero = np.zeros((row - len_each_data2, col))
                each_data2 = np.concatenate((zero, each_data2), axis=0)
            else:
                each_data2 = each_data2[len_each_data2-row:]

            num = 0
            matrix2 = np.zeros((row,col))
            r = self.radius
            for i in each_label2:
                if i != 0:
                    y_index = int((i - VRange[0])/VRange[2])
                    for j in range(len(matrix2)):
                        matrix2[j, num] = np.exp(-((y_index-j)**2)/(2*r**2))
                num += 1

            each_data = np.array([each_data1, each_data2])
            each_label = np.array([matrix1, matrix2])
            each_data = each_data[:, :self.data_size[0], :self.data_size[1]]
            each_label = each_label[:, :self.data_size[0], :self.data_size[1]]

            validation_data.append(each_data)
            validation_label.append(each_label)

        validation_data = np.array(validation_data)
        validation_data = validation_data.transpose((0,2,3,1))
        validation_label = np.array(validation_label)
        validation_label = validation_label.transpose((0,2,3,1))

        return validation_data, validation_label, file_list

    def get_test_file(self):
        filename_list = []
        for file in os.listdir(self.config.test_data_path + '/' + 'group_image'):
            filename = os.path.splitext(file)[0]
            filename_list.append(filename)
        filename_list = np.array(filename_list)
        filename_list.sort()

        return filename_list

    def get_disp_matrix(self, file_path, size):
        ''' Read a dispersion matrix(image).

        Attributes:
            file_path: File path.
            size ([int, int]): Expected matrix size. Zero padding or cutting   
                if the matrix in the file is not consist with you want. 

        Raises:
            Exception: Wrong input size.

        Returns:
            A numpy array with the size of 'size'.
        '''

        input_matrix = np.loadtxt(file_path)
        input_matrix = input_matrix[:size[0], :size[1]]
        input_size = input_matrix.shape

        if input_size[0] <= size[0]:
            st = size[0] - input_size[0]
            matrix = np.zeros(size)
            if input_size[1] >= size[1]:
                matrix[int(st):, :] = input_matrix[:, :size[1]]
            else:
                matrix[int(st):, :input_size[1]] = input_matrix
        else:
            raise Exception('Wrong input size!')

        return matrix
    
    def get_label_matrix(self, file_path, size):
        ''' Read a dispersion curve and generate a label prob matrix.

        Attributes:
            file_path: File path.
            size ([int, int]): Expected matrix size. ssss

        Returns:
            A numpy array with the size of 'size'.
        '''
        
        try:
            disp_curve = np.loadtxt(file_path)
            disp_curve = disp_curve[:size[1], 1]
        except:
            disp_curve = np.zeros(size[1])
        
        matrix = np.zeros(size)
        for i in range(len(disp_curve)):
            vel = disp_curve[i]
            if vel != 0:
                y_index = int((vel - self.config.range_V[0])/self.config.dV)
                for j in range(size[0]):
                    matrix[j, i] = np.exp(-((y_index - j)**2)/(2*self.radius**2))

        return matrix
