#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @Author: Shaobo Yang
 @Time:12/12/2019 20:10 PM
 @Email: yang0123@mail.ustc.edu.cn
"""

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from config import Config
from train_cnn import CNN
from plot.plot_test import plot_test
from qc import qc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Pick(object):
    """ Pick the dispersion curves.

    DisperPicker configuration：
    
    Abbreviations:
        G: group velocity
        C: phase velocity
        T: period
        V: velocity

    Attributes (Omit those can be found in config.py):
        range_T: Period range [start, end, num]
        range_V: Velocity range [start, end, num]
    
    Data path:
        result_path: Result path.

    Picking thresholds:
        ref_T (int): Find the local maximum points in this column of C dispersion image. 
            This para can be set to 'None' to use the default value.
        ref_T2 ([int, int]): Use these columns to calculate the average probability of C curves.   
            This para can be set to [] to use the default value.
        min_len (int): Accept the dispersion curves if it's length (number of points) is  
            larger than this parameter.

    Detailed picking thresholds:
        disp_G_value (float): Accept the G points if G dispersion image value is larger 
            than this.
        mean_confidence_G (float): Extend the G curve if average G probability value is  
            larger than this parameter.
        begin (int): Sometimes the short period G dispersion image is not stable. To pick a   
            more smooth G dispersion curve, the G curve shorter than this parameter period 
            (number of points) can be traced using the long period curve. If this is not  
            zero, the 'forward' parameter must be 'True' to left extend the G curve.
        forward (True or False): Whether left extend the G curve.  
        backward (True or False): Whether right extend the G curve. 
        max_dv_G, max_dv_C (float): the G curve will be stop if the velocity deviation   
            between two period is larger this parameter.
        slow_G, slow_C (int): G dispersion image prefer to find a smaller v as T is smaller.
        v_max, v_min (float): Limit the extracted v from v_min to v_max.
    
    """

    def __init__(self):

        self.config = Config()
        self.model = CNN()
        self.sess = tf.Session()
        # Load the trained model.
        with open(self.config.root + '/DisperPicker/saver/checkpoint') as file:
            line = file.readline()
            ckpt = line.split('"')[1]
            ckpt = ckpt.split('/')[-1]
        saver = tf.train.Saver()
        # ckpt = '-10000'
        saver.restore(self.sess, self.config.root + '/DisperPicker/saver/' + ckpt)
        print('\nRestored CNN model from checkpoint: ' + ckpt)
        
        self.range_T = self.config.range_T       # [start, end, num]
        self.range_V = self.config.range_V 
        self.dT = self.config.dT
        self.dV = self.config.dV
        self.input_size = self.config.input_size
        self.result_path = self.config.result_path
        self.test_data_path = self.config.test_data_path

        # Check result path
        if not os.path.exists('{}/pick_result'.format(self.result_path)):
            os.makedirs('{}/pick_result'.format(self.result_path))
        if not os.path.exists('{}/plot'.format(self.result_path)):
            os.makedirs('{}/plot'.format(self.result_path))

        # Picking thresholds
        self.ref_T = None
        self.ref_T2 = [] 
        self.confidence_G = self.config.confidence_G 
        self.mean_confidence_C = self.config.mean_confidence_C
        self.confidence_C = self.config.confidence_C
        self.min_len = self.config.min_len 

        # Another detailed thresholds
        self.disp_G_value = 0
        self.mean_confidence_G = 0.3
        self.begin = 0
        self.forward = True 
        self.backward = True 
        self.max_dv_G = 0.15 
        self.max_dv_C = 0.23 
        self.slow_G = 0 
        self.slow_C = 0 
        self.v_max = self.range_V[1] - 0.1 
        self.v_min = self.range_V[0] + 0.1 


    def pick(self, group_image, phase_image, sta_info, snr, file_list, ct, save_result, 
                 group_velocity_label=None, phase_velocity_label=None):
        """ Pick the dispersion curves using the trained CNN.

        Attributes (Omit those can be found in config.py):
            group_image (np.array): shape = [file_num, v_num, T_num]
            phase_image (np.array): shape = [file_num, v_num, T_num]
            sta_info (np.array): shape = [file_num, 5], each row includes:
                             [StaDist, Longitude_A, Latitude_A, Longitude_B, Latitude_B]
            snr (np.array): shape = [file_num]
            file_list (list): shape = [file_num]
            ct (flaot): Add distance constraint to extracted dispersion curves.
            save_result (bool): save the picking results or not.

            test (bool): If you want to test the performance of DisperPicker, you can set  
                this para to True. You have to place the label (group_velocity and phase_velocity)
                in 'test_data_path' to test the DisperPicker, and when you run the pick.py, 
                DisperPicker will compare the reault with the label. 
                If you only want to use DisperPicker to pick dispersion curves, this should be False.
            group_velocity_label (np.array): shape = [batch_size, num]
            phase_velocity_label (np.array): shape = [batch_size, num]

        """
        
        self.batch_size = np.array(group_image).shape[0]
        sta_info = np.array(sta_info)
        group_velocity = []
        phase_velocity = []
        self.test = self.config.test 
        self.radius = self.config.radius
        self.group_velocity_label = group_velocity_label
        self.phase_velocity_label = phase_velocity_label

        # Check batch size
        if self.config.batch_size != self.batch_size:
            raise ValueError('Incorrect batch number. Please check the batch number in config.py')

        # print('Start picking!')

        # Extract the dispersion curves for each batch.
        batch_input = np.array([group_image, phase_image])    # [channel, file, V, T]
        batch_input = batch_input.transpose((1, 2, 3, 0))               # [file, V, T, channel]

        # batch_pred_prob: Predicted probability maps.
        batch_pred_prob = self.model.predict(sess=self.sess, input=batch_input)    

        # transpose: [file, V, T, channel] to [channel, file, V, T]
        batch_pred_prob = batch_pred_prob.transpose((3, 0, 1, 2))   
        batch_pred_probG = batch_pred_prob[0]
        batch_pred_probC = batch_pred_prob[1]

        # Calculate the loss value.
        if self.test:
            true_vG_all = self.group_velocity_label[:, :self.input_size[1]]
            true_vC_all = self.phase_velocity_label[:, :self.input_size[1]]
            r = self.radius
            true_probG = np.zeros((self.batch_size, self.input_size[0], self.input_size[1]))
            for i in range(self.batch_size):
                for j, v in enumerate(true_vG_all[i]):
                    if v != 0:
                        y_index = int((v - self.range_V[0])/self.dV)
                        for k in range(self.input_size[0]):
                            true_probG[i, k, j] = np.exp(-((y_index-k)**2)/(2*r**2))
            true_probC = np.zeros((self.batch_size, self.input_size[0], self.input_size[1]))
            for i in range(self.batch_size):
                for j, v in enumerate(true_vC_all[i]):
                    if v != 0:
                        y_index = int((v - self.range_V[0])/self.dV)
                        for k in range(self.input_size[0]):
                            true_probC[i, k, j] = np.exp(-((y_index-k)**2)/(2*r**2))
            batch_label = np.array([true_probG, true_probC])    # [channel, file, V, T]
            loss = np.sum((batch_label - batch_pred_prob)*(batch_label - 
                                    batch_pred_prob), axis=2)
            loss = np.mean(loss)
            self.loss = loss
            print(f'  * Test loss = {loss:.3f}')

        # Save the results.
        for i in range(self.batch_size):
            file_name = file_list[i]
            dist = sta_info[i, 0]
            if not self.ref_T:                
                self.ref_T = int(0.6*min(self.range_T[2]-1, round((dist/1.5/3.2 - self.range_T[0])/self.dT)))

            # Extract group velocity.
            true_vG = true_vG_all[i] if self.test else None
            pred_probG = np.array(batch_pred_probG[i])
            max_probG = np.max(pred_probG, axis=0)
            disp_G = batch_input[i, :, :, 0]
            pred_vG = []
            for j in range(len(max_probG)):
                if max_probG[j] > self.confidence_G:
                    column = pred_probG[:, j]
                    # Find the index of the max value.
                    index_all = np.where(column==max_probG[j])
                    if len(index_all) % 2 == 0:
                        index = np.median(index_all[:-1])
                    else:
                        index = np.median(index_all)
                    if disp_G[int(index), j] > self.disp_G_value:
                        pred_vG.append(index*self.dV)
                    else:
                        pred_vG.append(0)
                else:
                    pred_vG.append(0)

            x = np.linspace(self.range_T[0], self.range_T[1], self.range_T[2])
            pred_vG = self.process_G(pred_vG, disp_G, pred_probG)

            # Find the T start and the end of the disp.
            # G_start is the T start of the group velocity.
            # G_end is the T end of the group velocity.
            pred_vG[np.where(pred_vG != 0)] += self.range_V[0]
            pred_vG = self.dist_constraint(pred_vG, dist, ct)
            pred_vG = pred_vG[:self.input_size[1]]
            none_zero_index = np.where(pred_vG != 0)
            if len(none_zero_index[0]) != 0:
                G_start = np.min(none_zero_index)
                G_end = np.max(none_zero_index)
            else:
                G_start = 0
                G_end = 0
            print('  * Group velocity period index range:', G_start, G_end) 
            group_velocity.append(pred_vG)
            prob_vG = np.zeros(self.input_size[1])
            for j in np.where(pred_vG != 0)[0]:
                vi = int((pred_vG[j] - self.range_V[0])/self.dV) + 1
                prob_vG[j] = pred_probG[vi, j]
            
            output_vG = []
            output_vG.append(x)
            output_vG.append(pred_vG[:self.input_size[1]])
            output_vG.append(snr[i])
            output_vG.append(prob_vG)
            output_vG = np.array(output_vG).T 
            if save_result:
                disp_name = '{}/pick_result/GDisp.{}.txt'.format(self.result_path, file_name)
                with open(disp_name, 'w') as f:
                    f.write(f'{sta_info[i, 1]:.8f}    {sta_info[i, 2]:.8f}\n')
                    f.write(f'{sta_info[i, 3]:.8f}    {sta_info[i, 4]:.8f}\n')
                    np.savetxt(f, output_vG, fmt="%1.2f  %1.3f  %.3f  %.3f")
            
            # Extract phase velocity.
            pred_probC = np.array(batch_pred_probC[i])
            disp_C = batch_input[i, :, :, 1]
            true_vC = true_vC_all[i] if self.test else None

            # random plot
            if save_result and random.random() <= self.config.random_plot_ratio:
                self.plot = True
            else:
                self.plot = False
            fig_name = '{}/plot/{}'.format(self.result_path, file_name)

            if self.ref_T2:
                pred_vC = self.pick_C(pred_probC, self.ref_T2[0], self.ref_T2[1], disp_C, 
                                      fig_name)
            else:
                pred_vC = self.pick_C(pred_probC, G_start, G_end, disp_C, fig_name)

            pred_vC = self.process_C(pred_vC, pred_probC)
            
            pred_vC[np.where(pred_vC != 0)] += self.range_V[0]
            pred_vC = self.dist_constraint(pred_vC, dist, ct)
            pred_vC = pred_vC[:self.input_size[1]]
            none_zero_index = np.where(pred_vC != 0)
            if len(none_zero_index[0]) != 0:
                C_start = np.min(none_zero_index)
                C_end = np.max(none_zero_index)
            else:
                C_start = 0
                C_end = 0
            print('  * Phase velocity period index range:', C_start, C_end)
            phase_velocity.append(pred_vC)
            prob_vC = np.zeros(self.input_size[1])
            for j in np.where(pred_vC != 0)[0]:
                vi = int((pred_vC[j] - self.range_V[0])/self.dV) + 1
                prob_vC[j] = pred_probC[vi, j]

            output_vC = []
            output_vC.append(x)
            output_vC.append(pred_vC[:self.input_size[1]])
            output_vC.append(snr[i])
            output_vC.append(prob_vC)
            output_vC = np.array(output_vC).T
            if save_result:
                disp_name = '{}/pick_result/CDisp.{}.txt'.format(self.result_path, file_name)
                with open(disp_name, 'w') as f:
                    f.write(f'{sta_info[i, 1]:.8f}    {sta_info[i, 2]:.8f}\n')
                    f.write(f'{sta_info[i, 3]:.8f}    {sta_info[i, 4]:.8f}\n')
                    np.savetxt(f, output_vC, fmt="%1.2f  %1.3f  %.3f  %.3f")

            # random plot
            if self.plot:
                print('  * Plot', file_name)
                plot_test(disp_G, pred_probG, pred_vG, disp_C, pred_probC, pred_vC,
                          fig_name, self.test, true_vG, true_vC)

            self.group_velocity = np.array(group_velocity)
            self.phase_velocity = np.array(phase_velocity)
            self.prob_vG = prob_vG
            self.prob_vC = prob_vC
        return np.array(group_velocity), np.array(phase_velocity), prob_vG, prob_vC


    def process_G(self, pred_vG, disp_G, pred_probG):
        """ Process the exrtacted group velocity dispersion curves.

        Args:
            pred_vG: Group velocity curve.
            disp_G: Group velocity dispersion image.
            pred_probG: Predicted group velocity probability map.
        
        Returns:
            Processed group velocity dispersion curve.

        """
        max_dv = self.max_dv_G
        v_max = self.v_max
        v_min = self.v_min
        row = self.input_size[0]
        col = self.input_size[1]
        slow = self.slow_G
        dV = self.dV
        begin = self.begin                  # Use b to end to trace 0-b
        forward = self.forward
        backward = self.backward

        # index1: From short period to long period.
        # index2: From long period to short period.
        index1 = np.arange(0, col, 1)
        index2 = np.arange(col - 1, -1, -1)
        start = 0
        end = col - 1

        # Process1: remove outliers.
        for j in index1[1:col - 1]:
            if abs(pred_vG[j] - pred_vG[j - 1]) > max_dv and abs(pred_vG[j] - pred_vG[j + 1]) > max_dv:
                pred_vG[j] = 0.5*(pred_vG[j - 1] + pred_vG[j + 1])

        # Correction
        for j in index1:
            j = int(j)
            max_probGrange = int((j/col*1 + 0.1)/dV)     # Search range.
            if pred_vG[j] + self.range_V[0] > v_min and pred_vG[j] + self.range_V[0] < v_max:
                key_index = int(round(pred_vG[j]/dV))
                for k in range(max_probGrange):
                    if key_index - k > 0 and key_index + k < self.range_V[2] - 1:
                        if (disp_G[key_index + k, j] >= disp_G[key_index + k - 1, j] and 
                                disp_G[key_index + k, j] >= disp_G[key_index + k + 1, j]):
                            key_index += k
                            break
                        if (disp_G[key_index - k, j] >= disp_G[key_index - k - 1, j] and 
                                disp_G[key_index - k, j] >= disp_G[key_index - k + 1, j]):
                            key_index -= k
                            break
                pred_vG[j] = key_index*dV

        # process2: remove unstable and only keep stable part.
        good_points = 0
        for j in index1[:col - 1]:
            if (abs(pred_vG[j + 1] - pred_vG[j]) > max_dv or pred_vG[j + 1] + self.range_V[0] > v_max or 
                    pred_vG[j + 1] + self.config.range_V[0] < v_min):
                start = j + 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 8:
                    break

        good_points = 0
        for j in index2[:col - 1]:
            if (abs(pred_vG[j] - pred_vG[j - 1]) > max_dv or pred_vG[j] + self.range_V[0] > v_max or 
                    pred_vG[j] + self.range_V[0] < v_min):
                end = j - 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 8:
                    break

        if start > 0:
            pred_vG[:start] = np.zeros(start)
        if end < col - 1:
            pred_vG[end + 1:] = np.zeros(col - 1 - end)

        # Process3: find the most stable stage.
        stage_index = []
        pred_vG = list(pred_vG)
        pred_vG.append(0)
        for j in index1:
            if len(stage_index) == 0:
                stage_index.append(j)
            if abs(pred_vG[j] - pred_vG[j + 1]) > max_dv:
                stage_index.append(j + 1)
        if len(stage_index) == 1:
            stage_index.append(col)
        if len(stage_index) == 0:
            stage_index.append(0)
            stage_index.append(col)

        stage_length = list(np.array(stage_index[1:]) - np.array(stage_index[:-1]))
        len_stage = len(stage_index)

        # Calculate the average probability for each stage
        stage_eng = []
        for j in range(len(stage_index) - 1):
            eng = 0
            length = 0
            for k in range(stage_index[j], stage_index[j + 1]):
                eng += pred_probG[int(pred_vG[k]/dV), k]
                length += 1
            if length >= self.min_len:
                stage_eng.append(eng/length)
            else:
                stage_eng.append(0)
        max_probGeng = np.max(stage_eng)
        max_probGindex = stage_eng.index(max_probGeng)
        end = stage_index[max_probGindex + 1]
        start = stage_index[max_probGindex]
        new_pred_vG = np.zeros(col)

        # no smooth
        if len_stage <= 10 and np.max(stage_length) >= self.min_len:
            print(f'  * Average value in G dispersion image: {max_probGeng:.3f}')
            new_pred_vG[max(start, begin):end] = pred_vG[max(start, begin):end]
            # Extend group velocity based on the stable stage.
            if max_probGeng >= self.mean_confidence_G:
                if max(start, begin) > 0 and forward:
                    i = list(range(max(start, begin)))
                    i.reverse()
                    for j in i:
                        key_index = int(round(new_pred_vG[j + 1]/dV))
                        max_probGrange = int(self.max_dv_G/dV)
                        for k in range(max_probGrange):
                            if key_index - k > 0 and key_index + k < self.range_V[2] - 1:
                                if (disp_G[key_index - k, j] >= disp_G[key_index - k - 1, j] and 
                                        disp_G[key_index - k, j] >= disp_G[key_index - k + 1, j]):
                                    key_index = key_index - k
                                    break
                                if k >= slow:
                                    if disp_G[key_index + k - slow, j] >= disp_G[key_index + k - slow - 1, j] and \
                                            disp_G[key_index + k - slow, j] >= disp_G[key_index + k - slow + 1, j]:
                                        key_index = key_index + k - slow
                                        break
                        if disp_G[key_index,j] >= self.disp_G_value and k < max_probGrange - 1:
                            new_pred_vG[j] = key_index*dV
                            start = j
                        else:
                            break

                if end < col and backward:
                    for j in range(end, col):
                        key_index = int(round(new_pred_vG[j - 1]/dV))
                        # max_probGrange = int((j/col*1.0 + 0.1)*500)
                        max_probGrange = int(self.max_dv_G/dV)    # Search in 0.1 range
                        for k in range(max_probGrange):
                            if key_index - k > 0 and key_index + k < self.range_V[2] - 1:
                                if (disp_G[key_index + k, j] >= disp_G[key_index + k - 1, j] and 
                                        disp_G[key_index + k, j] >= disp_G[key_index + k + 1, j]):
                                    key_index = key_index + k
                                    break
                                if k >= slow:
                                    if (disp_G[key_index - k + slow, j] >= disp_G[key_index - k + slow - 1, j] and
                                            disp_G[key_index - k + slow, j] >= disp_G[key_index - k + slow + 1, j]):
                                        key_index = key_index - k + slow
                                        break
                        if disp_G[key_index, j] >= self.disp_G_value and k < max_probGrange - 1:
                            new_pred_vG[j] = key_index*dV
                            end = j
                        else:
                            break
        return new_pred_vG


    def pick_C(self, map_C, start_T, end_T, disp_C, name):
        """Extracted phase velocity dispersion curves.

        Args:
            map_C: Phase velocity probability image.
            start_T，end_T: Use these columns to calculate the average probability of C curves.
            disp_C: Phase velocity disprsion image.
            name: Figure name.
        
        Returns:
            Extracted raw phase velocity dispersion curve.
        """
        col = self.input_size[1]
        name = name + '_pc.jpg'
        dV = self.dV
        dT = self.dT
        boundary = 25                   # up the lower boundary

        # find potential phase curve     # todo: scipy.find_peaks
        ref_points = []   # size: n*1
        ref_col = disp_C[:, self.ref_T]
        slow = self.slow_C
        for j in range(1, self.range_V[2] - 1):
            if ref_col[j] >= ref_col[j - 1] and ref_col[j] >= ref_col[j + 1]:
                if ref_col[j] == ref_col[j - 1]:
                    ref_points.append(j - 1)
                else:
                    ref_points.append(j)

        potential_c = []
        # print(ref_points)

        for each_refp in ref_points:
            each_curve = [[self.ref_T, each_refp]]
            # trace before the reference point
            before = list(range(self.ref_T))
            before.reverse()
            key_index = each_refp
            for j in before:        # loop for each column
                max_probGrange = int((0.2)/dV)
                if key_index > self.range_V[2] - 1 or key_index < 0:
                    break
                if (disp_C[key_index, j] >= disp_C[key_index + 1, j] and disp_C[key_index, j] >= 
                        disp_C[key_index - 1, j]):
                    if disp_C[key_index, j] == disp_C[key_index - 1, j]:
                        key_index = key_index - 1
                    else:
                        key_index = key_index
                    each_curve.insert(0, [j, key_index])
                else:
                    for k in range(max_probGrange)[1:]:
                        exist = False
                        if key_index - k > 0 and key_index - k < self.range_V[2] - 1:
                            if (disp_C[key_index - k, j] >= disp_C[key_index - k - 1, j] and 
                                    disp_C[key_index - k, j] >= disp_C[key_index - k + 1, j]):
                                key_index = key_index - k
                                each_curve.insert(0, [j, key_index])
                                exist = True
                                break
                        if k >= slow:
                            if key_index + k - slow < self.range_V[2] - 1 and key_index + k - slow >= 0:
                                if (disp_C[key_index + k - slow, j] >= disp_C[key_index + k - slow - 1, j] and 
                                        disp_C[key_index + k - slow, j] >= disp_C[key_index + k - slow + 1, j]):
                                    key_index = key_index + k - slow
                                    each_curve.insert(0, [j, key_index])
                                    exist = True
                                    break
                    if not exist:
                        break
                # boundary stop
                if key_index <=  boundary or key_index >= self.range_V[2] - boundary:
                    break

            # Trace after the reference point
            after = list(range(col)[self.ref_T + 1:])
            key_index = each_refp
            for j in after:        # loop for each column
                max_probGrange = int((j/col*1.0 + 0.3)/dV)
                if disp_C[key_index, j] >= disp_C[key_index + 1, j] and disp_C[key_index, j] >= disp_C[key_index - 1, j]:
                    if disp_C[key_index, j] == disp_C[key_index + 1, j]:
                        key_index = key_index + 1
                    else:
                        key_index = key_index
                    each_curve.append([j, key_index])
                else:
                    for k in range(max_probGrange)[1:]:
                        exist = False
                        if key_index + k > 0 and key_index + k < self.range_V[2] - 1:
                            if disp_C[key_index + k, j] >= disp_C[key_index + k - 1, j] and \
                                    disp_C[key_index + k, j] >= disp_C[key_index + k + 1, j]:
                                key_index = key_index + k
                                each_curve.append([j, key_index])
                                exist = True
                                break
                        if k >= slow:
                            if key_index - k + slow > 0 and key_index - k + slow < self.range_V[2] - 1:
                                if disp_C[key_index - k + slow, j] >= disp_C[key_index - k + slow - 1, j] and \
                                        disp_C[key_index - k + slow, j] >= disp_C[key_index - k + slow + 1, j]:
                                    key_index = key_index - k + slow
                                    each_curve.append([j, key_index])
                                    exist = True
                                    break
                    if not exist:
                        break

                # boundary stop
                if key_index <= boundary + 1 or key_index >= self.range_V[2] - boundary - 2:
                    break
            potential_c.append(each_curve)

        # plot each potential phase curve
        if self.plot and len(potential_c) != 0:
            # plt.figure(figsize=(5, 3), clear=True)
            # plt.tight_layout()
            # fontsize = 12

            # x1 = np.linspace(self.range_T[0], self.range_T[1], self.range_T[2])
            # y1 = np.linspace(self.range_V[0], self.range_V[1], self.range_V[2])
            # plt.pcolor(x1, y1, disp_C, shading='auto', cmap='jet', vmin=-1, vmax=1.05)
            # plt.colorbar()
            # plt.xlabel('Period (s)', fontsize=fontsize)
            # plt.ylabel('Phase Velocity (km/s)', fontsize=fontsize)
            # plt.title('C spectrogram', fontsize=fontsize)

            # for each in potential_c:
            #     # print(each)
            #     y2 = np.array(each)[:, 1]*dV + np.ones(len(each))*self.range_V[0]
            #     x2 = np.array(each)[:, 0]*dT + self.range_T[0]
            #     plt.plot(x2, y2, '-wo', markersize=1)

            # for i in ref_points:
            #     plt.plot(x1[self.ref_T], i*dV + self.range_V[0], 'wo', markersize=5)

            # # plt.show()
            # plt.savefig(name, bbox_inches='tight', dpi=300)
            # plt.close()
            print('  * Potential phase curve number:', len(potential_c))

        # find the best phase curve
        pred_vG = np.zeros(col)
        potential_c2 = []
        if len(potential_c) != 0:
            confidence = []
            each_len = []
            for each in potential_c:
                each_conf = 0
                num = 0
                for each_item in each:
                    if each_item[0] >= start_T and each_item[0] <= end_T:
                        each_conf += map_C[each_item[1], each_item[0]]
                        num += 1
                if num >= self.min_len:
                    confidence.append(each_conf)
                    potential_c2.append(each)
                    each_len.append(num)

            if len(confidence) != 0:
                max_probGindex = confidence.index(max(confidence))
                print(f'  * Max average C probability value:{ confidence[max_probGindex]/each_len[max_probGindex]:.3f}')
                if confidence[max_probGindex]/each_len[max_probGindex] >= self.mean_confidence_C:
                    for each2 in potential_c2[max_probGindex]:
                        pred_vG[each2[0]] = each2[1]*dV

        return pred_vG


    def process_C(self, pred_vC, map_C):
        """Process phase velocity dispersion curves.

        Args:
            pred_vC: Phase velocity dispersion curve.
            map_C: Phase velocity probability image.
        
        Returns:
            Processed phase velocity dispersion curve.
        """
        max_dv = self.max_dv_C
        v_max = self.v_max
        v_min = self.v_min
        row = self.input_size[0]
        col = self.input_size[1]
        slow = self.slow_G
        dV = self.dV
        dT = self.dT

        # process
        index1 = np.arange(0, col, 1)
        index2 = np.arange(col - 1, -1, -1)
        start = 0
        end = col - 1

        # process1: remove unstable and only save stable part
        good_points = 0
        for j in index1[:col - 1]:
            if (abs(pred_vC[j + 1] - pred_vC[j]) > max_dv or pred_vC[j + 1] + self.range_V[0] > 
                    v_max or pred_vC[j + 1] + self.range_V[0] < v_min):
                start = j + 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 8:
                    break
        good_points = 0
        for j in index2[:col - 1]:
            if (abs(pred_vC[j] - pred_vC[j - 1]) > max_dv or pred_vC[j] + self.range_V[0] > 
                    v_max or pred_vC[j] + self.range_V[0] < v_min):
                end = j - 1
                good_points = 0
            else:
                good_points = good_points + 1
                if good_points >= 8:
                    break

        if start > 0:
            pred_vC[:start] = np.zeros(start)
        if end < col - 1:
            pred_vC[end + 1:] = np.zeros(col - 1 - end)

        # process2: remove unstable and only keep stable part.
        stage_index = []
        pred_vC = list(pred_vC)
        pred_vC.append(0)
        for j in index1[:col]:
            if len(stage_index) == 0:
                stage_index.append(j)
            if abs(pred_vC[j] - pred_vC[j + 1]) > max_dv:
                stage_index.append(j + 1)
        if len(stage_index) == 1:
            stage_index.append(col)
        if len(stage_index) == 0:
            stage_index.append(0)
            stage_index.append(col)

        stage_length = list(np.array(stage_index[1:]) - np.array(stage_index[:-1]))
        len_stage = len(stage_index)
        stage_eng = []
        for j in range(len(stage_index) - 1):
            sum_eng = 0
            start = stage_index[j]
            end = stage_index[j + 1]
            num = stage_length[j]
            for k in range(start, end):
                sum_eng += map_C[int(pred_vC[k]/dV)][k]
            if num >= self.min_len:
                stage_eng.append(sum_eng/num)
            else:
                stage_eng.append(0)

        max_probGindex = stage_eng.index(np.max(stage_eng))
        start = stage_index[max_probGindex]
        end = stage_index[max_probGindex + 1]
        for k in range(end - 1, -1, -1):
            if map_C[int(pred_vC[k]/dV)][k] >= self.confidence_C:
                break
        new_pred_vG = np.zeros(col)
        
        if len_stage <= 5 and stage_length[max_probGindex] >= self.min_len:
            new_pred_vG[start:k + 1] = pred_vC[start:k + 1]

        return new_pred_vG

    def dist_constraint(self, pred_v, dist, ratio=2.0):
        """Add distance constraint to extracted dispersion curves.
        Distance must be larger than ratio*v*T

        Args:
            pred_v: Dispersion curve.
            dist: Station pair distance.
            ratio: Threshold.
        
        Returns:
            Dispersion curves with distance constraint.
        """
        range_T = self.range_T
        T = np.linspace(range_T[0], range_T[1], range_T[2])
        for i in range(len(T)):      
            if pred_v[i] != 0:
                if dist/ratio/pred_v[i] < T[i]:
                    break
        new_curve = np.zeros(len(T))
        new_curve[:i + 1] = pred_v[:i + 1]

        return new_curve
