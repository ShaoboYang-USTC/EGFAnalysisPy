from genericpath import exists
import os
import math
import numpy as np
import sys
sys.path.append('DisperPicker')
from pick import Pick
from plot.plot_test2 import plot_test
import qc
from config import Config
import EGFAnalysisTimeFreq

config = Config()
pick = Pick()
file_list = os.listdir(config.waveform_path)
## single test:
# file_list = ['FD01_FD16.dat']
file_list.sort()
file_num = len(file_list)
T = np.linspace(config.range_T[0], config.range_T[1], config.range_T[2])
print(f'\nFile number: {file_num}\n')
all_G = []
all_C = []

for index in range(file_num):
    print('\n', '------------------------------------------------')
    print(f'{index + 1}/{file_num}, {file_list[index]}')
    print('\nT-variable = no')
    batch_group_image = np.zeros((1, config.input_size[0], config.input_size[1]))
    batch_phase_image = np.zeros((1, config.input_size[0], config.input_size[1]))
    if config.test:
        key = os.path.splitext(file_list[index])[0].split('_')
        if os.path.exists(config.test_data_path + '/group_velocity/' + key[0] + '.' + key[1] + '.txt'):
            batch_group_v = np.loadtxt(config.test_data_path + '/group_velocity/' + key[0] + 
                                       '.' + key[1] + '.txt')[:config.input_size[1]]
        else:
            batch_group_v = np.zeros((1, config.input_size[1]))
        if os.path.exists(config.test_data_path + '/phase_velocity/' + key[0] + '.' + key[1] + '.txt'):
            batch_phase_v = np.loadtxt(config.test_data_path + '/phase_velocity/' + key[0] + 
                                       '.' + key[1] + '.txt')[:config.input_size[1]]
        else:
            batch_phase_v = np.zeros((1, config.input_size[1]))
        batch_group_v = np.array([batch_group_v[:, 1]])
        batch_phase_v = np.array([batch_phase_v[:, 1]])
    file_path = os.path.join(config.waveform_path, file_list[index])
    file_name = os.path.splitext(file_list[index])[0]
    # 1. No time variable filtering
    gfcn = EGFAnalysisTimeFreq.gfcn_analysis(DataFileName=file_path, StartT=config.StartT, EndT=config.EndT,  
                                             DeltaT=config.DeltaT, StartV=config.StartV, 
                                             EndV=config.EndV, DeltaV=config.DeltaV, 
                                             GreenFcnObjects=config.GreenFcnObjects, 
                                             WinAlpha=config.WinAlpha, NoiseTime=config.NoiseTime, 
                                             MinSNR=config.MinSNR)

    sta_info = [gfcn.StaDist, gfcn.Longitude_A, gfcn.Latitude_A, gfcn.Longitude_B, gfcn.Latitude_B]
    group_image = gfcn.GroupVelocityImgCalculate()
    group_image = group_image[:config.input_size[0], :config.input_size[1]]
    if True not in np.isnan(group_image):
        batch_group_image[0, config.input_size[0]-group_image.shape[0]:, :group_image.shape[1]] = group_image
    snr = gfcn.SNR_T
    SNRIndex = gfcn.SNRIndex
    phase_image = gfcn.PhaseVelocityImgCalculate(TimeVariableFilter=gfcn.TimeVariableFilterType.no)
    phase_image = phase_image[:config.input_size[0], :config.input_size[1]]
    if True not in np.isnan(phase_image):
        batch_phase_image[0, config.input_size[0]-phase_image.shape[0]:, :group_image.shape[1]] = phase_image

    print(f'\n  Start picking')
    pick.mean_confidence_G = 0
    group_velocity, phase_velocity, _, _ = pick.pick(group_image=batch_group_image, 
                                                phase_image=batch_phase_image, sta_info=[sta_info], 
                                                snr=[snr], file_list=[file_name], ct=0.01, save_result=False, 
                                                group_velocity_label=batch_group_v if config.test else None, 
                                                phase_velocity_label=batch_phase_v if config.test else None)

    # 2. Use time variable filtering
    print('\nT-variable = obs')
    if np.count_nonzero(group_velocity[0]) > 0:
        # gfcn.GroupVelocityImgCalculate()
        gfcn.GroupDisperCurve = group_velocity[0]
        phase_image = gfcn.PhaseVelocityImgCalculate(TimeVariableFilter=gfcn.TimeVariableFilterType.obs)
        phase_image = phase_image[:config.input_size[0], :config.input_size[1]]
        if True not in np.isnan(phase_image):
            batch_phase_image[0, config.input_size[0]-phase_image.shape[0]:, :group_image.shape[1]] = phase_image
    print(f'\n  Start picking')
    pick.mean_confidence_G = 0.3
    group_velocity, phase_velocity, prob_G, prob_C = pick.pick(group_image=batch_group_image, 
                                                               phase_image=batch_phase_image, sta_info=[sta_info], 
                                                               snr=[snr], file_list=[file_name], ct=2, 
                                                               save_result=True, 
                                                               group_velocity_label=batch_group_v if config.test else None, 
                                                               phase_velocity_label=batch_phase_v if config.test else None)

    # 3. Quality control                               
    print('\n  Quality control')
    if np.count_nonzero(snr) == 0:  
        snr = np.ones(config.input_size[1])
    group_velocity = qc.qc(group_velocity[0], snr, v_range=[0.6, 3.5], diff_range=[-0.07, 0.08],  
                           upward=0.4, each_stage_upward=0.3, min_len=config.min_len, skip=False)
    phase_velocity = qc.qc(phase_velocity[0], snr, v_range=[0.6, 3.8], diff_range=[-0.1, 0.1],  
                           upward=0, each_stage_upward=0.2, min_len=config.min_len, skip=True)
    all_G.append(group_velocity)
    all_C.append(phase_velocity)
    format_G = np.array([T, group_velocity, snr, prob_G]).T
    format_C = np.array([T, phase_velocity, snr, prob_C]).T
    fig_name = '{}/qc_plot/{}'.format(config.result_path, file_name)
    plot_test(batch_group_image[0], group_velocity, batch_phase_image[0], phase_velocity, fig_name)
    disp_name = '{}/qc_result/GDisp.{}.txt'.format(config.result_path, file_name)
    with open(disp_name, 'w') as f:
        f.write(f'{sta_info[1]:.8f}    {sta_info[2]:.8f}\n')
        f.write(f'{sta_info[3]:.8f}    {sta_info[4]:.8f}\n')
        np.savetxt(f, format_G, fmt="%1.2f  %1.3f  %.3f  %.3f")
    disp_name = '{}/qc_result/CDisp.{}.txt'.format(config.result_path, file_name)
    with open(disp_name, 'w') as f:
        f.write(f'{sta_info[1]:.8f}    {sta_info[2]:.8f}\n')
        f.write(f'{sta_info[3]:.8f}    {sta_info[4]:.8f}\n')
        np.savetxt(f, format_C, fmt="%1.2f  %1.3f  %.3f  %.3f")

# 4. Statistic
all_G = np.array(all_G)
all_C = np.array(all_C)
print('Phase velocity: ')
qc.plot_disp(all_C, config.range_T, config.result_path + '/all_C.png')
qc.mean(all_C, config.range_T, config.result_path + '/mean_C.png')
print('Group velocity: ')
qc.plot_disp(all_G, config.range_T, config.result_path + '/all_G.png')
qc.mean(all_G, config.range_T, config.result_path + '/mean_G.png')
