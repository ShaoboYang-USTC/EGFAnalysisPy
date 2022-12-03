import numpy as np
import matplotlib.pyplot as plt
import random

def qc(disp, snr, v_range, diff_range, upward, each_stage_upward, min_len, skip=False):
    ''' Dispersion curve quality control.
    
    Args:
        disp (size = [1*T_sample]): input dispersion data. 
        snr (size = [1*T_sample]): SNR or dispersion data.
        v_range ([v_min, v_max]): velocity range.
        diff_range: ([diff_min, diff_max]): dv range.
        upward (float): min upward. if v[T_max] - v[T_min] > upward.
        each_stage_upward (float): for each stage, same as 'upward'.
        min_len (int): min length of dispersion curve.
        skip (bool): allow stage skip or not.

    Returns:
        Quality controled dispersion curve.
    '''

    v_min, v_max = v_range           
    each_stage_upward = each_stage_upward
    min_diff, max_diff = diff_range
    velocity = disp
    new_velocity = np.zeros(len(velocity))

    # process
    # process 1: remove outliers
    zero_index = np.where((velocity > v_max)|(velocity < v_min))
    velocity[zero_index] = 0

    # process 2: remove large difference
    stage = np.zeros(len(velocity) - 1)
    diff = np.diff(velocity)
    for j in range(len(diff)):
        if diff[j] < max_diff and diff[j] > min_diff and velocity[j+1] != 0:
            stage[j] = 1
    if skip:
        for j in range(len(stage))[2:-2]:
            if (stage[j] == 0 and stage[j-1] == 1 and 
                    stage[j-2] == 1 and stage[j+1] == 1 and stage[j+2] == 1):
                stage[j] = 1

    start = []
    end = []
    new_stage = True
    for j in range(len(stage)):
        if stage[j] == 1 and new_stage == True:
            start.append(j)
            new_stage = False
        if stage[j] != 1 and new_stage == False:
            end.append(j)
            new_stage = True
        if stage[j] == 1 and new_stage == False and j == len(stage)-1:
            end.append(j+1)
    stage_len = np.array(end) - np.array(start) + 1
    stage_snr = np.zeros(len(stage_len))
    for j in range(len(stage_len)):
        stage_snr[j] = np.sum(snr[start[j]:end[j]+1])

    # sort to use larger stage prefer
    if len(stage_len):
        info = list(zip(stage_snr, stage_len, start, end))
        # print(info)
        info.sort(reverse=True)
        stage_snr, stage_len, start, end = zip(*info)
        if stage_len[0] >= min_len:
            valid_velocity = velocity[start[0]:end[0]+1]
            start2 = [0]
            end2 = []
            for j in range(len(valid_velocity))[1:-1]:
                if valid_velocity[j] >= valid_velocity[j-1] and valid_velocity[j] >= valid_velocity[j+1]:
                    end2.append(j)
                    start2.append(j)
                elif valid_velocity[j] <= valid_velocity[j-1] and valid_velocity[j] <= valid_velocity[j+1]:
                    end2.append(j)
                    start2.append(j)
            end2.append(len(valid_velocity)-1)
            valid = []
            for j in range(len(start2)):
                if valid_velocity[start2[j]] - valid_velocity[end2[j]] < each_stage_upward:
                    valid.append(1)
                else:
                    valid.append(0)

            # process 3: remove downward stages.
            new_stage = True
            valid_start2 = []
            valid_end2 = []
            for j in range(len(valid)):
                if valid[j] == 1 and new_stage == True:
                    valid_start2.append(j)
                    new_stage = False
                if valid[j] != 1 and new_stage == False:
                    valid_end2.append(j-1)
                    new_stage = True
                if valid[j] == 1 and new_stage == False and j == len(valid)-1:
                    valid_end2.append(j)

            true_start = []
            true_end = []
            true_length = []
            for j in range(len(valid_start2)):
                true_start.append(start2[valid_start2[j]])
                true_end.append(end2[valid_end2[j]])
                true_length.append(end2[valid_end2[j]]-start2[valid_start2[j]]+1)

            if len(true_length) > 0:
                max_index = true_length.index(max(true_length))
                final_velocity = np.zeros(len(valid_velocity))
                final_velocity[true_start[max_index]:true_end[max_index]+1] = \
                               valid_velocity[true_start[max_index]:true_end[max_index]+1]

                if (true_length[max_index] >= min_len and 
                        final_velocity[true_start[max_index]] - final_velocity[true_end[max_index]] < upward):
                    new_velocity[start[0]:end[0]+1] = final_velocity

    return new_velocity

def plot_disp(disp, T_range, fig_name):

    fontsize = 20
    disp_num = 0
    plt.figure(figsize=(8, 5), clear=True)
    disp = np.array(disp)
    T = np.linspace(T_range[0], T_range[1], T_range[2])
    all_color = ['-k','-b','-g','-y','-r','-m','-c']
    for k in range(len(disp)):
        start = 0
        end = 0
        II = np.where(disp[k] != 0)
        if len(II[0]) != 0:
            disp_num += 1
            start = np.min(II)
            end = np.max(II)
        color = random.sample(all_color, 1)[0]
        plt.plot(T[start:end], disp[k, start:end], color, linewidth=1.5)
    print(f'None zero dispersion curve number: {disp_num}')
    print(f'All dispersion curve number: {len(disp)}')
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Velocity (km/s)',fontsize=fontsize)
    plt.ylim((0, 5))
    plt.yticks([0, 1, 2, 3, 4, 5])
    plt.tick_params(labelsize=15)

    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()

def mean(disp, T_range, fig_name):

    velocity = np.zeros(T_range[2])
    num = np.zeros(T_range[2])
    mean = np.zeros(T_range[2])

    for v in disp:
        II = np.where(v != 0)
        velocity[II] += v[II]
        num[II] += 1

    II = np.where(num != 0)
    mean[II] = velocity[II] / num[II]
    print(f'Dispersion data point number: {num}')
    # print(mean)

    fontsize = 18
    T = np.linspace(T_range[0], T_range[1], T_range[2])
    plt.plot(T, mean, '-go', linewidth=2, markersize=3)

    plt.xlabel('T (s)', fontsize=fontsize)
    plt.ylabel('Velocity (km/s)', fontsize=fontsize)
    plt.ylim((1, 4))
    plt.tick_params(labelsize=15)

    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()
