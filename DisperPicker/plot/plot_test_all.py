import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import Config
# np.set_printoptions(threshold=np.nan)

def plot_test(fig1, prob_G, curve_G, fig2, prob_C, curve_C, name, 
                test=False, true_G=None, true_C=None):
    """ Plot the figures of the test process.

    Args:
        fig: Group and phase dispersion images.
        prob_G: Group velocity probability image.
        curve_G: Predicted group velocity curve.
        prob_C: Phase velocity probability image.
        curve_C: Predicted phase velocity curve.
        name: Image storage name.
        test: If test is True, you must assign a value to true_G and true_C.
        true_G: Ground truth of the group velocity dispersion curve.
        true_C: Ground truth of the phase velocity dispersion curve.
    """

    fontsize = 18
    figformat = '.jpg'

    plt.figure(figsize=(12, 16), clear=True)
    plt.tick_params(labelsize=15)

    range_T = Config().range_T
    range_V = Config().range_V

    plt.subplot(421)
    image = fig1

    z_max = np.abs(image).max()
    x1 = np.linspace(range_T[0],range_T[1],range_T[2])
    y1 = np.linspace(range_V[0],range_V[1],range_V[2])

    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=0, vmax=z_max+0.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('G disp spectrogram',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(422)
    x2 = x1
    y2 = y1

    plt.pcolor(x2, y2, prob_G, shading='auto', cmap='jet', vmin=0, vmax=z_max+0.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('Predicted G',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(423)      # plot the dispersion curve before correction
    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=0, vmax=z_max+0.05)
    x4 = x1
    prob_G = np.array(prob_G)
    max = np.max(prob_G, axis=0)
    prob_G = prob_G.T
    y4=[]
    for i in range(len(max)):
        index = list(prob_G[i]).index(max[i])
        y4.append(index*Config().dV+range_V[0])
    plt.plot(x4,y4,'--k', linewidth=3, label='Predicted')
    plt.colorbar()
    plt.ylim((range_V[0],range_V[1]))
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('Group velocity',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(424)       # after correction
    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=0, vmax=z_max+0.05)
    x4 = np.linspace(range_T[0],range_T[1],range_T[2])

    if test:
        b, e = line_interval(true_G)
        plt.plot(x4[b:e],true_G[b:e],'-w', linewidth=3, label='Ground truth')
    b, e = line_interval(curve_G)
    plt.plot(x4[b:e],curve_G[b:e],'--k', linewidth=3, label='Disperpicker')
    if test:
        plt.legend(loc=0, fontsize=15)
    plt.colorbar()
    plt.ylim((range_V[0],range_V[1]))
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('Group velocity',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(425)
    image = fig2

    z_max = np.array(image).max()
    z_min = np.array(image).min()
    x1 = np.linspace(range_T[0],range_T[1],range_T[2])
    y1 = np.linspace(range_V[0],range_V[1],range_V[2])

    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=z_min, vmax=z_max+0.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('C disp spectrogram',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(426)
    x2 = x1
    y2 = y1

    plt.pcolor(x2, y2, prob_C, shading='auto', cmap='jet', vmin=0, vmax=z_max+0.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('Predicted C',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(427)         # before correction
    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=z_min, vmax=z_max+0.05)
    x4 = x1
    prob_C = np.array(prob_C)
    max = np.max(prob_C, axis=0)
    prob_C = prob_C.T
    y4=[]
    for i in range(len(max)):
        index = list(prob_C[i]).index(max[i])
        y4.append(index*Config().dV+range_V[0])
    plt.plot(x4,y4,'--k', linewidth=3, label='Predicted')
    plt.colorbar()
    plt.ylim((range_V[0],range_V[1]))
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('Phase velocity',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(428)           # after correction
    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=z_min, vmax=z_max+0.05)
    x4 = np.linspace(range_T[0],range_T[1],range_T[2])
    if test:
        b, e = line_interval(true_C)
        plt.plot(x4[b:e],true_C[b:e],'-w', linewidth=3, label='Ground truth')
    b, e = line_interval(curve_C)
    plt.plot(x4[b:e],curve_C[b:e],'--k', linewidth=3, label='DisperPicker')
    b, e = line_interval(curve_G)
    # plt.plot(x4[b:e],curve_G[b:e],'-c', linewidth=2, label='G velocity')
    if test:
        plt.legend(loc=0, fontsize=15)
    plt.colorbar()
    plt.ylim((range_V[0],range_V[1]))
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('Phase velocity',fontsize=fontsize)

    plt.tick_params(labelsize=15)
    plt.tight_layout()

    plt.savefig(name+figformat, bbox_inches='tight', dpi=300)
    plt.close()

def line_interval(curve):
    start = 0
    end = Config().range_T[2]
    for each in curve:
        if each != 0:
            break
        start += 1

    reverse = list(curve)
    reverse.reverse()
    for each in reverse:
        if each != 0:
            break
        end -= 1
    return start, end


if __name__ == '__main__':
    pass
