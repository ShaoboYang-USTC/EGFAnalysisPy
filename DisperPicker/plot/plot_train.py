import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import Config
# np.set_printoptions(threshold=np.nan)

def plot_train(fig, curve1, curve2, name):
    """ Plot the figures of the training process.

    Args:
        fig: Group and phase dispersion images.
        curve1: Predicted probability images.
        curve2: Label probability images.
        name: Image storage name.
    """

    range_V = Config().range_V                   # velocity range
    range_T = Config().range_T                   # period range

    fontsize = 18
    figformat = '.png'

    plt.figure(figsize=(12, 16), clear=True)
    plt.tick_params(labelsize=15)
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(421)
    image = fig[0]

    z_max = np.array(image).max()
    z_min = np.array(image).min()
    x1 = np.linspace(range_T[0],range_T[1],range_T[2])
    y1 = np.linspace(range_V[0],range_V[1],range_V[2])

    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=z_min, vmax=z_max+0.05)
    plt.colorbar()
    ture_G = []
    curve2[0] = np.array(curve2[0])
    max = np.max(curve2[0], axis=0)
    curve2[0] = curve2[0].T
    for i in range(len(max)):
        index = list(curve2[0][i]).index(max[i])
        ture_G.append(index*Config().dV+range_V[0])
    b, e = line_interval(ture_G, range_T, range_V)
    plt.plot(x1[b:e],ture_G[b:e],'-wo', linewidth=2, markersize=3, label='label')
    curve2[0] = curve2[0].T

    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('G disp spectrogram',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(422)
    x2 = x1
    y2 = y1

    plt.pcolor(x2, y2, curve1[0], shading='auto', cmap='jet', vmin=0, vmax=1.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('Predicted G',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(423)
    x3 = x1
    y3 = y1

    plt.pcolor(x3, y3, curve2[0], shading='auto', cmap='jet', vmin=0, vmax=1.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('Label G',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(424)
    x4 = x1
    curve1[0] = np.array(curve1[0])
    max = np.max(curve1[0], axis=0)
    curve1[0] = curve1[0].T
    y4=[]
    for i in range(len(max)):
        index = list(curve1[0][i]).index(max[i])
        y4.append(index*Config().dV+range_V[0])
    plt.plot(x4,y4,'-ko', linewidth=2, markersize=3, label='Predicted')

    x4 = x1
    #curve2[0] = np.array(curve2[0])
    #max = np.max(curve2[0], axis=0)
    #curve2[0] = curve2[0].T
    #y4=[]
    #for i in range(len(max)):
    #    index = list(curve2[0][i]).index(max[i])
    #    y4.append(index/500)
    plt.pcolor(x1, y1, image, cmap='jet', shading='auto', vmin=z_min, vmax=z_max+0.05)
    plt.colorbar()
    b, e = line_interval(ture_G, range_T, range_V)
    plt.plot(x4[b:e],ture_G[b:e],'-wo', linewidth=2, markersize=3, label='Label')
    xrefer = x4[b:e]
    yrefer = ture_G[b:e]
    plt.ylim((range_V[0],range_V[1]))
    plt.legend(loc=0,fontsize=14)

    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('Group velocity',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(425)
    image = fig[1]

    z_max = np.array(image).max()
    z_min = np.array(image).min()
    x1 = np.linspace(range_T[0],range_T[1],range_T[2])
    y1 = np.linspace(range_V[0],range_V[1],range_V[2])

    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=z_min, vmax=z_max+0.05)
    plt.colorbar()
    ture_C = []
    curve2[1] = np.array(curve2[1])
    max = np.max(curve2[1], axis=0)
    curve2[1] = curve2[1].T
    for i in range(len(max)):
        index = list(curve2[1][i]).index(max[i])
        ture_C.append(index*Config().dV+range_V[0])
    b, e = line_interval(ture_C, range_T, range_V)
    plt.plot(x1[b:e],ture_C[b:e],'-wo', linewidth=2, markersize=3, label='Label')
    curve2[1] = curve2[1].T
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('C disp spectrogram',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(426)
    x2 = x1
    y2 = y1

    plt.pcolor(x2, y2, curve1[1], shading='auto', cmap='jet', vmin=0, vmax=1.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('Predicted C',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(427)
    x3 = x1
    y3 = y1

    plt.pcolor(x3, y3, curve2[1], shading='auto', cmap='jet', vmin=0, vmax=1.05)
    plt.colorbar()
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('Label C',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(428)
    x4 = x1
    curve1[1] = np.array(curve1[1])
    max = np.max(curve1[1], axis=0)
    curve1[1] = curve1[1].T
    y4=[]
    for i in range(len(max)):
        index = list(curve1[1][i]).index(max[i])
        y4.append(index*Config().dV+range_V[0])
    plt.plot(x4,y4,'-ko', linewidth=2, markersize=3, label='Predicted')

    x4 = x1
    #curve2[1] = np.array(curve2[1])
    #max = np.max(curve2[1], axis=0)
    #curve2[1] = curve2[1].T
    #y4=[]
    #for i in range(len(max)):
    #    index = list(curve2[1][i]).index(max[i])
    #    y4.append(index/500)
    plt.pcolor(x1, y1, image, shading='auto', cmap='jet', vmin=z_min, vmax=z_max+0.05)
    plt.colorbar()
    b, e = line_interval(ture_C, range_T, range_V)
    plt.plot(x4[b:e],ture_C[b:e],'-wo', linewidth=2, markersize=3, label='Label')

    # plt.plot(xrefer,yrefer,'-co', linewidth=1.5, markersize=2, label='G velocity')

    plt.ylim((range_V[0],range_V[1]))
    plt.legend(loc=0,fontsize=14)

    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Phase Velocity (km/s)',fontsize=fontsize)
    plt.title('Phase velocity',fontsize=fontsize)
    plt.tick_params(labelsize=15)
    plt.tight_layout()

    plt.savefig(name+figformat, bbox_inches='tight', dpi=300)
    plt.close()

def line_interval(curve, range_T, range_V):

    start = 0
    end = range_T[-1]
    for each in curve:
        if each != range_V[0]:
            break
        start += 1

    reverse = list(curve)
    reverse.reverse()
    for each in reverse:
        if each != range_V[0]:
            break
        end -= 1
    return start, end

if __name__ == '__main__':
    pass
