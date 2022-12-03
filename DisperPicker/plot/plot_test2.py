import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import Config
# np.set_printoptions(threshold=np.nan)

def plot_test(fig1, curve_G, fig2, curve_C, name, test=False, true_G=None, true_C=None):
    """ Plot the figures of the test process.

    Args:
        fig: Group and phase dispersion images.
        curve_G: Predicted group velocity curve.
        curve_C: Predicted phase velocity curve.
        name: Image storage name.
        test: If test is True, you must assign a value to true_G and true_C.
        true_G: Ground truth of the group velocity dispersion curve.
        true_C: Ground truth of the phase velocity dispersion curve.
    """

    fontsize = 18
    figformat = '.jpg'

    plt.figure(figsize=(12, 4), clear=True)
    plt.tick_params(labelsize=15)

    range_T = Config().range_T
    range_V = Config().range_V

    plt.subplot(121)
    x = np.linspace(range_T[0],range_T[1],range_T[2])
    y = np.linspace(range_V[0],range_V[1],range_V[2])

    image = fig1
    z_max = np.abs(image).max()
    plt.pcolor(x, y, image, shading='auto', cmap='jet', vmin=0, vmax=z_max+0.05)

    if test:
        b, e = line_interval(true_G)
        plt.plot(x[b:e],true_G[b:e],'-w', linewidth=3, label='Ground truth')
    b, e = line_interval(curve_G)
    plt.plot(x[b:e],curve_G[b:e],'--k', linewidth=3, label='Disperpicker')
    plt.legend(loc=0, fontsize=15)
    plt.colorbar()
    plt.ylim((range_V[0],range_V[1]))
    plt.xlabel('Period (s)',fontsize=fontsize)
    plt.ylabel('Group Velocity (km/s)',fontsize=fontsize)
    plt.title('Group velocity',fontsize=fontsize)
    plt.tick_params(labelsize=15)

    plt.subplot(122)         # after correction
    image = fig2
    z_max = np.array(image).max()
    z_min = np.array(image).min()        
    plt.pcolor(x, y, image, shading='auto', cmap='jet', vmin=z_min, vmax=z_max+0.05)
    if test:
        b, e = line_interval(true_C)
        plt.plot(x[b:e],true_C[b:e],'-w', linewidth=3, label='Ground truth')
    b, e = line_interval(curve_C)
    plt.plot(x[b:e],curve_C[b:e],'--k', linewidth=3, label='DisperPicker')
    b, e = line_interval(curve_G)
    plt.plot(x[b:e],curve_G[b:e],'--w', linewidth=3, label='Group disp')
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
    none_zero_index = np.where(curve != 0)
    if len(none_zero_index[0]) != 0:
        start = np.min(none_zero_index)
        end = np.max(none_zero_index)
    else:
        start = 0
        end = 0

    return start, end


if __name__ == '__main__':
    pass
