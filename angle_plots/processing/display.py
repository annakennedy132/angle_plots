from matplotlib import pyplot as plt
import numpy as np
from angle_plots.processing import traja_alt
    
def stim_plot(length_seconds, n_highs, signal, threshold, y_limit=True):
    # Convert to NumPy arrays if needed
    length_seconds = np.asarray(length_seconds)
    n_highs = np.asarray(n_highs)
    signal = np.asarray(signal)

    fig, axs = plt.subplots(2, 1)

    axs[0].step(length_seconds, n_highs)
    axs[0].axhline(y=threshold, color='r', linestyle='-')
    if y_limit:
        axs[0].set_ylim(bottom=0, top=threshold * 3)
    axs[0].set_title("Number of bright pixels")

    axs[1].step(length_seconds, signal)
    axs[1].set_title("Signal")
    axs[1].set_ylim(bottom=0, top=1.2)
    axs[1].set_yticks([1.0], ["On"])

    fig.tight_layout()
    plt.show()

    return fig

def save_stim_plot(fig, video_name):
    signal_name = video_name.removesuffix("_signal.avi") + '_stim-analysis.jpg'
    fig.savefig(signal_name, format='jpg')
    plt.close()
    
def close_current_plot():
    
    plt.close()
    
def get_cmap(n, name='hsv'):
    
    return plt.cm.get_cmap(name, n)

def histogram(vector, color, bins):
    
    dvector = np.diff(vector)
    dvector = dvector[np.isfinite(dvector)]
    plt.hist(dvector,color=color,histtype='step',bins=bins)

def trip_grid(tracking, grid_bins=20, show=False, save=False):
    
    hist, image = traja_alt.trip_grid(tracking, bins=grid_bins);
    plt.gca().invert_yaxis()
    fig = plt.gcf()
    
    if show: plt.show()
    
    return fig

def time_plot_on_image(ax, tracking, fps, pcutoff, image_file=None, schedule=None, length=None, colormap='jet', offset=(0, 0), show=False, close=True):
    ''' Plots poses vs time; pose x vs pose y; histogram of differences and likelihoods.'''

    if image_file is not None:
        im = plt.imread(image_file)
        im = ax.imshow(im, cmap='gray')
        
    total_time = sum(schedule) if schedule is not None else length
    
    colors = get_cmap(total_time*fps, name = colormap)

    Index = tracking['likelihood'].values > pcutoff
    x = tracking['x'].values[Index]
    y = tracking['y'].values[Index]
    colors = [i/fps for i, x in enumerate(Index) if x]
    
    plt.scatter(x, y, c=colors, cmap='viridis', s=5, vmin=0, vmax=total_time)   # 'viridis' is one of the available colormaps
    colorbar = plt.colorbar()
    colorbar.set_label('Time (s)')  # Set the label for the colorbar
    
    
    if schedule is not None:
        
        stim_frame = schedule[0]*fps
    
        if tracking['likelihood'].values[stim_frame] > pcutoff:
       
            plt.scatter(tracking['x'].values[stim_frame],tracking['y'].values[stim_frame],c='red', s=50, marker='x', label='stim on')

    if show: plt.show()
    
    if close: plt.close()
    
def two_plots(fig, ax1, x, data1, data2, x_label, data1_label, data2_label, speed_lim=800, stim_lim=1.2, show=False, close=True):
    # Create some mock data
    
    color = 'tab:red'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(data1_label, color=color)
    ax1.set_ylim(bottom=0, top=speed_lim)
    ax1.plot(x, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # if data2 is not None:
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(data2_label, color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(bottom=0, top=stim_lim)
    ax2.plot(x, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    if show: plt.show()
    
    if close: plt.close()