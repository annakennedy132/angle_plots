import matplotlib.pyplot as plt
import numpy as np

def two_plots(fig, ax1, x, data1, data2, data1_colour, x_label, data1_label, data2_label, data1_lim, stim_lim=1.2, show=False, close=True):
    
    color = data1_colour
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(data1_label, color=color)
    ax1.set_ylim(data1_lim)
    ax1.plot(x, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # if data2 is not None:
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel(data2_label, color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(bottom=0, top=stim_lim)
    ax2.plot(x, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    if show: plt.show()
    if close: plt.close()

def plot_polar_chart(fig, ax, angles, bins, direction=1, zero="E", show=False, close=True):

    angles_float = [float(angle) for angle in angles if angle is not None]
    angles_radians = np.deg2rad(angles_float)
        
    hist, bins = np.histogram(angles_radians, bins=bins)
    if np.max(hist) != 0:
        hist_norm = hist / np.max(hist)
    else:
        print("Error: No angle values provided for plotting polar chart.")
        return fig, ax

    bars = ax.bar(bins[:-1], hist_norm, width=((2*np.pi)/len(bins)),
                    edgecolor="navy", alpha=0.5)

    for bar, height in zip(bars, hist_norm):
        bar.set_facecolor(plt.cm.viridis(height))

    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))  # 8 ticks evenly spaced, excluding the endpoint
    labels = ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°']
    ax.set_xticklabels(labels)
        
    ax.set_theta_direction(direction)
    ax.set_theta_zero_location(zero)
    fig.tight_layout()

    if show: plt.show()
    if close: plt.close()

    return fig, ax

def plot_coords(fig, ax, coords, xlabel, ylabel, gridsize, vmin, vmax, xmin, xmax, ymin, ymax, show=False, close=True):
    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]

    mean_x = np.nanmean(x_values)
    mean_y = np.nanmean(y_values)
    mean_coord = (mean_x, mean_y)
    
    # Calculate the extent of the plot area
    extent = [xmin, xmax, ymin, ymax]
    
    hb = ax.hexbin(x_values, y_values, gridsize=gridsize, cmap='inferno', vmin=vmin, vmax=vmax, extent=extent, mincnt=0)
    fig.colorbar(hb, ax=ax, label='Frequency')
    ax.set_ylim(ymin+25, ymax-25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Heatmap of Coordinates')
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(mean_coord[0], mean_coord[1], color='red', marker="x", label='Mean Coordinate')
    ax.legend()

    if show: plt.show()
    if close: plt.close()

    return fig