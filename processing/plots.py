import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import parse

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

    angles = [float(angle) for angle in angles if isinstance(angle, str) and angle.strip() != '' and angle.strip().lower() != 'nan']

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

def plot_coords(fig, ax, coords, xlabel=None, ylabel=None, gridsize=None, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, show=True, close=False):

    coords = [coord for coord in coords if not (isinstance(coord, (list, tuple)) and
                                                not (np.isnan(coord[0]) and np.isnan(coord[1])))]
    
    # Extract x and y values
    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]

    # Plot coordinates
    ax.hexbin(x_values, y_values, gridsize=gridsize, cmap='inferno', vmin=vmin, vmax=vmax)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if xmin and xmax: ax.set_xlim(xmin, xmax)
    if ymin and ymax: ax.set_ylim(ymin, ymax)

    if show: plt.show()
    if close: plt.close(fig)

    return fig

def plot_string_coords(fig, ax, coords, xlabel=None, ylabel=None, gridsize=None, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, show=True, close=False):
    
    # Parse string coordinates to tuples of floats
    coords = [parse.parse_coord(coord) for coord in coords]
    coords = [coord for coord in coords if coord is not np.nan]

    # Extract x and y values
    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]

    # Plot coordinates
    ax.hexbin(x_values, y_values, gridsize=gridsize, cmap='inferno', vmin=vmin, vmax=vmax)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if xmin and xmax: ax.set_xlim(xmin, xmax)
    if ymin and ymax: ax.set_ylim(ymin, ymax)

    if show: plt.show()
    if close: plt.close(fig)

    return fig
    
def plot_one_scatter_trendline(fig, ax, data, x_label, y_label, title, label, color='blue', marker_size=20, show=False, close=True):
    sns.regplot(x=range(len(data)), y=data, ax=ax, scatter=True, label=label,
                scatter_kws={'color': color, 'alpha': 0.7, 's': marker_size}, ci=None)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show: plt.show()
    if close: plt.close()

    return fig

def plot_two_scatter_trendline(fig, ax, data1, data2, x_label, y_label, title, group1_label, group2_label, 
                           group1_color='blue', group2_color='orange', marker_size=20, show=False, close=True):
    sns.regplot(x=range(len(data1)), y=data1, ax=ax, scatter=True, label=group1_label,
                scatter_kws={'color': group1_color, 'alpha': 0.7, 's': marker_size}, ci=None)
    
    sns.regplot(x=range(len(data2)), y=data2, ax=ax, scatter=True, label=group2_label,
                scatter_kws={'color': group2_color, 'alpha': 0.7, 's': marker_size}, ci=None)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show: plt.show()
    if close: plt.close()

    return fig

def plot_three_scatter_trendline(fig, ax, data1, data2, data3, x_label, y_label, title, 
                           group1_label, group2_label, group3_label,
                           group1_color='blue', group2_color='orange', group3_color='green', marker_size=20, show=False, close=True):
    sns.regplot(x=range(len(data1)), y=data1, ax=ax, scatter=True, label=group1_label,
                scatter_kws={'color': group1_color, 'alpha': 0.7, 's': marker_size}, ci=None)
    
    sns.regplot(x=range(len(data2)), y=data2, ax=ax, scatter=True, label=group2_label,
                scatter_kws={'color': group2_color, 'alpha': 0.7, 's': marker_size}, ci=None)
    
    sns.regplot(x=range(len(data3)), y=data3, ax=ax, scatter=True, label=group3_label,
                scatter_kws={'color': group3_color, 'alpha': 0.7, 's': marker_size}, ci=None)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show: plt.show()
    if close: plt.close()

    return fig

def plot_bar_two_groups(fig, ax, data1, data2, labels, x_label, y_label, title, group1_label, group2_label, color1='blue', color2='orange', show=False, close=True):
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, data1, width, label=group1_label, color=color1)
    ax.bar(x + width/2, data2, width, label=group2_label, color=color2)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    if show: plt.show()
    if close: plt.close()

    return fig

def plot_bar_two_groups(fig, ax, data1, data2, data3, labels, x_label, y_label, title, group1_label, group2_label, group3_label, color1='blue', color2='orange', color3="red", show=False, close=True):
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, data1, width, label=group1_label, color=color1)
    ax.bar(x + width/2, data2, width, label=group2_label, color=color2)
    ax.bar(x + width/2, data3, width, label=group3_label, color=color3)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    if show: plt.show()
    if close: plt.close()

    return fig