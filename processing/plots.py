import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import parse

def two_plots(fig, ax1, x, data1, data2, data1_colour, x_label, data1_label, data2_label, data1_lim, title=None, stim_lim=1.2, show=False, close=True):
    
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

    if title:
        ax1.set_title(title)
    
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

def plot_coords(fig, ax, coords, xlabel, ylabel, gridsize, vmin, vmax, xmin, xmax, ymin, ymax, show_coord=None, show=False, close=True):
    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]

    extent = [xmin, xmax, ymin, ymax]
    
    hb = ax.hexbin(x_values, y_values, gridsize=gridsize, cmap='inferno', vmin=vmin, vmax=vmax, extent=extent, mincnt=0)
    fig.colorbar(hb, ax=ax, label='Frequency')
    ax.set_ylim(ymin+25, ymax-25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Heatmap of Coordinates')
    ax.set_aspect('equal', adjustable='box')
    if show_coord:
        ax.scatter(show_coord[0], show_coord[1], color='red', marker="x")

    if show: plt.show()
    if close: plt.close()

    return fig

def plot_str_polar_chart(fig, ax, angles, bins, direction=1, zero="E", show=False, close=True):
    angles = [float(angle) for angle in angles if isinstance(angle, str) and angle.strip() != '' and angle.strip().lower() != 'nan']
    plot_polar_chart(fig, ax, angles, bins, direction, zero, show=show, close=close)

def plot_str_coords(fig, ax, coords, xlabel=None, ylabel=None, gridsize=None, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, colorbar=False, show=True, close=False):
    
    # Parse string coordinates to tuples of floats
    coords = [parse.parse_coord(coord) for coord in coords]
    coords = [coord for coord in coords if coord is not np.nan]

    # Extract x and y values
    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]

    # Plot coordinates
    hb = ax.hexbin(x_values, y_values, gridsize=gridsize, cmap='inferno', vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax], mincnt=0)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if xmin and xmax: ax.set_xlim(xmin, xmax)
    if ymin and ymax: ax.set_ylim(ymin, ymax)

    if colorbar:
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Frequency')

    if show: plt.show()
    if close: plt.close(fig)

    return fig

def time_plot(fig, ax, coordinates, fps=30, xlim=None, ylim=None, show=False, close=True):
    total_time = len(coordinates[0]) / fps
    
    colors = np.linspace(0, total_time, len(coordinates[0]))  # Color based on time
    
    for coords in coordinates:
        ax.scatter([coord[0] for coord in coords], [coord[1] for coord in coords], c=colors, cmap='viridis', s=0.25, vmin=0, vmax=total_time)

    colorbar = plt.colorbar(ax.collections[0], ax=ax)
    colorbar.set_label('Time (s)')

    ax.set_xlabel("x coordinates")
    ax.set_ylabel("y coordinates") 

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show:
        plt.show()
    
    if close:
        plt.close()

    return fig

def plot_scatter_trendline(fig, ax, data1_x, data1_y, x_label, y_label, title, color='blue', marker_size=20, show=False, close=True):
    sns.regplot(x=data1_x, y=data1_y, ax=ax, scatter=True,
                scatter_kws={'color': color, 'alpha': 0.7, 's': marker_size}, 
                line_kws={'color': color}, ci=None)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.grid(True, alpha=0.3)

    if show: 
        plt.show()
    if close: 
        plt.close()

    return fig

def plot_two_scatter_trendline(fig, ax, data1_x, data1_y, data2_x, data2_y, x_label, y_label, title, group1_label, group2_label, 
                           group1_color='blue', group2_color='green', marker_size=20, show=False, close=True):
    sns.regplot(x=data1_x, y=data1_y, ax=ax, scatter=True, label=group1_label,
                scatter_kws={'color': group1_color, 'alpha': 0.7, 's': marker_size},line_kws={'color': group1_color}, ci=None)
    
    sns.regplot(x=data2_x, y=data2_y, ax=ax, scatter=True, label=group2_label,
                scatter_kws={'color': group2_color, 'alpha': 0.7, 's': marker_size},line_kws={'color': group2_color}, ci=None)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show: 
        plt.show()
    if close: 
        plt.close()

    return fig

def plot_bar_two_groups(fig, ax, data1, data2, x_label, y_label, title, bar1_label, bar2_label,
                        color1='blue', color2='green', ylim=None, bar_width=0.5, show=False, close=True):
    x = np.array([0, 1])
    
    mean1 = np.nanmean(data1)
    mean2 = np.nanmean(data2)

    ax.bar(x[0], mean1, label=bar1_label, color=color1, alpha=0.6, width=bar_width, zorder=1)
    ax.bar(x[1], mean2, label=bar2_label, color=color2, alpha=0.6, width=bar_width, zorder=1)

    # Plot bar outlines with opaque edges
    ax.bar(x[0], mean1, color='none', edgecolor=color1, linewidth=2, alpha=1, width=bar_width, zorder=2)
    ax.bar(x[1], mean2, color='none', edgecolor=color2, linewidth=2, alpha=1, width=bar_width, zorder=2)
    
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([bar1_label, bar2_label])
    ax.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    if show: plt.show()
    if close: plt.close()

    return fig

def plot_binned_bar_chart(fig, ax, data_x, data_y, bin_edges, xlabel, ylabel, title, color, y_limit=None, show=False, close=True):

    # Create a DataFrame with data_x and data_y
    df = pd.DataFrame({'x': data_x, 'y': data_y})

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers

    # Group y_values into bins and calculate average
    grouped_data = []
    for i in range(len(bin_edges) - 1):
        mask = (df['x'] >= bin_edges[i]) & (df['x'] < bin_edges[i+1])
        grouped_data.append(df.loc[mask, 'y'].mean())

    ax.bar(bin_centers, grouped_data, width=np.diff(bin_edges), color=color, alpha=0.6, edgecolor=color, linewidth=2, zorder=1)
    ax.bar(bin_centers, grouped_data, width=np.diff(bin_edges), color='none', edgecolor=color, alpha=0.8, linewidth=2, zorder=2) 

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if y_limit is not None:
        ax.set_ylim(y_limit)

    if show:
        plt.show()
    if close:
        plt.close()

    return fig

def plot_grouped_bar_chart(fig, ax, data1, data2, data3, data4, labels, xlabel, ylabel, title, colors, bar_width=0.35, show=False, close=True):

    x = np.arange(len(labels))
    bar_positions = np.arange(len(labels))
    
    datasets = [data1, data2, data3, data4]

    for bar_position, data, color, label in zip(bar_positions, datasets, colors, labels):
        ax.bar(bar_position, np.nanmean(data), width=bar_width, color=color, alpha=0.6, edgecolor=color, linewidth=2, label=label, zorder=1)
        ax.bar(bar_position, np.nanmean(data), width=bar_width, color='none', alpha=0.8, edgecolor=color, linewidth=2, label=label, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax

def plot_two_lines(fig, ax, data1, data2, label1, label2, color1, color2, xlabel='X', ylabel='Y', title='Line Plot', ylim=None, show=False, close=True):
    x = np.arange(len(data1))
    ax.plot(x, data1, color=color1, label=label1)
    ax.plot(x, data2, color=color2, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(ylim)

    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax

def plot_one_line(fig, ax, data1, label1, color1, xlabel='X', ylabel='Y', title='Line Plot', xlim=None, ylim=None, show=False, close=True):
    x = np.arange(len(data1))
    ax.plot(x, data1, color=color1, label=label1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax