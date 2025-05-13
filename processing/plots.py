import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import parse

def two_plots(fig, ax1, x, data1, data2, data1_colour, x_label, data1_label, data2_label, data1_lim, title=None, stim_lim=1.2, show=False, close=True, show_axes="both"):
    
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

    if show_axes == 'both':
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
    elif show_axes == 'none':
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
    
    if show: plt.show()
    if close: plt.close()

def plot_polar_chart(fig, ax, angles, bins, direction=-1, zero="E", show=False, close=True):
    valid_angles = []
    for angle in angles:
        try:
            if angle is not None and np.isfinite(float(angle)):
                valid_angles.append(float(angle))
        except (ValueError, TypeError):
            continue

    if not valid_angles:
        print("Error: No valid angle values provided for plotting polar chart.")
        return fig, ax

    angles_radians = np.deg2rad(valid_angles)
        
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

    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    labels = ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°']
    ax.set_xticklabels(labels)
        
    ax.set_theta_direction(direction)
    ax.set_theta_zero_location(zero)
    fig.tight_layout()

    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax

def plot_coords(fig, ax, coords, xlabel=None, ylabel=None, gridsize=None, vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None, show=True, close=False, colorbar=True):
    
    fig.set_constrained_layout(True)

    coords = [coord for coord in coords if isinstance(coord, tuple) and len(coord) == 2 and not np.isnan(coord[0]) and not np.isnan(coord[1])]
    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]

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

def time_plot(fig, ax, coordinates, fps=30, xlim=None, ylim=(700, 50), show=False, close=True, colorbar=True):

    total_time = len(coordinates[0]) / fps
    colors = np.linspace(0, total_time, len(coordinates[0]))

    for coords in coordinates:
        # Extract x and y coordinates
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        # Filter out NaN values
        valid_indices = [i for i, (x, y) in enumerate(zip(x_coords, y_coords)) if not (np.isnan(x) or np.isnan(y))]
        # Use the valid indices to filter x and y coordinates
        filtered_x_coords = [x_coords[i] for i in valid_indices]
        filtered_y_coords = [y_coords[i] for i in valid_indices]
        filtered_colors = [colors[i] for i in valid_indices]

        ax.scatter(filtered_x_coords, filtered_y_coords, c=filtered_colors, cmap='viridis', s=0.25, vmin=0, vmax=total_time)

    if colorbar:
        colorbar = plt.colorbar(ax.collections[0], ax=ax)
        colorbar.set_label('Time (s)')

    ax.set_xlabel("x coordinates")
    ax.set_ylabel("y coordinates")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if show:
        plt.show()
    
    if close:
        plt.close()

    return fig

def cmap_plot(fig, axes, data1, data2, sort_data1, sort_data2, title1, title2, ylabel, ylim, cmap="viridis", fps=30, vmin=100, vmax=600, cbar_dim=[0.93, 0.11, 0.015, 0.53]):
    
    wt_avg_data = [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in data1]) for i in range(max(map(len, data1)))]
    blind_avg_data = [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in data2]) for i in range(max(map(len, data2)))]
        
    num_frames = len(wt_avg_data)  #Get number of frames from speed data
    frame_time = (1. / fps)
    x_ticks = np.linspace(0, num_frames, 4).astype(int)  # Adjust number of ticks as needed
    x_labels = (x_ticks * frame_time) - 5
    
    # Plot average speeds above the heatmaps
    axes[0, 0].plot(wt_avg_data, color='red')
    axes[0, 0].set_title(title1)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].set_ylim(ylim)
    axes[0, 0].spines['left'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['bottom'].set_visible(False)
    axes[0, 0].get_xaxis().set_visible(False)

    axes[0, 1].plot(blind_avg_data, color='red')
    axes[0, 1].set_title(title2)
    axes[0, 1].set_ylim(ylim)
    axes[0, 1].spines['left'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['bottom'].set_visible(False)
    axes[0, 1].get_xaxis().set_visible(False)

    #sort and separate speed by age
    wt_sorted_data = [data for _, data in sorted(zip(sort_data1, data1))]
    blind_sorted_data = [data for _, data in sorted(zip(sort_data2, data2))]

    # Plot the heatmaps below the average speeds
    sns.heatmap(wt_sorted_data, ax=axes[1, 0], cmap=cmap, cbar=False, vmin=vmin, vmax=vmax)
    axes[1, 0].set_ylabel('Trial')
    axes[1, 0].axvline(150, color='black', linewidth=2)  # Stimulus start
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xticks(x_ticks)
    axes[1, 0].set_xticklabels(x_labels)

    sns.heatmap(blind_sorted_data, ax=axes[1, 1], cmap=cmap, cbar=False, vmin=vmin, vmax=vmax)
    axes[1, 1].set_ylabel('Trial')
    axes[1, 1].axvline(150, color='black', linewidth=2)
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xticks(x_ticks)
    axes[1, 1].set_xticklabels(x_labels)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes(cbar_dim)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Speed (pps)', rotation=270, labelpad=10)
    cbar.outline.set_visible(False) 
    
    return fig, axes 

def scatter_plot_with_stats(fig, ax,coords, point_color='darkgrey', mean_marker='o', x_limits=None, y_limits=None, show=False, close=True):

    coords = [parse.parse_coord(coord) for coord in coords]# Parse string coordinates to tuples of floats
    coords = [coord for coord in coords if coord is not np.nan]
    x_coords = [coord[0] for coord in coords] # Extract x and y coordinates
    y_coords = [coord[1] for coord in coords]
    mean_x = np.nanmean(x_coords) # Calculate mean and standard deviation
    mean_y = np.nanmean(y_coords)
    std_x = np.nanstd(x_coords)
    std_y = np.nanstd(y_coords)
    
    ax.scatter(x_coords, y_coords, color=point_color, s=10)
    ax.plot([mean_x - std_x, mean_x + std_x], [mean_y, mean_y], color='black')
    ax.plot([mean_x, mean_x], [mean_y - std_y, mean_y + std_y], color='black')
    ax.scatter([mean_x - std_x, mean_x + std_x], [mean_y, mean_y], color='black', marker='|', s=100)
    ax.scatter([mean_x, mean_x], [mean_y - std_y, mean_y + std_y], color='black', marker='_', s=100)
    ax.scatter(mean_x, mean_y, color='black', s=50, marker=mean_marker, zorder=3)

    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)
    
    if show:
        plt.show()
    if close:
        plt.close()
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return fig, ax

def plot_bar_two_groups(fig, ax, data1, data2, x_label, y_label, bar1_label, bar2_label,
                        color1, color2, ylim=None, bar_width=0.2, points=True, 
                        log_y=False, error_bars=False, show=False, close=True, title=None):
    
    mean1 = np.nanmean(data1)
    mean2 = np.nanmean(data2)
    std1 = np.nanstd(data1)
    std2 = np.nanstd(data2)
    
    x = [0.2, 0.5]
    ax.margins(x=0.1)
    ax.bar(x[0], mean1, bar_width, color=color1,yerr=std1 if error_bars else None, error_kw=dict(ecolor=color1, capsize=5))
    ax.bar(x[1], mean2, bar_width, color=color2, yerr=std2 if error_bars else None, error_kw=dict(ecolor=color2, capsize=5))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks([x[0],x[1]])
    ax.set_xticklabels([bar1_label, bar2_label])

    if ylim is not None:
        ax.set_ylim(ylim)
    if log_y:
        ax.set_yscale('log')
    if points:
        ax.scatter(np.full_like(data1, x[0]), data1, color=color1, s=10)
        ax.scatter(np.full_like(data2, x[1]), data2, color=color2, s=10)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if show:
        plt.show()
    if close:
        plt.close()

    return fig

def plot_bar_four_groups(fig, ax, data1, data2, data3, data4, xticks, labels, colors, bar_width=0.2, error_bars=False, points=False, log_y=False, ylim=(0,None), show=False, close=True):
    
    mean1 = np.nanmean(data1)
    mean2 = np.nanmean(data2)
    mean3 = np.nanmean(data3)
    mean4 = np.nanmean(data4)
    std1 = np.nanstd(data1)
    std2 = np.nanstd(data2)
    std3 = np.nanstd(data3)
    std4 = np.nanstd(data4)
    
    x = [0.2, 0.5, 0.8, 1.1]
    ax.margins(x=0.1)
    ax.bar(x[0], mean1, bar_width, color=colors[0],yerr=std1 if error_bars else None, error_kw=dict(ecolor=colors[0], capsize=5))
    ax.bar(x[1], mean2, bar_width, color=colors[1], yerr=std2 if error_bars else None, error_kw=dict(ecolor=colors[1], capsize=5))
    ax.bar(x[2], mean3, bar_width, color=colors[2],yerr=std3 if error_bars else None, error_kw=dict(ecolor=colors[2], capsize=5))
    ax.bar(x[3], mean4, bar_width, color=colors[3], yerr=std4 if error_bars else None, error_kw=dict(ecolor=colors[3], capsize=5))
        
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.legend(labels, loc='best')
    if ylim is not None:
        ax.set_ylim(ylim)
    if log_y:
        ax.set_yscale("log")
    if points:
        ax.scatter(np.full_like(data1, x[0]), data1, color=colors[0], s=10)
        ax.scatter(np.full_like(data2, x[1]), data2, color=colors[1], s=10)
        ax.scatter(np.full_like(data3, x[2]), data3, color=colors[2], s=10)
        ax.scatter(np.full_like(data4, x[3]), data4, color=colors[3], s=10)
                
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax
    
def plot_grouped_bar_chart(fig, ax, data1, data2, data3, data4, xticks, labels, colors, bar_width=0.2, error_bars=False, points=False, log_y=False, ylim=(0,None), show=False, close=True):
    x = [0.2, 0.5]
    mean1 = np.nanmean(data1)
    mean2 = np.nanmean(data2)
    mean3 = np.nanmean(data3)
    mean4 = np.nanmean(data4)
     
    std1 = np.nanstd(data1)
    std2 = np.nanstd(data2)
    std3 = np.nanstd(data3)
    std4 = np.nanstd(data4)
    
    ax.margins(x=0.1)
    ax.bar((x[0] - bar_width/2), mean1, bar_width, color=colors[0],yerr=std1 if error_bars else None, error_kw=dict(ecolor=colors[0], capsize=5))
    ax.bar((x[0]+ bar_width/2), mean2, bar_width, color=colors[1], yerr=std2 if error_bars else None, error_kw=dict(ecolor=colors[1], capsize=5))
    ax.bar((x[1] - bar_width/2), mean3, bar_width, color=colors[2], yerr=std3 if error_bars else None, error_kw=dict(ecolor=colors[2], capsize=5))
    ax.bar((x[1] + bar_width/2), mean4, bar_width, color=colors[3], yerr=std4 if error_bars else None, error_kw=dict(ecolor=colors[3], capsize=5))
    
    if points:
        ax.scatter(np.full_like(data1, x[0] - bar_width/2), data1, color=colors[0], s=10)
        ax.scatter(np.full_like(data2, x[0] + bar_width/2), data2, color=colors[0], s=10)
        ax.scatter(np.full_like(data3, x[1] - bar_width/2), data3, color=colors[1], s=10)
        ax.scatter(np.full_like(data4, x[1] + bar_width/2), data4, color=colors[1], s=10)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.legend(labels, loc='best')
    if log_y:
        ax.set_yscale("log")
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax