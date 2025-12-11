import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

from escape_plotter.utils import parse

from scipy.stats import ttest_ind, linregress, t, skew, median_test, norm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib import transforms
from itertools import combinations
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def plot_coords(fig, ax, coords,
                         xlabel=None, ylabel=None,
                         gridsize=100, smooth=None,
                         vmin=None, vmax=None,
                         xmin=None, xmax=None,
                         ymin=None, ymax=None,
                         show=False, close=False,
                         colorbar=True):

    fig.set_constrained_layout(True)
    
    coords = [
    c for c in coords
    if isinstance(c, tuple) and len(c) == 2
    and not np.isnan(c[0]) and not np.isnan(c[1])
]
    vmax = len(coords) / 3000.0

    coords = [
        c for c in coords
        if isinstance(c, tuple) and len(c) == 2
        and not np.isnan(c[0]) and not np.isnan(c[1])
    ]
    if not coords:
        return fig

    x_values = np.array([c[0] for c in coords])
    y_values = np.array([c[1] for c in coords])

    # Fix: Ensure histogram2d doesn't fail when min==max
    eps = 1e-9
    xlow, xhigh = xmin, xmax
    ylow, yhigh = ymin, ymax
    if xlow == xhigh:
        xhigh = xlow + eps
    if ylow == yhigh:
        yhigh = ylow + eps

    if smooth is None or smooth <= 0:
        # -------------------
        # Standard HEXBIN
        # -------------------
        hb = ax.hexbin(
            x_values, y_values,
            gridsize=gridsize, cmap='inferno',
            vmin=vmin, vmax=vmax,
            extent=[xmin, xmax, ymin, ymax],
            mincnt=0
        )

        if xmin is not None and xmax is not None: ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None: ax.set_ylim(ymin, ymax)

        if colorbar:
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Frequency')
            cb.outline.set_visible(False)

    else:
        # -------------------
        # Smoothed Heatmap
        # -------------------
        ylow, yhigh = sorted([ymin, ymax])
        xlow, xhigh = sorted([xmin, xmax])

        
        counts, yedges, xedges = np.histogram2d(
            y_values, x_values,
            bins=gridsize,
            range=[[ylow, yhigh], [xlow, xhigh]]
        )
        counts_smooth = gaussian_filter(counts, sigma=smooth)

        img = ax.imshow(
            counts_smooth,
            origin='upper',
            extent=[xmin, xmax, ymin, ymax],
            cmap='inferno',
            vmin=vmin,
            vmax=vmax,
            aspect='equal'
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if colorbar:
            cb = fig.colorbar(img, ax=ax)
            cb.set_label("Frequency")
            cb.outline.set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig

def time_plot(fig, ax, coordinates, cbar_dim=None, fps=30, xlim=None, ylim=(700, 50), show=False, close=True, colorbar=True, cmap="viridis"):
    total_time = len(coordinates[0]) / fps
    colors = np.linspace(0, total_time, len(coordinates[0]))

    for coords in coordinates:
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        valid_indices = [i for i, (x, y) in enumerate(zip(x_coords, y_coords)) if not (np.isnan(x) or np.isnan(y))]
        filtered_x_coords = [x_coords[i] for i in valid_indices]
        filtered_y_coords = [y_coords[i] for i in valid_indices]
        filtered_colors = [colors[i] for i in valid_indices]

        sc = ax.scatter(filtered_x_coords, filtered_y_coords, c=filtered_colors, cmap=cmap, s=0.25, vmin=0, vmax=total_time)

    if colorbar and cbar_dim is not None:
        cbar_ax = fig.add_axes(cbar_dim)
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Time (s)')
        cbar.outline.set_visible(False)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if show:
        plt.show()
    if close:
        plt.close()

    return fig

def regression_plot(fig, ax, x, y, label, color, x_label, y_label, title,
                    text_index=0, show=False, close=True,
                    stats=True, scatter=False, legend_loc='best', ci=0.95):
    """
    Plot regression line and optionally scatter points.
    Prints slope stats and, if multiple groups are plotted on the same axes,
    also prints pairwise slope difference tests automatically.
    """
    if not hasattr(ax, "_regression_results"):
        ax._regression_results = []
        ax._regression_header_printed = False

    # --- drop NaNs ---
    pairs = [(a, b) for a, b in zip(x, y) if not (np.isnan(a) or np.isnan(b))]
    if len(pairs) < 2:
        return
    x, y = map(np.array, zip(*pairs))
    n = len(x)

    # --- regression ---
    res = linregress(x, y)
    slope, intercept, r_value, p_value, std_err = res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr
    r2 = r_value ** 2

    # --- slope CI ---
    ci_low = ci_high = None
    if n > 2 and np.isfinite(std_err):
        alpha = 1 - ci
        tcrit = t.ppf(1 - alpha/2, df=n - 2)
        ci_low, ci_high = slope - tcrit * std_err, slope + tcrit * std_err
    else:
        tcrit = None

    # --- plot ---
    sns.regplot(
        x=x,
        y=y,
        ax=ax,
        scatter=False,                 # <- no scatter here
        line_kws={'color': color},
        label=label                    # <- legend will show a line
    )

    # --- optional scatter on top, but no legend label ---
    if scatter:
        ax.scatter(
            x, y,
            color=color,
            alpha=0.7,
            s=10
        )

    print(f"{label}: slope={slope:.2f} r2={r2:.2f}, p={p_value:.3g}")

    # --- format axes ---
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc=legend_loc, fontsize='x-small', frameon=False)

    # --- pretty printing (fallback to "y vs x" if no title) ---
    if stats:
        # choose header label
        if isinstance(title, str) and title.strip():
            header_title = title.strip()
        else:
            header_title = f"{y_label} vs {x_label}"

        # print header once per axis
        if not ax._regression_header_printed:
            print(f"\n=== Regression stats: {header_title} ===")
            print("Label          n    Slope      95% CI slope          Intercept    r^2      p")
            print("-" * 80)
            ax._regression_header_printed = True

        # CI as string
        if ci_low is not None:
            ci_str = f"({ci_low:7.3f}, {ci_high:7.3f})"
        else:
            ci_str = " " * 21

        # row
        print(f"{label:12s}  {n:3d}  {slope:7.3f}  {ci_str:21s}  {intercept:9.3f}  {r2:5.3f}  {p_value:7.3g}")

    # --- store for later pairwise comparison ---
    ax._regression_results.append({
        'label': label, 'slope': slope, 'se': std_err,
        'ci_low': ci_low, 'ci_high': ci_high, 'tcrit': tcrit
    })

    # --- when figure is closed, print slope comparisons (neater) ---
    if close:
        results = ax._regression_results
        if len(results) > 1:
            if isinstance(title, str) and title.strip():
                comparison_title = title.strip()
            else:
                comparison_title = f"{y_label} vs {x_label}"

            print(f"\n--- Pairwise slope comparisons: {comparison_title} ---")
            print("Group A        Group B        ΔSlope     z        p")
            print("-" * 60)

            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    a, b = results[i], results[j]

                    # fallback SE from CI if needed
                    if a['se'] is not None and np.isfinite(a['se']):
                        se1 = a['se']
                    elif (a['ci_high'] is not None and a['ci_low'] is not None
                          and a['tcrit'] not in (None, 0)):
                        se1 = (a['ci_high'] - a['ci_low']) / (2 * a['tcrit'])
                    else:
                        se1 = np.nan

                    if b['se'] is not None and np.isfinite(b['se']):
                        se2 = b['se']
                    elif (b['ci_high'] is not None and b['ci_low'] is not None
                          and b['tcrit'] not in (None, 0)):
                        se2 = (b['ci_high'] - b['ci_low']) / (2 * b['tcrit'])
                    else:
                        se2 = np.nan

                    if not (np.isfinite(se1) and np.isfinite(se2)):
                        continue

                    z = (a['slope'] - b['slope']) / np.sqrt(se1**2 + se2**2)
                    p = 2 * (1 - norm.cdf(abs(z)))
                    print(f"{a['label']:12s}  {b['label']:12s}  {a['slope']-b['slope']:7.3f}  {z:6.3f}  {p:7.3g}")
            print("\n")

        plt.close(fig)

    if show:
        plt.show()

    return fig, ax

def cmap_plot_with_average(
    fig, axes, data1, data2=None,
    sort_data1=None, sort_data2=None,
    title1="", title2="", ylabel="", ylim=None, 
    cbar_label="", length=0, cmap="viridis", fps=30,
    vmin=100, vmax=600, cbar_dim=(0.93, 0.11, 0.015, 0.53),
    smooth=False, norm=False,
    add_cbar=True
):
    
        max_len = max(map(len, data1)) if data1 else 0
        avg_data = [
            np.nanmean([lst[i] if i < len(lst) else np.nan for lst in data1])
            for i in range(max_len)
        ]

        axes[0].plot(avg_data, color="red")
        axes[0].set_title(title1)
        axes[0].set_ylabel(ylabel)
        axes[0].set_ylim(ylim)
        for s in ("left", "right", "top", "bottom"):
            axes[0].spines[s].set_visible(False)
        axes[0].get_xaxis().set_visible(False)

        data1_sorted = [d for _, d in sorted(zip(sort_data1, data1))]
        num_frames = max(map(len, data1_sorted)) if data1_sorted else 0

        frame_time = 1.0 / max(1, length)
        x_ticks = np.linspace(0, num_frames, 3, dtype=int)
        x_labels = x_ticks * frame_time

        # --- IMPORTANT: keep handle to the heatmap QuadMesh ---
        hm = sns.heatmap(
            data1_sorted,
            ax=axes[1],
            cmap=cmap,
            cbar=False,      # <- no auto-cbar here
            vmin=vmin,
            vmax=vmax,
        )
        for spine in axes[1].spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.5)

        axes[1].set_ylabel("Trial")
        axes[1].set_xticks(x_ticks)
        axes[1].set_xticklabels(x_labels)
        axes[1].set_yticks([])
        axes[1].set_xlabel("Normalised Time")
        
        if add_cbar:
            norm_ = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_)
            cbar_ax = inset_axes(axes[1], width="7%", height="100%", loc="right", 
                                 bbox_to_anchor=(0.2, 0., 1, 1), bbox_transform=axes[1].transAxes, borderpad=0)
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(cbar_label, rotation=270, labelpad=10)
            cbar.outline.set_visible(False)

        return fig, axes
    
def cmap_plot(fig, ax, data, sort_data=None, title="", ylabel="", xlabel="",
              cbar_label="", cmap="viridis", fps=30,
              vmin=100, vmax=600, norm=False, show=False, close=True):

    # sort trials
    sort_data = sort_data or list(range(len(data)))
    sorted_data_1 = [d for _, d in sorted(zip(sort_data, data))]

    # x-axis in seconds
    num_frames = len(data[0])          # assuming data is (trials, frames)
    frame_time = 1.0 / max(1, fps)
    x_ticks = np.linspace(0, num_frames, 5, dtype=int)
    x_labels = (x_ticks * frame_time) - 5

    # heatmap
    im = sns.heatmap(sorted_data_1, ax=ax, cmap=cmap, cbar=False,
                     vmin=vmin, vmax=vmax)

    ax.set_ylabel(ylabel)
    if not norm:
        ax.axvline(150, color="white", linewidth=2)

    ax.set_yticks([])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=10)

    # individual colorbar for this axes
    cbar = fig.colorbar(im.collections[0], ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=10)
    cbar.outline.set_visible(False)
    
    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax

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
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def plot_bar(fig, ax, data_groups, x_label, y_label, bar_labels,
             colors, ylim=None, bar_width=0.1, points=False, 
             log_y=False, error_bars=False, show=False, close=True, 
             title=None, stats=True, comparisons=None, equal_var=False):

    num_groups = len(data_groups)
    means = [np.nanmean(group) for group in data_groups]
    stds  = [np.nanstd(group)  for group in data_groups]

    # X positions and margins
    x_positions = np.linspace(0.2, 0.2 + (num_groups - 1) * (bar_width + 0.05), num_groups)
    ax.margins(x=0.2)

    # Bars (+ optional scatter points)
    for i in range(num_groups):
        ax.bar(
            x_positions[i], means[i], bar_width,
            color=colors[i],
            yerr=stds[i] if error_bars else None,
            error_kw=dict(ecolor=colors[i], capsize=5),
            alpha=1)
        if points:
            xi = np.full_like(np.asarray(data_groups[i], dtype=float), x_positions[i], dtype=float)
            ax.scatter(xi, data_groups[i], color=colors[i], s=10)

    # Labels, ticks (intentionally ignore `title` to avoid layout shifts)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, rotation=45, fontsize='small')

    # Y scale/limits
    if ylim is not None:
        ax.set_ylim(ylim)
    if log_y:
        ax.set_yscale('log')

    # Clean spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ==== Stats annotations that DO NOT change y-limits ====
    if stats and num_groups > 1:
        # Decide which comparisons to show
        if comparisons is None:
            if num_groups == 2:
                comparisons = [(0, 1)]
            elif num_groups == 4:
                # Your requested set:
                comparisons = [(0, 1), (0, 2), (1, 3), (2, 3)]
            else:
                comparisons = list(combinations(range(num_groups), 2))

        # Save current limits so nothing we add will autoscale
        orig_ylim = ax.get_ylim()

        # blended transform: x in data coords, y in axes coords (0..1)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        # Place rows just above the top frame, stepping upward
        y_base = 1.01      # first row just above axes
        y_step = 0.06      # gap between rows

        for idx, (i, j) in enumerate(comparisons):
            # Welch's t-test
            stat, p_val = ttest_ind(
                data_groups[i], data_groups[j],
                equal_var=equal_var, nan_policy='omit'
            )

            if p_val < 0.001:
                sig_label = '***'
            elif p_val < 0.01:
                sig_label = '**'
            elif p_val < 0.05:
                sig_label = '*'
            else:
                sig_label = 'ns'

            # Short line between bars i and j
            line_pad = bar_width * 0.3
            x0 = x_positions[i] + line_pad
            x1 = x_positions[j] - line_pad
            y_here = y_base + idx * y_step

            # Horizontal line (y in axes coords)
            ax.plot([x0, x1], [y_here, y_here],
                    transform=trans, clip_on=False, lw=0.6, color='black')

            # Centered text above the line (y in axes coords)
            ax.text((x0 + x1) / 2.0, y_here, sig_label,
                    transform=trans, clip_on=False,
                    ha='center', va='bottom', fontsize=6)

        # Restore original limits explicitly (belt-and-braces)
        ax.set_ylim(orig_ylim)

    # Show/close
    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax

def plot_grouped_bar(fig, ax, grouped_data, xticks, labels, colors,
                     bar_width=0.2, error_bars=False, log_y=False, ylim=(0, None), y_label=None, legend_loc='best',
                     show=False, close=True):
    """
    Plots grouped bar chart with two bars per group.

    Parameters:
    - grouped_data: list of tuples [(data1_group1, data2_group1), ...]
    - xticks: labels for groups on x-axis
    - labels: labels for bars (e.g. ("Nest", "Arena"))
    - colors: list of base colors, one per group
              (the second bar will be a lighter version of the base color)
    """
    num_groups = len(grouped_data)
    group_spacing = 0.3
    start_x = 0.2
    x_centers = [start_x + i * group_spacing for i in range(num_groups)]

    ax.margins(x=0.1)

    for i, (data1, data2) in enumerate(grouped_data):
        mean1, mean2 = np.nanmean(data1), np.nanmean(data2)
        std1, std2 = np.nanstd(data1), np.nanstd(data2)

        x1 = x_centers[i] - bar_width/2
        x2 = x_centers[i] + bar_width/2

        base = mcolors.to_rgb(colors[i])
        lighter = tuple(0.5 + 0.5*np.array(base))  # simple lighten: halfway to white

        ax.bar(x1, mean1, bar_width, color=base,
               yerr=std1 if error_bars else None,
               error_kw=dict(ecolor=base, capsize=5) if error_bars else {})
        ax.bar(x2, mean2, bar_width, color=lighter,
               yerr=std2 if error_bars else None,
               error_kw=dict(ecolor=lighter, capsize=5) if error_bars else {})

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(xticks, rotation=45, fontsize='small')
    ax.legend(labels, loc=legend_loc, fontsize='x-small', frameon=False)

    if log_y:
        ax.set_yscale("log")
        
    if y_label:
        ax.set_ylabel(y_label)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax

def plot_timecourse(
    fig, ax, speeds_by_type, mouse_types, color=None,
    frame_time=None,
    title="",
    x_label=None, y_label="", ylim=(0, None),
    alpha_fill=0.25, linewidth=1.8, legend_loc='upper left',
    smooth=None,
    shade=True, show=False, close=True):

    # determine T from the first non-empty mouse type
    T = None
    for group in speeds_by_type:
        if len(group) > 0:
            T = np.asarray(group[0]).shape[-1]
            break
    if T is None:
        ax.set_title(title)
        ax.set_xlabel(x_label or "Frame")
        ax.set_ylabel(y_label)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if show: plt.show()
        if close: plt.close()
        return fig, ax

    # build x-axis
    if frame_time is None:
        x = np.arange(T)
        default_x_label = "Frame"
    else:
        x = np.arange(T) / frame_time
        default_x_label = "Time (s)" if frame_time < 100 else "Normalized time"

    # color handling
    if isinstance(color, (list, tuple)) and len(color) > 1:
        colors = color
    else:
        n = len(mouse_types)
        base_color = color
        colors = [base_color for _ in range(n)]
        alphas = np.linspace(1.0, 0.2, n)
    
    # smoothing kernel (if requested)
    if smooth is not None and smooth > 0:
        sigma = float(smooth)
        truncate = 3.0
        radius = int(truncate * sigma + 0.5)
        xk = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (xk / sigma) ** 2)
        kernel /= kernel.sum()
    else:
        kernel = None

    for i, (mt_label, color, trials) in enumerate(zip(mouse_types, colors, speeds_by_type)):
        if len(trials) == 0:
            continue

        A = np.asarray(trials, dtype=float)
        if A.ndim == 1:
            A = A[None, :]

        valid = (~np.isnan(A)) & (A != 0)

        n_eff  = valid.sum(axis=0).astype(float)                 # count of valid samples
        A_fill = np.where(valid, A, 0.0)                         # zero out invalid entries

        # mean (safe divide; frames with no data -> NaN)
        sum_  = A_fill.sum(axis=0)
        mean  = np.divide(sum_, n_eff, out=np.full_like(sum_, np.nan), where=n_eff > 0)

        # sample variance (ddof=1) on valid entries only
        sumsq = (A_fill * A_fill).sum(axis=0)
        var_num = sumsq - (sum_**2) / np.where(n_eff > 0, n_eff, 1.0)
        var_den = n_eff - 1.0
        var = np.divide(var_num, var_den, out=np.zeros_like(var_num), where=var_den > 0)
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var)

        # in-house smoothing (ignoring zeros)
        if kernel is not None:
            mask_mean = ~np.isnan(mean)
            valid = np.convolve(mask_mean.astype(float), kernel, mode='same')
            smoothed = np.convolve(np.nan_to_num(mean) * mask_mean, kernel, mode='same')
            mean = np.divide(smoothed, valid, out=np.nan*np.ones_like(smoothed), where=valid > 0)

            mask_sd = ~np.isnan(sd)
            valid_sd = np.convolve(mask_sd.astype(float), kernel, mode='same')
            smoothed_sd = np.convolve(np.nan_to_num(sd) * mask_sd, kernel, mode='same')
            sd = np.divide(smoothed_sd, valid_sd, out=np.nan*np.ones_like(smoothed_sd), where=valid_sd > 0)

        # transparency for single-color case
        alpha_line = alphas[i] if 'alphas' in locals() else 1.0
        alpha_fill_i = alpha_fill * alpha_line

        ax.plot(x, mean, color=color, lw=linewidth, alpha=alpha_line, label=mt_label)
        if shade:
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=alpha_fill_i, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel(x_label or default_x_label)
    ax.set_ylabel(y_label)
    ax.legend(mouse_types, loc=legend_loc, fontsize='x-small', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(ylim)

    if show:
        plt.show()
    if close:
        plt.close()
    return fig, ax

def plot_histogram(
    fig, ax, data_list, labels, colors,
    bins=20, xlabel="Value", ylabel="Frequency (%)",
    show_median=True, xlim=None, alpha=0.45,
    show=False, close=True, print_skew=True
):
    rng = np.random.default_rng(0)  # reproducible bootstraps
    stats = {}

    all_data = np.concatenate([
        np.asarray(d, dtype=float)[~np.isnan(d)]
        for d in data_list if d is not None and len(d) > 0
    ]) if any(data_list) else np.array([])

    bin_edges = np.histogram_bin_edges(all_data, bins=bins)
    group_stats = []

    # store medians (and colors) so we can label later
    medians = []
    median_colors = []

    for data, label, color in zip(data_list, labels, colors):
        if data is None or len(data) == 0:
            continue
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        if len(data) == 0:
            continue

        weights = np.ones_like(data) * 100 / len(data)
        ax.hist(data, bins=bin_edges, weights=weights,
                color=color, alpha=alpha, label=label)

        med = np.median(data)
        boot = [np.median(rng.choice(data, size=len(data), replace=True)) for _ in range(2000)]
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        sk = skew(data, nan_policy='omit')

        group_stats.append({
            "label": label,
            "n": len(data),
            "median": med,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "skew": sk,
        })

        if show_median:
            ax.axvline(med, linestyle="--", color=color, linewidth=1.5)
            medians.append(med)
            median_colors.append(color)

    # ----- place ALL median labels AFTER axes limits are final -----
    if show_median and medians:
        # x in data coords, y in axes coords (0–1)
        transform = ax.get_xaxis_transform()
        for med, color in zip(medians, median_colors):
            ax.text(
                med, 1.02, f"{med:.2f}",   # 2% above top, SAME for all
                transform=transform,
                ha="center", va="bottom",
                fontsize=8, color="black",
                rotation=70,
                clip_on=False,            # don't clip at top
            )

    # --- stats printing (unchanged) ---
    if print_skew and group_stats:
        header_title = xlabel
        print(f"\n=== Histogram group stats: {header_title} ===")
        print("Group         n    Median    95% CI median         Skew")
        print("-" * 80)
        for gs in group_stats:
            ci_str = f"({gs['ci_low']:7.3f}, {gs['ci_high']:7.3f})"
            print(f"{gs['label']:12s}  {gs['n']:3d}  {gs['median']:7.3f}  "
                  f"{ci_str:21s}  {gs['skew']:7.3f}")
        print("\n")

    # ... rest of your function unchanged ...
    valid_data = []
    for lbl, dat in zip(labels, data_list):
        if dat is None or len(dat) == 0:
            continue
        arr = np.asarray(dat, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        valid_data.append((lbl, arr))

    if len(valid_data) > 1:
        print("\n--- Pairwise Median Tests (Mood’s) ---")
        print("Group A       Group B       p-value    Common median")
        print("-" * 60)
        for (l1, d1), (l2, d2) in combinations(valid_data, 2):
            stat, p, common_med, table = median_test(d1, d2)
            print(f"{l1:12s}  {l2:12s}  {p:8.4f}    {common_med:7.3f}")
            stats[(l1, l2)] = {"p_median_test": p, "common_median": common_med}
        print("\n")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    handles = [Line2D([0], [0], color=c, linestyle='-', linewidth=2) for c in colors]
    ax.legend(handles, labels, loc='best', fontsize='x-small', frameon=False)

    plt.tight_layout()

    if show:
        plt.show()
    if close:
        plt.close()

    return fig, ax

def plot_pie_chart(fig, ax, data, title=None, labels=None, colors=None, autopct='%1.1f%%'):
    
    # --- data & labels ---
    if isinstance(data, dict):
        labels = list(data.keys())
        values = list(data.values())
    else:
        values = np.asarray(data)
        if labels is None:
            labels = [f"Value {i+1}" for i in range(len(values))]

    n = len(values)

    # --- color handling (similar spirit to plot_timecourse) ---
    # allow colors to be a single string or a list
    if isinstance(colors, (list, tuple)) and len(colors) > 1:
        pie_colors = colors[:n]
        alphas = None  # use default alpha = 1 for all
    else:
        # single base color (string or 1-element list)
        base_color = colors[0] if isinstance(colors, (list, tuple)) else colors
        # same base color for all wedges
        pie_colors = [base_color] * n
        # different alphas for each wedge (e.g. 1.0 -> 0.2)
        alphas = np.linspace(1.0, 0.2, n)

    wedges, _, autotexts = ax.pie(
        values,
        labels=None,           # legend handles labels
        autopct=autopct,
        colors=pie_colors,
        startangle=90
    )

    # --- apply per-wedge alpha after the fact (no RGBA, no to_rgba) ---
    if alphas is not None:
        for w, a in zip(wedges, alphas):
            w.set_alpha(a)

    # --- legend & labels ---
    ax.legend(
        wedges, labels,
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1, 0.5),
        fontsize='x-small'
    )
    for t in autotexts:
        t.set_color("black")
        t.set_fontsize(10)

    ax.axis("equal")
    ax.set_title(title or "", fontsize=10)
    plt.tight_layout()

    return fig, ax

def plot_trial_grid_paths(
    before_trials, 
    stim_trials, 
    *,
    title="",
    drawer=None,
    areas=None,
    scores=None,
    show_lines=False,
    max_lines=450,
    x_tolerance=30,   # <-- NEW: only match before-points within this Δx
    self_obj=None
):
    """
    Plot per-trial before/stim paths in a grid.

    Parameters
    ----------
    before_trials, stim_trials : list of (N,2) arrays
    drawer : optional callable (before, stim, ax, area, score)
        Custom renderer, e.g. your plot_path_with_area().
    areas, scores : list[float] or None
        Used by drawer or to annotate panels.
    show_lines : bool
        If True, draw up to `max_lines` closest-point lines from stim→before.
    max_lines : int
        Maximum number of connection lines per trial.
    x_tolerance : float or None
        If not None, only consider before-points whose x-coordinate is
        within ±x_tolerance of the stim point's x (same units as paths,
        typically pixels).
    self_obj : object
        Optional, if you want to auto-append fig to self.imgs.
    """
    
    n = min(len(before_trials), len(stim_trials))
    if n == 0:
        return None, None

    n_cols = 6
    n_rows = math.ceil(n / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    if title:
        fig.suptitle(title, fontsize=14)
    if self_obj is not None and hasattr(self_obj, "imgs"):
        self_obj.imgs.append(fig)

    axs = np.ravel(axs)
    for j, ax in enumerate(axs):
        if j >= n:
            ax.axis("off")
            continue

        before = np.asarray(before_trials[j], float)
        stim   = np.asarray(stim_trials[j], float)
        if before.size == 0 or stim.size == 0:
            ax.axis("off")
            continue

        ax.set_facecolor("#f0f0f0")
        ax.set_xlim(80, 790)
        ax.set_ylim(700, 80)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # custom drawer takes over if provided
        if drawer is not None:
            area = areas[j] if areas and j < len(areas) else None
            score = scores[j] if scores and j < len(scores) else None
            drawer(before, stim, ax, area, score)
        else:
            # default simple plot
            ax.plot(before[:, 0], before[:, 1], color="grey")
            ax.plot(stim[:, 0],   stim[:, 1],   color="blue")

            if show_lines:
                # connect sampled stim points to their closest before points,
                # optionally restricted by x_tolerance
                b_arr = before  # use full array, no removal
                idxs = np.linspace(
                    0, len(stim) - 1,
                    min(max_lines, len(stim)),
                    dtype=int
                )

                for idx in idxs:
                    e = stim[idx]  # (x, y) of escape/stim point

                    # filter candidates by x_tolerance if requested
                    if x_tolerance is not None:
                        mask = np.abs(b_arr[:, 0] - e[0]) <= x_tolerance
                        candidates = b_arr[mask]
                        if len(candidates) == 0:
                            candidates = b_arr
                        
                    # squared distances to candidate before points
                    dists = np.sum((candidates - e) ** 2, axis=1)
                    k = int(np.argmin(dists))
                    b = candidates[k]

                    # draw connection line
                    ax.plot(
                        [e[0], b[0]], [e[1], b[1]],
                        color="red", alpha=0.3, lw=0.5
                    )

            if scores and j < len(scores):
                ax.text(
                    0.05, 0.95, f"{scores[j]:.2f}",
                    transform=ax.transAxes,
                    fontsize=8, color="red", va="top", ha="left",
                    bbox=dict(
                        facecolor="white", alpha=0.7,
                        edgecolor="none", boxstyle="round,pad=0.3"
                    )
                )

    plt.close(fig)
    return fig, axs


