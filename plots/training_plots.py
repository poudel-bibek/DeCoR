import os
import sys

# Ensure project root is on sys.path so absolute imports (utils, config, etc.) work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import xml.etree.ElementTree as ET
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from utils import get_averages
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Path helpers — resolve relative to project root regardless of CWD
# ---------------------------------------------------------------------------
def _out(*parts):
    """Join parts relative to the plots/ directory (for output files)."""
    return os.path.join(_SCRIPT_DIR, *parts)

def count_consecutive_ones_filtered(actions):
    """
    Helper function to count consecutive occurrences of 1's in the action list.
    The first action (corresponding to intersection) is ignored.
    Returns a list where each element is the length of a consecutive sequence of 1's.

    Example:
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 1] → [2, 1, 3]
    """
    if not actions or len(actions) <= 1:
        return []

    counts = []
    count = 0

    # Start from the second action (index 1)
    for action in actions[1:]:
        if action == 1:
            count += 1
        else:
            if count > 0:
                counts.append(count)
                count = 0

    # Don't forget to add the last sequence if it ends with 1's
    if count > 0:
        counts.append(count)

    return counts

def plot_avg_consecutive_ones(file_path, output_path=None):
    """
    Creates a clean, professional plot of the average sum of consecutive occurrences of '1's
    per training iteration with a vibrant appearance.

    Parameters:
        file_path (str): Path to the JSON file containing the data.
        output_path (str): Path to save the output PDF file.
    """
    if output_path is None:
        output_path = _out("sampled_actions_retro.pdf")

    # Load data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Compute the average sum of consecutive 1's per iteration
    avg_consecutive_ones_per_iteration = []
    iterations = []

    for iteration, actions_list in data.items():
        iteration = int(iteration)  # Convert iteration key to integer
        consecutive_ones = [count_consecutive_ones_filtered(action_list) for action_list in actions_list]

        # Calculate the sum of consecutive 1's for each sample, then average across samples
        sums_of_consecutive_ones = [sum(seq) for seq in consecutive_ones if seq]
        avg_consecutive_ones = np.mean(sums_of_consecutive_ones) if sums_of_consecutive_ones else 0

        iterations.append(iteration)
        avg_consecutive_ones_per_iteration.append(avg_consecutive_ones)

    # Sort by iteration
    iterations, avg_consecutive_ones_per_iteration = zip(*sorted(zip(iterations, avg_consecutive_ones_per_iteration)))
    iterations = np.array(iterations)
    avg_consecutive_ones_per_iteration = np.array(avg_consecutive_ones_per_iteration)

    # Set base font size
    fs = 24  # Base font size - adjust this to change all font sizes proportionally

    # Set up the figure with a clean style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    # Set background color
    ax.set_facecolor('white')

    # Calculate y-axis limits with some padding
    y_min = min(avg_consecutive_ones_per_iteration) * 0.9
    y_max = max(avg_consecutive_ones_per_iteration) * 1.1

    # Calculate x-axis limits with added margins
    x_min = min(iterations) - (max(iterations) - min(iterations)) * 0.05  # 5% margin on left
    x_max = max(iterations) + (max(iterations) - min(iterations)) * 0.05  # 5% margin on right

    # Set axis limits
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    # Format y-axis with one decimal place
    def format_with_decimals(x, pos):
        return f'{x:.1f}'

    ax.yaxis.set_major_formatter(FuncFormatter(format_with_decimals))

    # Add light grid lines with slightly more visibility
    ax.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Use a more vibrant blue for the data points
    VIBRANT_BLUE = '#2E5EAA'  # More vibrant blue for data points

    # Create scatter plot with more vibrant, semi-transparent circles
    scatter = ax.scatter(iterations, avg_consecutive_ones_per_iteration,
                        s=110, edgecolors=VIBRANT_BLUE, facecolors='none',
                        linewidth=2.0, alpha=0.75, zorder=3)

    # Fit a trend line
    z = np.polyfit(iterations, avg_consecutive_ones_per_iteration, 1)
    p = np.poly1d(z)

    # Create x values for the trend line (only within the data range)
    x_trend = np.linspace(min(iterations), max(iterations), 100)
    y_trend = p(x_trend)

    # Use a very dark blue color for the trend line - almost navy blue
    VERY_DARK_BLUE = '#0A2472'  # Very dark blue/navy color

    # Plot the trend line as a solid, very dark line
    trend_line = ax.plot(x_trend, y_trend, color=VERY_DARK_BLUE, linewidth=4.0, zorder=4)

    # Set labels with increased font size and more vibrant color
    LABEL_COLOR = '#1A1A1A'  # Slightly lighter than pure black for better contrast
    ax.set_xlabel('Training Iteration', fontsize=fs*1.2, labelpad=10, color=LABEL_COLOR)
    ax.set_ylabel('# of Synchronized Green Signals', fontsize=fs*1.2, labelpad=10, color=LABEL_COLOR)

    # Line for trend line - use the very dark blue color
    trend_line_handle = mlines.Line2D([], [], color=VERY_DARK_BLUE, linewidth=4.0,
                                     label='Trend Line')

    # Add the legend with the proper handles
    ax.legend(handles=[trend_line_handle],
             loc='upper right', frameon=True, framealpha=0.9,
             edgecolor='#CCCCCC', fontsize=fs)

    # Add padding between y-axis and tick labels
    ax.tick_params(axis='y', pad=8)  # Add padding between y-axis and y-tick labels

    # Customize tick parameters with larger font size and more vibrant color
    ax.tick_params(axis='both', colors=LABEL_COLOR, labelsize=fs)

    # Add a subtle border around the plot with slightly more visible color
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#AAAAAA')  # Slightly darker border
        ax.spines[spine].set_linewidth(1.2)  # Slightly thicker border

    # Add more padding around the entire plot
    plt.tight_layout(pad=2.0)

    # Save with extra padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()

    print(f"Plot saved to {output_path}")

def plot_control_results(*json_paths, in_range_demand_scales):
    """
    """
    fs = 17 
    COLORS = {
        'Signalized':   '#F4B400', 
        'Unsignalized': '#4285F4',   
        'RL (Ours)':    '#0F9D58',   
    }
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Open Sans', 'Arial', 'DejaVu Sans'],
        'text.color':         '#202124',
        'axes.edgecolor':     '#dadce0',
        'axes.linewidth':     1.0,
        'axes.titlesize':     fs + 2,
        'axes.titleweight':   'bold',
        'axes.labelsize':     fs,
        'xtick.color':        '#5f6368',
        'ytick.color':        '#5f6368',
        'xtick.labelsize':    fs - 1,
        'ytick.labelsize':    fs - 1,
        'grid.color':         '#e8eaed',
        'grid.linewidth':     0.8,
        'grid.linestyle':     '--',
        'legend.frameon':     False,
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
    })

    fig = plt.figure(figsize=(16, 7))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.22)
    ax_pa = fig.add_subplot(gs[0,0])
    ax_pt = fig.add_subplot(gs[1,0], sharex=ax_pa)
    ax_va = fig.add_subplot(gs[0,1])
    ax_vt = fig.add_subplot(gs[1,1], sharex=ax_va)
    panels = [ax_pa, ax_pt, ax_va, ax_vt]

    all_scales = []
    for path in json_paths:
        scales = get_averages(path, total=False)[0]
        all_scales.extend(scales)
    all_scales = np.array(all_scales)

    unique_scales = np.sort(np.unique(all_scales))

    ax_pa.set_title('Pedestrian')
    ax_va.set_title('Vehicle')

    if len(json_paths) == 3:
        tl_idx = [i for i, p in enumerate(json_paths) if 'tl' in p.lower()][0]
        us_idx = [i for i, p in enumerate(json_paths) if 'unsignalized' in p.lower()][0]
        rl_idx = [i for i, p in enumerate(json_paths) if 'ppo' in p.lower()][0]
        json_paths = [json_paths[tl_idx], json_paths[us_idx], json_paths[rl_idx]]
        method_labels = ['Signalized', 'Unsignalized', 'RL (Ours)']
    else:
        tl_idx = [i for i, p in enumerate(json_paths) if 'tl' in p.lower()][0]
        rl_idx = [i for i, p in enumerate(json_paths) if 'ppo' in p.lower()][0]
        json_paths = [json_paths[tl_idx], json_paths[rl_idx]]
        method_labels = ['Signalized', 'RL (Ours)']


    x_min, x_max = all_scales.min(), all_scales.max()
    x_margin = 0.05 * (x_max - x_min)

    for ax in panels:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    valid_min_scale = min(in_range_demand_scales)
    valid_max_scale = max(in_range_demand_scales)

    for ax in panels:
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], valid_min_scale, facecolor='grey', alpha=0.25, zorder=-2)
        ax.axvspan(valid_max_scale, xlim[1], facecolor='grey', alpha=0.25, zorder=-2)

    legend_handles = []
    for _, (path, label) in enumerate(zip(json_paths, method_labels)):
        color = COLORS[label]

        scales, veh_avg_mean, ped_avg_mean, _, veh_avg_std, ped_avg_std, _ = get_averages(path, total=False)
        _, veh_tot, ped_tot, _, veh_tot_std, ped_tot_std, _ = get_averages(path, total=True)

        h_pa = ax_pa.plot(scales, ped_avg_mean, color=color, lw=2.5, label=label, zorder=2)[0]
        ax_pa.fill_between(scales,
                           ped_avg_mean - ped_avg_std,
                           ped_avg_mean + ped_avg_std,
                           color=color, alpha=0.2, zorder=2)

        ax_pt.plot(scales, ped_tot/1000, color=color, lw=2.5, zorder=2)
        ax_pt.fill_between(scales,
                           (ped_tot - ped_tot_std)/1000,
                           (ped_tot + ped_tot_std)/1000,
                           color=color, alpha=0.2, zorder=2)

        ax_va.plot(scales, veh_avg_mean, color=color, lw=2.5, zorder=2)
        ax_va.fill_between(scales,
                           veh_avg_mean - veh_avg_std,
                           veh_avg_mean + veh_avg_std,
                           color=color, alpha=0.2, zorder=2)

        ax_vt.plot(scales, veh_tot/1000, color=color, lw=2.5, zorder=2)
        ax_vt.fill_between(scales,
                           (veh_tot - veh_tot_std)/1000,
                           (veh_tot + veh_tot_std)/1000,
                           color=color, alpha=0.2, zorder=2)

        legend_handles.append(h_pa)
    
    scales_to_show = unique_scales[::2]
    labels = []
    for s in scales_to_show:
        if abs(s * 10 - round(s * 10)) < 1e-6:
            labels.append(f"{s:.1f}x")
        else:
            labels.append(f"{s:.2f}x")

    # Set major ticks only at the locations we want to label
    ax_pt.set_xticks(scales_to_show)
    ax_vt.set_xticks(scales_to_show)

    ax_pt.set_xticklabels(labels)
    ax_vt.set_xticklabels(labels)

    for ax in panels:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y')
        ax.set_xticks(unique_scales, minor=True)
        ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)
        ax.grid(which='major', axis='x', linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)

    fig.text(0.03, 0.76, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.03, 0.29, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.52, 0.76, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.52, 0.29, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs+1)

    ax_pt.set_xlabel('Demand Scale', fontsize=fs+1) # (× original)
    ax_vt.set_xlabel('Demand Scale', fontsize=fs+1) # (× original)

    ax_pa.tick_params(labelbottom=False)
    ax_va.tick_params(labelbottom=False)

    n_yticks = 6
    for ax in panels:
        # ax.set_ylim(bottom=-0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    fig.legend(legend_handles, method_labels,
               ncol=len(method_labels),
               loc='lower center',
               bbox_to_anchor=(0.5, -0.08),  # push slightly below panels
               frameon=True,
               edgecolor='#dadce0',
               fontsize=fs)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.11, wspace=0.10, hspace=0.12)
    plt.savefig(_out("consolidated_control_results.pdf"), bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_design_results(*json_paths, in_range_demand_scales):
    """
    """
    original_pedestrian_demand = 2222.80
    COLORS = {'Design Agent': '#0F9D58', 
            'Real-world': '#4285F4'}

    fs = 17
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Open Sans', 'Arial', 'DejaVu Sans'],
        'text.color':         '#202124',
        'axes.edgecolor':     '#dadce0',
        'axes.linewidth':     1.0,
        'axes.titlesize':     fs + 2,
        'axes.titleweight':   'bold',
        'axes.labelsize':     fs,
        'xtick.color':        '#5f6368',
        'ytick.color':        '#5f6368',
        'xtick.labelsize':    fs - 1,
        'ytick.labelsize':    fs - 1,
        'grid.color':         '#e8eaed',
        'grid.linewidth':     0.8,
        'grid.linestyle':     '--',
        'legend.frameon':     False,
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
    })

    fig = plt.figure(figsize=(9, 8)) # Slightly wider/taller for better label spacing
    gs  = GridSpec(2, 1, figure=fig, hspace=0.12) # Adjusted spacing if needed
    ax_avg = fig.add_subplot(gs[0, 0])
    ax_tot = fig.add_subplot(gs[1, 0], sharex=ax_avg)
    panels = [ax_avg, ax_tot]

    ax_avg.set_title('Pedestrian')

    if len(json_paths) != 2:
        raise ValueError('plot_design_results expects exactly two json paths')

    all_scales = []
    design_scales = None
    for path in json_paths:
        scales = get_averages(path, total=False)[0]
        all_scales.extend(scales)
        if design_scales is None: # Store scales from the first path (Design Agent)
             design_scales = scales
    all_scales = np.array(all_scales)
    unique_scales = np.sort(np.unique(all_scales))

    x_min, x_max = unique_scales.min(), unique_scales.max()
    x_margin = 0.05 * (x_max - x_min)

    # Remove top/right spines and set x-axis limits
    for ax in panels:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    # Shade out-of-range demand scales before drawing gridlines
    vmin, vmax = min(in_range_demand_scales), max(in_range_demand_scales)
    for ax in panels:
        xlim = ax.get_xlim()  # axis limits already set
        ax.axvspan(xlim[0], vmin, facecolor='grey', alpha=0.25, zorder=-2)
        ax.axvspan(vmax, xlim[1], facecolor='grey', alpha=0.25, zorder=-2)

    # Draw gridlines on top of shading
    for ax in panels:
        ax.grid(True, axis='y')  # Only horizontal grid lines
        ax.set_xticks(unique_scales, minor=True)
        ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.8,
                alpha=0.7, zorder=-5)
        ax.grid(which='major', axis='x', linestyle='--', linewidth=0.8,
                alpha=0.7, zorder=-5)

    legend_handles = []
    labels = ['Design Agent', 'Real-world'] # Assuming order matches json_paths
    for path, label in zip(json_paths, labels):
        scales, _, _, avg_vals, _, _, avg_std = get_averages(path, total=False)
        _, _, _, tot_vals, _, _, tot_std = get_averages(path, total=True)
        color = COLORS[label]

        h = ax_avg.plot(scales, avg_vals, color=color, lw=2.5, label=label, zorder=2)[0]
        ax_avg.fill_between(scales, avg_vals - avg_std, avg_vals + avg_std, color=color, alpha=0.2, zorder=2)

        tot_k = tot_vals / 1000.0
        tot_k_std = tot_std / 1000.0
        ax_tot.plot(scales, tot_k, color=color, lw=2.5, zorder=2)
        ax_tot.fill_between(scales, tot_k - tot_k_std, tot_k + tot_k_std, color=color, alpha=0.2, zorder=2)

        legend_handles.append(h)

    scales_to_show = unique_scales[::2] # Show every other major tick
    if unique_scales[-1] not in scales_to_show:
         scales_to_show = np.append(scales_to_show, unique_scales[-1])

    ax_tot.set_xticks(scales_to_show)
    
    x_labs = [f"{s:.1f}x" if abs(s * 10 - round(s * 10)) < 1e-6 else f"{s:.2f}x" for s in scales_to_show]
    ax_tot.set_xticklabels(x_labs)

    ax_tot.set_xlabel('Demand Scale', fontsize=fs + 1)
    ax_avg.tick_params(labelbottom=False) # Hide x-labels on top plot

    # Uniform Y-ticks: integer, padded bottom
    n_yticks = 6 # Match control plot
    for ax in panels:
        if ax == ax_avg:
            ax.set_ylim(bottom=50, top=120)
        else:
            ax.set_ylim(bottom=-0.5) # Match control plot padding
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    fig.text(0.01, 0.76, 'Average Arrival Time (s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.01, 0.29, 'Total Arrival Time (×10³ s)', va='center', rotation='vertical', fontsize=fs+1)

    fig.legend(legend_handles, labels,
               ncol=len(labels),
               loc='lower center',
               bbox_to_anchor=(0.5, -0.08), # Adjusted anchor
               frameon=True, # Keep frame off
               edgecolor='#dadce0',
               fontsize=fs)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.10, wspace=0.10, hspace=0.12) # Adjust margins
    plt.savefig(_out('consolidated_design_results.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_consolidated_insights(sampled_actions_file_path, conflict_json_file_path, switching_freq_data_path):
    """
    Creates a consolidated figure with three subplots:
    1. Left: Bar chart of mean conflicts across demand scales with error bars
    2. Middle: Plot of average consecutive ones over training iterations
    3. Right: TL as horizontal line and RL as histogram for switching frequency (TL switching frequency is obtained analytically as 54 for 600 timestep horizon)

    Parameters:
    - sampled_actions_file_path: Path to JSON file containing action data
    - conflict_json_file_path: Path to JSON file containing conflict data
    - switching_freq_data: Dictionary containing switching frequency data (optional)
    """
    # Function to process data from json
    def process_json_data(json_data, key):
        # Extract data by demand scale
        data = {}
        for demand_scale, runs in json_data.items():
            values = [run_data[key] for run_index, run_data in runs.items()]
            data[float(demand_scale)] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        return data

    # Load conflict data
    with open(conflict_json_file_path, 'r') as f:
        conflict_json_data = json.load(f)

    # Process conflict data
    processed_conflict_data = process_json_data(conflict_json_data, "total_conflicts")

    # Set base font size
    fs = 23

    # Set consistent number of y-ticks for all subplots
    n_ticks = 5  # Define the number of y-ticks to use across all subplots

    # Set up the figure with a 1x3 grid
    fig = plt.figure(figsize=(24, 6.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1])

    # Create subplots
    ax_near_accidents = fig.add_subplot(gs[0, 0])
    ax_consecutive_ones = fig.add_subplot(gs[0, 1])
    ax_switching_freq = fig.add_subplot(gs[0, 2])

    # Set style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    # Define colors - updated middle plot colors
    BRIGHT_BLUE = '#0078D7'  # New bright blue for middle plot trend line
    VIBRANT_BLUE = '#2E5EAA'  # Keep for scatter points
    SALMON = '#E29587'  # Subtle salmon for TL/Unsignalized
    SEA_GREEN = '#85B79D'  # Subtle sea green for RL

    # ========== LEFT SUBPLOT: Conflict events across demand scales ==========
    # Filter demand scales to only include the specified levels
    selected_demand_scales = [0.5, 1.0, 1.5, 2.0, 2.5]
    filtered_scales = [scale for scale in selected_demand_scales if scale in processed_conflict_data]

    conflict_means = [processed_conflict_data[scale]["mean"] for scale in filtered_scales]
    conflict_stds = [processed_conflict_data[scale]["std"] for scale in filtered_scales]

    # Even more subtle gradient - using shades of orange/coral with less intensity
    colors = [
        '#FDE5D2',  # Very pale orange for 0.5x
        '#FDCBAD',  # Lighter orange for 1.0x
        '#FCB08A',  # Light salmon for 1.5x
        '#FC9774',  # Salmon for 2.0x
        '#FB7D5B'   # Darker salmon for 2.5x
    ]

    # Make sure we have enough colors
    if len(colors) < len(filtered_scales):
        colors = colors * (len(filtered_scales) // len(colors) + 1)
    colors = colors[:len(filtered_scales)]

    # Create bar positions
    x_positions = np.arange(len(filtered_scales))
    width = 0.5

    # Create bar chart with MORE PROMINENT error bars
    bars = ax_near_accidents.bar(x_positions, conflict_means, width, color=colors,
                               edgecolor='#333333', linewidth=1.0,
                               yerr=conflict_stds, capsize=8, error_kw={'elinewidth': 2.5, 'ecolor': '#333333', 'capthick': 2.5})

    # Add data labels to the left of the top of each bar
    # for i, bar in enumerate(bars):
    #     height = bar.get_height() + 9
    #     # Position text to the left of the bar top
    #     ax_near_accidents.text(bar.get_x() + 0.25*width, height,
    #                          f'{int(conflict_means[i])}', ha='right', va='center',
    #                          fontsize=fs-4)

    labelsize = fs-4
    # Set x-ticks at the bar positions with the appropriate labels
    ax_near_accidents.set_xticks(x_positions)
    ax_near_accidents.set_xticklabels([f'{scale}x' for scale in filtered_scales], fontsize=labelsize)

    # Styling
    ax_near_accidents.set_ylabel('# of Conflicts in Unsignalized', fontsize=fs)  # Updated label
    ax_near_accidents.set_xlabel('Demand Scale', fontsize=fs)
    ax_near_accidents.tick_params(axis='both', labelsize=labelsize)

    # Set y-limit with headroom for labels and error bars
    ax_near_accidents.set_ylim(0, max(conflict_means + np.array(conflict_stds)) * 1.1)  # More headroom for labels

    # Make grid match middle plot (light lines behind data)
    ax_near_accidents.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_near_accidents.set_axisbelow(True)

    # Remove top and right spines to match middle plot
    ax_near_accidents.spines['top'].set_visible(False)
    ax_near_accidents.spines['right'].set_visible(False)

    # Set consistent y-ticks
    ax_near_accidents.yaxis.set_major_locator(MaxNLocator(n_ticks))

    # ========== MIDDLE SUBPLOT: Average consecutive ones plot ==========
    # Load data
    with open(sampled_actions_file_path, "r") as file:
        data = json.load(file)

    # Compute the average sum of consecutive 1's per iteration
    avg_consecutive_ones_per_iteration = []
    iterations = []

    for iteration, actions_list in data.items():
        iteration = int(iteration)
        consecutive_ones = [count_consecutive_ones_filtered(action_list) for action_list in actions_list]
        sums_of_consecutive_ones = [sum(seq) for seq in consecutive_ones if seq]
        avg_consecutive_ones = np.mean(sums_of_consecutive_ones) if sums_of_consecutive_ones else 0
        iterations.append(iteration)
        avg_consecutive_ones_per_iteration.append(avg_consecutive_ones)

    # Sort by iteration
    iterations, avg_consecutive_ones_per_iteration = zip(*sorted(zip(iterations, avg_consecutive_ones_per_iteration)))
    iterations = np.array(iterations)
    avg_consecutive_ones_per_iteration = np.array(avg_consecutive_ones_per_iteration)

    # Set background color
    ax_consecutive_ones.set_facecolor('white')

    # Calculate y-axis limits with padding
    y_min = 3.3  # Set explicitly to 3.2 to match the lowest data point
    y_max = 4.1  # Set explicitly to 4.1 to provide headroom for highest points

    # Calculate x-axis limits with margins
    x_min = min(iterations) - (max(iterations) - min(iterations)) * 0.05
    x_max = max(iterations) + (max(iterations) - min(iterations)) * 0.05

    # Set axis limits
    ax_consecutive_ones.set_ylim(y_min, y_max)
    ax_consecutive_ones.set_xlim(x_min, x_max)

    # Format y-axis with one decimal place
    ax_consecutive_ones.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))

    # Add light grid lines
    ax_consecutive_ones.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_consecutive_ones.set_axisbelow(True)

    # Remove top and right spines
    ax_consecutive_ones.spines['top'].set_visible(False)
    ax_consecutive_ones.spines['right'].set_visible(False)

    # Create scatter plot - KEEPING ORIGINAL COLORS
    scatter = ax_consecutive_ones.scatter(iterations, avg_consecutive_ones_per_iteration,
                                        s=110, edgecolors=VIBRANT_BLUE, facecolors='none',
                                        linewidth=2.0, alpha=0.75, zorder=3)

    # Fit a trend line
    z = np.polyfit(iterations, avg_consecutive_ones_per_iteration, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(iterations), max(iterations), 100)
    y_trend = p(x_trend)

    # Calculate 95% confidence interval
    n = len(iterations)
    x_mean = np.mean(iterations)
    y_mean = np.mean(avg_consecutive_ones_per_iteration)

    # Sum of squares
    ss_xx = np.sum((iterations - x_mean)**2)
    ss_xy = np.sum((iterations - x_mean) * (avg_consecutive_ones_per_iteration - y_mean))
    ss_yy = np.sum((avg_consecutive_ones_per_iteration - y_mean)**2)

    # Regression slope and intercept
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # Standard error of estimate
    y_hat = slope * iterations + intercept
    se = np.sqrt(np.sum((avg_consecutive_ones_per_iteration - y_hat)**2) / (n - 2))

    # Confidence interval
    alpha = 0.05  # 95% confidence interval
    t_val = stats.t.ppf(1 - alpha/2, n - 2)

    # Calculate confidence bands
    x_eval = x_trend
    ci = t_val * se * np.sqrt(1/n + (x_eval - x_mean)**2 / ss_xx)
    y_upper = y_trend + ci
    y_lower = y_trend - ci

    # Plot the trend line with new bright blue color
    trend_line = ax_consecutive_ones.plot(x_trend, y_trend, color=BRIGHT_BLUE, linewidth=4.0, zorder=4, label='Trend Line')

    # Add confidence interval with shading
    confidence_interval = ax_consecutive_ones.fill_between(x_trend, y_lower, y_upper,
                                                         color=BRIGHT_BLUE, alpha=0.2,
                                                         zorder=2, label='95% Confidence Interval')

    # Set labels
    ax_consecutive_ones.set_xlabel('Training Episode', fontsize=fs)
    ax_consecutive_ones.set_ylabel('Synchronized Green Signals', fontsize=fs)

    # Create legend with both trend line and confidence interval
    trend_line_handle = mlines.Line2D([], [], color=BRIGHT_BLUE, linewidth=4.0,
                                    label='Trend Line')
    ci_handle = mpatches.Patch(facecolor=BRIGHT_BLUE, alpha=0.2,
                              label='95% Confidence Interval')

    ax_consecutive_ones.legend(handles=[trend_line_handle, ci_handle],
                            loc='upper right', frameon=True, framealpha=0.9,
                            edgecolor='#CCCCCC', fontsize=fs-4)

    # Tick parameters
    ax_consecutive_ones.tick_params(axis='both', labelsize=labelsize)

    # Set consistent y-ticks with fixed 0.1 interval to ensure we have 3.6 tick
    ax_consecutive_ones.yaxis.set_major_locator(MultipleLocator(0.2))

    # ========== RIGHT SUBPLOT: Switching frequency with TL as horizontal line ==========

    # Load frequency data
    with open(switching_freq_data_path, 'r') as f:
        frequency_json_data = json.load(f)

    # Process frequency data
    processed_frequency_data = process_json_data(frequency_json_data, "total_switches")

    frequency_demands = [0.5, 1.0, 1.5, 2.0, 2.5]
    filtered_demands = [demand for demand in frequency_demands if demand in processed_frequency_data]

    frequency_means = [processed_frequency_data[demand]["mean"] for demand in filtered_demands]
    frequency_stds = [processed_frequency_data[demand]["std"] for demand in filtered_demands]

    # Create placeholder data with TL having same value across demand scales
    tl_value = 54  # Same value for all demand scales

    # Get x positions for grouped bars
    x = np.arange(len(filtered_demands))
    width = 0.5  # Width of bars - keep the same

    # Create subtle gradient for RL bars
    rl_colors = [
        '#CFEAD6',  # Lower level lighter green
        '#A8D5BA',  # Lightest sea green
        '#8CCB9B',  # Light sea green
        '#73C17E',  # Medium sea green
        '#5AB663'   # Deeper sea green
    ]

    # Ensure we have enough colors
    if len(rl_colors) < len(filtered_demands):
        rl_colors = rl_colors * (len(filtered_demands) // len(rl_colors) + 1)
    rl_colors = rl_colors[:len(filtered_demands)]

    # Set up the plot with a discontinuous y-axis
    ax_switching_freq.set_facecolor('white')

    # Function to transform values to the broken y-axis scale
    def transform_y(y):
        # Map values to a discontinuous scale:
        # 0-54 maps to 0-0.2 (bottom 20% of plot)
        # 260-320 maps to 0.3-1.0 (top 70% of plot)
        if y <= 54:
            return y / 54 * 0.2
        else:
            return 0.3 + (y - 260) / (320 - 260) * 0.7

    # Plot the bars with standard deviations
    for i, (mean, std) in enumerate(zip(frequency_means, frequency_stds)):
        # Calculate bar height in the transformed space
        bar_height = transform_y(mean) - transform_y(0)

        # Draw the bar
        bar = ax_switching_freq.bar(x[i], bar_height, width=width,
                                   bottom=transform_y(0),
                                   color=rl_colors[i],
                                   edgecolor='#333333',
                                   linewidth=1.0)

        # Add error bars
        # Calculate the std dev in the transformed space
        yerr = transform_y(mean + std) - transform_y(mean)

        # Draw error bar
        ax_switching_freq.errorbar(x[i], transform_y(mean), yerr=yerr,
                                  fmt='none', ecolor='#333333', capsize=8,
                                  elinewidth=2.5, capthick=2.5)

    # Add the TL horizontal line
    tl_line = ax_switching_freq.axhline(y=transform_y(tl_value), color=SALMON, linewidth=3, linestyle='-', zorder=5)

    # Get the y-axis line width to match the break marks to it
    axis_line_width = ax_switching_freq.spines['left'].get_linewidth()

    # Create break marks for the y-axis
    # Position of the break in the transformed scale
    break_pos = 0.31  # middle of the gap between 0.2 and 0.3

    # Draw break marks on the left y-axis only
    # Increased spacing between diagonal lines
    gap = 0.020  # Increased gap between the diagonal lines
    d = 0.03    # Size of the diagonal lines

    # First create a white rectangle to "erase" part of the axis
    # This ensures the break appears as a true gap in the axis
    rect_height = gap * 1.5  # Height of white rectangle
    rect_width = d * 2.0     # Width of white rectangle

    # Draw white background rectangle to create a clean break
    white_patch = plt.Rectangle((-rect_width/2, break_pos-rect_height/2), rect_width, rect_height,
                              facecolor='white', edgecolor='none', transform=ax_switching_freq.transAxes,
                              clip_on=False, zorder=10)
    ax_switching_freq.add_patch(white_patch)

    # Then draw the diagonal lines centered on the axis
    # Make sure line width matches the axis line width
    kwargs = dict(transform=ax_switching_freq.transAxes, color='black',
                 clip_on=False, linewidth=axis_line_width, zorder=11)

    # Upper diagonal line
    ax_switching_freq.plot([-d/2, d/2], [break_pos+gap/2, break_pos+gap/2 + d], **kwargs)

    # Lower diagonal line
    ax_switching_freq.plot([-d/2, d/2], [break_pos-gap/2, break_pos-gap/2 + d], **kwargs)

    # Set the y-ticks at the actual data values
    yticks = [0, tl_value, 275, 300]
    yticklabels = [str(int(y)) for y in yticks]

    ax_switching_freq.set_yticks([transform_y(y) for y in yticks])
    ax_switching_freq.set_yticklabels(yticklabels, fontsize=labelsize)

    # Create legend handles
    tl_handle = mlines.Line2D([], [], color=SALMON, linewidth=3, linestyle='-', label='Signalized')
    rl_handle = mpatches.Patch(facecolor=rl_colors[1], edgecolor='#333333', linewidth=1.0, label='RL (Ours)')

    # Styling
    ax_switching_freq.set_ylabel('Switching Frequency', fontsize=fs)
    ax_switching_freq.set_xlabel('Demand Scale', fontsize=fs)
    ax_switching_freq.set_xticks(x)

    # Format x-ticks to show demand scale
    demand_labels = [f"{d}x" for d in filtered_demands]
    ax_switching_freq.set_xticklabels(demand_labels, fontsize=labelsize)

    ax_switching_freq.tick_params(axis='both', labelsize=labelsize)

    # Make grid match middle plot (light lines behind data)
    ax_switching_freq.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_switching_freq.set_axisbelow(True)

    # Remove top and right spines to match middle plot
    ax_switching_freq.spines['top'].set_visible(False)
    ax_switching_freq.spines['right'].set_visible(False)

    # Set uniform margins in right subplot
    # Calculate the margin to add on each side (half the width of a bar)
    margin = 0.7
    # Set the x-limits to create uniform margins
    ax_switching_freq.set_xlim(-margin, len(filtered_demands) - 1 + margin)

    # Set y-limits for the plot
    ax_switching_freq.set_ylim(0, 1.05)  # Provide headroom for the legend

    # Add legend in the top right corner
    ax_switching_freq.legend(handles=[tl_handle, rl_handle], fontsize=fs-4, loc='upper right',
                           bbox_to_anchor=(1.0, 1.01))

    # ========== Add (a), (b), (c) labels centered below each subplot ==========
    # Get the exact position of each subplot after tight_layout
    bbox1 = ax_near_accidents.get_position()
    bbox2 = ax_consecutive_ones.get_position()
    bbox3 = ax_switching_freq.get_position()
    # bcbar = cbar.ax.get_position(fig)

    x1 = 0.17
    x2 = 0.5
    x3 = 0.77

    # Common y position for labels
    label_y = 0.1 # Adjusted slightly from previous attempts

    fig.text(x1, label_y, "(a)", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    fig.text(x2, label_y, "(b)", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    fig.text(x3, label_y, "(c)", ha="center", va="bottom", fontsize=fs, fontweight="bold")

    # ========== Figure-level adjustments ==========
    plt.subplots_adjust(wspace=0.23, bottom=0.1)  # Adjusted bottom margin to make room for labels

    # Save figure
    plt.savefig(_out("consolidated_insights.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()
    return fig

def gmm_to_video():
    """
    """
    pass 

def graph_to_video():
    """
    """
    pass 





### _crop_graph PLACEHOLDER_START
    pos_orig = nx.get_node_attributes(G_orig, "pos")
    if len(pos_orig) != G_orig.number_of_nodes():
        pos_orig = nx.spring_layout(G_orig, seed=42) # Fallback
        nx.set_node_attributes(G_orig, pos_orig, "pos")

    # Calculate original coordinate ranges for jitter scaling
    y_coords_orig = np.array([coord[1] for coord in pos_orig.values()])
    y_range = y_coords_orig.ptp() if len(y_coords_orig) > 1 else 1.0
    jitter_std_dev = y_range * 0.005 # 0.5% of y-range

    low, high = np.percentile(y_coords_orig, [lower, upper])

    nodes_inside = {n for n, (_, y) in pos_orig.items() if low <= y <= high}

    # Start with subgraph of nodes inside the range and edges between them
    H = G_orig.subgraph(nodes_inside).copy()
    pos_H = {n: pos_orig[n] for n in H.nodes()}

    boundary_nodes_data = []
    boundary_node_counter = 0

    # Find edges crossing the boundary in the original graph
    for u, v in G_orig.edges():
        if u in nodes_inside and v in nodes_inside:
            continue # Skip edges fully inside

        uy = pos_orig[u][1]
        vy = pos_orig[v][1]
        u_inside = low <= uy <= high
        v_inside = low <= vy <= high

        if u_inside != v_inside: # Found a crossing edge
            inside_node = u if u_inside else v
            outside_node = v if u_inside else u
            ux, uy = pos_orig[inside_node]
            vx, vy = pos_orig[outside_node]

            y_boundary = -1
            if vy < low:
                y_boundary = low
            elif vy > high:
                y_boundary = high
            else:
                continue # Should not happen

            # Calculate intersection x-coordinate
            x_intersect = ux # Default for vertical
            if abs(vy - uy) > 1e-9: # Avoid division by zero
                if abs(vx - ux) > 1e-9: # Not vertical
                    t = (y_boundary - uy) / (vy - uy)
                    x_intersect = ux + t * (vx - ux)

            boundary_pos = (x_intersect, y_boundary)
            boundary_nodes_data.append((boundary_pos, inside_node))

    # Add unique boundary nodes and connecting edges to H
    processed_boundaries = {} # Cache boundary points: rounded_pos -> node_id
    for boundary_pos, inside_node in boundary_nodes_data:
        x_intersect, y_boundary = boundary_pos
        rounded_pos = (round(x_intersect, 6), round(y_boundary, 6))

        if rounded_pos not in processed_boundaries:
            # Add random vertical jitter
            y_jitter = np.random.normal(0, jitter_std_dev)
            final_y = y_boundary + y_jitter
            final_boundary_pos = (x_intersect, final_y) # Jitter applied to y

            new_boundary_node_id = f"boundary_{boundary_node_counter}"
            boundary_node_counter += 1
            H.add_node(new_boundary_node_id)
            pos_H[new_boundary_node_id] = final_boundary_pos # Store jittered position
            processed_boundaries[rounded_pos] = new_boundary_node_id # Use original rounded pos for lookup
            boundary_node_id = new_boundary_node_id
        else:
            boundary_node_id = processed_boundaries[rounded_pos]

        # Add edge from inside node to the boundary node
        if inside_node in H:
             H.add_edge(inside_node, boundary_node_id)

    return H, pos_H


def plot_gmm_top_down(gmm_pkl_path: str, 
                               location_range: tuple[float, float] = (0.0, 1.05), 
                               thickness_range: tuple[float, float] = (0.0, 1.05), 
                               fs: int = 19, 
                               num_grid_points: int = 100,
                               contour_levels: int = 20):
    """
    Plots the top-down view of a GMM distribution with markers from a .pkl file.
    The output is saved as 'gmm_flat.png'.

    Args:
        gmm_pkl_path (str): Path to the pickle file containing (gmm_object, markers_data).
        location_range (tuple[float, float]): Min and max for the location (x-axis).
        thickness_range (tuple[float, float]): Min and max for the thickness (y-axis).
        fs (int): Base font size.
        num_grid_points (int): Number of points for the grid in each dimension.
        contour_levels (int): Number of levels for the contour plot.
    """
    with open(gmm_pkl_path, "rb") as f:
        gmm_single, markers = pickle.load(f)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()

    label_color = '#202124'
    tick_color = '#5f6368'
    # Return to more subtle grid lines
    grid_color = (0.0, 0.0, 0.0, 0.55)  

    xmin, xmax = location_range
    ymin, ymax = thickness_range
    X_grid = np.linspace(xmin, xmax, num_grid_points)
    Y_grid = np.linspace(ymin, ymax, num_grid_points)
    X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)

    device = gmm_single.component_distribution.loc.device

    positions = torch.tensor(np.column_stack([X_mesh.ravel(), Y_mesh.ravel()]), 
                           dtype=torch.float32,
                           device=device)

    with torch.no_grad():
        Z_log_prob = gmm_single.log_prob(positions).detach().cpu()
    Z = np.exp(Z_log_prob.numpy()).reshape(X_mesh.shape)
    
    z_min = Z.min()
    z_ptp = Z.ptp()
    if z_ptp == 0:
        Z_norm = np.zeros_like(Z) if z_min == 0 else np.ones_like(Z) * (z_min / (z_min + 1e-9) )
    else:
        Z_norm = (Z - z_min) / z_ptp

    # Manual grid lines with low zorder 
    ax.set_axisbelow(True)
    
    # Draw horizontal grid lines with subtler appearance
    grid_y_ticks = np.arange(0.0, 1.1, 0.2)
    for y in grid_y_ticks:
        ax.axhline(y=y, color=grid_color, linestyle=(0, (5, 5)), linewidth=0.5, zorder=-10)  # Back to original linewidth
    
    # Draw vertical grid lines with subtler appearance
    grid_x_ticks = np.arange(0.0, 1.1, 0.2)
    for x in grid_x_ticks:
        ax.axvline(x=x, color=grid_color, linestyle=(0, (5, 5)), linewidth=0.5, zorder=-10)  # Back to original linewidth

    cmap = plt.get_cmap("coolwarm", 256)
    # Adjust contour opacity for better balance
    contour = ax.contourf(X_mesh, Y_mesh, Z_norm, levels=contour_levels, cmap=cmap, alpha=0.85, zorder=1)  # Moderate opacity
    
    cbar = plt.colorbar(contour, ax=ax, shrink=1.0, aspect=20, pad=0.05)
    cbar.set_label('Normalized Density', fontweight='bold', fontsize=fs-2, color=label_color)
    cbar.ax.tick_params(labelsize=fs-2, colors=tick_color)

    if hasattr(gmm_single, 'component_distribution') and hasattr(gmm_single.component_distribution, 'loc'):
        means = gmm_single.component_distribution.loc.detach().cpu().numpy()
        # Keep royal blue as it was requested
        royal_blue = '#0066ff'  # Royal blue hex color
        ax.scatter(means[:, 0], means[:, 1], 
                   c=royal_blue,
                   marker='o', 
                   s=120, 
                   edgecolors='black', 
                   linewidths=0.7, 
                   label='Component Means', 
                   zorder=2)

    if markers is not None:
        locations, thicknesses = markers
        ax.scatter(locations, thicknesses, 
                   c='red',
                   marker='x', 
                   s=120, 
                   label='Samples Drawn', 
                   zorder=3)

        x_range_plot = ax.get_xlim()[1] - ax.get_xlim()[0]
        offset_x = x_range_plot * 0.04

        for i, (loc, thick) in enumerate(zip(locations, thicknesses)):
            ax.text(loc + offset_x, thick, f'C{i+1}',
                    fontsize=fs-5, 
                    ha='left', 
                    va='center',
                    zorder=4)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=2, 
                       frameon=True, fancybox=True, facecolor='white', edgecolor='#cccccc',
                       framealpha=1.0, fontsize=fs-2, borderpad=0.6, labelspacing=0.4)
    legend.set_zorder(10)

    ax.set_xlabel('Location', fontweight='bold', fontsize=fs, color=label_color, labelpad=10)
    ax.set_ylabel('Width', fontweight='bold', fontsize=fs, color=label_color, labelpad=10)
    ax.set_title('GMM Distribution', fontweight='bold', fontsize=fs, color=label_color, pad=15)
    
    ax.tick_params(axis='both', which='major', labelsize=fs-2, colors=tick_color)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(tick_color)
    ax.spines['bottom'].set_color(tick_color)
    
    ax.set_xlim(location_range)
    ax.set_ylim(thickness_range)

    output_filename = _out("gmm_flat.png")
    
    plt.subplots_adjust(bottom=0.2)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"GMM top-down plot saved to {output_filename}")


# plot_gmm_top_down(gmm_pkl_path = "./runs/May09_12-21-15/gmm_iterations/gmm_i_eval14400000_b0_data.pkl")
