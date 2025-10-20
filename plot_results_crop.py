import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import deepinv as dinv
from pathlib import Path
import argparse

sns.set_theme()
colors = sns.color_palette("colorblind")

plt.rcParams.update({
    'lines.markersize': 3,
    'lines.linewidth': 2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
})

CYCLE_MARKER_SIZE = 20
PSNR = dinv.metric.PSNR()


def scatter_small(ax, *args, **kwargs):
    kwargs.setdefault("s", CYCLE_MARKER_SIZE)
    return ax.scatter(*args, **kwargs)


def load_results(exp_dir):
    results_path = os.path.join(exp_dir, "results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results found at {results_path}")
    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    for method_name, result in data['results'].items():
        if result['reconstruction'] is not None:
            result['reconstruction'] = torch.from_numpy(result['reconstruction'])
    data['x_true'] = torch.from_numpy(data['x_true'])
    data['y'] = torch.from_numpy(data['y'])
    return data


def crop_results(data, max_len):
    """Coupe les listes (loss, psnr, times, cycles) à max_len éléments"""
    if max_len is None:
        return data
    for method_name, result in data['results'].items():
        for key in ['loss', 'psnr', 'times']:
            if key in result and result[key] is not None:
                result[key] = result[key][:max_len]
        # cycles : on garde uniquement ceux < max_len
        if 'cycles' in result and result['cycles'] is not None:
            result['cycles'] = [c for c in result['cycles'] if c <= max_len]
    return data


def plot_all_metrics(data, exp_dir, max_len=None):
    results = data['results']
    multilevel_iter = data['multilevel_iter']
    method_colors = data['method_info']['colors']
    method_linestyles = data['method_info']['linestyles']
    method_markers = data['method_info']['markers']

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    # 1. Loss vs Iter
    for method_name, result in results.items():
        if result['loss']:
            axes[0].plot(result['loss'],
                         color=method_colors[method_name],
                         linestyle=method_linestyles[method_name],
                         label=method_name)
            if result['cycles']:
                valid_cycles = [i for i in result['cycles'] if i - 1 < len(result['loss'])]
                scatter_small(
                    axes[0],
                    valid_cycles,
                    [result['loss'][i - 1] for i in valid_cycles],
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                )

    axes[0].axvline(x=multilevel_iter, color='red', linestyle='--',
                    label=f"End of ML iterations ({multilevel_iter})")
    axes[0].set(xlabel='Iteration', ylabel='Loss', title='Loss over Iterations')
    axes[0].legend(frameon=True)
    axes[0].grid(True)

    # 2. Loss vs Time
    for method_name, result in results.items():
        if result['loss'] and result['times']:
            axes[1].plot(result['times'], result['loss'],
                         color=method_colors[method_name],
                         linestyle=method_linestyles[method_name],
                         label=method_name)
            if result['cycles']:
                valid_cycles = [i for i in result['cycles'] if i - 1 < len(result['times'])]
                cycle_times = [result['times'][i - 1] for i in valid_cycles]
                cycle_losses = [result['loss'][i - 1] for i in valid_cycles]
                scatter_small(
                    axes[1],
                    cycle_times,
                    cycle_losses,
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                )

    axes[1].set(xlabel='CPU time (s)', ylabel='Loss', title='Loss over CPU Time')
    axes[1].legend(frameon=True)
    axes[1].grid(True)

    # 3. PSNR vs Iter
    for method_name, result in results.items():
        if result['psnr']:
            axes[2].plot(result['psnr'],
                         color=method_colors[method_name],
                         linestyle=method_linestyles[method_name],
                         label=method_name)
            if result['cycles']:
                valid_cycles = [i for i in result['cycles'] if i - 1 < len(result['psnr'])]
                scatter_small(
                    axes[2],
                    valid_cycles,
                    [result['psnr'][i - 1] for i in valid_cycles],
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                )

    axes[2].axvline(x=multilevel_iter, color='red', linestyle='--',
                    label=f"End of ML iterations ({multilevel_iter})")
    axes[2].set(xlabel='Iteration', ylabel='PSNR (dB)', title='PSNR over Iterations')
    axes[2].legend(frameon=True)
    axes[2].grid(True)

    # 4. PSNR vs Time
    for method_name, result in results.items():
        if result['psnr'] and result['times']:
            axes[3].plot(result['times'], result['psnr'],
                         color=method_colors[method_name],
                         linestyle=method_linestyles[method_name],
                         label=method_name)
            if result['cycles']:
                valid_cycles = [i for i in result['cycles'] if i - 1 < len(result['times'])]
                cycle_times = [result['times'][i - 1] for i in valid_cycles]
                cycle_psnrs = [result['psnr'][i - 1] for i in valid_cycles]
                scatter_small(
                    axes[3],
                    cycle_times,
                    cycle_psnrs,
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                )

    axes[3].set(xlabel='CPU time (s)', ylabel='PSNR (dB)', title='PSNR over CPU Time')
    axes[3].legend(frameon=True)
    axes[3].grid(True)

    plt.tight_layout()
    suffix = f"_crop{max_len}" if max_len else ""
    combined_path = os.path.join(exp_dir, f"all_plots_combined{suffix}.pdf")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved at: {combined_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot results with optional cropping.")
    parser.add_argument("exp_dir", help="Dossier de l'expérience à tracer")
    parser.add_argument("--crop", "-c", type=int, default=None,
                        help="Coupe les courbes à N points maximum")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    max_len = args.crop

    if not os.path.exists(exp_dir):
        print(f"Error: Directory {exp_dir} does not exist")
        sys.exit(1)

    print(f"Loading results from {exp_dir}...")
    data = load_results(exp_dir)
    data = crop_results(data, max_len)

    print("Generating metric plots...")
    plot_all_metrics(data, exp_dir, max_len=max_len)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()