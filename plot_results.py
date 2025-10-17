import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import deepinv as dinv
from pathlib import Path

# Configuration du style
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
    """Charge les résultats depuis le dossier d'expérience"""
    results_path = os.path.join(exp_dir, "results.pkl")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results found at {results_path}")

    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    # Reconvertir les numpy arrays en tensors si nécessaire
    for method_name, result in data['results'].items():
        if result['reconstruction'] is not None:
            result['reconstruction'] = torch.from_numpy(result['reconstruction'])

    data['x_true'] = torch.from_numpy(data['x_true'])
    data['y'] = torch.from_numpy(data['y'])

    return data

def plot_all_metrics(data, exp_dir, only_cycles=False):
    """Génère tous les plots de métriques"""
    results = data['results']
    params = data['params']
    multilevel_iter = data['multilevel_iter']
    method_colors = data['method_info']['colors']
    method_linestyles = data['method_info']['linestyles']
    method_markers = data['method_info']['markers']

    # Créer les subplots
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    # 1. Loss vs Iterations
    for method_name, result in results.items():
        if result['loss']:
            axes[0].plot(result['loss'],
                        color=method_colors[method_name],
                        linestyle=method_linestyles[method_name],
                        label=method_name)
            if result['cycles']:
                scatter_small(
                    axes[0],
                    result['cycles'],
                    [result['loss'][i-1] for i in result['cycles']],
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                    label=f"cycles_{method_name}"
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
                cycle_times = [result['times'][i-1] for i in result['cycles']]
                cycle_losses = [result['loss'][i-1] for i in result['cycles']]
                scatter_small(
                    axes[1],
                    cycle_times,
                    cycle_losses,
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                    label=f"cycles_{method_name}"
                )

    axes[1].set(xlabel='CPU time (s)', ylabel='Loss', title='Loss over CPU Time')
    axes[1].legend(frameon=True)
    axes[1].grid(True)

    # 3. PSNR vs Iterations
    for method_name, result in results.items():
        if result['psnr']:
            axes[2].plot(result['psnr'],
                        color=method_colors[method_name],
                        linestyle=method_linestyles[method_name],
                        label=method_name)
            if result['cycles']:
                scatter_small(
                    axes[2],
                    result['cycles'],
                    [result['psnr'][i-1] for i in result['cycles']],
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                    label=f"cycles_{method_name}"
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
                cycle_times = [result['times'][i-1] for i in result['cycles']]
                cycle_psnrs = [result['psnr'][i-1] for i in result['cycles']]
                scatter_small(
                    axes[3],
                    cycle_times,
                    cycle_psnrs,
                    color=method_colors[method_name],
                    marker=method_markers.get(method_name, "o"),
                    label=f"cycles_{method_name}"
                )

    axes[3].set(xlabel='CPU time (s)', ylabel='PSNR (dB)', title='PSNR over CPU Time')
    axes[3].legend(frameon=True)
    axes[3].grid(True)

    # Sauvegarder le plot combiné
    plt.tight_layout()
    combined_path = os.path.join(exp_dir, "all_plots_combined.pdf")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved at: {combined_path}")

    # Sauvegarder chaque plot individuellement
    plot_names = ['loss_vs_iterations', 'loss_vs_time', 'psnr_vs_iterations', 'psnr_vs_time']

    for i, plot_name in enumerate(plot_names):
        fig_individual = plt.figure(figsize=(8, 6))
        ax_individual = fig_individual.add_subplot(111)

        for line in axes[i].get_lines():
            ax_individual.plot(line.get_xdata(), line.get_ydata(),
                              color=line.get_color(),
                              label=line.get_label(),
                              linewidth=line.get_linewidth(),
                              linestyle=line.get_linestyle())

        for collection in axes[i].collections:
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                ax_individual.scatter(offsets[:, 0], offsets[:, 1],
                                     color=collection.get_facecolors()[0],
                                     s=collection.get_sizes()[0])

        ax_individual.set_xlabel(axes[i].get_xlabel())
        ax_individual.set_ylabel(axes[i].get_ylabel())
        ax_individual.set_title(axes[i].get_title())
        ax_individual.legend(frameon=True)
        ax_individual.grid(True)

        plt.tight_layout()
        individual_path = os.path.join(exp_dir, f"{plot_name}.pdf")
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close(fig_individual)
        print(f"Saved: {individual_path}")

    plt.show()

def plot_reconstructions(data, exp_dir):
    """Génère le plot des reconstructions"""
    results = data['results']
    x_true = data['x_true']
    y = data['y']

    images = [x_true, y]
    titles = ["Original", "Observation"]
    subtitles = ["PSNR:", f"{PSNR(y, x_true).item():.2f} dB"]

    for method_name, res in results.items():
        x_rec = res["reconstruction"]
        psnr = res["psnr"]
        if x_rec is not None:
            images.append(x_rec)
            titles.append(f"{method_name}")
            if psnr:
                subtitles.append(f"{psnr[-1]:.2f} dB")

    dinv.utils.plot(
        images,
        titles=titles,
        subtitles=subtitles,
        cmap="gray",
        tight=False,
        save_fn=os.path.join(exp_dir, "all_reconstructions.pdf")
    )
    print(f"Reconstructions saved at: {os.path.join(exp_dir, 'all_reconstructions.pdf')}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <exp_dir>")
        print("Example: python plot_results.py experiments_results/compare_methods/exp_2025-01-15_10-30")
        sys.exit(1)

    exp_dir = sys.argv[1]

    if not os.path.exists(exp_dir):
        print(f"Error: Directory {exp_dir} does not exist")
        sys.exit(1)

    print(f"Loading results from {exp_dir}...")
    data = load_results(exp_dir)

    print("Generating metric plots...")
    plot_all_metrics(data, exp_dir)

    print("Generating reconstruction plots...")
    plot_reconstructions(data, exp_dir)

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()