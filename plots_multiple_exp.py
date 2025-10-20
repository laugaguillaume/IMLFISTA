import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.rcParams.update({
    'lines.linewidth': 2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
})

def load_results(exp_dir):
    path = os.path.join(exp_dir, "results.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results.pkl found in {exp_dir}")
    with open(path, "rb") as f:
        return pickle.load(f)

def compare_experiments(exp_dirs, labels=None):
    if labels is None:
        labels = [os.path.basename(e) for e in exp_dirs]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    colors = sns.color_palette("colorblind", len(exp_dirs))

    for exp_dir, label, color in zip(exp_dirs, labels, colors):
        data = load_results(exp_dir)
        results = data["results"]

        # suppose qu’il y a une seule méthode par expérience
        # sinon tu peux boucler sur results.items()
        method_name = list(results.keys())[0]
        res = results[method_name]

        # Loss vs Iter
        if res["loss"]:
            axes[0].plot(res["loss"], label=label, color=color)

        # Loss vs Time
        if res["loss"] and res["times"]:
            axes[1].plot(res["times"], res["loss"], label=label, color=color)

        # PSNR vs Iter
        if res["psnr"]:
            axes[2].plot(res["psnr"], label=label, color=color)

        # PSNR vs Time
        if res["psnr"] and res["times"]:
            axes[3].plot(res["times"], res["psnr"], label=label, color=color)

    titles = [
        "Loss over Iterations",
        "Loss over CPU Time",
        "PSNR over Iterations",
        "PSNR over CPU Time",
    ]
    xlabels = ["Iteration", "CPU Time (s)", "Iteration", "CPU Time (s)"]
    ylabels = ["Loss", "Loss", "PSNR (dB)", "PSNR (dB)"]

    for ax, title, xlabel, ylabel in zip(axes, titles, xlabels, ylabels):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("compare_experiments.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved combined comparison as compare_experiments.pdf")

if __name__ == "__main__":
    # Exemple d'utilisation :
    exp_dirs = ["exp1", "exp2", "exp3", "exp4"]
    compare_experiments(exp_dirs, labels=["Exp 1", "Exp 2", "Exp 3", "Exp 4"])
