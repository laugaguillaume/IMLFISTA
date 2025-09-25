import deepinv as dinv
import torch
import time
import numpy as np
import pywt
import matplotlib.pyplot as plt
from pathlib import Path
import platform
import seaborn as sns

import os
import json
from datetime import datetime

from block.block import BlockCoordinateDescent

# Plot settings
sns.set_theme()
sns.color_palette("colorblind")
colors = sns.color_palette("colorblind")

PSNR = dinv.metric.PSNR()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

x_true = dinv.utils.load_example("butterfly.png", device=device)

# Wavelet parameters
J = 3
wv_type = 'haar'

# Physics
sigma = 0.01
noise_model = dinv.physics.GaussianNoise(sigma=sigma)
physics = dinv.physics.Inpainting(tensor_size=x_true.shape[1:], mask=0.7, device=device, noise_model=noise_model)
seed = torch.manual_seed(0)  # Random seed for reproducibility

# Observation
y = physics(x_true)

dinv.utils.plot([x_true, y], titles=['Original', 'Observation'], cmap='gray')

# Objective function
data_fidelity = dinv.optim.L2()
prior = dinv.optim.L1Prior()
reg_weight = 1e-4

# Parameters
n_iter = 100  # Réduire pour faire des tests plus rapides
Anorm2 = physics.compute_norm(x_true).item()
stepsize = 0.2/Anorm2
update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'
print(f"Stepsize: {stepsize}")

# Créer les répertoires d'expérience
cbp = True
if not cbp:
    if platform.system() == "Darwin":
        EXPERIMENTS_ROOT = Path("/Users/edgardesainte-mareville/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/blocks")
    else:
        EXPERIMENTS_ROOT = Path("/home/edgar/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/blocks")
else:
    EXPERIMENTS_ROOT = Path(__file__).resolve().parent / "../experiments_results/blocks"

EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
exp_name = f"exp_comparison_{timestamp}_J{J}_mode{update_mode}_reg{reg_weight}_niter{n_iter}_sigma{sigma}"
exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
os.makedirs(exp_dir, exist_ok=True)

# Paramètres de l'expérience
params = {
    "image": "butterfly.png",
    "physics": "Inpainting + Gaussian noise",
    "sigma": sigma,
    "reg_weight": reg_weight,
    "n_iter": n_iter,
    "stepsize": stepsize,
    "J": J,
    "wavelet": wv_type,
    "update_mode": update_mode,
    "comment": "Comparison between optimized and non-optimized BCD"
}
params_path = os.path.join(exp_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=4)

x0 = y.clone()

print("=" * 60)
print("COMPARAISON OPTIMISÉE vs NON-OPTIMISÉE")
print("=" * 60)

# Version NON-OPTIMISÉE (ton code original)
print("\n1. Running BCD NON-OPTIMISÉE...")
bcd_original = BlockCoordinateDescent(
    x_true.shape, wv_type=wv_type, physics=physics,
    data_fidelity=data_fidelity, prior=prior,
    max_levels=J, stepsize=stepsize, device=device
)

start_time_orig = time.time()
x_recon_orig, loss_orig, times_orig, cycles_orig, psnr_orig = bcd_original.run(
    y, x0.clone(), x_true=x_true, n_iter=n_iter, n_iter_coarse=10,
    reg_weight=reg_weight, update_mode=update_mode, metrics=True
)
end_time_orig = time.time()
total_time_orig = end_time_orig - start_time_orig

# Version OPTIMISÉE
print("\n2. Running BCD OPTIMISÉE...")
bcd_optimized = BlockCoordinateDescent(
    x_true.shape, wv_type=wv_type, physics=physics,
    data_fidelity=data_fidelity, prior=prior,
    max_levels=J, stepsize=stepsize, device=device
)

start_time_opt = time.time()
x_recon_opt, loss_opt, times_opt, cycles_opt, psnr_opt = bcd_optimized.run_optimized(
    y, x0.clone(), x_true=x_true, n_iter=n_iter, n_iter_coarse=10,
    reg_weight=reg_weight, update_mode=update_mode, metrics=True
)
end_time_opt = time.time()
total_time_opt = end_time_opt - start_time_opt

# Version FB pour référence
print("\n3. Running FB pour référence...")
n_iter_fb = min(len(loss_orig), len(loss_opt))
x_recon_fb, loss_fb, times_fb, psnr_fb = bcd_original.FB(
    y, num_iterations=n_iter_fb, reg_weight=reg_weight, metrics=True
)

# Calcul des métriques de performance
print("\n" + "=" * 60)
print("RÉSULTATS DE PERFORMANCE")
print("=" * 60)

speedup = total_time_orig / total_time_opt
psnr_final = {
    'observation': PSNR(y, x_true).item(),
    'FB': PSNR(x_recon_fb, x_true).item(),
    'BCD_original': PSNR(x_recon_orig, x_true).item(),
    'BCD_optimized': PSNR(x_recon_opt, x_true).item(),
}

# Différence de reconstruction (doit être quasi-nulle si les algos sont équivalents)
reconstruction_diff = torch.norm(x_recon_orig - x_recon_opt).item()
max_loss_diff = max([abs(l1 - l2) for l1, l2 in zip(loss_orig, loss_opt)])

print(f"Temps total NON-OPTIMISÉE : {total_time_orig:.2f}s")
print(f"Temps total OPTIMISÉE     : {total_time_opt:.2f}s")
print(f"Speedup                   : {speedup:.2f}x")
print(f"")
print(f"PSNR Observation         : {psnr_final['observation']:.2f} dB")
print(f"PSNR FB                   : {psnr_final['FB']:.2f} dB")
print(f"PSNR BCD NON-OPTIMISÉE    : {psnr_final['BCD_original']:.2f} dB")
print(f"PSNR BCD OPTIMISÉE        : {psnr_final['BCD_optimized']:.2f} dB")
print(f"")
print(f"Différence reconstruction : {reconstruction_diff:.2e} (doit être ~0)")
print(f"Différence loss max       : {max_loss_diff:.2e} (doit être ~0)")

# Sauvegarder les métriques
performance_metrics = {
    "total_time_original": total_time_orig,
    "total_time_optimized": total_time_opt,
    "speedup": speedup,
    "psnr_final": psnr_final,
    "reconstruction_difference": reconstruction_diff,
    "max_loss_difference": max_loss_diff,
    "iterations": n_iter,
    "final_losses": {
        "original": loss_orig[-1],
        "optimized": loss_opt[-1],
        "fb": loss_fb[-1]
    }
}

with open(os.path.join(exp_dir, "performance_metrics.json"), "w") as f:
    json.dump(performance_metrics, f, indent=4)

# Graphiques de comparaison
psnrs_display = [f"{p:.2f}" for p in [
    psnr_final['observation'], psnr_final['FB'],
    psnr_final['BCD_original'], psnr_final['BCD_optimized']
]]

# Créer une figure avec 6 sous-graphiques
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot 1: Loss vs iterations
axes[0].plot(loss_orig, color=colors[0], label="BCD NON-OPTIMISÉE", linewidth=2)
axes[0].plot(loss_opt, color=colors[1], label="BCD OPTIMISÉE", linewidth=2, linestyle='--')
axes[0].plot(loss_fb, color=colors[2], label="FB", linewidth=2)
axes[0].scatter(cycles_orig, [loss_orig[i-1] for i in cycles_orig if i-1 < len(loss_orig)],
            color=colors[0], marker="o", s=60, alpha=0.7)
axes[0].scatter(cycles_opt, [loss_opt[i-1] for i in cycles_opt if i-1 < len(loss_opt)],
            color=colors[1], marker="x", s=60, alpha=0.7)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss vs Iterations")
axes[0].legend(frameon=True)
axes[0].set_yscale('log')

# Plot 2: Loss vs temps
axes[1].plot(times_orig, loss_orig, color=colors[0], label='BCD NON-OPTIMISÉE', linewidth=2)
axes[1].plot(times_opt, loss_opt, color=colors[1], label='BCD OPTIMISÉE', linewidth=2, linestyle='--')
axes[1].plot(times_fb, loss_fb, color=colors[2], label='FB', linewidth=2)
axes[1].set_xlabel('Temps CPU (s)')
axes[1].set_ylabel('Loss')
axes[1].set_title(f'Loss vs Temps\nSpeedup: {speedup:.2f}x')
axes[1].legend(frameon=True)
axes[1].set_yscale('log')

# Plot 3: PSNR vs iterations
axes[2].plot(psnr_orig, color=colors[0], label="BCD NON-OPTIMISÉE", linewidth=2)
axes[2].plot(psnr_opt, color=colors[1], label="BCD OPTIMISÉE", linewidth=2, linestyle='--')
axes[2].plot(psnr_fb, color=colors[2], label="FB", linewidth=2)
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("PSNR (dB)")
axes[2].set_title("PSNR vs Iterations")
axes[2].legend(frameon=True)

# Plot 4: PSNR vs temps
axes[3].plot(times_orig, psnr_orig, color=colors[0], label='BCD NON-OPTIMISÉE', linewidth=2)
axes[3].plot(times_opt, psnr_opt, color=colors[1], label='BCD OPTIMISÉE', linewidth=2, linestyle='--')
axes[3].plot(times_fb, psnr_fb, color=colors[2], label='FB', linewidth=2)
axes[3].set_xlabel('Temps CPU (s)')
axes[3].set_ylabel('PSNR (dB)')
axes[3].set_title('PSNR vs Temps')
axes[3].legend(frameon=True)

# Plot 5: Différence des losses
loss_diff = [abs(l1 - l2) for l1, l2 in zip(loss_orig, loss_opt)]
axes[4].semilogy(loss_diff, color=colors[3], linewidth=2)
axes[4].set_xlabel("Iteration")
axes[4].set_ylabel("Différence absolue des losses")
axes[4].set_title(f"Vérification: |Loss_orig - Loss_opt|\nMax diff: {max_loss_diff:.2e}")
axes[4].grid(True, alpha=0.3)

# Plot 6: Temps par itération (histogramme)
time_per_iter_orig = [times_orig[i+1] - times_orig[i] for i in range(len(times_orig)-1)]
time_per_iter_opt = [times_opt[i+1] - times_opt[i] for i in range(len(times_opt)-1)]

avg_time_orig = np.mean(time_per_iter_orig)
avg_time_opt = np.mean(time_per_iter_opt)

axes[5].bar(['NON-OPTIMISÉE', 'OPTIMISÉE'], [avg_time_orig, avg_time_opt],
            color=[colors[0], colors[1]], alpha=0.7)
axes[5].set_ylabel('Temps moyen par itération (s)')
axes[5].set_title(f'Temps par itération\nGain: {avg_time_orig/avg_time_opt:.2f}x')

# Ajouter les valeurs sur les barres
axes[5].text(0, avg_time_orig + avg_time_orig*0.05, f'{avg_time_orig:.4f}s',
            ha='center', va='bottom')
axes[5].text(1, avg_time_opt + avg_time_opt*0.05, f'{avg_time_opt:.4f}s',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(exp_dir, "comparison_plots.pdf"), bbox_inches='tight')
plt.show()

# Images de reconstruction
dinv.utils.plot([
    x_true, y, x_recon_fb, x_recon_orig, x_recon_opt,
    torch.abs(x_recon_orig - x_recon_opt) * 10  # Différence amplifiée
],
titles=[
    'Original',
    f'Observation\nPSNR: {psnrs_display[0]}',
    f'FB\nPSNR: {psnrs_display[1]}',
    f'BCD NON-OPT\nPSNR: {psnrs_display[2]}',
    f'BCD OPTIMISÉE\nPSNR: {psnrs_display[3]}',
    f'Différence x10\n||diff||: {reconstruction_diff:.2e}'
],
cmap='gray',
save_fn=os.path.join(exp_dir, "reconstruction_comparison.pdf")
)

print(f"\nRésultats sauvegardés dans : {exp_dir}")
print("Fichiers générés :")
print("  - params.json : Paramètres de l'expérience")
print("  - performance_metrics.json : Métriques de performance détaillées")
print("  - comparison_plots.pdf : Graphiques de comparaison")
print("  - reconstruction_comparison.pdf : Images de reconstruction")