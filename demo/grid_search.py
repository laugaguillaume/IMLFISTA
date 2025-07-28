"""
Grid Search for param_regularization
"""

# pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
import deepinv as dinv
import torch
from torchvision.io import read_image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import copy
import time
from deepinv.loss.metric import PSNR
from deepinv.models import Denoiser
from multilevel.multilevel import ParametersMultilevel, MultiLevelWavelets, MultiLevel, WaveletDenoiserConditional

perf_psnr = PSNR()

plt.rcParams["text.usetex"] = True  # Activate LaTeX rendering

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")

# Load image
file_name = "butterfly.png"
url = f"https://huggingface.co/datasets/deepinv/images/resolve/main/{file_name}?download=true"
x_true = dinv.utils.load_url_image(url=url, img_size=256).to(device)

# Define the Forward Operator: study case of deblurring + Gaussian noise
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")
seed = torch.manual_seed(0)  # Random seed for reproducibility

sigma = 0.01
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Construct observation
y = physics(x_true)
back = physics.A_adjoint(y)

# Define data fidelity term
data_fidelity = dinv.optim.L2()

# Define prior
args_prior = "TV"
# args_prior = "Wavelet"

# Define single level algorithm
args_algo = "FISTA"
# args_algo = "FB"

if args_prior == "TV":
    criterion = 1e-5
    n_it_max = 50
    prior = dinv.optim.TVPrior(def_crit=criterion, n_it_max=n_it_max)
    denoiser = prior.prox
elif args_prior == "Wavelet":
    prior = dinv.optim.WaveletPrior(level=4, wv="haar", p=1, device=device)
    denoiser = prior.prox

# Define algorithm parameters
random_tensor = torch.randn(x_true.shape).to(device)
Anorm2 = physics.compute_norm(random_tensor)
param_gamma = torch.ones(1, device=device) / Anorm2  # Set the step-size

if args_algo == "FISTA":
    d = 1
    param_gamma = 0.95 * param_gamma
    param_gamma_ML = 1.95 * param_gamma
elif args_algo == "FB":
    d = 0
    param_gamma = 1.95 * param_gamma
    param_gamma_ML = param_gamma

param_iter = 200  # Reduced number of iterations for grid search
a = 2.1  # inertia parameter

# Define multilevel parameters
levels = 4  # number of levels
param_coarse_iter = 5  # number of iterations at coarse level
max_multilevel_iter = 5  # maximum number of multilevel iterations at fine level
cst_grad = None  # only used at coarser levels. stays none at fine level.
info_transfer = "daubechies8"  # type of information transfer

# ============= GRID SEARCH PARAMETERS =============
# Define range of regularization parameters to test
#param_reg_values = [1e-7]
# Alternative: logarithmic spacing
param_reg_values = np.logspace(-7, 2, 10)

print(f"Testing regularization parameters: {param_reg_values}")

# Storage for results
results = {
    'ML': {},
    'SL': {},
    'ML_cond': {}
}

def run_algorithm(param_regularization, algorithm_type='ML'):
    """
    Run a single algorithm with given regularization parameter
    
    Args:
        param_regularization: regularization parameter value
        algorithm_type: 'ML', 'SL', or 'ML_cond'
    
    Returns:
        tuple: (psnr_values, final_psnr, final_image)
    """
    # Initialize
    xk = back.clone()
    zk = back.clone()
    psnr_values = np.zeros(param_iter)
    
    # Setup multilevel parameters if needed
    if algorithm_type in ['ML', 'ML_cond']:
        args_multilevel = ParametersMultilevel(
            target_shape=x_true.shape[-3:],
            levels=levels,
            max_ML_steps=1,
            param_coarse_iter=param_coarse_iter,
            step_size=param_gamma_ML,
            info_transfer=info_transfer,
            prior=prior,
            denoiser=denoiser,
            data_fidelity=data_fidelity,
            physics=physics,
            observation=y,
            device=device,
        )
    
    with torch.no_grad():
        for k in range(param_iter):
            xk_prev = xk.clone()
            
            if algorithm_type == 'ML':
                # Multilevel FISTA
                if k < max_multilevel_iter:
                    zk = MultiLevel(
                        zk, levels, levels - 1, args_multilevel,
                        param_regularization, cst_grad, device,
                    )
                xk = zk - param_gamma * data_fidelity.grad(zk, y, physics)
                if isinstance(prior, dinv.optim.TVPrior):
                    xk = prior.prox(xk, gamma=param_gamma * param_regularization)
                else:
                    xk = prior.prox(xk, gamma=[param_gamma * param_regularization])
                    
            elif algorithm_type == 'ML_cond':
                # Multilevel with conditional denoiser
                if k < max_multilevel_iter:
                    zk = MultiLevelWavelets(
                        zk, levels, levels - 1, args_multilevel,
                        param_regularization, cst_grad, device,
                    )
                xk = zk - param_gamma * data_fidelity.grad(zk, y, physics)
                denoiser_cond = WaveletDenoiserConditional(level=levels, wv="db8", device=device, non_linearity="soft")
                xk = denoiser_cond(xk, gamma=param_regularization * param_gamma)
                
            elif algorithm_type == 'SL':
                # Single level
                xk = zk - param_gamma * data_fidelity.grad(zk, y, physics)
                if isinstance(prior, dinv.optim.TVPrior):
                    xk = prior.prox(xk, gamma=param_gamma * param_regularization)
                else:
                    xk = prior.prox(xk, gamma=[param_gamma * param_regularization])
            
            psnr_values[k] = perf_psnr(x_true, xk).item()
            
            # Update momentum term
            if d == 0:
                zk = xk
            else:
                zk = xk + (((k + a) / a) ** d - 1) / ((k + 1 + a) / a) ** d * (xk - xk_prev)
    
    return psnr_values, psnr_values[-1], xk.clone()

# ============= RUN GRID SEARCH =============
print("Starting grid search...")

for i, param_reg in enumerate(param_reg_values):
    print(f"\nTesting param_regularization = {param_reg} ({i+1}/{len(param_reg_values)})")
    
    # Test ML algorithm
    print("  Running ML...")
    psnr_ml, final_psnr_ml, final_img_ml = run_algorithm(param_reg, 'ML')
    results['ML'][param_reg] = {
        'psnr_curve': psnr_ml,
        'final_psnr': final_psnr_ml,
        'final_image': final_img_ml
    }
    
    # Test SL algorithm
    print("  Running SL...")
    psnr_sl, final_psnr_sl, final_img_sl = run_algorithm(param_reg, 'SL')
    results['SL'][param_reg] = {
        'psnr_curve': psnr_sl,
        'final_psnr': final_psnr_sl,
        'final_image': final_img_sl
    }
    
    # Test ML with conditional denoiser
    print("  Running ML with conditional denoiser...")
    psnr_ml_cond, final_psnr_ml_cond, final_img_ml_cond = run_algorithm(param_reg, 'ML_cond')
    results['ML_cond'][param_reg] = {
        'psnr_curve': psnr_ml_cond,
        'final_psnr': final_psnr_ml_cond,
        'final_image': final_img_ml_cond
    }
    
    print(f"  Final PSNRs - ML: {final_psnr_ml:.3f}, SL: {final_psnr_sl:.3f}, ML_cond: {final_psnr_ml_cond:.3f}")

print("\nGrid search completed!")

# ============= PLOT RESULTS =============

# Plot 1: PSNR curves for different regularization parameters (ML algorithm)
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(param_reg_values)))

for i, param_reg in enumerate(param_reg_values):
    plt.plot(results['ML'][param_reg]['psnr_curve'], 
             color=colors[i], 
             label=f'lambda = {param_reg:.1e}',
             linewidth=2)

plt.title('PSNR Evolution for Different Regularization Parameters (ML Algorithm)')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot 2: Comparison of final PSNRs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

final_psnrs_ml = [results['ML'][param_reg]['final_psnr'] for param_reg in param_reg_values]
final_psnrs_sl = [results['SL'][param_reg]['final_psnr'] for param_reg in param_reg_values]
final_psnrs_ml_cond = [results['ML_cond'][param_reg]['final_psnr'] for param_reg in param_reg_values]

# Linear scale
ax1.plot(param_reg_values, final_psnrs_ml, 'o-', label='ML', linewidth=2, markersize=8)
ax1.plot(param_reg_values, final_psnrs_sl, 's--', label='SL', linewidth=2, markersize=8)
ax1.plot(param_reg_values, final_psnrs_ml_cond, '^:', label='ML cond', linewidth=2, markersize=8)
ax1.set_xlabel('Regularization Parameter')
ax1.set_ylabel('Final PSNR (dB)')
ax1.set_title('Final PSNR vs Regularization Parameter')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Log scale
ax2.semilogx(param_reg_values, final_psnrs_ml, 'o-', label='ML', linewidth=2, markersize=8)
ax2.semilogx(param_reg_values, final_psnrs_sl, 's--', label='SL', linewidth=2, markersize=8)
ax2.semilogx(param_reg_values, final_psnrs_ml_cond, '^:', label='ML cond', linewidth=2, markersize=8)
ax2.set_xlabel('Regularization Parameter (log scale)')
ax2.set_ylabel('Final PSNR (dB)')
ax2.set_title('Final PSNR vs Regularization Parameter (Log Scale)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# Plot 3: Best reconstruction images
best_param_ml = max(param_reg_values, key=lambda x: results['ML'][x]['final_psnr'])
best_param_sl = max(param_reg_values, key=lambda x: results['SL'][x]['final_psnr'])
best_param_ml_cond = max(param_reg_values, key=lambda x: results['ML_cond'][x]['final_psnr'])

print(f"\nBest regularization parameters:")
print(f"ML: {best_param_ml:.1e} (PSNR: {results['ML'][best_param_ml]['final_psnr']:.3f} dB)")
print(f"SL: {best_param_sl:.1e} (PSNR: {results['SL'][best_param_sl]['final_psnr']:.3f} dB)")
print(f"ML_cond: {best_param_ml_cond:.1e} (PSNR: {results['ML_cond'][best_param_ml_cond]['final_psnr']:.3f} dB)")

# Display best reconstructions
best_psnrs = np.array([
    perf_psnr(x_true, y).item(),
    results['ML'][best_param_ml]['final_psnr'],
    results['SL'][best_param_sl]['final_psnr'],
    results['ML_cond'][best_param_ml_cond]['final_psnr']
])
best_psnrs = np.round(best_psnrs, 3)

dinv.utils.plot(
    [x_true, y, 
     results['ML'][best_param_ml]['final_image'],
     results['SL'][best_param_sl]['final_image'],
     results['ML_cond'][best_param_ml_cond]['final_image']],
    titles=[
        "Original",
        f"Observation\nPSNR: {best_psnrs[0]} dB",
        f"Best ML (lambda={best_param_ml:.1e})\nPSNR: {best_psnrs[1]} dB",
        f"Best SL (lambda={best_param_sl:.1e})\nPSNR: {best_psnrs[2]} dB",
        f"Best ML_cond (lambda={best_param_ml_cond:.1e})\nPSNR: {best_psnrs[3]} dB"
    ],
    cmap="gray",
)

# Plot 4: PSNR evolution comparison for best parameters
plt.figure(figsize=(12, 8))
plt.plot(results['ML'][best_param_ml]['psnr_curve'], '-', linewidth=2, 
         label=f'ML (lambda={best_param_ml:.1e})')
plt.plot(results['SL'][best_param_sl]['psnr_curve'], '--', linewidth=2, 
         label=f'SL (lambda={best_param_sl:.1e})')
plt.plot(results['ML_cond'][best_param_ml_cond]['psnr_curve'], ':', linewidth=2, 
         label=f'ML_cond (lambda={best_param_ml_cond:.1e})')

plt.title('PSNR Evolution for Best Regularization Parameters')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\nGrid search analysis completed!")