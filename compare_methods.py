import os
import platform
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepinv as dinv
import time

from block.block import BlockCoordinateDescent
from multilevel.multilevel import ParametersMultilevel, MultiLevel, MultiLevelWavelets
from multilevel.multilevel_initialization import ml_init_pnp

PSNR = dinv.metric.PSNR()

device = torch.device('cpu')
x_true = dinv.utils.load_example("butterfly.png", device=device)

#%%------ MODEL -----%%
# Physics
blur_variance = (2, 2)
filter_0 = dinv.physics.blur.gaussian_blur(sigma=blur_variance, angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")
seed = torch.manual_seed(0)  # Random seed for reproducibility

sigma = 0.01
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Observation
y = physics(x_true)

# Objective function
data_fidelity = dinv.optim.L2()
prior_type = "TV"  # "TV", "L1", "L1_wavelet"


#%%------ PARAMETERS -----%%
n_iter = 100
reg_weight = 1e-2
Anorm2 = physics.compute_norm(x_true).item()
stepsize = 0.1/Anorm2

J = 3
wv_type = 'haar'

# For multilevel algorithms
multilevel_iter = 5 # Number of multilevel iterations at the fine level
n_coarse_steps = 5  # Number of coarse steps per level in each multilevel iteration

# For BCD algorithms
update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'

#%%----- INITIALIZE ALGORITHMS ARGUMENTS -----%%

# Plug and Play denoiser
denoiser_0 = dinv.models.DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
denoiser = dinv.models.EquivariantDenoiser(denoiser_0, random=True)

if prior_type == "L1":
    prior = dinv.optim.L1Prior()
elif prior_type == "TV":
    prior = dinv.optim.TVPrior(n_it_max=50)
elif prior_type == "L1_wavelet":
    prior = dinv.optim.WaveletPrior(level=J, wv=wv_type, p=1, device=device)

# Block coordinate descent setup
bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize)

# Multilevel parameters
cst_grad = None

args_multilevel = ParametersMultilevel(
    target_shape=x_true.shape[-3:],
    levels=J,
    max_ML_steps=1,
    param_coarse_iter=n_coarse_steps, # Number of coarse iterations
    step_size=stepsize,
    info_transfer=wv_type,
    prior=prior,
    denoiser=denoiser,
    data_fidelity=data_fidelity,
    physics=physics,
    observation=y,
    device=device,
)

#%%----- Experiment directory setup to save parameters and figures -----%%
if platform.system() == "Darwin":  # macOS
    EXPERIMENTS_ROOT = "/Users/edgardesainte-mareville/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/compare_methods"
else:  # Linux ou autre
    EXPERIMENTS_ROOT = "/home/edgar/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/compare_methods"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
exp_name = f"exp_{timestamp}_J{J}_mode{update_mode}_reg{reg_weight}_niter{n_iter}_sigma{sigma}"
exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
os.makedirs(exp_dir, exist_ok=True)

params = {
    "image": "butterfly.png",
    "physics": f'Blur (variance {blur_variance}) + Gaussian noise (sigma={sigma})',
    "prior": prior_type,
    "sigma": sigma,
    "reg_weight": reg_weight,
    "n_iter": n_iter,
    "multilevel_iter": multilevel_iter,
    "stepsize": stepsize,
    "J": J,
    "wavelet": wv_type,
    "update_mode": update_mode,
    "comment": "With approx thresholding in BCD"
}
params_path = os.path.join(exp_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=4)

#%%%--- Reconstruction %%%---

def run_FB(x0, y, x_true=None, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    loss, psnr, times = [], [], []
    start = time.process_time()

    stepsizeATy = params['stepsize'] * physics.A_adjoint(y)

    xk = x0.clone()
    for k in range(params['n_iter']):
        xk = xk - params['stepsize'] * (physics.A_adjoint(physics.A(xk))) + stepsizeATy
        xk = prior.prox(xk, gamma=params['stepsize'] * params['reg_weight'])

        if x_true is not None:
            current_psnr = PSNR(xk, x_true).item()
            psnr.append(current_psnr)
        current_loss =  data_fidelity.fn(xk, y, physics).item() + reg_weight * prior.fn(xk).item()
        loss.append(current_loss)
        times.append(time.process_time() - start)

    return xk, loss, psnr, times

def run_MLFB(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params, args_multilevel=args_multilevel):
    loss, psnr, times = [], [], []
    xk = x0.clone()
    levels = params['J']
    param_regularization = params['reg_weight']
    param_gamma = params['stepsize']

    start = time.time()
    with torch.no_grad():
        for k in range(params['n_iter']):
            #xk_prev = xk.clone()

            if k < params['multilevel_iter']:
                xk = MultiLevel(
                    xk,
                    levels,
                    levels - 1,
                    args_multilevel,
                    param_regularization,
                    cst_grad,
                    device,
                )
            xk = xk - param_gamma * data_fidelity.grad(xk, y, physics)
            if isinstance(prior, dinv.optim.TVPrior):
                xk = prior.prox(xk, gamma=param_gamma * param_regularization)
            else:
                xk = prior.prox(xk, gamma=[param_gamma * param_regularization])

            current_loss = data_fidelity(
                xk, y, physics
            ) + param_regularization * prior.fn(xk)
            loss.append(current_loss.item())
            times.append(time.time() - start)
            current_psnr = PSNR(xk, x_true).item()
            psnr.append(current_psnr)

            if k % 10 == 0:
                print(f"crit ML[{k}] / snr ML[{k}]: {loss[k]} / {psnr[k]}")

    recon = xk.clone()
    return recon, loss, psnr, times

def run_BCD(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize)
    recon, loss, times = bcd.run(y, xk, x_true=x_true, n_iter=params['n_iter'], reg_weight=params['reg_weight'], update_mode='MLFB', metrics=True)
    return recon, loss, psnr, times

def run_PnP(x0, y, x_true=None, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params, denoiser=denoiser):
    loss, psnr, times = None, [], []
    start = time.process_time()

    stepsizeATy = params['stepsize'] * physics.A_adjoint(y)

    xk = x0.clone()
    for k in range(params['n_iter']):
        xk = xk - params['stepsize'] * (physics.A_adjoint(physics.A(xk))) + stepsizeATy
        xk = denoiser(xk, sigma=params['reg_weight'])

        if x_true is not None:
            current_psnr = PSNR(xk, x_true).item()
            psnr.append(current_psnr)
        times.append(time.process_time() - start)

    return xk, loss, psnr, times

def run_MLPnP(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_MLFBcond(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_BCDcond(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize)
    recon, loss, times = bcd.run(y, xk, x_true=x_true, n_iter=params['n_iter'], reg_weight=params['reg_weight'], update_mode='MLFBcond', metrics=True)
    return recon, loss, psnr, times


#%%--- Run methods %%%---
x0 = y.clone()

methods = {
    "FB": run_FB,
    "MLFB": run_MLFB,
    "BCD": run_BCD,
    "PnP": run_PnP,
    #"MLPnP": run_MLPnP,
    #"MLFBcond": run_MLFBcond,
    "BCDcond": run_BCDcond
}

results = {}
for method_name, method_func in methods.items():
    print(f"Running {method_name}...")
    params['update_mode'] = method_name
    x_rec, loss, psnr, times = method_func(x0, y, x_true=x_true)
    results[method_name] = {
        "reconstruction": x_rec,
        "loss": loss,
        "psnr": psnr,
        "times": times
    }
    if psnr:
        print(f"Final PSNR for {method_name}: {psnr[-1]:.2f} dB")
    else:
        print(f"No PSNR computed for {method_name}")
    if x_rec is not None:
        dinv.utils.plot(
            [x_true, y, x_rec],
            titles=['Original', 'Observation', f'{method_name} Reconstruction'],
            cmap='gray',
            save_fn=os.path.join(exp_dir, f"{method_name}_reconstruction.pdf")
        )
    else:
        print(f"No reconstruction available for {method_name}")

#%%%--- Plotting results %%%---

# PSNR vs Iterations
plt.figure(figsize=(10, 6))
for method_name, result in results.items():
    if result['psnr']:
        plt.plot(result['psnr'], label=method_name)
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Iteration for Different Methods')
plt.legend()
plt.grid()
plt.savefig(os.path.join(exp_dir, "psnr_iter_comparison.pdf"))
plt.show()

# Loss vs Iterations
plt.figure(figsize=(10, 6))
for method_name, result in results.items():
    if result['loss']:
        plt.plot(result['loss'], label=method_name)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration for Different Methods')
plt.legend()
plt.grid()
plt.savefig(os.path.join(exp_dir, "loss_iter_comparison.pdf"))
plt.show()

# PSNR vs Time
plt.figure(figsize=(10, 6))
for method_name, result in results.items():
    if result['psnr']:
        plt.plot(result['times'], result['psnr'], label=method_name)
plt.xlabel('CPU time (s)')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Time for Different Methods')
plt.legend()
plt.grid()
plt.savefig(os.path.join(exp_dir, "psnr_time_comparison.pdf"))
plt.show()

# Loss vs Time
plt.figure(figsize=(10, 6))
for method_name, result in results.items():
    if result['loss']:
        plt.plot(result['times'], result['loss'], label=method_name)
plt.xlabel('CPU time (s)')
plt.ylabel('Loss')
plt.title('Loss vs Time for Different Methods')
plt.legend()
plt.grid()
plt.savefig(os.path.join(exp_dir, "loss_time_comparison.pdf"))
plt.show()
