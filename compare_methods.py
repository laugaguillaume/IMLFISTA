import os
import platform
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepinv as dinv
import time
import seaborn as sns

from block.block import BlockCoordinateDescent
from multilevel.multilevel import ParametersMultilevel, MultiLevel, MultiLevelWavelets
from multilevel.multilevel_initialization import ml_init_pnp

# Plot settings
sns.set_theme()
sns.color_palette("colorblind")
colors = sns.color_palette("colorblind")

PSNR = dinv.metric.PSNR()

device = torch.device('cpu')
x_true = dinv.utils.load_example("butterfly.png", device=device)

#%%------ MODEL -----%%
# Physics
sigma = 0.01
noise_model = dinv.physics.GaussianNoise(sigma=sigma)
physics = dinv.physics.Inpainting(tensor_size=x_true.shape[1:], mask=0.5, device=device, noise_model=noise_model)
seed = torch.manual_seed(0)  # Random seed for reproducibility

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
wv_type = 'daubechies8'

# For multilevel algorithms
multilevel_iter = 5 # Number of multilevel iterations at the fine level
n_coarse_steps = 5  # Number of coarse steps per level in each multilevel iteration

# For BCD algorithms
update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'

#%%----- INITIALIZE ALGORITHMS ARGUMENTS -----%%

# Plug and Play denoiser
denoiser_0 = dinv.models.DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
denoiser_pnp = dinv.models.EquivariantDenoiser(denoiser_0, random=True)
prior_pnp = dinv.optim.prior.PnP(denoiser=denoiser_pnp)

if prior_type == "L1":
    prior = dinv.optim.L1Prior()
    denoiser = prior.prox
elif prior_type == "TV":
    prior = dinv.optim.TVPrior(n_it_max=50)
    denoiser = prior.prox
elif prior_type == "L1_wavelet":
    prior = dinv.optim.WaveletPrior(level=J, wv=wv_type, p=1, device=device)
    denoiser = prior.prox

# Block coordinate descent setup
bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize)

# Multilevel parameters
cst_grad = None

args_multilevel = ParametersMultilevel(
    target_shape=x_true.shape[-3:],
    levels=J,
    max_ML_steps=1,
    param_coarse_iter=n_coarse_steps, # Number of coarse iterations
    step_size=torch.tensor(stepsize),
    info_transfer=wv_type,
    prior=prior,
    denoiser=denoiser,
    data_fidelity=data_fidelity,
    physics=physics,
    observation=y,
    device=device,
)

args_multilevel.info_transfer = 'daubechies8'

# Initialize coarse physics
coarse_physics = {f'level{J}': physics}
coarse_data = physics.mask.data 

for i in range(J-1, 0, -1):
    coarse_data = args_multilevel.information_transfer.to_coarse(coarse_data, coarse_data.shape)
    coarse_physics[f'level{i}'] = dinv.physics.Inpainting(
        tensor_size=coarse_data.shape[1:],
        mask=coarse_data, 
        device=physics.mask.device 
    )

args_multilevel.coarse_physics = coarse_physics

# Initialize coarse observations
observations = {f'level{J}': y}
current_obs = y.clone()
for i in range(J-1, 0, -1):
    current_obs = args_multilevel.information_transfer.to_coarse(
        current_obs, 
        current_obs.shape[-3:]
    )
    observations[f'level{i}'] = current_obs

args_multilevel.observations = observations

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
    "physics": f'Inpainting (mask: 50%) + Gaussian noise (sigma={sigma})',
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
    
    start = time.process_time()
    
    with torch.no_grad():
        for k in range(params['n_iter']):
            
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
            
            # Gradient step
            xk = xk - param_gamma * data_fidelity.grad(xk, y, physics)
            
            # Proximal step
            if isinstance(prior, dinv.optim.TVPrior):
                xk = prior.prox(xk, gamma=param_gamma * param_regularization)
            else:
                xk = prior.prox(xk, gamma=[param_gamma * param_regularization])

            # Compute loss and PSNR
            current_loss = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
            loss.append(current_loss.item())
            
            current_psnr = PSNR(xk, x_true).item()
            psnr.append(current_psnr)
            
            times.append(time.process_time() - start)

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

def run_MLPnP(x0, y, x_true=None, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params, denoiser=denoiser):
    loss, psnr, times = [], [], []
    
    with torch.no_grad():
        print("initialize ML PnP ...")
        init = x0.clone()
        levels = params['J']
        regularization = params['reg_weight']
        max_ML_steps = params['multilevel_iter']
        step_size = params['stepsize']
        
        start = time.process_time()
        
        # Initial PSNR
        if x_true is not None:
            PSNR_init = PSNR(init, x_true).item()
            psnr.append(PSNR_init)

        # ML initialization
        args_multilevel.param_coarse_iter = 5
        ml_init = ml_init_pnp(init, levels, levels - 1, args_multilevel, regularization, denoiser, device)
        
        if x_true is not None:
            PSNR_ML_init = PSNR(ml_init, x_true).item()
            psnr.append(PSNR_ML_init)

        print("solver is running ...")
        args_multilevel.param_coarse_iter = 3

        xk = ml_init
        
        # Main iteration loop
        for k in range(params['n_iter']):
            xk_prev = xk.clone()
            
            if k < max_ML_steps:
                cst_grad = None  # coherence not required on finest level
                uk = MultiLevel(xk_prev, levels, levels-1, args_multilevel, regularization, cst_grad, device)
            else:
                uk = xk_prev
                
            xk = uk - step_size * data_fidelity.grad(uk, y, physics)
            xk = denoiser(xk, sigma=regularization)
            
            # Compute metrics
            current_loss = data_fidelity(xk, y, physics) + regularization * prior.fn(xk)
            loss.append(current_loss.item())
            
            if x_true is not None:
                current_psnr = PSNR(xk, x_true).item()
                psnr.append(current_psnr)
            
            times.append(time.process_time() - start)
            
            if k % 10 == 0:
                print(f"psnr ML[{k}]: {psnr[-1] if x_true is not None else 'N/A'}")
                
        print("done.")

    recon = xk.clone()
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
    #"BCD": run_BCD,
    "PnP": run_PnP,
    "MLPnP": run_MLPnP,
    #"MLFBcond": run_MLFBcond,
    #"BCDcond": run_BCDcond
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
plt.axvline(x=multilevel_iter, color='red', linestyle='--', label=f"End of Multilevel iterations (total : {multilevel_iter})")
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
plt.axvline(x=multilevel_iter, color='red', linestyle='--', label=f"End of Multilevel iterations (total : {multilevel_iter})")
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
