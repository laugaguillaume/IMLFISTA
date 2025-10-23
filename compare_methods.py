# TODO: - fix problem of total num_iter in BCD methods
#       - Save plots FB vs BCD FB and MLFB vs BCD MLFB
#       - Optimize code (avoid recomputing gradients etc)
#       - Optimize code (efficiently compute A_H)


import os
import platform
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepinv as dinv
import time
import seaborn as sns
import pywt
import pickle
import threading, psutil
import argparse

from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from block.block import BlockCoordinateDescent
from block.utils import wavelet_numpy_to_torch, wavelet_torch_to_numpy
from multilevel.multilevel import ParametersMultilevel, MultiLevel, MultiLevelWavelets, WaveletDenoiserConditional
from multilevel.multilevel_initialization import ml_init_pnp
from multilevel.utils import WaveletPriorCustom

# --- ARGUMENTS PARSING ------

parser = argparse.ArgumentParser(description="Run inverse problem experiments with different configurations.")

parser.add_argument("--physics", type=str, choices=["inpainting", "deblurring"], default="deblurring", help="Type of physics model to use.")
parser.add_argument("--prior", type=str, choices=["TV", "L1", "L1_wavelet"], default="L1_wavelet", help="Type of prior to use.")
parser.add_argument("--reg_weight", type=float, default=0.01, help="Regularization weight λ.")
parser.add_argument("--stepsize", type=float, default=0.1, help="Stepsize for the gradient descent.")
parser.add_argument("--sigma", type=float, default=0.01, help="Noise level for GaussianNoise.")
parser.add_argument("--J", type=int, default=4, help="Number of wavelet levels.")
parser.add_argument("--n_iter", type=int, default=50, help="Number of iterations for the solver.")
parser.add_argument("--n_coarse_steps", type=int, default=5, help="Number of coarse steps per level in multilevel iterations.")
parser.add_argument("--image_size", type=str, choices=["small", "big"], default="small", help="Select which image size to use.")
parser.add_argument("--methods", type=str, nargs="+", default=["FB"], help="List of methods to run (e.g. FB MLFB BCD_FB).")

args = parser.parse_args()

print(args.physics)

# --- MEMORY WATCHDOG (avoid out of memory errors) ------

def memory_watchdog(limit_gb=8, check_interval=5):
    """Function to monitor memory usage and terminate the program if it exceeds a limit."""
    process = psutil.Process(os.getpid())
    while True:
        used_gb = process.memory_info().rss / 1e9
        if used_gb > limit_gb:
            print(f"Memory limit exceeded: {used_gb:.2f} GB > {limit_gb} GB. Terminating the program.")
            os._exit(1)
        time.sleep(check_interval)

threading.Thread(target=memory_watchdog, args=(12, 5), daemon=True).start()

# --- SETUP -----
sns.set_theme()
sns.color_palette("colorblind")
colors = sns.color_palette("colorblind")

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

seed = torch.manual_seed(0)  # Random seed for reproducibility

PSNR = dinv.metric.PSNR()

# --- LOAD IMAGE ----- %%

# Ground truth
if args.image_size == "small":
    x_true = dinv.utils.load_example("butterfly.png", device=device)
else:
    x_true = dinv.utils.load_image("pillars_of_creation.png", img_size=2048, device=device)

print(f'x_true.shape: {x_true.shape}')

#%%------ MODEL -----%%
# Physics
sigma = args.sigma  # Noise level
noise_model = dinv.physics.GaussianNoise(sigma=sigma)

if args.physics == 'inpainting':
    physics = dinv.physics.Inpainting(img_size=x_true.shape[1:], mask=0.8, device=device, noise_model=noise_model)
elif args.physics == 'deblurring':
    filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
    physics = dinv.physics.Blur(filter_0, padding="reflect")
physics.noise_model = noise_model

# Observation
y = physics(x_true)

# Objective function
data_fidelity = dinv.optim.L2()
prior_type = args.prior  # "TV", "L1", "L1_wavelet"


#%%------ PARAMETERS -----%%
n_iter = args.n_iter
reg_weight = args.reg_weight
Anorm2 = physics.compute_norm(x_true).item()
stepsize = args.stepsize/Anorm2

J = args.J         # Number of wavelet levels
levels = J+1  # Same but the Multilevel function uses levels=J+1
filter = 'daubechies8'
wv_type = 'db8'

# For multilevel algorithms
multilevel_iter = 15 #int(0.1 * n_iter)  # Number of multilevel iterations at the fine level
n_coarse_steps = args.n_coarse_steps  # Number of coarse steps per level in each multilevel iteration

# For BCD algorithms
update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'

#%%----- INITIALIZE ALGORITHMS ARGUMENTS -----%%

# Plug and Play denoiser
denoiser_0 = dinv.models.DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
denoiser_pnp = dinv.models.EquivariantDenoiser(denoiser_0, random=True)
prior_pnp = dinv.optim.prior.PnP(denoiser=denoiser_pnp)

# Conditional Denoiser
denoiser_cond = WaveletDenoiserConditional(level=J, wv=wv_type, device=device, non_linearity="soft")

if prior_type == "L1":
    prior = dinv.optim.L1Prior()
    denoiser = prior.prox
elif prior_type == "TV":
    prior = dinv.optim.TVPrior(n_it_max=50)
    denoiser = prior.prox
elif prior_type == "L1_wavelet":
    # prior = dinv.optim.WaveletPrior(level=J, wv=wv_type, p=1, mode='periodic', device=device)
    prior = WaveletPriorCustom(level=J, wv=wv_type, p=1, device=device)
    denoiser = prior.prox

# Block coordinate descent setup
prior_l1 = dinv.optim.L1Prior()
bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)
#n_iter_BCD = n_iter
n_iter_BCD = int(n_iter/(n_coarse_steps*J))  # To have roughly the same number of fine updates as other methods

# Multilevel parameters
cst_grad = None

args_multilevel = ParametersMultilevel(
    target_shape=x_true.shape[-3:],
    levels=levels,
    max_ML_steps=1,
    param_coarse_iter=n_coarse_steps, # Number of coarse iterations
    step_size=torch.tensor(stepsize),
    info_transfer=filter,
    prior=prior,
    denoiser=denoiser,
    data_fidelity=data_fidelity,
    physics=physics,
    observation=y,
    device=device,
)

args_multilevel.info_transfer = filter
wv_type = args_multilevel.information_transfer.wavelet_type

x_true_coarse = wavelet_numpy_to_torch(pywt.wavedec2(x_true.detach().cpu().numpy(), wavelet=wv_type, level=J, mode='periodization'))[0].to(device)

# Initialize coarse physics
if isinstance(physics, dinv.physics.Inpainting):
    coarse_physics = {f'level{levels}': physics}
    coarse_data = physics.mask.data.to(device)

    for i in range(levels-1, 0, -1):
        coarse_data = args_multilevel.information_transfer.to_coarse(coarse_data, coarse_data.shape)
        coarse_physics[f'level{i}'] = dinv.physics.Inpainting(
            img_size=coarse_data.shape[1:],
            mask=coarse_data,
            device=device
        )

    coarsest_physics = coarse_physics[f'level{1}']
    args_multilevel.coarse_physics = coarse_physics

    coarse_operator_norm = coarsest_physics.compute_norm(x_true_coarse).item()
    print(f'Fine level operator norm: {Anorm2}, coarsest level operator norm: {coarse_operator_norm}')

# Initialize coarse observations
observations = {f'level{levels}': y}
current_obs = y.clone().to(device)
for i in range(levels-1, 0, -1):
    current_obs = args_multilevel.information_transfer.to_coarse(
        current_obs,
        current_obs.shape[-3:]
    )
    observations[f'level{i}'] = current_obs.to(device)

args_multilevel.observations = observations

#%%----- Experiment directory setup to save parameters and figures -----%%
cbp = True

if not cbp:
    if platform.system() == "Darwin":
        EXPERIMENTS_ROOT = Path("/Users/edgardesainte-mareville/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/compare_methods")
    else:
        EXPERIMENTS_ROOT = Path("/home/edgar/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/compare_methods")
else:
    EXPERIMENTS_ROOT = Path(__file__).resolve().parent / "experiments_results/compare_methods"  # relatif au repo

EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
exp_name = (
    f"exp_{timestamp}_J{J}_reg{reg_weight}_sigma{sigma}_niter{n_iter}"
    f"_ncoarse{n_coarse_steps}_{args.image_size}"
)
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
    "n_coarse_steps": n_coarse_steps,
    "stepsize": stepsize,
    "J": J,
    "wavelet": wv_type,
    "update_mode": update_mode,
    "comment": "With approx thresholding in BCD"
}
params_path = os.path.join(exp_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=4)

biggest_multilevel_iter = 0

#%%%--- Reconstruction %%%---

only_cycles = False  # Whether to only keep the values at the end of each cycle for BCD methods

def run_FB(x0, y, x_true=None, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    extend_value = (J+1) * n_coarse_steps

    n = x0.shape[-1] * x0.shape[-2]
    N = x0.shape[-1]
    filter_size = 16  # for db8

    loss, psnr, times, cost = [data_fidelity.fn(x0, y, physics).item() + params['reg_weight'] * prior.fn(x0).item()], [PSNR(x0, x_true).item()], [0], [4*N**3]

    start = time.process_time()

    stepsizeATy = params['stepsize'] * physics.A_adjoint(y)
    xk = x0.clone().to(device)
    with torch.no_grad():
        with tqdm(range(1000), desc="FB") as t:
        #with tqdm(range(params['n_iter']), desc="FB") as t:
            for k in t:
                # Gradient step
                xk = xk - params['stepsize'] * (physics.A_adjoint(physics.A(xk))) + stepsizeATy
                # Proximal step
                xk = prior.prox(xk, gamma=params['stepsize'] * params['reg_weight'])

                current_loss_val = data_fidelity.fn(xk, y, physics).item() + params['reg_weight'] * prior.fn(xk).item()
                loss.extend([current_loss_val] * extend_value)
                current_time_val = time.process_time() - start
                times.extend([current_time_val] * extend_value)
                cost.extend([cost[-1] + 4*N**3 + N**2] * extend_value)
                if x_true is not None:
                    current_psnr_val = PSNR(xk, x_true).item()
                    psnr.extend([current_psnr_val] * extend_value)


    # --- Plot ---
    print(cost[0])
    #cost, loss = cost[1:], loss[1:]
    plt.figure(figsize=(6,4))
    plt.plot(cost, loss)
    plt.xlabel("Coût (opérations)")
    plt.ylabel("Loss")
    plt.title("Loss en fonction du coût")
    plt.grid(True)
    plt.show()

    recon = xk.clone()
    cycles = None
    return recon, loss, psnr, times, cycles, cost

def run_MLFB(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params, args_multilevel=args_multilevel):
    extend_value = 1#(((J+1)*(J+2))//2) * n_coarse_steps

    loss, psnr, times = [data_fidelity.fn(x0, y, physics).item() + params['reg_weight'] * prior.fn(x0).item()], [PSNR(x0, x_true).item()], [0]

    levels = args_multilevel.levels
    param_regularization = params['reg_weight']
    param_gamma = params['stepsize']

    global biggest_multilevel_iter

    start = time.process_time()
    stepsizeATy = params['stepsize'] * physics.A_adjoint(y)
    xk = x0.clone().to(device)
    with torch.no_grad():
        with tqdm(range(params['n_iter'])) as t:
            for k in t:
                    cst_grad_device = None if cst_grad is None else cst_grad.to(device)
                    if k < params['multilevel_iter']:
                        xk, intermediate_losses, intermediate_times, intermediate_psnrs = MultiLevel(
                            xk,
                            levels,
                            levels - 1,
                            args_multilevel,
                            param_regularization,
                            cst_grad_device,
                            device=device,
                            x_true=x_true,
                            xk_finest=xk
                        )
                        #loss += intermediate_losses
                        #times += intermediate_times
                        #psnr += intermediate_psnrs
                        biggest_multilevel_iter = len(loss)

                    # Gradient step
                    xk = xk - params['stepsize'] * (physics.A_adjoint(physics.A(xk))) + stepsizeATy

                    # Proximal step
                    if isinstance(prior, dinv.optim.TVPrior) or isinstance(prior, WaveletPriorCustom):
                        xk = prior.prox(xk, gamma=param_gamma * param_regularization)
                    else:
                        xk = prior.prox(xk, gamma=[param_gamma * param_regularization])

                    # Compute metrics
                    current_loss_val = data_fidelity.fn(xk, y, physics).item() + params['reg_weight'] * prior.fn(xk).item()
                    loss.extend([current_loss_val] * extend_value)
                    current_time_val = time.process_time() - start
                    times.extend([current_time_val] * extend_value)
                    if x_true is not None:
                        current_psnr_val = PSNR(xk, x_true).item()
                        psnr.extend([current_psnr_val] * extend_value)

    recon = xk.clone()
    cycles = None
    cost = None
    return recon, loss, psnr, times, cycles, cost

def run_BCD_MLFB(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):
    prior_l1 = dinv.optim.L1Prior()
    len_cycle = J+1
    n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))
    n_cycles = params['n_iter']

    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y, xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='MLFB', metrics=True)

    cycles = [1] + cycles

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    cost = None
    return recon, loss, psnr, times, cycles, cost

def run_BCD_cyclic(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):
    prior_l1 = dinv.optim.L1Prior()
    len_cycle = J+1
    n_cycles=params['n_iter']
    #n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))

    n = x0.shape[-1] * x0.shape[-2]
    N = x0.shape[-1]

    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y, xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='cyclic', metrics=True)

    cost = [34*N**3]
    cost += [K*4*N**3 for K in range(1, len(loss))]

    cycles = [1] + cycles

    print(len(cost))

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    print(cost[0])

    #loss, cost = loss[1:], cost[1:]

    plt.figure(figsize=(6,4))
    plt.plot(cost, loss)
    plt.xlabel("Coût (opérations)")
    plt.ylabel("Loss")
    plt.title("Loss en fonction du coût")
    plt.grid(True)
    plt.show()

    return recon, loss, psnr, times, cycles, cost

def run_PnP(x0, y, x_true=None, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):

    loss, psnr, times = None, [PSNR(x0, x_true).item()], [0]
    start = time.process_time()

    stepsizeATy = params['stepsize'] * physics.A_adjoint(y)
    xk = x0.clone().to(device)

    with torch.no_grad():
            with tqdm(range(params['n_iter']), desc="PnP") as t:
                for k in t:
                    xk = xk - params['stepsize'] * (physics.A_adjoint(physics.A(xk))) + stepsizeATy
                    xk = denoiser_pnp(xk, sigma=params['reg_weight'])

                    if x_true is not None:
                        current_psnr = PSNR(xk, x_true).item()
                        psnr.append(current_psnr)
                        t.set_postfix(psnr=f"{current_psnr:.4f} dB")
                    times.append(time.process_time() - start)

    recon = xk.clone()
    cycles = None
    return recon, loss, psnr, times, cycles, cost

def run_MLPnP(x0, y, x_true=None, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):

    loss, psnr, times = None, [PSNR(x0, x_true).item()], [0]

    args_multilevel.prior = prior_pnp
    args_multilevel.denoiser = denoiser_pnp

    with torch.no_grad():
        print("initialize ML PnP ...")
        init = x0.clone()
        levels = params['J']+1
        step_size = params['stepsize']

        start = time.process_time()

        # ML initialization
        ml_init = ml_init_pnp(init, levels, levels - 1, args_multilevel, params['reg_weight'], denoiser_pnp, device)

        print("solver is running ...")

        xk = ml_init

        # Main iteration loop
        with torch.no_grad():
            with tqdm(range(params['n_iter']), desc="MLPnP") as t:
                for k in t:
                    xk_prev = xk.clone()

                    if k < params['multilevel_iter']:
                        print('Multilevel iteration ', k+1, '/', params['multilevel_iter'])
                        cst_grad = None  # coherence not required on finest level
                        uk, _, _, _ = MultiLevel(xk_prev, levels, levels-1, args_multilevel, params['reg_weight'], cst_grad, device)
                    else:
                        uk = xk_prev


                    xk = uk - step_size * data_fidelity.grad(uk, y, physics)
                    xk = denoiser_pnp(xk, sigma=params['reg_weight'])

                    # Compute metrics
                    if x_true is not None:
                        current_psnr = PSNR(xk, x_true).item()
                        psnr.append(current_psnr)
                        t.set_postfix(psnr=f"{current_psnr:.4f} dB")

                    times.append(time.process_time() - start)

        print("done.")

    recon = xk.clone()
    cycles = None
    return recon, loss, psnr, times, cycles, cost

def run_MLFBcond(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params, args_multilevel=args_multilevel):
    loss, psnr, times = None, [PSNR(x0, x_true).item()], [0]
    xk = x0.clone()
    levels = args_multilevel.levels
    param_regularization = params['reg_weight']
    param_gamma = params['stepsize']

    args_multilevel.prior = prior
    args_multilevel.denoiser = denoiser

    start = time.process_time()

    with torch.no_grad():
        with tqdm(range(params['n_iter']), desc="MLFBcond") as t:
            for k in t:

                if k < params['multilevel_iter']:
                    xk = MultiLevelWavelets(
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
                xk = denoiser_cond(xk, gamma=params['stepsize'] * params['reg_weight'])

                # Compute metrics
                current_psnr = PSNR(xk, x_true).item()
                t.set_postfix(psnr=f"{current_psnr:.4f} dB")
                psnr.append(current_psnr)

                times.append(time.process_time() - start)

    recon = xk.clone()
    cycles = None
    return recon, loss, psnr, times, cycles, cost

def run_BCDcond(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):
    prior_l1 = dinv.optim.L1Prior()
    len_cycle = 2*J+1
    n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))
    n_cycles = params['n_iter']

    xk = x0.clone()

    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y, xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='MLFBcond', metrics=True)

    cycles = [1] + cycles

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    cost = None
    return recon, loss, psnr, times, cycles, cost

def run_BCD_FB(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):

    prior_l1 = dinv.optim.L1Prior()
    len_cycle = J+1
    n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))
    n_cycles = params['n_iter']

    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y=y, x0=xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='FB', metrics=True)

    cycles = [1] + cycles

    # To only keep the values at the end of each cycle
    if only_cycles:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    print(len(loss))
    cost = None
    return recon, loss, psnr, times, cycles, cost

def run_BCD_cyclic_cond(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):
    prior_l1 = dinv.optim.L1Prior()
    len_cycle = J+1
    n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))

    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y, xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='cyclic', use_conditional_thresholding=True, metrics=True)

    cycles = [1] + cycles

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    return recon, loss, psnr, times, cycles, cost

def run_GD_all_levels(x0, y=y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    xk = x0.clone().to(device)
    param_regularization = params['reg_weight']

    args_multilevel.prior = prior
    args_multilevel.denoiser = denoiser

    xk = MultiLevelWavelets(
                        xk,
                        levels,
                        levels - 1,
                        args_multilevel,
                        param_regularization,
                        cst_grad,
                        device,
                    )

    recon = xk.clone()
    return recon, None, None, None, None, None

def run_coarse_GD(x0, y=y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    xk = x0.clone().to(device)
    coarsest_physics = coarse_physics[f'level{1}']
    coarsest_observation = observations[f'level{1}']

    xk_coeffs_np = pywt.wavedec2(xk.detach().numpy(), wavelet=params['wavelet'], level=params['J'], mode='periodization')
    xk_coeffs = wavelet_numpy_to_torch(xk_coeffs_np)
    approx = xk_coeffs[0].clone().to(device)

    print(approx.shape, coarsest_observation.shape)

    for k  in range(params['n_coarse_steps']):
        approx = approx - params['stepsize'] * coarsest_physics.A_adjoint(approx) * (coarsest_physics.A(approx) - coarsest_observation)

    xk_coeffs[0] = approx
    xk = torch.from_numpy(pywt.waverec2(wavelet_torch_to_numpy(xk_coeffs), wavelet=params['wavelet'], mode='periodization'))

    xk = denoiser_cond(xk)

    return xk, None, None, None, None, None

def run_coarse_GD_iter(x0, y=y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    xk = x0.clone().to(device)
    n_iter=params['n_iter']
    n_iter_coarse = params['n_coarse_steps']
    coarsest_physics = coarse_physics[f'level{1}']
    coarsest_observation = observations[f'level{1}']

    with torch.no_grad():
        with tqdm(range(n_iter), desc="Coarse GD") as t:
            for k in t:
                xk_coeffs_np = pywt.wavedec2(xk.detach().numpy(), wavelet=params['wavelet'], level=params['J'], mode='periodization')
                xk_coeffs = wavelet_numpy_to_torch(xk_coeffs_np)
                approx = xk_coeffs[0].clone().to(device)

                for l  in range(n_iter_coarse):
                    approx = approx - params['stepsize'] * coarsest_physics.A_adjoint(approx) * (coarsest_physics.A(approx) - coarsest_observation)

                xk_coeffs[0] = approx
                xk = torch.from_numpy(pywt.waverec2(wavelet_torch_to_numpy(xk_coeffs), wavelet=params['wavelet'], mode='periodization'))
                xk = denoiser_cond(xk)

    recon = xk.clone()
    return recon, None, None, None, None, None

'''
results = {}
params['n_coarse_steps'] = 20
args_multilevel.param_coarse_iter = params['n_coarse_steps']

alpha = 1e-3
params['stepsize'] = alpha / Anorm2
args_multilevel.step_size = torch.tensor(params['stepsize'])
for param_regularization in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    params['reg_weight'] = param_regularization
    print(f"Running with regularization weight: {param_regularization}")

    # on stocke les deux reconstructions
    recon_coarse_GD, _, _, _, _ = run_coarse_GD(x0=y)
    psnr_coarse_GD = PSNR(recon_coarse_GD, x_true).item()
    #recon_GD_all, _, _, _, _ = run_GD_all_levels(x0=y)
    #psnr_GD_all = PSNR(recon_GD_all, x_true).item()

    results[param_regularization] = {
        "coarse_GD": recon_coarse_GD,
        #"GD_all": recon_GD_all,
        "PSNR_coarse_GD": psnr_coarse_GD,
        #"PSNR_GD_all": psnr_GD_all
    }

#recon_coarse_GD_iter, _, _, _, _ = run_coarse_GD_iter(x0=y)

for param_regularization, recons_dict in results.items():
    titles = [
        "Ground truth",
        f"Observation (PSNR={PSNR(y, x_true).item():.2f} dB)",
        f"Coarse GD (reg={param_regularization}, PSNR={recons_dict['PSNR_coarse_GD']:.2f})",
        #f"GD all levels (reg={param_regularization}, PSNR={recons_dict['PSNR_GD_all']:.2f})"
    ]

    dinv.utils.plot([x_true, y, recons_dict["coarse_GD"]], titles=titles, suptitle=f"Regularization weight: {param_regularization}, n_iter_coarse={params['n_coarse_steps']}, stepsize={alpha}/L", cmap="gray", save_fn=os.path.join(exp_dir, f"coarse_GD_vs_GD_all_reg{param_regularization}.pdf"), tight=True)

#recon_approx = run_coarse_GD(x0=y)

import sys
sys.exit()'''


#%%--- Run methods %%%---

# Complete list of all methods
all_methods = {
    "FB": run_FB,
    "MLFB": run_MLFB,
    "PnP": run_PnP,
    "MLPnP": run_MLPnP,
    "MLFBcond": run_MLFBcond,
    "BCD_FB": run_BCD_FB,
    "BCD_MLFB": run_BCD_MLFB,
    "BCDcyclic": run_BCD_cyclic,
    "BCD_MLFB_details": run_BCDcond,
    "BCDcyclic_cond": run_BCD_cyclic_cond
}

# Filter the methods to run based on user input
methods = {name: all_methods[name] for name in args.methods if name in all_methods}

results = {}
for method_name, method_func in methods.items():
    x0 = y.clone()
    print(f"Running {method_name}...")
    params['update_mode'] = method_name
    x_rec, loss, psnr, times, cycles, cost = method_func(x0, y, x_true=x_true)
    print("Method: ", method_name, " Loss: ", len(loss) if loss is not None else 'No loss', " PSNR: ", len(psnr), " Times: ", len(times))
    results[method_name] = {
        "reconstruction": x_rec,
        "loss": loss,
        "psnr": psnr,
        "times": times,
        "cycles": cycles if not only_cycles else None,
        "cost": cost
    }
    if psnr:
        print(f"Final PSNR for {method_name}: {psnr[-1]:.2f} dB")
    else:
        print(f"No PSNR computed for {method_name}")

# === Define method colors, linestyles, and markers ===
method_colors = {
    "FB": colors[0],
    "MLFB": colors[0],
    "BCD_MLFB": colors[4],
    "PnP": colors[1],
    "MLPnP": colors[1],
    "MLFBcond": colors[2],
    "BCD_MLFB_details": colors[2],
    "BCDcyclic": colors[3],
    "BCD_FB": colors[0],
    "BCDcyclic_cond": colors[2],
}

method_linestyles = {
    "FB": "-",
    "MLFB": "--",
    "BCD_MLFB": "-.",
    "PnP": "-",
    "MLPnP": "--",
    "MLFBcond": "--",
    "BCD_MLFB_details": "-.",
    "BCDcyclic": "-.",
    "BCD_FB": "-.",
    "BCDcyclic_cond": "-.",
}

method_markers = {
    "BCD_MLFB": "o",
    "BCD_MLFB_details": "o",
    "BCDcyclic": "o",
    "BCD_FB": "o",
    "BCDcyclic_cond": "o"
}


#%%%--- Saving results %%%---

results_data = {
    'results': results,  # Contient tous les résultats des méthodes
    'params': params,    # Paramètres de l'expérience
    'x_true': x_true.cpu().numpy(),  # Image de référence
    'y': y.cpu().numpy(),            # Observation
    'multilevel_iter': multilevel_iter,
    'method_info': {
        'colors': method_colors,
        'linestyles': method_linestyles,
        'markers': method_markers
    }
}

# Convert tensors to numpy arrays for pickle compatibility
for method_name, result in results_data['results'].items():
    if result['reconstruction'] is not None:
        result['reconstruction'] = result['reconstruction'].cpu().numpy()
    # Les loss, psnr, times sont déjà en listes Python normalement

# Save results using pickle
results_path = os.path.join(exp_dir, "results.pkl")
with open(results_path, 'wb') as f:
    pickle.dump(results_data, f)

print(f"Results saved to: {results_path}")

# Save parameters as JSON for easy reference
params_path = os.path.join(exp_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=4)

print(f"Parameters saved to: {params_path}")
print(f"\nTo plot results, run: python3 plot_results.py {exp_dir}")