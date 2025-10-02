# TODO: - fix problem of total num_iter in BCD methods
#       - Save plots FB vs BCD FB and MLFB vs BCD MLFB
#       - Optimize code (avoid recomputing gradients etc)
#       - Optimize code (efficiently compute A_H)


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
from pathlib import Path
from tqdm import tqdm
import pywt

from block.block import BlockCoordinateDescent
from block.utils import wavelet_numpy_to_torch, wavelet_torch_to_numpy
from multilevel.multilevel import ParametersMultilevel, MultiLevel, MultiLevelWavelets, WaveletDenoiserConditional
from multilevel.multilevel_initialization import ml_init_pnp
from multilevel.utils import WaveletPriorCustom

# Plot settings
sns.set_theme()
sns.color_palette("colorblind")
colors = sns.color_palette("colorblind")

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print(f"Using device: {device}")

PSNR = dinv.metric.PSNR()

# Ground truth
x_true = dinv.utils.load_example("butterfly.png", device=device)
print(x_true.shape)
#x_true = dinv.utils.load_image('pillars_of_creation.png', img_size=2048, device=device)

#%%------ MODEL -----%%
# Physics
sigma = 0.01
noise_model = dinv.physics.GaussianNoise(sigma=sigma)
physics = dinv.physics.Inpainting(img_size=x_true.shape[1:], mask=0.8, device=device, noise_model=noise_model)
seed = torch.manual_seed(0)  # Random seed for reproducibility

# Observation
y = physics(x_true)

# Objective function
data_fidelity = dinv.optim.L2()
prior_type = "L1_wavelet"  # "TV", "L1", "L1_wavelet"


#%%------ PARAMETERS -----%%
n_iter = 600
reg_weight = 0.1
Anorm2 = physics.compute_norm(x_true).item()
stepsize = 0.05/Anorm2

J = 3         # Number of wavelet levels
levels = J+1  # Same but the Multilevel function uses levels=J+1
filter = 'daubechies8'
wv_type = 'db8'

# For multilevel algorithms
multilevel_iter = 10 #int(0.1 * n_iter)  # Number of multilevel iterations at the fine level
n_coarse_steps = 5  # Number of coarse steps per level in each multilevel iteration

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
    #prior = dinv.optim.WaveletPrior(level=J, wv=wv_type, p=1, mode='periodic', device=device)
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

# Initialize coarse physics
coarse_physics = {f'level{levels}': physics}
coarse_data = physics.mask.data.to(device)

#dinv.utils.plot(physics.mask.data)

for i in range(levels-1, 0, -1):
    coarse_data = args_multilevel.information_transfer.to_coarse(coarse_data, coarse_data.shape)
    coarse_physics[f'level{i}'] = dinv.physics.Inpainting(
        img_size=coarse_data.shape[1:],
        mask=coarse_data,
        device=device
    )

coarsest_physics = coarse_physics[f'level{1}']
#dinv.utils.plot(coarsest_physics.mask.data)

x_true_coarse = wavelet_numpy_to_torch(pywt.wavedec2(x_true.detach().cpu().numpy(), wavelet=wv_type, level=J, mode='periodization'))[0].to(device)

coarse_operator_norm = coarsest_physics.compute_norm(x_true_coarse).item()
print(f'Fine level operator norm: {Anorm2}, coarsest level operator norm: {coarse_operator_norm}')

'''mask_wavelet = pywt.wavedec2(physics.mask.data.cpu().numpy(), wavelet=wv_type, level=J, mode='periodization')
mask_wavelet = wavelet_numpy_to_torch(mask_wavelet)
dinv.utils.plot(mask_wavelet[1][0][0])'''

args_multilevel.coarse_physics = coarse_physics

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

only_cycles = True  # Whether to only keep the values at the end of each cycle for BCD methods

def run_FB(x0, y, x_true=None, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params):
    only_cycles = False

    loss, psnr, times = [data_fidelity.fn(x0, y, physics).item() + params['reg_weight'] * prior.fn(x0).item()], [PSNR(x0, x_true).item()], [0]
    n = x0.shape[-1] * x0.shape[-2]
    filter_size = 8  # for db8
    cost = [n**3 + n**2]
    start = time.process_time()

    stepsizeATy = params['stepsize'] * physics.A_adjoint(y)

    xk = x0.clone().to(device)
    with torch.no_grad():
        with tqdm(range(params['n_iter']), desc="FB") as t:
            for k in t:
                cost.append(cost[-1] + n**2 + n*filter_size**2 + n)
                xk = xk - params['stepsize'] * (physics.A_adjoint(physics.A(xk))) + stepsizeATy
                xk = prior.prox(xk, gamma=params['stepsize'] * params['reg_weight'])

                if only_cycles:
                    if k % (n_coarse_steps*J) == 0:
                        if x_true is not None:
                            current_psnr = PSNR(xk, x_true).item()
                            psnr.append(current_psnr)
                        current_loss =  data_fidelity.fn(xk, y, physics).item() + params['reg_weight'] * prior.fn(xk).item()
                        t.set_postfix(loss=f"{current_loss:.4f}")
                        loss.append(current_loss)
                        times.append(time.process_time() - start)
                else:
                    current_loss_val = data_fidelity.fn(xk, y, physics).item() + params['reg_weight'] * prior.fn(xk).item()
                    loss.extend([current_loss_val] * 1)

                    if x_true is not None:
                        current_psnr_val = PSNR(xk, x_true).item()
                        psnr.extend([current_psnr_val] * 1)

                    current_time_val = time.process_time() - start
                    times.extend([current_time_val] * 1)


    # --- Plot ---
    plt.figure(figsize=(6,4))
    plt.plot(cost, loss)
    plt.xlabel("Coût (opérations)")
    plt.ylabel("Loss")
    plt.title("Loss en fonction du coût")
    plt.grid(True)
    plt.show()

    recon = xk.clone()
    cycles = None
    print(len(loss))
    return recon, loss, psnr, times, cycles, cost

def run_MLFB(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, prior=prior, params=params, args_multilevel=args_multilevel):

    import cProfile
    import pstats
    from pstats import SortKey

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.process_time()
    loss, psnr, times = [data_fidelity.fn(x0, y, physics).item() + params['reg_weight'] * prior.fn(x0).item()], [PSNR(x0, x_true).item()], [start]
    xk = x0.clone().to(device)
    levels = args_multilevel.levels
    param_regularization = params['reg_weight']
    param_gamma = params['stepsize']
    global biggest_multilevel_iter

    stepsizeATy = params['stepsize'] * physics.A_adjoint(y)

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

                    profiler.disable()

                    # Afficher les résultats
                    '''stats = pstats.Stats(profiler)
                    stats.sort_stats(SortKey.CUMULATIVE)
                    stats.print_stats(20)  # Top 20 des fonctions les plus gourmandes'''

                    # Gradient step
                    xk = xk - params['stepsize'] * (physics.A_adjoint(physics.A(xk))) + stepsizeATy

                    # Proximal step
                    if isinstance(prior, dinv.optim.TVPrior) or isinstance(prior, WaveletPriorCustom):
                        xk = prior.prox(xk, gamma=param_gamma * param_regularization)
                    else:
                        xk = prior.prox(xk, gamma=[param_gamma * param_regularization])

                    # Compute metrics
                    current_loss = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
                    t.set_postfix(loss=f"{current_loss.item():.4f}")
                    loss.append(current_loss.item())

                    current_psnr = PSNR(xk, x_true).item()
                    psnr.append(current_psnr)

                    times.append(time.process_time())

    recon = xk.clone()
    times = [t - start for t in times]  # Convert to elapsed time
    cycles = None
    return recon, loss, psnr, times, cycles

def run_BCD_MLFB(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):
    prior_l1 = dinv.optim.L1Prior()
    len_cycle = J+1
    n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))

    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y, xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='MLFB', metrics=True)

    cycles = [1] + cycles

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    return recon, loss, psnr, times, cycles

def run_BCD_cyclic(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):
    only_cycles = True
    prior_l1 = dinv.optim.L1Prior()
    len_cycle = J+1
    n_cycles=40
    #n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))

    n = x0.shape[-1] * x0.shape[-2]

    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y, xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='cyclic', metrics=True)

    cost = [n**3 + n**2 + ((2/(3*4**J) +1/3))*n**2 + (2/(3*4**J) +1/3)*n**3 + (1/4**(2*J) + (1-1/4**J)**2/9 + (2*(1-1/4**J))/(3*4**J))*n**3]
    cost += [cost[0] + k*((1/4**(2*J) + (1-1/4**J)**2/9 + (2*(1-1/4**J)/(3*4**J)))*n**2 + (2/(3*4**J) + 1/3)*n ) for k in range(1, len(cycles)+1)]

    cycles = [1] + cycles

    print(len(cost))

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    print(len(loss))

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
    return recon, loss, psnr, times, cycles

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
    return recon, loss, psnr, times, cycles

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
    return recon, loss, psnr, times, cycles

def run_BCDcond(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):
    prior_l1 = dinv.optim.L1Prior()
    len_cycle = 2*J+1
    n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))

    xk = x0.clone()

    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y, xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='MLFBcond', metrics=True)

    cycles = [1] + cycles

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    return recon, loss, psnr, times, cycles

def run_BCD_FB(x0, y, x_true=x_true, physics=physics, data_fidelity=data_fidelity, params=params):

    prior_l1 = dinv.optim.L1Prior()
    len_cycle = J+1
    n_cycles = int(params['n_iter'] / (len_cycle * params['n_coarse_steps'] * (params['J'] + 1)))

    xk = x0.clone()
    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior_l1, max_levels=J, stepsize=stepsize)

    recon, loss, times, cycles, psnr = bcd.run(y=y, x0=xk, x_true=x_true, n_iter=n_cycles, n_iter_coarse=params['n_coarse_steps'], reg_weight=params['reg_weight'], update_mode='FB', metrics=True)

    cycles = [1] + cycles

    # To only keep the values at the end of each cycle
    if only_cycles == True:
        loss = [loss[index-1] for index in cycles]
        psnr = [psnr[index-1] for index in cycles]
        times = [times[index-1] for index in cycles]

    print(len(loss))
    return recon, loss, psnr, times, cycles

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

    return recon, loss, psnr, times, cycles

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
    return recon, None, None, None, None

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

    return xk, None, None, None, None

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
    return recon, None, None, None, None

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

methods = {
    "FB": run_FB,
    #"MLFB": run_MLFB,
    #"PnP": run_PnP,
    #"MLPnP": run_MLPnP,
    #"MLFBcond": run_MLFBcond,
    #"BCD_FB": run_BCD_FB,
    #"BCD_MLFB": run_BCD_MLFB,
    "BCDcyclic": run_BCD_cyclic,
    #"BCDcond": run_BCDcond,
    #"BCDcyclic_cond": run_BCD_cyclic_cond
}

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

#%%%--- Plotting results %%%---

# Définir les couleurs par groupe de méthodes
method_colors = {
    "FB": colors[0],             # Bleu
    "MLFB": colors[0],           # Bleu
    "BCD_MLFB": colors[4],       # Jaune
    "PnP": colors[1],            # Orange
    "MLPnP": colors[1],          # Orange
    "MLFBcond": colors[2],       # Vert
    "BCDcond": colors[2],        # Vert
    "BCDcyclic": colors[3],      # Rouge
    "BCD_FB": colors[0],         # Bleu
    "BCDcyclic_cond": colors[2], # Vert
}

# Définir les styles de ligne
method_linestyles = {
    "FB": "-",             # Ligne pleine
    "MLFB": "--",          # Tirets
    "BCD_MLFB": "-.",           # Point-tiret
    "PnP": "-",            # Ligne pleine
    "MLPnP": "--",         # Tirets
    "MLFBcond": "--",      # Tirets
    "BCDcond": "-.",       # Point-tiret
    "BCDcyclic": "-.",     # Point-tiret
    "BCD_FB": "-.",        # Point-tiret
    "BCDcyclic_cond": "-." # Point-tiret
}

# Définir les marqueurs pour les cycles
method_markers = {
    "BCD_MLFB": "o",
    "BCDcond": "o",
    "BCDcyclic": "o",
    "BCD_FB": "o",
    "BCDcyclic_cond": "o"
}

# Créer une figure avec 4 sous-graphiques côte à côte
fig, axes = plt.subplots(1, 4, figsize=(28, 6))

# Plot 1: Loss vs Iterations
for method_name, result in results.items():
    if result['loss']:
        axes[0].plot(result['loss'],
                    color=method_colors[method_name],
                    linestyle=method_linestyles[method_name],
                    label=method_name,
                    linewidth=2)
        # Ajouter les marqueurs de cycles pour les méthodes BCD
        if result['cycles'] is not None and len(result['cycles']) > 0:
            axes[0].scatter(result['cycles'],
                          [result['loss'][i-1] for i in result['cycles']],
                          color=method_colors[method_name],
                          marker=method_markers.get(method_name, "o"),
                          s=80,
                          label=f"cycles_{method_name}")

axes[0].axvline(x=multilevel_iter, color='red', linestyle='--',
               label=f"End of Multilevel iterations for ML methods (total : {multilevel_iter})")
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss over Iterations')
axes[0].legend(frameon=True)
axes[0].grid(True)

# Plot 2: Loss vs Time
for method_name, result in results.items():
    if result['loss'] and result['times']:
        axes[1].plot(result['times'], result['loss'],
                    color=method_colors[method_name],
                    linestyle=method_linestyles[method_name],
                    label=method_name,
                    linewidth=2)
        # Ajouter les marqueurs de cycles pour les méthodes BCD
        if result['cycles'] is not None and len(result['cycles']) > 0:
            cycle_times = [result['times'][i-1] for i in result['cycles']]
            cycle_losses = [result['loss'][i-1] for i in result['cycles']]
            axes[1].scatter(cycle_times, cycle_losses,
                          color=method_colors[method_name],
                          marker=method_markers.get(method_name, "o"),
                          s=80,
                          label=f"cycles_{method_name}")

axes[1].set_xlabel('CPU time (s)')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss over CPU Time')
axes[1].legend(frameon=True)
axes[1].grid(True)

# Plot 3: PSNR vs Iterations
for method_name, result in results.items():
    if result['psnr']:
        axes[2].plot(result['psnr'],
                    color=method_colors[method_name],
                    linestyle=method_linestyles[method_name],
                    label=method_name,
                    linewidth=2)
        # Ajouter les marqueurs de cycles pour les méthodes BCD
        if result['cycles'] is not None and len(result['cycles']) > 0:
            axes[2].scatter(result['cycles'],
                          [result['psnr'][i-1] for i in result['cycles']],
                          color=method_colors[method_name],
                          marker=method_markers.get(method_name, "o"),
                          s=80,
                          label=f"cycles_{method_name}")

axes[2].axvline(x=multilevel_iter, color='red', linestyle='--',
               label=f"End of Multilevel iterations for ML methods (total : {multilevel_iter})")
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('PSNR (dB)')
axes[2].set_title('PSNR over Iterations')
axes[2].legend(frameon=True)
axes[2].grid(True)

# Plot 4: PSNR vs Time
for method_name, result in results.items():
    if result['psnr'] and result['times']:
        axes[3].plot(result['times'], result['psnr'],
                    color=method_colors[method_name],
                    linestyle=method_linestyles[method_name],
                    label=method_name,
                    linewidth=2)
        # Ajouter les marqueurs de cycles pour les méthodes BCD
        if result['cycles'] is not None and len(result['cycles']) > 0:
            cycle_times = [result['times'][i-1] for i in result['cycles']]
            cycle_psnrs = [result['psnr'][i-1] for i in result['cycles']]
            axes[3].scatter(cycle_times, cycle_psnrs,
                          color=method_colors[method_name],
                          marker=method_markers.get(method_name, "o"),
                          s=80,
                          label=f"cycles_{method_name}")

axes[3].set_xlabel('CPU time (s)')
axes[3].set_ylabel('PSNR (dB)')
axes[3].set_title('PSNR over CPU Time')
axes[3].legend(frameon=True)
axes[3].grid(True)

plt.savefig(os.path.join(exp_dir, f"all_plots_combined.pdf"), dpi=300, bbox_inches='tight')

# Save each plot individually
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

    # Ajouter les scatter plots (marqueurs de cycles) s'ils existent
    for collection in axes[i].collections:
        ax_individual.scatter(collection.get_offsets()[:, 0],
                            collection.get_offsets()[:, 1],
                            color=collection.get_facecolors()[0],
                            marker=collection.get_paths()[0] if len(collection.get_paths()) > 0 else 'o',
                            s=80)

    ax_individual.set_xlabel(axes[i].get_xlabel())
    ax_individual.set_ylabel(axes[i].get_ylabel())
    ax_individual.set_title(axes[i].get_title())
    ax_individual.legend(frameon=True)
    ax_individual.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{plot_name}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_individual)

# Ajouter 2 nouveaux sous-graphiques pour Cost
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Plot 5: Loss vs Cost
for method_name, result in results.items():
    if result['loss'] and result['cost']:
        print("Plotting cost for method:", method_name)
        axes2[0].plot(result['cost'], result['loss'],
                    color=method_colors[method_name],
                    linestyle=method_linestyles[method_name],
                    label=method_name,
                    linewidth=2)
        # Ajouter les marqueurs de cycles
        if result['cycles'] is not None and len(result['cycles']) > 0:
            cycle_costs = [result['cost'][i-1] for i in result['cycles']]
            cycle_losses = [result['loss'][i-1] for i in result['cycles']]
            axes2[0].scatter(cycle_costs, cycle_losses,
                            color=method_colors[method_name],
                            marker=method_markers.get(method_name, "o"),
                            s=80,
                            label=f"cycles_{method_name}")

axes2[0].set_xlabel('Cost (operations count)')
axes2[0].set_ylabel('Loss')
axes2[0].set_title('Loss over Cost')
axes2[0].legend(frameon=True)
axes2[0].grid(True)

# Plot 6: PSNR vs Cost
for method_name, result in results.items():
    if result['psnr'] and result['cost']:
        axes2[1].plot(result['cost'], result['psnr'],
                    color=method_colors[method_name],
                    linestyle=method_linestyles[method_name],
                    label=method_name,
                    linewidth=2)
        if result['cycles'] is not None and len(result['cycles']) > 0:
            cycle_costs = [result['cost'][i-1] for i in result['cycles']]
            cycle_psnrs = [result['psnr'][i-1] for i in result['cycles']]
            axes2[1].scatter(cycle_costs, cycle_psnrs,
                            color=method_colors[method_name],
                            marker=method_markers.get(method_name, "o"),
                            s=80,
                            label=f"cycles_{method_name}")

axes2[1].set_xlabel('Cost (operations count)')
axes2[1].set_ylabel('PSNR (dB)')
axes2[1].set_title('PSNR over Cost')
axes2[1].legend(frameon=True)
axes2[1].grid(True)

plt.savefig(os.path.join(exp_dir, "loss_psnr_vs_cost.pdf"), dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig2)


# Plot the reconstructions together
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
    else:
        print(f"No reconstruction for {method_name}")

dinv.utils.plot(
    images,
    titles=titles,
    subtitles=subtitles,
    cmap="gray",
    tight=False,
    save_fn=os.path.join(exp_dir, "all_reconstructions.pdf")
)