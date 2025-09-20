#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025
@author: Guillaume Lauga
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

'''
A faire :
Deux approches :
x_k+1 = x_k + tau * P(a^* - a0, détails propres-d0)
= P (a0 + tau(a^* - a0), d0 + tau*(détails propres-d0))
x_k+1 = P(a^*, détails propres),
Regarder si tau = 1 (alors les deux approches sont équivalentes)
'''

perf_psnr = PSNR()

plt.rcParams["text.usetex"] = True  # Activate LaTeX rendering

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")

#%% ----- Initialization -----

file_name = "butterfly.png"

# Download an image
url = f"https://huggingface.co/datasets/deepinv/images/resolve/main/{file_name}?download=true"
x_true = dinv.utils.load_url_image(url=url).to(device)

# Define linear operator
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")
seed = torch.manual_seed(0)  # Random seed for reproducibility

'''sigma = 0.01
noise_model = dinv.physics.GaussianNoise(sigma=sigma)
physics = dinv.physics.Inpainting(tensor_size=x_true.shape[1:], mask=0.8, device=device, noise_model=noise_model)
seed = torch.manual_seed(0)  # Random seed for reproducibility'''

sigma = 0.01

# Define noise
#physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Construct observation and display original image
y = physics(x_true)
yML = y.clone()
back = physics.A_adjoint(y)

dinv.utils.plot([x_true, y, back], titles=['original','observation','backprojection'])

# Define data fidelity term
data_fidelity = dinv.optim.L2()

#%% ----- Parameters -----

# Define prior
args_prior = "TV"
# args_prior = "Wavelet"

# Define single level algorithm
# args_algo = "FISTA"
args_algo = "FB"

if args_prior == "TV":
    print('Using TV prior')
    criterion = 1e-5  # Parameters for computing the TV prior
    n_it_max = 50
    prior = dinv.optim.TVPrior(def_crit=criterion, n_it_max=n_it_max)
    denoiser = prior.prox
elif args_prior == "Wavelet":
    print('Using Wavelet prior')
    prior = dinv.optim.WaveletPrior(level=4, wv="haar", p=1, device=device)
    denoiser = prior.prox

# Define regularization parameter
# param_regularization = 2*sigma**2 # From the Bayesian interpretation
param_regularization = 1e-2

# Define algorithm parameters
random_tensor = torch.randn(x_true.shape).to(device)
Anorm2 = physics.compute_norm(random_tensor)
param_gamma = torch.ones(1, device=device) / Anorm2  # Set the step-size
if args_algo == "FISTA":
    print('Using FISTA algorithm')
    d = 1
    param_gamma = 0.95 * param_gamma
    param_gamma_ML = 1.95 * param_gamma
elif args_algo == "FB":
    print('Using FB algorithm')
    d = 0
    #param_gamma = 1.95 * param_gamma
    param_gamma = 0.1 * param_gamma # Initial value : 0.95 * param_gamma
    param_gamma_ML = param_gamma

'''param_gamma = 0.95 * torch.ones(1, device=device) / Anorm2  # For coherence we use this step-size for all algorithms
param_gamma_gamma_ML = 1.95 * param_gamma'''
param_iter = 10000  # number of iterations
a = 2.1  # inertia parameter

# Define multilevel parameters
levels = 4                     # number of levels
param_coarse_iter = 5          # number of iterations at coarse level
max_multilevel_iter = 5        # maximum number of  multilevel iterations at fine level
cst_grad = None                # only used at coarser levels. stays none at fine level
info_transfer = "daubechies8"  # type of information transfer

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


#%% ----- I) Multilevel Iterations with Conditional Denoiser -----

crit_ML_cond = 1e10 * np.ones(param_iter)
psnr_ML_cond = perf_psnr(back, x_true).item() * np.ones(param_iter)
diff_ML_cond = []
data_fidelity_crit_ML_cond = []

denoiser_cond = WaveletDenoiserConditional(level=levels, wv="db8", device=device, non_linearity="soft")

xk = back.clone()

start = time.time()
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()

        if k < max_multilevel_iter:
            # Multilevel step
            xk = MultiLevelWavelets(
                xk,
                levels,
                levels - 1,
                args_multilevel,
                param_regularization,
                cst_grad,
                device,
            )
        # Fine gradient step
        xk = xk - param_gamma * data_fidelity.grad(xk, y, physics)
        # Fine proximal step (using conditional denoiser as the prox)
        xk = denoiser_cond(xk, gamma=param_regularization * param_gamma)

        psnr_ML_cond[k] = perf_psnr(x_true, xk).item()
        data_fidelity_crit_ML_cond.append(data_fidelity(xk, y, physics).item())
        diff_ML_cond.append(torch.norm(xk - xk_prev).item())

        if k % 10 == 0:
            print(f"crit ML Cond[{k}] / snr ML Cond[{k}]: No obj. fun. / {psnr_ML_cond[k]}")

x_est_ml_cond = xk.clone()
end = time.time()
time_ML_cond = end - start

plt.plot(psnr_ML_cond)
plt.show()

dinv.utils.plot(x_est_ml_cond)

import sys
sys.exit()
#%% ----- II) Multilevel FB -----

crit_ML = 1e10 * np.ones(param_iter)
psnr_ML = 1e10 * np.ones(param_iter)
diff_ML = []
data_fidelity_crit_ML = []

xk = back.clone()

start = time.time()
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()

        if k < max_multilevel_iter:
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

        crit_ML[k] = data_fidelity(
            xk, y, physics
        ) + param_regularization * prior.fn(xk)
        psnr_ML[k] = perf_psnr(x_true, xk).item()
        data_fidelity_crit_ML.append(data_fidelity(xk, y, physics).item())
        diff_ML.append(torch.norm(xk - xk_prev).item())

        if k % 10 == 0:
            print(f"crit ML[{k}] / snr ML[{k}]: {crit_ML[k]} / {psnr_ML[k]}")

x_est_ml = xk.clone()
end = time.time()
time_ML = end - start


#%% ----- III) Multilevel FB with Conditional Denoiser (with coherence) -----

crit_ML_cond_coherence = 1e10 * np.ones(param_iter)
psnr_ML_cond_coherence = 1e10 * np.ones(param_iter)
diff_ML_cond_coherence = []
data_fidelity_crit_ML_cond_coherence = []

denoiser_cond = WaveletDenoiserConditional(level=levels, wv="db8", device=device, non_linearity="soft")

xk = back.clone()

start = time.time()
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()

        if k < max_multilevel_iter:
            # Multilevel step
            xk = MultiLevelWavelets(
                xk,
                levels,
                levels - 1,
                args_multilevel,
                param_regularization,
                cst_grad,
                device,
                use_coherence=True,
                use_initial_linesearch=True,
            )

        # Fine gradient step
        xk = xk - param_gamma * data_fidelity.grad(xk, y, physics)
        if isinstance(prior, dinv.optim.TVPrior):
            xk = prior.prox(xk, gamma=param_gamma * param_regularization)
        else:
            xk = prior.prox(xk, gamma=[param_gamma * param_regularization])
        # Fine proximal step (using conditional denoiser as the prox)
        # xk = denoiser_cond(xk, gamma=param_regularization * param_gamma)

        psnr_ML_cond_coherence[k] = perf_psnr(x_true, xk).item()
        data_fidelity_crit_ML_cond_coherence.append(data_fidelity(xk, y, physics).item())
        diff_ML_cond_coherence.append(torch.norm(xk - xk_prev).item())

        if k % 10 == 0:
            print(f"crit ML Cond + Coherence[{k}] / snr ML Cond + Coherence[{k}]: No obj. fun. / {psnr_ML_cond_coherence[k]}")

x_est_ml_cond_coherence = xk.clone()
end = time.time()
time_ML_cond_coherence = end - start


#%% ----- IV) Classical single level iterations -----

crit_SL = 1e10 * np.ones(param_iter)
psnr_SL = 1e10 * np.ones(param_iter)
d = 0
diff_SL = []
data_fidelity_crit_SL = []

xk = back.clone()

start = time.time()
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        xk = xk - param_gamma * data_fidelity.grad(xk, y, physics)

        if isinstance(prior, dinv.optim.TVPrior):
            xk = prior.prox(xk, gamma=param_gamma * param_regularization)
            crit_SL[k] = data_fidelity(
                xk, y, physics
            ) + param_regularization * prior.fn(
                xk
            )  # Ajouter FISTA
        else:
            xk = prior.prox(xk, gamma=[param_gamma * param_regularization])
            crit_SL[k] = data_fidelity(
                xk, y, physics
            ) + param_regularization * prior.fn(xk)

        psnr_SL[k] = perf_psnr(x_true, xk).item()
        data_fidelity_crit_SL.append(data_fidelity(xk, y, physics).item())
        diff_SL.append(torch.norm(xk - xk_prev).item())

        if k % 10 == 0:
            print(f"crit SL[{k}] / snr SL[{k}]: {crit_SL[k]} / {psnr_SL[k]}")

x_est_sl = xk.clone()
end = time.time()
time_SL = end - start


#%% ----- V) Single level iterations with conditional denoiser as prox -----

psnr_SL_cond = 1e10 * np.ones(param_iter)
diff_SL_cond = []
data_fidelity_crit_SL_cond = []

xk = back.clone()

start = time.time()
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        xk = xk - param_gamma * data_fidelity.grad(xk, y, physics)
        xk = denoiser_cond(xk, gamma=param_regularization * param_gamma)

        psnr_SL_cond[k] = perf_psnr(x_true, xk).item()
        diff_SL_cond.append(torch.norm(xk - xk_prev).item())
        data_fidelity_crit_SL_cond.append(data_fidelity(xk, y, physics).item())

        if k % 10 == 0:
            print(f"crit SL cond[{k}] / snr SL Cond[{k}]: No obj. fun. / {psnr_SL_cond[k]}")

x_est_sl_cond = xk.clone()
end = time.time()
time_SL_cond = end - start


#%% ----- VI) Single level Plug and Play -----

param_gamma_pnp = 0.95 * torch.ones(1, device=device) / Anorm2  # Set the step-size for PnP
param_regularization_pnp = sigma  # Regularization parameter for PnP

psnr_SL_pnp = 1e10 * np.ones(param_iter)
diff_SL_pnp = []
data_fidelity_crit_SL_pnp = []

xk = back.clone()

start = time.time()
denoiser_pnp = dinv.models.DRUNet(
    in_channels=3,
    out_channels=3,
    pretrained='download',
    device=device
)

with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        xk = xk - param_gamma_pnp * data_fidelity.grad(xk, y, physics)
        xk = denoiser_pnp(xk, sigma=param_regularization_pnp) # We need to know sigma for the denoising step

        psnr_SL_pnp[k] = perf_psnr(x_true, xk).item()
        diff_SL_pnp.append(torch.norm(xk - xk_prev).item())
        data_fidelity_crit_SL_pnp.append(data_fidelity(xk, y, physics).item())

        if k % 10 == 0:
            print(f"crit SL PnP[{k}] / snr SL PnP[{k}]: No obj. fun. / {psnr_SL_pnp[k]}")

x_est_sl_pnp = xk.clone()
end = time.time()
time_SL_pnp = end - start


#%% ----- Display results -----

# Plot ML vs SL objective function evolution
plt.figure(figsize=(10, 5))
plt.plot(crit_ML, linestyle="-", color="blue", label="ML")
plt.plot(crit_SL, linestyle="--", color="green", label="SL")
plt.title("Convergence of ML and SL Algorithms : objective function")
plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot the error evolution for each algorithm
plt.figure(figsize=(10, 5))
plt.plot(diff_ML, linestyle="-", color="blue", label=f"ML {args_algo}")
plt.plot(diff_SL, linestyle="--", color="green", label="SL")
plt.plot(diff_ML_cond, linestyle=":", color="red", label="ML with conditional denoiser")
plt.plot(diff_SL_cond, linestyle=":", color="orange", label="SL with conditional denoiser")
plt.plot(diff_SL_pnp, linestyle="--", color="purple", label="SL PnP")
plt.plot(diff_ML_cond_coherence, linestyle=":", color="brown", label="ML with conditional denoiser (+ coherence)")
plt.title("Convergence of ML and SL Algorithms")
plt.xlabel("Iteration")
plt.ylabel(r"$\|x_k - x_{k-1}\|_2$")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot PSNR and data fidelity evolutions for each algorithm
plt.figure(figsize=(10, 5))
plt.plot(psnr_ML, linestyle="-", color="blue", label=f"ML {args_algo}")
plt.plot(psnr_SL, linestyle="--", color="green", label="SL")
plt.plot(psnr_ML_cond, linestyle=":", color="red", label="ML with conditional denoiser")
plt.plot(psnr_SL_cond, linestyle=":", color="orange", label="SL with conditional denoiser")
plt.plot(psnr_SL_pnp, linestyle="--", color="purple", label="SL PnP")
plt.plot(psnr_ML_cond_coherence, linestyle=":", color="brown", label="ML with conditional denoiser (+ coherence)")
plt.title("PSNR of ML and SL Algorithms")
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot data fidelity evolution for each algorithm
plt.figure(figsize=(10, 5))
plt.plot(data_fidelity_crit_ML, linestyle="-", color="blue", label=f"ML {args_algo}")
plt.plot(data_fidelity_crit_SL, linestyle="--", color="green", label="SL")
plt.plot(data_fidelity_crit_ML_cond, linestyle=":", color="red", label="ML with conditional denoiser")
plt.plot(data_fidelity_crit_SL_cond, linestyle=":", color="orange", label="SL with conditional denoiser")
plt.plot(data_fidelity_crit_SL_pnp, linestyle="--", color="purple", label="SL PnP")
plt.plot(data_fidelity_crit_ML_cond_coherence, linestyle=":", color="brown", label="ML with conditional denoiser (+ coherence)")
plt.title("Data Fidelity of ML and SL Algorithms")
plt.xlabel("Iteration")
plt.ylabel(r"$\|Ax_k - y\|_2$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot the reconstructions
psnrs = np.array(
    [perf_psnr(x_true, x).item() for x in [y, x_est_sl, x_est_ml, x_est_ml_cond, x_est_sl_cond, x_est_sl_pnp, x_est_ml_cond_coherence]]
)
psnrs = np.round(psnrs, 3)

best_idx = np.argmax(psnrs)

def format_title(name, psnr, time, idx, best_idx):
    psnr_str = f"$\\mathbf{{{psnr}}}$" if idx == best_idx else f"{psnr}"
    return f"{name}\n PSNR: {psnr_str}\n Time: {time:.2f}s"

titles = [
    "original",
    format_title("observation", psnrs[0], time_ML_cond, 0, best_idx),
    format_title("reconstruction single level", psnrs[1], time_SL, 1, best_idx),
    format_title(f"ML {args_algo}", psnrs[2], time_ML, 2, best_idx),
    format_title("ML w/ conditional denoiser", psnrs[3], time_ML_cond, 3, best_idx),
    format_title("SL w/ conditional denoiser", psnrs[4], time_SL_cond, 4, best_idx),
    format_title("SL PnP", psnrs[5], time_SL_pnp, 5, best_idx),
    format_title("ML w/ conditional denoiser (+ coherence)", psnrs[6], time_ML_cond_coherence, 6, best_idx),
]

dinv.utils.plot(
    [x_true, y, x_est_sl, x_est_ml, x_est_ml_cond, x_est_sl_cond, x_est_sl_pnp, x_est_ml_cond_coherence],
    titles=titles,
    cmap="gray",
)

print(f"Final value the difference in objective function: {crit_ML[-1] - crit_SL[-1]}")