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

perf_psnr = PSNR()

plt.rcParams["text.usetex"] = True  # Activate LaTeX rendering

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")

#%% ----- Initialization -----

# Download an image
url = f"https://huggingface.co/datasets/deepinv/images/resolve/main/{file_name}?download=true"
x_true = dinv.utils.load_url_image(url=url).to(device)
print(f"Image shape: {x_true.shape}")

# Define linear operator
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")
seed = torch.manual_seed(0)  # Random seed for reproducibility

sigma = 0.01

# Define noise
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

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
args_algo = "FISTA"
# args_algo = "FB"

if args_prior == "TV":
    criterion = 1e-5  # Parameters for computing the TV prior
    n_it_max = 50
    prior = dinv.optim.TVPrior(def_crit=criterion, n_it_max=n_it_max)
    denoiser = prior.prox
elif args_prior == "Wavelet":
    prior = dinv.optim.WaveletPrior(level=4, wv="haar", p=1, device=device)
    denoiser = prior.prox

# Define regularization parameter
#param_regularization = 2*sigma**2 # From the Bayesian interpretation
param_regularization = 1e-6

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

param_iter = 35  # number of iterations
a = 2.1  # inertia parameter

# Define multilevel parameters
levels = 4                     # number of levels
param_coarse_iter = 5          # number of iterations at coarse level
max_multilevel_iter = 5        # maximum number of  multilevel iterations at fine level
cst_grad = None                # only used at coarser levels. stays none at fine level
info_transfer = "daubechies8"  # type of information transfer


xk = back.clone()
zk = back.clone()
if isinstance(prior, dinv.optim.prior.PnP):
    x_denoiser = denoiser(xk, param_gamma * param_regularization)
    initial_value = torch.zeros(1, device=device)
else:
    initial_value = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)

initial_snr_value = perf_psnr(x_true, xk).item()

#%% ----- Multilevel Iterations with Conditional Denoiser -----
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
crit_ML_cond = 1e10 * np.ones(param_iter)
psnr_ML_cond = 1e10 * np.ones(param_iter)
diff_ML_cond = []

denoiser_cond = WaveletDenoiserConditional(level=levels, wv="db8", device=device, non_linearity="soft")

with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()

        if k < max_multilevel_iter:
            # Multilevel step
            zk = MultiLevelWavelets(
                zk,
                levels,
                levels - 1,
                args_multilevel,
                param_regularization,
                cst_grad,
                device,
            )
        # Fine gradient step
        xk = zk - param_gamma * data_fidelity.grad(zk, y, physics)
        # Fine proximal step (using conditional denoiser as the prox)
        xk = denoiser_cond(xk, gamma=param_regularization * param_gamma)

        psnr_ML_cond[k] = perf_psnr(x_true, xk).item()

        if k % 10 == 0:
            print(f"crit ML[{k}] / snr ML[{k}]: {crit_ML_cond[k]} / {psnr_ML_cond[k]}")

        if d == 0:
            zk = xk
        else:
            zk = xk + (((k + a) / a) ** d - 1) / ((k + 1 + a) / a) ** d * (xk - xk_prev)

        diff_ML_cond.append(torch.norm(xk - xk_prev).item())

x_est_ml_cond = xk.clone()

#%% ----- Multilevel FISTA -----

# Iterations ML
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
crit_ML = 1e10 * np.ones(param_iter)
psnr_ML = 1e10 * np.ones(param_iter)
diff_ML = []

xk = back.clone()
zk = back.clone()
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        if k < max_multilevel_iter:
            zk = MultiLevel(
                zk,
                levels,
                levels - 1,
                args_multilevel,
                param_regularization,
                cst_grad,
                device,
            )
        xk = zk - param_gamma * data_fidelity.grad(zk, y, physics)
        if isinstance(prior, dinv.optim.TVPrior):
            xk = prior.prox(xk, gamma=param_gamma * param_regularization)
            crit_ML[k] = data_fidelity(
                xk, y, physics
            ) + param_regularization * prior.fn(xk)
        else:
            xk = prior.prox(xk, gamma=[param_gamma * param_regularization])
            crit_ML[k] = data_fidelity(
                xk, y, physics
            ) + param_regularization * prior.fn(xk)
        psnr_ML[k] = perf_psnr(x_true, xk).item()
        if k % 10 == 0:
            print(f"crit ML[{k}] / snr ML[{k}]: {crit_ML[k]} / {psnr_ML[k]}")
        if d == 0:
            zk = xk
        else:
            zk = xk + (((k + a) / a) ** d - 1) / ((k + 1 + a) / a) ** d * (xk - xk_prev)
        diff_ML.append(torch.norm(xk - xk_prev).item())

x_est_ml = xk.clone()


#%% ----- Classical single level iterations -----
if args_prior == "TV":
    prior = dinv.optim.TVPrior(def_crit=criterion, n_it_max=n_it_max)
    denoiser = prior.prox
elif args_prior == "Wavelet":
    prior = dinv.optim.WaveletPrior(level=4, wv="db8", p=1, device=device)
    denoiser = prior.prox
xk = back.clone()
zk = back.clone()
crit_SL = 1e10 * np.ones(param_iter)
psnr_SL = 1e10 * np.ones(param_iter)
d = 0
diff_SL = []

with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        xk = zk - param_gamma * data_fidelity.grad(zk, y, physics)

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

        if k % 10 == 0:
            print(f"crit SL[{k}] / snr SL[{k}]: {crit_SL[k]} / {psnr_SL[k]}")

        if d == 0:
            zk = xk
        else:
            print((k) / (k + 1 + a))
            print((((k + a) / a) ** d - 1) / ((k + 1 + a) / a) ** d)
            zk = xk + (((k + a) / a) ** d - 1) / ((k + 1 + a) / a) ** d * (xk - xk_prev)
        diff_SL.append(torch.norm(xk - xk_prev).item())

#%% ----- Display results -----

# Plot ML vs SL crit
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


plt.figure(figsize=(10, 5))
plt.plot(diff_ML, linestyle="-", color="blue", label="ML")
plt.plot(diff_SL, linestyle="--", color="green", label="SL")
plt.plot(diff_ML_cond, linestyle=":", color="red", label="ML with conditional denoiser")
plt.title("Convergence of ML and SL Algorithms")
plt.xlabel("Iteration")
plt.ylabel(r"$\|x_k - x_{k-1}\|_2$")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(psnr_ML, linestyle="-", color="blue", label="ML")
plt.plot(psnr_SL, linestyle="--", color="green", label="SL")
plt.plot(psnr_ML_cond, linestyle=":", color="red", label="ML with conditional denoiser")
plt.title("PSNR of ML and SL Algorithms")
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

psnrs = np.array(
    [perf_psnr(x_true, x).item() for x in [y, xk, x_est_ml, x_est_ml_cond]]
)
psnrs = np.round(psnrs, 3)

dinv.utils.plot(
    [x_true, y, xk, x_est_ml, x_est_ml_cond],
    titles=[
        "original",
        f"observation\n PSNR: {psnrs[0]}",
        f"reconstruction single level\n PSNR: {psnrs[1]}",
        f"ML FISTA\n PSNR: {psnrs[2]}",
        f"ML w/ conditional denoiser\n PSNR: {psnrs[3]}",
    ],
    cmap="gray",
)

print(f"Final value the difference in objective function: {crit_ML[-1] - crit_SL[-1]}")