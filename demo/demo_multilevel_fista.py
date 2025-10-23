#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025
@author: Guillaume Lauga
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
import deepinv as dinv
import torch
from torchvision.io import read_image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import copy
from deepinv.loss.metric import PSNR
from multilevel.multilevel import ParametersMultilevel, MultiLevel
from tqdm import tqdm
perf_psnr = PSNR()

# Define device
device = torch.device("cpu")
print(f'device is {device}')


# image_path = "path_to_image/image.png"  # Replace with your image path
# image_file = read_image(image_path)
# x_true = image_file.unsqueeze(0).to(torch.float32).to(device)/255

# Download an image
file_name = "butterfly.png"
url = f"https://huggingface.co/datasets/deepinv/images/resolve/main/{file_name}?download=true"
x_true = dinv.utils.load_url_image(url=url, img_size=256).to(device)
# x_true = x_true[:, :, ::4, ::4]  # downsample by a factor of 4
# Define the Forward Operator: study case of deblurring + Gaussian noise
#-----------------------------------------------------------------------
# Load a forward operator $A$ and generate some (noisy) measurements.
# The full list of operators is available here:
# (https://deepinv.github.io/deepinv/deepinv.physics.html).

# Define linear operator
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding='reflect')
seed = torch.manual_seed(0) # Random seed for reproducibility

# physics = dinv.physics.Inpainting(mask=0.5, tensor_size=x_true.shape[1:])
sigma = 0.01

# Define noise
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Construct observation and display original image
y = physics(x_true)
yML = y.clone()
back = physics.A_adjoint(y)

# dinv.utils.plot([x_true, y, back], titles=['original','observation','backprojection'])



# Reconstruction PnP with Forward-Backward algorithm
#---------------------------------------------------


# Define data fidelity term
data_fidelity = dinv.optim.L2()

# Define prior
# args_prior = "TV"
args_prior = "Wavelet"

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


# Define regularization parameter
# param_regularization = 2*sigma**2
param_regularization = 1e-2

# Define algorithm parameters
random_tensor   = torch.randn(x_true.shape).to(device)
Anorm2          = physics.compute_norm(random_tensor)
param_gamma     = torch.ones(1, device= device)/ Anorm2         # Set the step-size
if args_algo == "FISTA":
    d = 1
    param_gamma = 0.95 * param_gamma
    param_gamma_ML = 1.95 * param_gamma
elif args_algo == "FB":
    d = 0
    param_gamma = 1.95 * param_gamma
    param_gamma_ML = param_gamma


param_iter      = 1000                # number of iterations
a               = 2.1                # inertia parameter

# Define multilevel parameters

levels          = 4                  # number of levels
param_coarse_iter = 5                # number of iterations at coarse level
max_multilevel_iter = 5                # maximum number of  multilevel iterations at fine level
cst_grad        = None                # only used at coarser levels. stays none at fine level.
info_transfer  = "daubechies8"            # type of information transfer
# info_transfer : plot filter_classes


xk = back.clone()
zk = back.clone()
if isinstance(prior, dinv.optim.prior.PnP):
    x_denoiser = denoiser(xk, param_gamma*param_regularization)
    initial_value = torch.zeros(1, device=device)
else:
    initial_value = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)

initial_snr_value = perf_psnr(x_true,xk).item()

# Iterations ML
args_multilevel = ParametersMultilevel(target_shape = x_true.shape[-3:], levels = levels, max_ML_steps = 1, param_coarse_iter = param_coarse_iter, step_size=param_gamma_ML, info_transfer=info_transfer, prior=prior, denoiser=denoiser, data_fidelity= data_fidelity, physics = physics, observation = y, device = device)
crit_ML = 1e10*np.ones(param_iter)
psnr_ML = 1e10*np.ones(param_iter)
with torch.no_grad():
    for k in tqdm(range(param_iter)):
        xk_prev = xk.clone()
        if k<max_multilevel_iter:
            zk = MultiLevel(zk, levels, levels-1, args_multilevel, param_regularization,cst_grad, device)
        xk = zk - param_gamma*data_fidelity.grad(zk, y, physics)
        if isinstance(prior, dinv.optim.TVPrior):
            xk = prior.prox(xk, gamma = param_gamma*param_regularization)
            crit_ML[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
        else:
            xk = prior.prox(xk, gamma = [param_gamma*param_regularization])
            crit_ML[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
        psnr_ML[k] = perf_psnr(x_true,xk).item()
        if k % 10 == 0: print(f"crit ML[{k}] / snr ML[{k}]: {crit_ML[k]} / {psnr_ML[k]}")
        if d==0:
            zk = xk
        else:
            zk = xk + ( ((k + a) / a )**d -1 ) / ((k+1+a)/a )**d * (xk - xk_prev)

x_est_ml = xk.clone()

if args_prior == "TV":
    prior = dinv.optim.TVPrior(def_crit=criterion, n_it_max=n_it_max)
    denoiser = prior.prox
elif args_prior == "Wavelet":
    prior = dinv.optim.WaveletPrior(level=4, wv="db8", p=1, device=device)
    denoiser = prior.prox
xk = back.clone()
zk = back.clone()
crit_SL = 1e10*np.ones(param_iter)
psnr_SL = 1e10*np.ones(param_iter)
d=0
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        xk = zk - param_gamma*data_fidelity.grad(zk, y, physics)
        if isinstance(prior, dinv.optim.TVPrior):
            xk = prior.prox(xk, gamma = param_gamma*param_regularization)
            crit_SL[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk) # Ajouter FISTA
        else:
            xk = prior.prox(xk, gamma = [param_gamma*param_regularization])
            crit_SL[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
        psnr_SL[k] = perf_psnr(x_true,xk).item()
        if k % 10 == 0: print(f"crit SL[{k}] / snr SL[{k}]: {crit_SL[k]} / {psnr_SL[k]}")
        if d==0:
            zk = xk
        else:
            print((k)/(k+1+a))
            print(( ((k + a) / a )**d -1 ) / ((k+1+a)/a )**d)
            zk = xk + ( ((k + a) / a )**d -1 ) / ((k+1+a)/a )**d * (xk - xk_prev)


# Compute some metrics
crit_min = min(np.min(crit_ML),np.min(crit_SL))

# Display results
psnrs = np.array([perf_psnr(x_true, xk).item() for xk in [y, xk, x_est_ml]])
psnrs = np.round(psnrs, 2)
dinv.utils.plot([x_true, y, xk, x_est_ml], titles=['original',f'observation\n PSNR: {psnrs[0]}',f'restored (SL)\n PSNR: {psnrs[1]}', f'restored (ML)\n PSNR: {psnrs[2]}'],figsize=[6,6])

fig, axs = plt.subplots(1, 3, figsize=(10, 4))  # 2 lignes, 1 colonne
axs[0].plot(np.concatenate((np.array(initial_value),crit_ML))/initial_value-crit_min*1.00001/initial_value, label='Multi-Level')  # ED : à quoi ça sert de diviser par initial_value ?
axs[0].plot(np.concatenate((np.array(initial_value),crit_SL))/initial_value-crit_min*1.00001/initial_value, label='Single-Level')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].set_title('objective function w.r.t iterations')
axs[1].plot(np.concatenate((np.array(initial_value),crit_ML)), label='Multi-Level')
axs[1].plot(np.concatenate((np.array(initial_value),crit_SL)), label='Single-Level')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].set_title('Log objective function w.r.t iterations')
axs[2].plot(np.concatenate((np.array([initial_snr_value]),psnr_ML)), label='Multi-Level')
axs[2].plot(np.concatenate((np.array([initial_snr_value]),psnr_SL)), label='Single-Level')
axs[2].legend()
axs[2].set_title('PSNR function w.r.t iterations')
plt.tight_layout()
plt.show()

print(f"Final value of the objective function for ML: {crit_ML[-1]}\n Final value of the objective function for SL: {crit_SL[-1]}")

diff_obj_fun = crit_ML - crit_SL
plt.figure(figsize=(10, 4))
plt.plot(diff_obj_fun, label='Difference in objective function (ML - SL)')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.xlabel('Iteration')
plt.ylabel('Difference in Objective Function')
plt.title(f'Difference in Objective Function between ML and SL\n Last value: {diff_obj_fun[-1]:.4f}')
plt.legend()
plt.tight_layout()
plt.show()