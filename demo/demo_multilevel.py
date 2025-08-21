#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:24:27 2024
@author: Nelly Pustelnik
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
perf_psnr = PSNR()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')

# Download an image
# image_path = "images/393035.jpg"
# image_path = "images/cameraman-image-1024.jpeg"
# image_file = read_image(image_path)
# x_true = image_file.unsqueeze(0).to(torch.float32).to(device)/255
# x_true = x_true[:, :, ::2, ::2]  # downsample by a factor of 4

url = f"https://huggingface.co/datasets/deepinv/images/resolve/main/butterfly.png?download=true"
x_true = dinv.utils.load_url_image(url=url).to(device)

# Define the Forward Operator: study case of deblurring + Gaussian noise
#-----------------------------------------------------------------------
# Load a forward operator $A$ and generate some (noisy) measurements.
# The full list of operators is available here:
# (https://deepinv.github.io/deepinv/deepinv.physics.html).

# Define linear operator
# filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
# physics = dinv.physics.Blur(filter_0, device=device, padding='reflect')
seed = torch.manual_seed(0) # Random seed for reproducibility

#physics = dinv.physics.Inpainting(mask=0.7, tensor_size=x_true.shape[1:])

filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")

sigma=0.01
# Define noise
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Display original and degraded image
y = physics(x_true)
back = physics.A_adjoint(y)
# dinv.utils.plot([x_true, y, back], titles=['original','observation','backprojection'])



# Reconstruction PnP with Forward-Backward algorithm
#---------------------------------------------------


# Define data fidelity term
data_fidelity = dinv.optim.L2()

# Define prior
args_prior = "TV"
# args_prior = "Wavelet"
# args_prior = "PnP"

if args_prior == "TV":
    prior = dinv.optim.TVPrior(def_crit=1e-6, n_it_max=50)
    denoiser = prior.prox
elif args_prior == "Wavelet":
    prior = dinv.optim.WaveletPrior(level=4, wv="db8", p=1, device=device)
    denoiser = prior.prox
elif args_prior == "PnP":
    denoiser = dinv.models.DRUNet(in_channels=3, out_channels=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', pretrained='download', device=None)
    # denoiser = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz", device=device)
    prior = dinv.optim.prior.PnP(denoiser=denoiser)
# xk_prox = prior.prox(back, gamma=0.005)

# Define regularization parameter
param_regularization    = 1e-6

# Define algorithm parameters
random_tensor   = torch.randn(x_true.shape).to(device)
Anorm2          = physics.compute_norm(random_tensor)
param_gamma     = 0.9* torch.ones(1, device= device)/ Anorm2         # Set the step-size
param_iter      = 100                 # number of iterations

# Define multilevel parameters

levels          = 3                  # number of levels
param_coarse_iter = 5                # number of iterations at coarse level
max_multilevel_iter = 5                # maximum number of  multilevel iterations at fine level
cst_grad        = None                # only used at coarser levels. stays none at fine level.
info_transfer  = "daubechies8"            # type of information transfer
# info_transfer : plot filter_classes
args_multilevel = ParametersMultilevel(target_shape = x_true.shape[-3:], levels = levels, max_ML_steps = 1, param_coarse_iter = param_coarse_iter, step_size=param_gamma, info_transfer=info_transfer, prior=prior, denoiser=denoiser, data_fidelity= data_fidelity, physics = physics, observation = y, device = device)

xk = back.clone()
if isinstance(prior, dinv.optim.prior.PnP):
    x_denoiser = denoiser(xk, param_gamma*param_regularization)
    initial_value = torch.zeros(1, device=device)
else:
    initial_value = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)

initial_snr_value = perf_psnr(x_true,xk).item()

crit_SL = 1e10*np.ones(param_iter)
psnr_SL = 1e10*np.ones(param_iter)
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        xk = xk - param_gamma*data_fidelity.grad(xk, y, physics)
        if isinstance(prior, dinv.optim.prior.PnP):
            if isinstance(denoiser, dinv.models.DRUNet):
                xk = denoiser(xk, sigma)
            if isinstance(denoiser, dinv.models.DnCNN):
                xk = denoiser(xk, param_gamma*param_regularization)
            crit_SL[k] = torch.linalg.norm(xk.flatten()-xk_prev.flatten())
        elif isinstance(prior, dinv.optim.TVPrior):
            xk = prior.prox(xk, gamma = param_gamma*param_regularization)
            crit_SL[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk) # Ajouter FISTA
        else:
            xk = prior.prox(xk, gamma = [param_gamma*param_regularization])
            crit_SL[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
        psnr_SL[k] = perf_psnr(x_true,xk).item()
        if k % 10 == 0: print(f"crit SL[{k}] / snr SL[{k}]: {crit_SL[k]} / {psnr_SL[k]}")


# Iterations ML

xk = back.clone()

crit_ML = 1e10*np.ones(param_iter)
psnr_ML = 1e10*np.ones(param_iter)
with torch.no_grad():
    for k in range(param_iter):
        xk_prev = xk.clone()
        if k<max_multilevel_iter:
            xk = MultiLevel(xk, levels, levels-1, args_multilevel, param_regularization, cst_grad, device)
        xk = xk - param_gamma*data_fidelity.grad(xk, y, physics)
        if isinstance(prior, dinv.optim.prior.PnP):
            if isinstance(denoiser, dinv.models.DRUNet):
                xk = denoiser(xk, sigma)
            if isinstance(denoiser, dinv.models.DnCNN):
                xk = denoiser(xk, param_gamma*param_regularization)
            crit_ML[k] = torch.linalg.norm(xk.flatten()-xk_prev.flatten())
        elif isinstance(prior, dinv.optim.TVPrior):
            xk = prior.prox(xk, gamma = param_gamma*param_regularization)
            crit_ML[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
        else:
            xk = prior.prox(xk, gamma = [param_gamma*param_regularization])
            crit_ML[k] = data_fidelity(xk, y, physics) + param_regularization * prior.fn(xk)
        psnr_ML[k] = perf_psnr(x_true,xk).item()
        if k % 10 == 0: print(f"crit ML[{k}] / snr ML[{k}]: {crit_ML[k]} / {psnr_ML[k]}")

# Compute some metrics
crit_min = min(np.min(crit_ML),np.min(crit_SL))

# Display results
dinv.utils.plot([x_true, y, xk], titles=['original','observation','restored'],figsize=[6,6])
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 2 lignes, 1 colonne
axs[0].plot(np.concatenate((np.array(initial_value),crit_ML)), label='Multi-Level') # Replace this value with the actual minimal value of the objective function if it has been computed.
axs[0].plot(np.concatenate((np.array(initial_value),crit_SL)), label='Single-Level')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].set_title('Objective function w.r.t iterations')
axs[1].plot(np.concatenate((np.array([initial_snr_value]),psnr_ML)), label='Multi-Level')
axs[1].plot(np.concatenate((np.array([initial_snr_value]),psnr_SL)), label='Single-Level')
axs[1].legend()
axs[1].set_title('PSNR function w.r.t iterations')
plt.tight_layout()
plt.show()