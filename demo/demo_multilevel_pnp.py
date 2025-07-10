#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2025
@author: Nils Laurent
"""
import matplotlib.pyplot as plt
#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv


import torch
from torchvision.io import read_image

from multilevel.info_transfer import DownsamplingTransfer, SincFilter
from multilevel.minimal_wrapper import perf_psnr
from multilevel.multilevel import ParametersMultilevel, MultiLevel
from multilevel.multilevel_initialization import ml_init_pnp
import deepinv as dinv

device = 'cpu'

# Load image from deepinv
x = dinv.utils.load_url_image(url=dinv.utils.get_image_url("butterfly.png"), img_size=256).to(device)

experiment_demosaicing = True  # True for demosaicing or False for inpainting

if experiment_demosaicing is True:  # demosaicing
    # Define linear operator and hyper parameters
    physics = dinv.physics.Demosaicing(
        x.shape[1:], device=device, noise_model=dinv.physics.GaussianNoise(0.1)
    )
    step_size = 1.17
    regularization = 0.0801
else:  # inpainting
    # Define linear operator and hyper parameters
    physics = dinv.physics.Inpainting(
        tensor_size=x.shape[1:], mask=0.5, device=device, noise_model=dinv.physics.GaussianNoise(0.1)
    )
    step_size = 1.0
    regularization = 0.0801

# generate measurement
y = physics(x)
back = physics.A_adjoint(y)

# ML PnP paper settings
#denoiser_0 = dinv.models.DnCNN(in_channels=3, out_channels=3, device=device, pretrained="download_lipschitz")
denoiser_0 = dinv.models.DRUNet(in_channels=3, out_channels=3, device=device, pretrained="download")
denoiser = dinv.models.EquivariantDenoiser(denoiser_0, random=True)
prior = dinv.optim.prior.PnP(denoiser=denoiser)
data_fidelity = dinv.optim.L2()
levels = 4
max_ML_steps = 2
info_transfer = "sinc"

args_multilevel = ParametersMultilevel(
    target_shape=x.shape[-3:],
    levels=levels,
    max_ML_steps=max_ML_steps,
    param_coarse_iter=-1,  # will be defined later
    step_size=step_size,
    info_transfer=info_transfer,
    prior=prior,
    denoiser=denoiser,
    data_fidelity=data_fidelity,
    physics=physics,
    observation=y,
    device=device
)

coarse_observations = {f'level{levels}': y}
coarse_y = y
for i in range(levels-1, 0, -1):
    ds = DownsamplingTransfer(SincFilter())
    coarse_y = ds.to_coarse(coarse_y, coarse_y.shape[-3:])
    coarse_observations[f'level{i}'] = coarse_y
args_multilevel.observations = coarse_observations

coarse_physics = {f'level{levels}': physics}
coarse_data = physics.mask.data
for i in range(levels-1, 0, -1):
    coarse_data = args_multilevel.information_transfer.to_coarse(coarse_data, coarse_data.shape)
    coarse_physics[f'level{i}'] = dinv.physics.Inpainting(
        tensor_size=coarse_data.shape[1:], mask=coarse_data, device=physics.mask.device
    )
args_multilevel.coarse_physics = coarse_physics

with torch.no_grad():
    print("initialize ...")
    init = back.clone()
    PSNR_init = perf_psnr(x, init).item()

    args_multilevel.param_coarse_iter = 5
    ml_init = ml_init_pnp(init, levels, levels - 1, args_multilevel, regularization, denoiser, device)
    PSNR_ML_init = perf_psnr(x, ml_init).item()

    print("solver is running ...")
    args_multilevel.param_coarse_iter = 3

    psnr_sequence = [PSNR_init, PSNR_ML_init]
    xk = ml_init
    for k in range(20):
        xk_prev = xk.clone()
        if k < max_ML_steps:
            cst_grad = None  # coherence not required on finest level
            uk = MultiLevel(xk_prev, levels, levels-1, args_multilevel, regularization, cst_grad, device)
        else:
            uk = xk_prev
        xk = uk - step_size*data_fidelity.grad(uk, y, physics)
        xk = denoiser(xk, sigma=regularization)
        psnr_sequence.append(perf_psnr(x, xk).item())
        if k % 10 == 0:
            print(f"psnr ML[{k}]: {psnr_sequence[-1]}")
    print("done.")

    x_IMLPNP = xk
    PSNR_out = psnr_sequence[-1]

plt.figure()
plt.plot(range(len(psnr_sequence)), psnr_sequence)

dinv.utils.plot(
    [x, x_IMLPNP, ml_init, y],
    titles=[
        "Original",
        f"IMLPNP - PSNR = {PSNR_out:.2f}dB",
        f"ML init PSNR = {PSNR_ML_init:.2f}dB",
        f"Measurements PSNR = {PSNR_init:.2f}dB",
    ],
    figsize=[10, 3]
)

