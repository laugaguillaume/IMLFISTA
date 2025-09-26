#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025
@author: Guillaume Lauga
"""

# pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

import deepinv as dinv
import torch
import torch.nn as nn
from torchvision.io import read_image
import numpy as np
import matplotlib as mpl
import scipy.io as sio
import copy
import pywt
import time
import os
from pathlib import Path
from multilevel.info_transfer import DownsamplingTransfer, create_filter
from deepinv.physics import Physics, Downsampling
from deepinv.models import Denoiser

PSNR = dinv.metric.PSNR()

from multilevel.utils import nabla, local_average, get_approximation_previous_scale, comparative_plot_wavelets, WaveletPriorCustom

def MultiLevel2(xk, level_max, levels, args_multilevel, param_regularization, cst_grad=None, device='cpu', x_true=None, xk_finest=None):
    """
    Multilevel step for image reconstruction
    """
    if not isinstance(args_multilevel, ParametersMultilevel):
        raise ValueError("args_multilevel must be an instance of ParametersMultilevel")
    if levels < 1:
        return xk
    # xk: current image
    # information_transfer: function to transfer information between levels
    # levels: number of levels
    # param_iter: number of iterations at coarse level
    # data_fidelity: data fidelity term
    # cst_grad: first order coherence term from previous level
    """
    Unpack parameters for readability
    """
    data_fidelity = args_multilevel.data_fidelity
    prior = args_multilevel.prior
    grad_prior = args_multilevel.grad_prior
    denoiser = args_multilevel.denoiser
    step_size = args_multilevel.step_size
    param_coarse_iter = args_multilevel.param_coarse_iter
    physics = args_multilevel.physics
    max_ML_steps = args_multilevel.max_ML_steps
    coarse_physics = args_multilevel.coarse_physics
    observations = args_multilevel.observations
    information_transfer = copy.deepcopy(args_multilevel.information_transfer)
    information_transfer._initialize_operator(xk, xk.shape[-3:])
    observation = observations[f"level{levels+1}"]
    physics = coarse_physics[f"level{levels+1}"]
    coarse_observation = observations[f"level{levels}"]
    coarse_physics = coarse_physics[f"level{levels}"]
    param_reg_fine = param_regularization
    param_reg_coarse = param_reg_fine / 4
    """
    Send information to the coarse level
    """
    if cst_grad is None:
        cst_grad_fine = None
    else:
        cst_grad_fine = cst_grad.clone()
    xk_coarse = information_transfer.to_coarse(xk, xk.shape[-3:])
    print(f"Level {levels}, base case, xk_coarse shape: {xk_coarse.shape}, coarse_physics shape: {coarse_physics.mask.shape}")
    cst_grad, coherence = compute_coherence(xk, xk_coarse, information_transfer, data_fidelity, grad_prior, cst_grad, physics, coarse_physics, observation, coarse_observation, param_reg_fine, param_reg_coarse)
    coherence = step_size*coherence.to(device)
    x0_coarse = xk_coarse.clone()
    step_coarse = 1

    """
    Optimize at coarse level
    """
    with torch.no_grad():
        for k in range(param_coarse_iter):
            if levels > 1 and k < max_ML_steps:
                print(f"Level {levels}, coarse iteration {k}, xk_coarse shape: {xk_coarse.shape}, coarse_physics shape: {coarse_physics.mask.shape}")
                xk_coarse, _, _, _ = MultiLevel(xk_coarse, level_max, levels-1, args_multilevel, param_reg_coarse, cst_grad) # Recursive call if levels > 1
            xk_coarse = xk_coarse -coherence \
                - step_size*data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics) \
                - step_size*grad_prior(xk_coarse, param_reg_coarse) # Coarse gradient descent

    # Coarse correction
    coarse_correction = xk_coarse - x0_coarse
    coarse_correction = information_transfer.to_fine(coarse_correction, xk.shape[-3:])
    xk, step_coarse = ML_linesearch(xk, level_max, levels+1, coarse_correction, cst_grad_fine, data_fidelity, observation, physics, grad_prior, denoiser, prior, param_reg_fine,  step_coarse*2)
    # xk = xk + step_coarse* coarse_correction
    # print(f"Step size at level {levels}: {step_coarse}")
    return xk, [], [], []


def MultiLevel(
    xk,
    level_max,
    levels,
    args_multilevel,
    param_regularization,
    cst_grad=None,
    device="cpu",
    x_true=None,
    xk_finest=None,
):
    """
    Multilevel step for image reconstruction
    """
    if not isinstance(args_multilevel, ParametersMultilevel):
        raise ValueError("args_multilevel must be an instance of ParametersMultilevel")
    if levels < 1:
        return xk
    # xk: current image
    # information_transfer: function to transfer information between levels
    # levels: number of levels
    # param_iter: number of iterations at coarse level
    # data_fidelity: data fidelity term
    # cst_grad: first order coherence term from previous level
    """
    Unpack parameters for readability
    """
    data_fidelity = args_multilevel.data_fidelity
    prior = args_multilevel.prior
    grad_prior = args_multilevel.grad_prior
    denoiser = args_multilevel.denoiser
    step_size = args_multilevel.step_size
    param_coarse_iter = args_multilevel.param_coarse_iter
    physics = args_multilevel.physics
    physics_fine = args_multilevel.physics
    max_ML_steps = args_multilevel.max_ML_steps
    coarse_physics = args_multilevel.coarse_physics
    observations = args_multilevel.observations
    information_transfer = copy.deepcopy(args_multilevel.information_transfer)
    information_transfer._initialize_operator(xk, xk.shape[-3:])
    observation = observations[f"level{levels+1}"]
    physics = coarse_physics[f"level{levels+1}"]
    coarse_observation = observations[f"level{levels}"]
    coarse_physics = coarse_physics[f"level{levels}"]
    param_reg_fine = param_regularization
    losses, times, psnr_list = [], [], []
    xk_finest = xk_finest.clone() if xk_finest is not None else None
    if isinstance(prior, dinv.optim.prior.PnP):
        param_reg_coarse = param_reg_fine
    else:
        param_reg_coarse = param_reg_fine / 4
    """
    Send information to the coarse level
    """
    if cst_grad is None:
        cst_grad_fine = None
    else:
        cst_grad_fine = cst_grad.clone()
    xk_coarse = information_transfer.to_coarse(xk, xk.shape[-3:])
    cst_grad, coherence = compute_coherence(
        xk,
        xk_coarse,
        information_transfer,
        data_fidelity,
        grad_prior,
        cst_grad,
        physics,
        coarse_physics,
        observation,
        coarse_observation,
        param_reg_fine,
        param_reg_coarse,
    )
    coherence = step_size * coherence.to(device)
    x0_coarse = xk_coarse.clone()
    step_coarse = 1

    """
    Optimize at coarse level
    """
    with torch.no_grad():
        for k in range(
            param_coarse_iter
        ):
            if levels > 1 and k < max_ML_steps:
                xk_coarse, losses_intermediate, times_intermediate, psnrs_intermediate = MultiLevel(
                    xk_coarse,
                    level_max,
                    levels - 1,
                    args_multilevel,
                    param_reg_coarse,
                    cst_grad,
                    x_true=x_true,
                    xk_finest=xk_finest,
                )  # Recursive call if levels > 1
                '''losses += losses_intermediate
                times += times_intermediate
                psnr_list += psnrs_intermediate'''

            if isinstance(prior, dinv.optim.prior.PnP):
                xk_coarse = prior.denoiser(
                    xk_coarse
                    - coherence
                    - step_size
                    * data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics),
                    param_reg_coarse,
                )
                '''losses.append(data_fidelity(information_transfer.to_fine(xk_coarse, xk.shape[-3:]), observation, physics).item())
                times.append(time.process_time())
                if x_true is not None:
                    psnr_list.append(PSNR(xk_fine, x_true).item())'''

            else:
                xk_coarse = (
                    xk_coarse
                    - coherence
                    - step_size
                    * data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics)
                    - step_size * grad_prior(xk_coarse, param_reg_coarse)
                )  # Coarse gradient descent

                '''observation_fine = observations[f"level{level_max}"]  # Fine observation
                diff = reconstruct_to_finest_level(xk_coarse-x0_coarse, levels, level_max, args_multilevel)
                obj_fun = lambda x: data_fidelity(x, observation_fine, physics_fine).item() + param_reg_fine * prior.fn(x).item()
                grad_obj_fun = lambda x: data_fidelity.grad(x, observation_fine, physics_fine) + param_reg_fine * grad_prior(x, param_reg_fine)
                xk_fine, tau = linesearch_armijo(xk_finest, diff, obj_fun, grad_obj_fun)
                print(f'Loss at coarse iteration {k} (level {levels}): {data_fidelity(xk_fine, observation_fine, physics_fine).item() + param_reg_fine * prior.fn(xk_fine).item()}')
                losses.append(data_fidelity(xk_fine, observation_fine, physics_fine).item() + param_reg_fine * prior.fn(xk_fine).item())
                times.append(time.process_time())
                if x_true is not None:
                    psnr_list.append(PSNR(xk_fine, x_true).item())'''

    # Coarse correction
    coarse_correction = xk_coarse - x0_coarse
    coarse_correction = information_transfer.to_fine(coarse_correction, xk.shape[-3:])
    xk, step_coarse = ML_linesearch(
        xk,
        level_max,
        levels + 1,
        coarse_correction,
        cst_grad_fine,
        data_fidelity,
        observation,
        physics,
        grad_prior,
        denoiser,
        prior,
        param_reg_fine,
        step_coarse * 2,
    )
    # xk = xk + step_coarse* coarse_correction
    # print(f"Step size at level {levels}: {step_coarse}")
    return xk, losses, times, psnr_list

def reconstruct_to_finest_level(xk_coarse, current_level, target_level, args_multilevel):
    if current_level >= target_level:
        return xk_coarse.clone()

    xk = xk_coarse.clone()

    for level in range(current_level, target_level):
        target_obs = args_multilevel.observations[f"level{level+1}"]
        target_shape = target_obs.shape[-3:]

        info_transfer = copy.deepcopy(args_multilevel.information_transfer)

        if xk.dim() == 3:  # Si pas de dimension batch, l'ajouter
            xk = xk.unsqueeze(0)
        info_transfer._initialize_operator(xk, target_shape)
        xk = info_transfer.to_fine(xk, target_shape)

    return xk


def MultiLevelWavelets(
    xk,
    level_max,
    levels,
    args_multilevel,
    param_regularization,
    cst_grad=None,
    device="cpu",
    use_coherence=False,
    use_initial_linesearch=False,
    no_intermediate_GD=False,
):
    """
    Multilevel step for image reconstruction with conditional wavelet thresholding.
    """
    if not isinstance(args_multilevel, ParametersMultilevel):
        raise ValueError("args_multilevel must be an instance of ParametersMultilevel")
    if levels < 1:
        return xk
    # xk: current image
    # information_transfer: function to transfer information between levels
    # levels: number of levels
    # param_iter: number of iterations at coarse level
    # data_fidelity: data fidelity term
    # cst_grad: first order coherence term from previous level
    """
    Unpack parameters for readability
    """
    data_fidelity = args_multilevel.data_fidelity
    prior = args_multilevel.prior
    grad_prior = args_multilevel.grad_prior
    denoiser = args_multilevel.denoiser
    step_size = args_multilevel.step_size
    param_coarse_iter = args_multilevel.param_coarse_iter
    physics = args_multilevel.physics
    max_ML_steps = args_multilevel.max_ML_steps
    coarse_physics = args_multilevel.coarse_physics
    observations = args_multilevel.observations
    information_transfer = copy.deepcopy(args_multilevel.information_transfer)
    information_transfer._initialize_operator(xk, xk.shape[-3:])
    observation = observations[f"level{levels+1}"]
    physics = coarse_physics[f"level{levels+1}"]
    coarse_observation = observations[f"level{levels}"]
    coarse_physics = coarse_physics[f"level{levels}"]
    param_reg_fine = param_regularization
    if isinstance(prior, dinv.optim.prior.PnP):
        param_reg_coarse = param_reg_fine
    else:
        param_reg_coarse = param_reg_fine / 4

    if cst_grad is None:
        cst_grad_fine = None
    else:
        cst_grad_fine = cst_grad.clone()

    """
    Send information to the coarse level
    """

    if use_initial_linesearch:
        xk_coarse = information_transfer.to_coarse(xk, xk.shape[-3:])

    else:

        # Instead of only getting the approximation, we also get the detail coefficients
        components = information_transfer.to_coarse_wavelet(xk)
        xk_coarse = components["LL"]
        LH, HL, HH = components["LH"], components["HL"], components["HH"]
        LH0, HL0, HH0 = copy.deepcopy(LH), copy.deepcopy(HL), copy.deepcopy(HH)

    # Initialize the multilevel iteration with the approximation coefficients
    x0_coarse = xk_coarse.clone()
    step_coarse = 1

    if use_coherence:
        cst_grad, coherence = compute_coherence(
        xk,
        xk_coarse,
        information_transfer,
        data_fidelity,
        grad_prior,
        cst_grad,
        physics,
        coarse_physics,
        observation,
        coarse_observation,
        param_reg_fine,
        param_reg_coarse,
    )
        coherence = step_size * coherence.to(device)

    """
    Optimize at coarse level
    """
    if no_intermediate_GD:
        with torch.no_grad():
            if levels == 1:
                for k in range(max_ML_steps):
                    # GD only at coarser scale
                    xk_coarse = (
                        xk_coarse
                        - step_size * data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics)
                )
            else:
                # no GD, only thresholding
                LH, HL, HH = conditional_thresholding(
                    {"LH": LH, "HL": HL, "HH": HH}, xk_coarse, global_threshold=param_reg_coarse
                )
    else:
        with torch.no_grad():
            for k in range(param_coarse_iter):
                if levels > 1 and k < max_ML_steps:
                    xk_coarse = MultiLevelWavelets(
                        xk_coarse,
                        level_max,
                        levels - 1,
                        args_multilevel,
                        param_reg_coarse,
                        cst_grad,
                        use_coherence=use_coherence,
                        use_initial_linesearch= use_initial_linesearch,
                        no_intermediate_GD=no_intermediate_GD,
                    )  # Recursive call if levels > 1

                if use_coherence:
                    # Coarse gradient descent with coherence term + grad prior
                    xk_coarse = (
                        xk_coarse
                        - coherence
                        - step_size
                        * data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics)
                        - step_size * grad_prior(xk_coarse, param_reg_coarse)
                    )  # Coarse gradient descent
                else:
                    # Coarse gradient descent but no coherence term nor grad prior
                    xk_coarse = (
                        xk_coarse
                        - step_size
                        * data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics)
                    )
                    # Check parameters
                    '''print(f"Gradient step at coarse level {levels}, iteration {k}")
                    print(f"Stepsize: {step_size}")
                    print(f"Regularization parameter: {param_reg_coarse}")'''

                    # Old coefficients if we need to compare
                    LH_prev, HL_prev, HH_prev = LH.clone(), HL.clone(), HH.clone()

                    # Threshold the detail coefficients based on the reconstructed approximation
                    LH, HL, HH = conditional_thresholding(
                        {"LH": LH, "HL": HL, "HH": HH}, xk_coarse, global_threshold=param_reg_coarse
                    )

            # Plot: detail coefficients before vs after thresholding
            # dinv.utils.plot([LH_prev, LH], titles=['LH previous', 'LH current'], cmap='gray', suptitle='LH Coarse Level')

    # Line search to find the optimal stepsize in the direction of coarse_correction
    if use_initial_linesearch: # Done with the objective function
        coarse_correction_fine = information_transfer.to_fine(xk_coarse - x0_coarse, xk.shape[-3:])
        xk, step_coarse = ML_linesearch(
            xk,
            level_max,
            levels + 1,
            coarse_correction_fine,
            cst_grad_fine,
            data_fidelity,
            observation,
            physics,
            grad_prior,
            denoiser,
            prior,
            param_reg_fine,
            step_coarse * 2,
        )

    else:
        # Coarse correction in the wavelet domain
        coarse_correction = xk_coarse - x0_coarse
        coarse_correction_components = {
            "LL": coarse_correction,
            "LH": LH - LH0,
            "HL": HL - HL0,
            "HH": HH - HH0,
        }
        coarse_correction_fine = information_transfer.to_fine_wavelet(
            coarse_correction_components
        )
        # Line search with the data fidelity only (and with Armijo condition)
        xk, step_coarse = linesearch_armijo(xk, p=coarse_correction_fine, obj_fun=lambda x: data_fidelity.fn(x, observation, physics), grad_obj_fun=lambda x: data_fidelity.grad(x, observation, physics))

    return xk

def linesearch(xk, p, obj_fun, tau_init=1.0, min_tau=1e-6, grad_obj_fun=None):
    tau = tau_init
    f_current = obj_fun(xk)
    x_new = xk + tau * p

    while obj_fun(x_new) > f_current and tau > min_tau:
        tau /= 2
        x_new = xk + tau * p

    return x_new, tau

def linesearch_armijo(xk, p, obj_fun, grad_obj_fun, tau_init=1.0, min_tau=1e-6, c=1e-4):
    tau = tau_init
    f_current = obj_fun(xk)
    grad_current = grad_obj_fun(xk)
    grad_dot_p = (grad_current * p).sum()
    x_new = xk + tau * p

    while obj_fun(x_new) > f_current + c * tau * grad_dot_p and tau > min_tau:  # Armijo condition
        tau /= 2
        x_new = xk + tau * p

    return x_new, tau

class ParametersMultilevel:
    def __init__(
        self,
        target_shape,
        levels,
        max_ML_steps,
        param_coarse_iter,
        step_size,
        info_transfer,
        prior,
        denoiser,
        data_fidelity,
        physics,
        observation,
        device,
    ):
        if not isinstance(levels, int) or levels < 1:
            raise ValueError("levels must be an integer and greater than 0")
        if isinstance(denoiser, dinv.models.DRUNet):
            if (
                target_shape[-2] // (2 ** (levels - 1)) < denoiser.m_head.out_channels
                and target_shape[-1] // (2 ** (levels - 1))
                < denoiser.m_head.out_channels
            ):
                raise ValueError(
                    "number of levels too high for DRUNet (min image size is {}x{}) - current min size is {}x{}. \n max number of levels for DRUNet is {}".format(
                        denoiser.m_head.out_channels,
                        denoiser.m_head.out_channels,
                        target_shape[-2] // (2 ** (levels - 1)),
                        target_shape[-1] // (2 ** (levels - 1)),
                        np.floor(
                            1
                            + np.min(
                                [
                                    np.log2(
                                        target_shape[-2] / denoiser.m_head.out_channels
                                    ),
                                    np.log2(
                                        target_shape[-1] / denoiser.m_head.out_channels
                                    ),
                                ]
                            )
                        ),
                    )
                )
        if isinstance(prior, dinv.optim.WaveletPrior):
            for i in range(levels):
                if target_shape[-2] % (2**i) != 0 or target_shape[-1] % (2**i) != 0:
                    print(
                        "Image size is not divisible by 2^(level-1) for all levels - denoiser(x) may not have the same size as x"
                    )
                    break
        self.levels = levels
        self.max_ML_steps = max_ML_steps
        self.param_coarse_iter = param_coarse_iter
        self.information_transfer = create_information_transfer(info_transfer, device)
        self.prior = prior
        self.denoiser = denoiser
        self.data_fidelity = data_fidelity
        self.physics = physics
        self.step_size = step_size
        self.observations = create_coarse_observations(
            observation, info_transfer, levels, device
        )
        self.coarse_physics = create_coarse_physics(
            physics, target_shape, levels, info_transfer, device
        )
        self.grad_prior = create_grad_prior(prior, denoiser, device)

        # self.gammas = {f'level{i}': self.default_gamma(i) for i in range(1, levels + 1)}


def create_information_transfer(filter, device):
    """
    Create the information transfer object
    """
    filter = create_filter(filter)  # choose your filter
    information_transfer = DownsamplingTransfer(filter)  # create operator
    information_transfer = information_transfer.to(device)

    return information_transfer


def create_coarse_observations(observation, filter_name, levels, device):
    """
    Create the coarse observations object
    """
    coarse_obs_dict = {f"level{levels}": observation}
    coarse_obs_iter = observation
    for i in range(levels - 1, 0, -1):
        ds = DownsamplingTransfer(create_filter(filter_name))
        coarse_obs_iter = ds.to_coarse(coarse_obs_iter, coarse_obs_iter.shape[-3:])
        coarse_obs_dict[f"level{i}"] = coarse_obs_iter
    return coarse_obs_dict


def create_coarse_physics(physics, img_shape, levels, filter, device):
    """
    Create the coarse physics object
    """
    scales = [2**i for i in range(0, levels)]
    scales = scales[::-1]
    filter = create_filter(filter)  # choose your filter
    information_transfer = DownsamplingTransfer(filter)
    filt_2d = information_transfer._getfilter().type(torch.float32)
    coarse_physics = {}
    coarse_physics[f"level{levels}"] = physics
    for i in range(levels - 1, 0, -1):
        coarse_physics[f"level{i}"] = MultilevelPhysics(
            physics, img_shape, scales[i - 1], filter=filt_2d, device=device
        )
    return coarse_physics


class Residual(nn.Module):
    def __init__(self, denoiser, prior):
        super().__init__()
        self.prior = prior
        self.denoiser = denoiser

        if isinstance(self.prior, dinv.optim.TVPrior):
            self.l12prior = dinv.optim.L12Prior()

    def forward(self, x, gamma):
        if isinstance(self.prior, dinv.optim.prior.PnP) or isinstance(self.prior, WaveletPriorCustom):
            return x - self.denoiser(x, gamma)
        if isinstance(self.prior, dinv.optim.TVPrior):
            Dx = self.prior.nabla(
                x
            )  # The TV operator is not normalized, so Lipschit constant is ||D||_2^2 = 8
            return (1/8 * gamma
                * self.prior.nabla_adjoint(Dx - self.l12prior.prox(Dx, gamma))
            )
        else:
            return x - self.denoiser(x, gamma=[gamma])
        '''if isinstance(self.prior, dinv.optim.WaveletPrior):
            return (

            )'''



def create_grad_prior(
    prior, denoiser, device
):  # ajouter option coherence pour la choisir
    """
    Create the gradient prior: automatically define the gradient of the smoothed fine level prior
    """
    grad_prior = Residual(denoiser=denoiser, prior=prior)
    return grad_prior.to(device)


class Upsampling(Downsampling):
    def Upsample(self, x, **kwargs):
        return super().A_adjoint(x, **kwargs)

    def Downsample(self, y, **kwargs):
        return super().A(y, **kwargs)

    def prox_l2(self, z, y, gamma, **kwargs):
        return super().prox_l2(z, y, gamma, **kwargs)


class MultilevelPhysics(Physics):
    def __init__(
        self, physics, img_shape, scale, filter="sinc", device="cpu", **kwargs
    ):
        super().__init__(noise_model=physics.noise_model, **kwargs)
        self.base = physics
        self.scale = scale
        self.img_shape = img_shape
        self.Upsampling = Upsampling(
            img_size=img_shape, filter=filter, factor=scale, device=device
        )

    def A(self, x, **kwargs):
        return self.Upsampling.Downsample(
            self.base.A(self.Upsampling.Upsample(x), **kwargs)
        )

    def A_adjoint(self, y, **kwargs):
        return self.Upsampling.Downsample(
            self.base.A_adjoint(self.Upsampling.Upsample(y), **kwargs)
        )


def compute_coherence(
    xk,
    xk_coarse,
    information_transfer,
    data_fidelity,
    grad_prior,
    cst_grad,
    physics,
    coarse_physics,
    observation,
    coarse_observation,
    reg_fine,
    reg_coarse,
):
    """
    Compute the coherence term for the multilevel optimization
    """
    if (
        cst_grad is None
    ):  # Store the coherence term of the fine level for the coarser levels
        gradient_fine_level = data_fidelity.grad(xk, observation, physics) + grad_prior(
            xk, reg_fine
        )
        coherence = (
            information_transfer.to_coarse(gradient_fine_level, xk.shape[-3:])
            - data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics)
            - grad_prior(xk_coarse, reg_coarse)
        )  # le paramètre scale n'est pas pris en compte !
        cst_grad = coherence.clone()
    else:
        gradient_fine_level = data_fidelity.grad(xk, observation, physics) + grad_prior(
            xk, reg_fine
        )
        cst_grad = information_transfer.to_coarse(cst_grad, xk.shape[-3:])
        coherence = (
            information_transfer.to_coarse(gradient_fine_level, xk.shape[-3:])
            + cst_grad
            - data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics)
        )
        -grad_prior(xk_coarse, reg_coarse)
    return cst_grad, coherence


def ML_linesearch(
    xk,
    level_max,
    levels,
    coarse_correction,
    coherence,
    data_fidelity,
    observation,
    physics,
    grad_prior,
    denoiser,
    prior,
    reg_fine,
    step_coarse=1,
):
    """
    Multilevel linesearch: ensures that the coarse correction is not too large
    """
    if isinstance(prior, dinv.optim.WaveletPrior):
        if levels == level_max:
            # Compute the current function value
            f_current = data_fidelity(xk, observation, physics) + reg_fine * prior.fn(
                xk
            )

            x_corrected = xk + step_coarse * coarse_correction
            while (
                data_fidelity(x_corrected, observation, physics)
                + reg_fine * prior.fn(x_corrected)
            ) > f_current:
                step_coarse /= 2
                x_corrected = (
                    xk + step_coarse * coarse_correction
                )  # Apply the step size
        else:
            f_current = (
                data_fidelity(xk, observation, physics)
                + reg_fine * torch.norm(denoiser(xk), p=1)
                + 1 / 2 * torch.norm(grad_prior(xk, reg_fine), 2) ** 2
                + torch.dot(torch.flatten(coherence), torch.flatten(xk))
            )
            x_corrected = xk + step_coarse * coarse_correction
            while (
                data_fidelity(x_corrected, observation, physics)
                + reg_fine * torch.norm(denoiser(x_corrected), p=1)
                + 1 / 2 * torch.norm(grad_prior(x_corrected, reg_fine), 2) ** 2
                + torch.dot(torch.flatten(coherence), torch.flatten(x_corrected))
            ) > f_current:
                step_coarse /= 2
                x_corrected = (
                    xk + step_coarse * coarse_correction
                )  # Apply the step size
        return x_corrected, step_coarse
    elif isinstance(prior, dinv.optim.TVPrior):
        l12prior = dinv.optim.L12Prior()
        if levels == level_max:
            # Compute the current function value
            f_current = data_fidelity(xk, observation, physics) + reg_fine * prior.fn(
                xk
            )
            x_corrected = xk + step_coarse * coarse_correction
            while (
                data_fidelity(x_corrected, observation, physics)
                + reg_fine * prior.fn(x_corrected)
            ) > f_current:
                step_coarse /= 2
                x_corrected = (
                    xk + step_coarse * coarse_correction
                )  # Apply the step size
        else:
            Dx = prior.nabla(xk)
            f_current = (
                data_fidelity(xk, observation, physics)
                + reg_fine * l12prior.fn(Dx)
                + 1 / 2 * torch.norm(Dx - l12prior.prox(Dx, reg_fine), 2) ** 2
            )
            x_corrected = xk + step_coarse * coarse_correction
            Dxk = prior.nabla(x_corrected)
            while (
                data_fidelity(x_corrected, observation, physics)
                + +reg_fine * l12prior.fn(Dxk)
                + 1 / 2 * torch.norm(Dxk - l12prior.prox(Dxk, reg_fine), 2) ** 2
            ) > f_current:
                step_coarse /= 2
                x_corrected = (
                    xk + step_coarse * coarse_correction
                )  # Apply the step size
                Dxk = prior.nabla(x_corrected)
        return x_corrected, step_coarse
    else:
        f_current = data_fidelity(xk, observation, physics)
        x_corrected = xk + step_coarse * coarse_correction
        while data_fidelity(x_corrected, observation, physics) > f_current:
            step_coarse /= 2
            x_corrected = xk + step_coarse * coarse_correction  # Apply the step size
        return x_corrected, step_coarse


def conditional_thresholding(
        details,
        approx,
        global_threshold
        ):
    """
    Conditional thresholding of the wavelet coefficients based on the gradient of the approximation.
    This function is used in the MultilevelWavelets class.

    Parameters
    ----------
    details : dict
        Dictionary containing the wavelet coefficients of the details (LH, HL, HH).
    approx : torch.Tensor
        The approximation coefficients of the wavelet transform.
    global_threshold : float
        The global threshold value for the wavelet coefficients, used to compute the other thresholds.
    """
    device, dtype = approx.device, approx.dtype

    l1_prior = dinv.optim.L1Prior()

    if isinstance(details, dict):
        LH = details["LH"]
        HL = details["HL"]
        HH = details["HH"]
    elif isinstance(details, torch.Tensor):
        LH, HL, HH = details[0], details[1], details[2]
    else:
        raise ValueError("Format de 'details' inattendu")

    # Wavelet transform of the approximation without downsampling
    stationnary_transform = pywt.swt2(approx.cpu().numpy(), wavelet="db8", level=1)
    grad_LH, grad_HL, grad_HH = (
        stationnary_transform[0][1][0],
        stationnary_transform[0][1][1],
        stationnary_transform[0][1][2],
    )

    grad_LH = torch.tensor(grad_LH, device=device, dtype=dtype)
    grad_HL = torch.tensor(grad_HL, device=device, dtype=dtype)
    grad_HH = torch.tensor(grad_HH, device=device, dtype=dtype)

    # Thresholds are inversely proportional to the gradient
    LH = l1_prior.prox(LH, global_threshold / grad_LH)
    HL = l1_prior.prox(HL, global_threshold / grad_HL)
    HH = l1_prior.prox(HH, global_threshold / grad_HH)

    return LH, HL, HH


class WaveletDenoiserConditional(Denoiser):
    """
    Wavelet denoising with conditional thresholding (inspired from DeepInv's WaveletDenoiser).
    This function is implemented with PyWavelets and PyTorch, and is able to handle the padding mode 'periodization' for the wavelet transform.
    """
    def __init__(
        self,
        level: int = 3,
        wv: str = "db8",
        device: torch.device = "cpu",
        non_linearity: str = "soft",
        mode: str = "periodization",
    ):
        super().__init__()
        self.level = level
        self.wv = wv
        self.device = device
        self.non_linearity = non_linearity
        self.mode = mode

    def dwt(self, x):
        dec_np = pywt.wavedec2(
            x, pywt.Wavelet(self.wv), mode=self.mode, level=self.level
        )
        dec = [torch.tensor(dec_np[0], device=self.device)]  # Approximation
        for detail in dec_np[1:]:
            dec.append(tuple(torch.tensor(c, device=self.device) for c in detail))
        return dec

    def iwt(self, coeffs):
        coeffs_np = [coeffs[0].cpu().numpy()]
        for detail in coeffs[1:]:
            coeffs_np.append(tuple(c.cpu().numpy() for c in detail))
        rec_np = pywt.waverec2(coeffs_np, pywt.Wavelet(self.wv), mode=self.mode)
        return torch.tensor(rec_np, device=self.device)

    def forward(self, x, gamma=1):
        coeffs_np = pywt.wavedec2(
            x.cpu().numpy(), pywt.Wavelet(self.wv), mode=self.mode, level=self.level
        )
        coeffs = [torch.tensor(coeffs_np[0], device=self.device)]  # Approximation
        for detail in coeffs_np[1:]:                               # Details
            coeffs.append(tuple(torch.tensor(c, device=self.device) for c in detail))

        approx = copy.deepcopy(coeffs[0])
        details = copy.deepcopy(coeffs[1:])

        # Initialize the list of thresholded coefficients with the approximation (not modified)
        coeffs_thresholded = [copy.deepcopy(approx)]

        for current_lvl in range(self.level):
            # Global threshold for the current level
            gamma_level = gamma / (2 ** (self.level - current_lvl))

            # Wavelet transform of the approximation without downsampling
            stationnary_transform = pywt.swt2(
                approx.cpu().numpy(), wavelet="sym8", level=1
            )

            grad_LH, grad_HL, grad_HH = (
                stationnary_transform[0][1][0],
                stationnary_transform[0][1][1],
                stationnary_transform[0][1][2],
            )

            grad_LH = torch.tensor(grad_LH, device=self.device)
            grad_HL = torch.tensor(grad_HL, device=self.device)
            grad_HH = torch.tensor(grad_HH, device=self.device)

            # Thresholds are inversely proportional to the gradient
            gamma_LH = gamma_level / torch.abs(grad_LH)
            gamma_HL = gamma_level / torch.abs(grad_HL)
            gamma_HH = gamma_level / torch.abs(grad_HH)
            gammas = [gamma_LH, gamma_HL, gamma_HH]

            details_thresholded = []
            # Threshold the 3 details coefficients
            for c in range(3):
                if self.non_linearity == "soft":
                    details_thresholded.append(
                        dinv.optim.L1Prior().prox(
                            details[current_lvl][c], gamma=gammas[c]
                        )
                    )
                elif self.non_linearity == "hard":
                    details_thresholded.append(
                        self.prox_l0(
                            details[current_lvl][c], gamma=gammas[c]
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown non-linearity: {self.non_linearity}. Use 'soft' or 'hard'."
                    )
            details_thresholded = tuple(details_thresholded)
            coeffs_thresholded.append(details_thresholded)

            # Reconstruct the approximation at the next scale with the thresholded details
            approx = self.iwt([approx, details_thresholded])

        return self.iwt(coeffs_thresholded)

    def prox_l0(
        self, x: torch.Tensor, gamma=0.1
    ):
        """
        Hard thresholding.
        """
        out = x.clone()
        out[out.abs() < gamma] = 0
        return out


# Previous wavelet thresholding / GD + thresholding class to compare methods

class WaveletThresholding:
    def __init__(self, y, prior, data_fidelity, physics, wavelet_type='haar', n_levels=3, stepsize=1e-3, device=torch.device('cpu'), grad_as_wavelet=False):
        self.y = y
        self.prior = prior
        self.data_fidelity = data_fidelity
        self.physics = physics
        self.wavelet_type = wavelet_type
        self.n_levels = n_levels
        self.stepsize = stepsize
        self.device = device
        self.grad_as_wavelet = grad_as_wavelet

    def threshold(self, wavelet_coeffs_noisy, wavelet_coeffs_true, gamma=1, conditional=True, n_iter_coarse= 20, direction_coeff=None, kernel_size=None, print_psnr=False, plot=None, grad_as_wavelet=False):

        target_shape = self.y.shape[-2:]
        filter = 'haar'
        self.information_transfer = DownsamplingTransfer(create_filter(filter)).to(self.device)

        # Génère coarse_physics
        self.coarse_physics = create_coarse_physics(
            self.physics, target_shape, self.n_levels, filter, self.device
        )

        self.grad_as_wavelet = grad_as_wavelet

        # Unpack and deep copy inputs
        approx_noisy, *details_noisy = copy.deepcopy(wavelet_coeffs_noisy)  # Noisy approx and details : to be modified
        approx = copy.deepcopy(approx_noisy)                                # Initial approximation (used for reconstruction)
        approx_init = copy.deepcopy(approx_noisy)                           # Initial approximation (untouched, used for final reconstruction)
        approx_true, *details_true = wavelet_coeffs_true                    # True approximation and details (used for comparison)

        for level in range(self.n_levels):

            if level == 0:
                filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
                level_physics = dinv.physics.Blur(filter_0, device=self.device, padding="reflect")

                y_coarse = approx_noisy

                approx_old = approx.clone()
                for k in range(n_iter_coarse):
                    grad = self.data_fidelity.grad(approx, y_coarse, level_physics)
                    approx = approx - self.stepsize * grad

                dinv.utils.plot([approx_true, approx_old, approx], titles=['Approx ground truth', 'Approx before GD', 'Approx after GD'])

            gamma_level = gamma / (2 ** (self.n_levels - level))
            if kernel_size:
                nabla_approx = local_average(approx, kernel_size=kernel_size)
            else:
                nabla_approx = nabla(approx)
            grad_hor, grad_ver = nabla_approx[..., 0], nabla_approx[..., 1]

            if self.grad_as_wavelet:
                # Computes the stationnary wavelet transform of the approximation coefficients (convolution without downsampling)
                stationnary_transform = pywt.swt2(approx.cpu().numpy(), wavelet=self.wavelet_type, level=1)
                grad_as_wavelets_hor, grad_as_wavelets_ver, grad_as_wavelets_diag = stationnary_transform[0][1][0], stationnary_transform[0][1][1], stationnary_transform[0][1][2]

                grad_hor = torch.tensor(grad_as_wavelets_hor, device=self.device, dtype=approx.dtype)
                grad_ver = torch.tensor(grad_as_wavelets_ver, device=self.device, dtype=approx.dtype)
                grad_diag = torch.tensor(grad_as_wavelets_diag, device=self.device, dtype=approx.dtype)

                gamma_hor = gamma_level / torch.sqrt(direction_coeff * grad_hor**2 + (1 - direction_coeff) * grad_ver**2)
                gamma_ver = gamma_level / torch.sqrt((1-direction_coeff) * grad_hor**2 + direction_coeff * grad_ver**2)
                gamma_diag = gamma_level / abs(grad_diag)

                gammas = [gamma_hor, gamma_ver, gamma_diag]

            if conditional:
                if direction_coeff:
                    gamma_hor = gamma_level / torch.sqrt(direction_coeff * grad_hor**2 + (1 - direction_coeff) * grad_ver**2)
                    gamma_ver = gamma_level / torch.sqrt((1-direction_coeff) * grad_hor**2 + direction_coeff * grad_ver**2)
                    gamma_diag = gamma_level / torch.sqrt(grad_hor**2 + grad_ver**2)

                    '''gamma_hor = gamma_level / torch.sqrt(direction_coeff * grad_hor**2 + (1 - direction_coeff) * grad_ver**2)
                    gamma_ver = gamma_level / torch.sqrt((1-direction_coeff) * grad_hor**2 + direction_coeff * grad_ver**2)
                    gamma_diag = gamma_level / abs(grad_diag)'''

                    gammas = [gamma_hor, gamma_ver, gamma_diag]
                else:
                    gamma_iso = gamma_level / torch.sqrt(grad_hor**2 + grad_ver**2)
                    gammas = [gamma_iso] * 3
            else:
                gammas = [gamma_level] * 3

            # Apply proximal operator to each subband
            for c in range(3):
                details_noisy[level][c] = self.prior.prox(details_noisy[level][c], gamma=gammas[c])

                if print_psnr:
                    print(f'PSNR true vs reconstructed details at scale {self.n_levels-level}, coefficient {c+1}: {PSNR()(details_noisy[level][c], details_true[level][c]).item():.3f}')

            if plot:
                comparative_plot_wavelets( # Print the PSNR between the true and updated coefficients
                    [
                        (approx, *details_noisy),
                        (approx, *wavelet_coeffs_noisy[1:]),
                        (approx_true, *details_true)
                        ],
                    level=level,
                    labels=[
                        'Updated coefficients',
                        'Noisy coefficients',
                        'True coefficients'
                        ],
                    save_fn=f'{plot}/coeffs_scale_{self.n_levels-level}.png'
                )

                # Compute the true value of the approximation coefficients for the next scale
                approx_true = get_approximation_previous_scale(
                    (approx_true, *details_true[level:]),
                    wavelet_type=self.wavelet_type
                )

            # Reconstruct current approximation from updated detail coefficients
            approx = get_approximation_previous_scale(
                (approx, *details_noisy[level:]),
                wavelet_type=self.wavelet_type
            )

        return (approx_init, *details_noisy)

    def gradient_descent(xk, y, level_physics, stepsize=1e-3, n_iter=5):
        for i in range(n_iter):
            xk = xk - stepsize * level_physics.A_adjoint(level_physics.A(xk) - y)
        return xk


if __name__ == "__main__":
    from deepinv.loss.metric import PSNR

    # Test Wavelet functions
    x = dinv.utils.load_url_image(
        url=dinv.utils.get_image_url("cameraman.png"), img_size=512, grayscale=True
    ).to(torch.device("cpu"))
    y = x + 0.1 * torch.randn_like(x)  # Add noise for testing

    denoiser = WaveletDenoiserConditional(level=3, wv="db8", device=torch.device("cpu"), non_linearity="soft")

    x_est = denoiser.forward(y, gamma=0.08)

    psnrs = [PSNR()(x, y).item(), PSNR()(x, x_est).item()]
    psnrs = [f"{psnr:.2f}" for psnr in psnrs]
    dinv.utils.plot(
        [x, y, x_est],
        titles=["Original", f"Noisy ({psnrs[0]})", f"Reconstructed ({psnrs[1]})"],
        cmap="gray",
        figsize=[6, 6],
    )
