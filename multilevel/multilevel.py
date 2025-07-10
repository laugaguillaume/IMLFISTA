#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025
@author: Guillaume Lauga
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

import deepinv as dinv
import torch
import torch.nn as nn
from torchvision.io import read_image
import numpy as np
import matplotlib as mpl
import scipy.io as sio
import copy
from multilevel.info_transfer import DownsamplingTransfer, create_filter
from deepinv.physics import Physics, Downsampling

def MultiLevel(xk, level_max, levels, args_multilevel, param_regularization, cst_grad=None ,  device = 'cpu'):
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
                xk_coarse = MultiLevel(xk_coarse, level_max, levels-1, args_multilevel, param_reg_coarse, cst_grad) # Recursive call if levels > 1

            if isinstance(prior, dinv.optim.prior.PnP):
                xk_coarse = prior.denoiser(
                    xk_coarse - coherence - step_size*data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics),
                    param_reg_coarse
                )
            else:
                xk_coarse = xk_coarse -coherence \
                    - step_size*data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics) \
                    - step_size*grad_prior(xk_coarse, param_reg_coarse) # Coarse gradient descent
            
    # Coarse correction
    coarse_correction = xk_coarse - x0_coarse
    coarse_correction = information_transfer.to_fine(coarse_correction, xk.shape[-3:])
    xk, step_coarse = ML_linesearch(xk, level_max, levels+1, coarse_correction, cst_grad_fine, data_fidelity, observation, physics, grad_prior, denoiser, prior, param_reg_fine,  step_coarse*2)
    # xk = xk + step_coarse* coarse_correction
    # print(f"Step size at level {levels}: {step_coarse}")
    return xk

class ParametersMultilevel:
    def __init__(self, target_shape, levels, max_ML_steps, param_coarse_iter, step_size, info_transfer, prior, denoiser, data_fidelity, physics, observation,device):
        if not isinstance(levels, int) or levels < 1:
            raise ValueError("levels must be an integer and greater than 0")
        if isinstance(denoiser, dinv.models.DRUNet):
            if target_shape[-2] // (2**(levels-1)) <denoiser.m_head.out_channels and target_shape[-1] // (2**(levels-1)) <denoiser.m_head.out_channels:
                raise ValueError("number of levels too high for DRUNet (min image size is {}x{}) - current min size is {}x{}. \n max number of levels for DRUNet is {}".format(denoiser.m_head.out_channels,denoiser.m_head.out_channels,target_shape[-2]//(2**(levels-1)),target_shape[-1]//(2**(levels-1)),np.floor(1+np.min([np.log2(target_shape[-2]/denoiser.m_head.out_channels),np.log2(target_shape[-1]/denoiser.m_head.out_channels)]))))
        if isinstance(prior,dinv.optim.WaveletPrior):
            for i in range(levels):
                if target_shape[-2] % (2**i) != 0 or target_shape[-1] % (2**i) != 0:
                    print("Image size is not divisible by 2^(level-1) for all levels - denoiser(x) may not have the same size as x")
                    break
        self.levels = levels
        self.max_ML_steps = max_ML_steps
        self.param_coarse_iter = param_coarse_iter
        self.information_transfer = create_information_transfer(info_transfer,device)
        self.prior = prior
        self.denoiser = denoiser
        self.data_fidelity = data_fidelity
        self.physics = physics
        self.step_size = step_size
        self.observations = create_coarse_observations(observation, info_transfer, levels, device)
        self.coarse_physics = create_coarse_physics(physics, target_shape, levels, info_transfer, device)
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

def create_coarse_observations(observation, filter, levels, device):
    """
    Create the coarse observations object
    """
    img_shape = observation.shape[-3:]
    scales = [2 ** i for i in range(0,levels)]
    scales = scales[::-1]
    filter=create_filter(filter) 
    information_transfer = DownsamplingTransfer(filter)
    k0 = information_transfer.filter_object.get_filter()
    filt_2d = information_transfer.set_2d_filter(k0, dtype=torch.float32).to(device)
    Upsamplings = [Upsampling(img_size=img_shape, filter=filt_2d, factor=factor, device=device) for factor in scales]
    # if not isinstance(info_transfer, DownsamplingTransfer):
    #     raise ValueError("info_transfer must be an instance of DownsamplingTransfer")
    if not isinstance(observation, torch.Tensor):
        raise ValueError("observation must be a torch.Tensor")
    coarse_observations = {}
    coarse_observations[f'level{levels}'] = observation
    for i in range(levels-1, 0, -1):
        # coarse_observations[f'level{i-1}'] =Upsamplings[i-1].Upsample(Upsamplings[i-1].Downsample(observation))
        coarse_observations[f'level{i}'] =Upsamplings[i-1].Downsample(observation)
    return coarse_observations

def create_coarse_physics(physics, img_shape, levels, filter, device):
    """
    Create the coarse physics object
    """
    scales = [2 ** i for i in range(0,levels)]
    scales = scales[::-1]
    filter=create_filter(filter)  # choose your filter
    information_transfer = DownsamplingTransfer(filter)
    k0 = information_transfer.filter_object.get_filter()
    filt_2d = information_transfer.set_2d_filter(k0, dtype=torch.float32).to(device)
    coarse_physics = {}
    coarse_physics[f'level{levels}'] = physics
    for i in range(levels-1, 0, -1):
        coarse_physics[f'level{i}'] =MultilevelPhysics(physics, img_shape, scales[i-1], filter=filt_2d, device=device)
    return coarse_physics


    
class Residual(nn.Module):
    def __init__(self, denoiser, prior):
        super().__init__()
        self.prior = prior
        self.denoiser = denoiser

        if isinstance(self.prior, dinv.optim.TVPrior):
            self.l12prior = dinv.optim.L12Prior()
    def forward(self, x, gamma):
        if isinstance(self.prior, dinv.optim.prior.PnP):
            return (x - self.denoiser(x, gamma) )
        if isinstance(self.prior, dinv.optim.TVPrior):
            Dx = self.prior.nabla(x) # The TV operator is not normalized, so Lipschit constant is ||D||_2^2 = 8
            return 1/8 * gamma * self.prior.nabla_adjoint( Dx - self.l12prior.prox(Dx,gamma) )
        else:
            return (x - self.denoiser(x, gamma=[gamma]) )
    
def create_grad_prior(prior, denoiser, device): # ajouter option coherence pour la choisir
    """
    Create the gradient prior: automatically define the gradient of the smoothed fine level prior
    """
    grad_prior = Residual(denoiser=denoiser, prior = prior)
    return grad_prior.to(device)

class Upsampling(Downsampling):
    def Upsample(self, x, **kwargs):
        return super().A_adjoint(x, **kwargs)

    def Downsample(self, y, **kwargs):
        return super().A(y, **kwargs)

    def prox_l2(self, z, y, gamma, **kwargs):
        return super().prox_l2(z, y, gamma, **kwargs)

class MultilevelPhysics(Physics):
    def __init__(self, physics, img_shape, scale, filter="sinc", device='cpu', **kwargs):
        super().__init__(noise_model=physics.noise_model, **kwargs)
        self.base = physics
        self.scale = scale
        self.img_shape = img_shape
        self.Upsampling = Upsampling(img_size=img_shape, filter=filter, factor=scale, device=device) 
    def A(self, x, **kwargs):
            return self.Upsampling.Downsample(self.base.A(self.Upsampling.Upsample(x), **kwargs))
    def A_adjoint(self, y,**kwargs):
            return self.Upsampling.Downsample(self.base.A_adjoint(self.Upsampling.Upsample(y), **kwargs))
    
def compute_coherence(xk, xk_coarse, information_transfer, data_fidelity, grad_prior, cst_grad, physics, coarse_physics, observation, coarse_observation, reg_fine, reg_coarse):
    """
    Compute the coherence term for the multilevel optimization
    """
    if cst_grad is None: # Store the coherence term of the fine level for the coarser levels
        gradient_fine_level =data_fidelity.grad(xk, observation, physics) +  grad_prior(xk,reg_fine)
        coherence = information_transfer.to_coarse(gradient_fine_level, xk.shape[-3:]) \
        - data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics) \
        - grad_prior(xk_coarse,reg_coarse) #le paramètre scale n'est pas pris en compte !
        cst_grad = coherence.clone()
    else:
        gradient_fine_level =data_fidelity.grad(xk, observation, physics) +  grad_prior(xk, reg_fine)
        cst_grad = information_transfer.to_coarse(cst_grad, xk.shape[-3:])
        coherence = information_transfer.to_coarse(gradient_fine_level, xk.shape[-3:]) + cst_grad \
        - data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics) 
        - grad_prior(xk_coarse, reg_coarse)
    return cst_grad, coherence

def ML_linesearch(xk, level_max, levels, coarse_correction, coherence, data_fidelity, observation, physics, grad_prior, denoiser, prior, reg_fine, step_coarse=1):
    """
    Multilevel linesearch: ensures that the coarse correction is not too large
    """
    if isinstance(prior, dinv.optim.WaveletPrior):
        if levels == level_max:
            # Compute the current function value
            f_current = data_fidelity(xk, observation, physics) + reg_fine * prior.fn(xk)

            x_corrected = xk + step_coarse * coarse_correction
            while  (data_fidelity(x_corrected, observation, physics) + reg_fine * prior.fn(x_corrected) ) > f_current:
                step_coarse /= 2
                x_corrected = xk + step_coarse * coarse_correction # Apply the step size
        else:
            f_current = data_fidelity(xk, observation, physics) + reg_fine * torch.norm(denoiser(xk),p=1) + 1/2*torch.norm(grad_prior(xk, reg_fine),2)**2 + torch.dot(torch.flatten(coherence), torch.flatten(xk))
            x_corrected = xk + step_coarse * coarse_correction
            while  (data_fidelity(x_corrected, observation, physics) + reg_fine * torch.norm(denoiser(x_corrected),p=1) + 1/2*torch.norm(grad_prior(x_corrected, reg_fine),2)**2 + torch.dot(torch.flatten(coherence), torch.flatten(x_corrected))) > f_current:
                step_coarse /= 2
                x_corrected = xk + step_coarse * coarse_correction # Apply the step size
        return x_corrected, step_coarse
    elif isinstance(prior, dinv.optim.TVPrior):
        l12prior = dinv.optim.L12Prior()
        if levels == level_max:
            # Compute the current function value
            f_current = data_fidelity(xk, observation, physics) + reg_fine * prior.fn(xk)
            x_corrected = xk + step_coarse * coarse_correction
            while  (data_fidelity(x_corrected, observation, physics) + reg_fine * prior.fn(x_corrected) ) > f_current:
                step_coarse /= 2
                x_corrected = xk + step_coarse * coarse_correction # Apply the step size
        else:
            Dx = prior.nabla(xk)
            f_current = data_fidelity(xk, observation, physics) + reg_fine * l12prior.fn(Dx) + 1/2*torch.norm(Dx - l12prior.prox(Dx,reg_fine),2)**2
            x_corrected = xk + step_coarse * coarse_correction
            Dxk = prior.nabla(x_corrected)
            while  (data_fidelity(x_corrected, observation, physics) + + reg_fine * l12prior.fn(Dxk) + 1/2*torch.norm(Dxk - l12prior.prox(Dxk,reg_fine),2)**2  ) > f_current:
                step_coarse /= 2
                x_corrected = xk + step_coarse * coarse_correction # Apply the step size
                Dxk = prior.nabla(x_corrected)
        return x_corrected, step_coarse
    else:
        f_current = data_fidelity(xk, observation, physics) 
        x_corrected = xk + step_coarse * coarse_correction
        while data_fidelity(x_corrected, observation, physics)> f_current:
            step_coarse /= 2
            x_corrected = xk + step_coarse * coarse_correction # Apply the step size
        return x_corrected, step_coarse
