import copy
from multilevel.multilevel import ParametersMultilevel
import torch


def ml_init_pnp(xk, level_max, levels, args_multilevel, param_regularization, denoiser, device='cpu'):
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
    """
    Unpack parameters for readability
    """
    data_fidelity = args_multilevel.data_fidelity
    step_size = args_multilevel.step_size
    param_coarse_iter = args_multilevel.param_coarse_iter
    observations = args_multilevel.observations
    information_transfer = copy.deepcopy(args_multilevel.information_transfer)
    information_transfer._initialize_operator(xk, xk.shape[-3:])
    coarse_observation = observations[f"level{levels}"]
    coarse_physics = args_multilevel.coarse_physics[f"level{levels}"]
    param_reg_fine = param_regularization
    param_reg_coarse = param_reg_fine  # coarse PnP use same denoiser
    """
    Send information to the coarse level
    """
    xk_coarse = information_transfer.to_coarse(xk, xk.shape[-3:])

    """
    Optimize at coarse level
    """
    with torch.no_grad():
        if levels > 1:
            xk_coarse = ml_init_pnp(xk_coarse, level_max, levels - 1, args_multilevel, param_reg_coarse, denoiser, device)

        for k in range(param_coarse_iter):
            # PnP scheme
            xk_coarse = xk_coarse - step_size * data_fidelity.grad(xk_coarse, coarse_observation, coarse_physics)
            xk_coarse = denoiser(xk_coarse, param_reg_coarse)

    # Upscale
    xk = information_transfer.to_fine(xk_coarse, xk.shape[-3:])
    return xk

def ml_init_pnp_with_solver(init, solver):
    x0 = init.clone()
    levels = solver.levels
    return ml_init_pnp(x0, levels, levels - 1, solver.args_multilevel, solver.regularization, solver.denoiser, solver.device)
