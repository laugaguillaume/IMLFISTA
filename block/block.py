import torch
import numpy as np
import pywt
import deepinv as dinv
import matplotlib.pyplot as plt
import copy
import time
import os
import json
from datetime import datetime
import platform
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Plot settings
sns.set_theme()
sns.color_palette("colorblind")
colors = sns.color_palette("colorblind")

PSNR = dinv.metric.PSNR()

from block.utils import wavelet_numpy_to_torch, wavelet_torch_to_numpy
from multilevel.utils import WaveletPriorCustom
from multilevel.multilevel import conditional_thresholding

class BlockCoordinateDescent():
    def __init__(self, img_size, wv_type, physics, data_fidelity, prior, max_levels, stepsize=1e-3, device='cpu'):
        self.img_size = img_size
        self.wv_type = wv_type
        self.physics = physics
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.max_levels = max_levels
        self.device = device

        self.stepsize = stepsize
        self.reg_weight = None
        self.y = None

        #self.wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device=self.device)
        self.wavelet_prior = WaveletPriorCustom(level=self.max_levels, wv=self.wv_type, device=self.device)
        self.proj = Projection(self.img_size, self.wv_type, self.max_levels, device=self.device)

    def update_blocks(self, x_wavelet, y, n_iter_coarse, updated_blocks, reg_weight, use_conditional_thresholding=False):
        # Update list is a list of tuples (level, mode) where mode is 'approx' or 'details'

        x_img = self.reconstruct_image(x_wavelet)

        grad = self.data_fidelity.grad(x_img, y, self.physics)
        self.grad_wavelet = self.img_to_wavelet(grad)

        '''
        y_wavelet = self.img_to_wavelet(y)
        approx = x_wavelet[0]
        APiVTa = self.physics.A(self.reconstruct_image(self.proj.project_adjoint(approx, mode="approx", level=0)))

        PiVTPiVy = self.proj.project_adjoint(  # - Pi_V^* Π_V y
                    self.proj.project(y_wavelet, mode="details", level=0),
                    mode="details", level=0
                )
        PiVTPiVy = self.reconstruct_image(PiVTPiVy)

        adj_img = self.physics.A_adjoint(APiVTa - PiVTPiVy)         # image tensor (Image domain)
        adj_wavelet = self.img_to_wavelet(adj_img)                  # list of coeffs (Wavelet domain)
        grad_proj = self.proj.project(adj_wavelet, mode="details", level=0) # list of 3 detail coeffs (Wavelet domain)

        grad_proj_img = self.reconstruct_image(self.proj.project_adjoint(grad_proj, mode="details", level=0)) # project_adjoint adds zeros to other coeffs
        #dinv.utils.plot([grad_proj_img])

        #print("Coherence shape :", grad_proj.shape)
        #print("Coherence norm :", torch.norm(grad_proj_img))
        '''

        for level, mode in updated_blocks:
            if mode == 'approx':
                coeff = self.proj.project(x_wavelet, mode, level=0)
                for i in range(n_iter_coarse):
                    coeff = coeff - self.stepsize * self.proj.project(self.grad_wavelet, mode, level=0)
                    #coeff = self.prior.prox(coeff, gamma=reg_weight*self.stepsize)
                    x_wavelet[0] = coeff

                    self.current_iter += 1
                    self.compute_metrics(x_wavelet)

            elif mode == 'details':
                coeff = self.proj.project(x_wavelet, mode, level=level)

                for i in range(n_iter_coarse):

                    if use_conditional_thresholding and self.current_iter <= 200:
                        for c in range(3):
                            coeff[c] = coeff[c] - self.stepsize * self.proj.project(self.grad_wavelet, mode, level=level)[c]
                        coeffs_same_level = x_wavelet[:level + 1]
                        approx_same_level = self.reconstruct_image(coeffs_same_level)
                        coeff = conditional_thresholding(coeff, approx_same_level, global_threshold=reg_weight)
                        coeff = torch.stack([coeff[0], coeff[1], coeff[2]], dim=0)
                        x_wavelet[level + 1] = coeff

                    else:
                        for c in range(3):
                            coeff[c] = coeff[c] - self.stepsize * self.proj.project(self.grad_wavelet, mode, level=level)[c]
                            coeff[c] = self.prior.prox(coeff[c], gamma=self.stepsize * reg_weight)
                            x_wavelet[level + 1][c] = coeff[c]

                    self.current_iter += 1
                    self.compute_metrics(x_wavelet)

            else:
                raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

        return x_wavelet

    def run(self, y, x0, x_true, n_iter, n_iter_coarse, reg_weight, update_mode='MLFB', use_conditional_thresholding=False, metrics=False):
        self.reg_weight = reg_weight
        self.y = y
        self.x_true = x_true

        print(f"BCD FB Initial PSNR in run method: {PSNR(x0, x_true).item()}")

        self.stepsizeATy = self.stepsize * self.physics.A_adjoint(self.y)

        self.losses = []
        self.times = []
        self.psnrs = []
        self.current_iter = 0
        self.cycles = []

        xk_wavelet = self.img_to_wavelet(x0)

        self.compute_metrics(x_wavelet=None, x_img=x0)
        print(self.psnrs)

        # Update list
        update_list = UpdateList(self.max_levels).get_list(type=update_mode)
        print(f"Update list ({update_mode}): {update_list}")

        if metrics:
            start = time.process_time()

        with torch.no_grad():
            with tqdm(range(n_iter), desc=f"BCD {update_mode}") as t:
                for it in t:
                    if metrics:
                        x_recon = self.reconstruct_image(xk_wavelet)
                        t.set_postfix_str(f"loss={self.data_fidelity.fn(x_recon, y, self.physics).item() + self.reg_weight * self.wavelet_prior.fn(x_recon).item():.2f}")

                    # Update detail coefficients from coarse to fine
                    for updated_blocks in update_list:
                        xk_wavelet = self.update_blocks(xk_wavelet, y, n_iter_coarse=n_iter_coarse, updated_blocks=updated_blocks, reg_weight=reg_weight, use_conditional_thresholding=use_conditional_thresholding)
                    self.cycles.append(self.current_iter)

        x_recon = self.reconstruct_image(xk_wavelet)

        if metrics:
            self.times = [t - start for t in self.times]  # Start at 0
            return x_recon, self.losses, self.times, self.cycles, self.psnrs
        return x_recon

    def FB(self, y, num_iterations, reg_weight, metrics=False):
        # Wavelet prior to compute the objective function values AND to use in the proximal step
        wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device=self.device)
        # wavelet_prior = dinv.optim.TVPrior(n_it_max=50)
        # The wavelet prior seems to introduce a decrease in the PSNR, that does not happen with TV prior...

        xk = copy.deepcopy(y)
        if metrics:
            loss, times, psnrs = [], [], []
            start = time.process_time()

        for it in range(num_iterations):
            if metrics:
                crit = self.data_fidelity.fn(xk, y, self.physics).item() + reg_weight * wavelet_prior.fn(xk).item()
                print(f"Iteration {it}/{num_iterations}, Crit: {crit:.2f}")
                loss.append(crit)
                times.append(time.process_time() - start)
                psnrs.append(PSNR(xk, self.x_true).item())
            xk = xk - self.stepsize * self.data_fidelity.grad(xk, y, self.physics)
            xk = wavelet_prior.prox(xk, gamma=reg_weight*self.stepsize)

        if metrics:
            return xk, loss, times, psnrs
        return xk

    def reconstruct_image(self, coeffs):
        """Reconstruct image from wavelet coefficients"""
        coeffs_np = wavelet_torch_to_numpy(coeffs)
        x = pywt.waverec2(coeffs_np, self.wv_type, mode='periodization')
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def img_to_wavelet(self, x):
        """Convert image to wavelet coefficients"""
        coeffs = pywt.wavedec2(x.cpu().numpy(), self.wv_type, level=self.max_levels, mode='periodization')
        return wavelet_numpy_to_torch(coeffs, device=self.device)

    def compute_metrics(self, x_wavelet, x_img=None):
        if x_img is None:
            x_img = self.reconstruct_image(x_wavelet)
        crit = self.data_fidelity.fn(x_img, self.y, self.physics).item() + self.reg_weight * self.wavelet_prior.fn(x_img).item()
        self.losses.append(crit)
        self.times.append(time.process_time())
        self.psnrs.append(PSNR(x_img, self.x_true).item())


class Projection():

    def __init__(self, image_size, wv_type, num_levels, device='cpu'):
        self.image_size = image_size
        self.wv_type = wv_type
        self.num_levels = num_levels
        self.device = device

    def project(self, coeffs, mode, level=0):
        if mode == 'details':
            return coeffs[level + 1]
        elif mode == 'approx':
            return coeffs[0]
        else:
            raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

    def project_adjoint(self, coeff, mode, level):
        # Coeff : one approximation coefficient tensor or one detail coefficient tuple (tuple of 3 detail tensors)
        zero = np.zeros(self.image_size)
        coeffs_zero = pywt.wavedec2(zero, self.wv_type, level=self.num_levels, mode='periodization')
        coeffs_zero = wavelet_numpy_to_torch(coeffs_zero, device=self.device)

        # Convert tuple to list for mutability
        coeffs_zero = list(coeffs_zero)

        if mode == 'approx':
            coeffs_zero[0] = coeff
        elif mode == 'details':
            coeffs_zero[level + 1] = coeff
        else:
            raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

        #coeffs_recon = wavelet_numpy_to_torch(pywt.wavedec2(zero, self.wv_type, level=self.num_levels))

        return coeffs_zero


class UpdateList():
    def __init__(self, max_levels):
        self.max_levels = max_levels

    def get_list(self, type='MLFB'):
        if type == 'MLFB':
            return self.create_update_list_MLFB()
        elif type == 'FB':
            return self.create_update_list_FB()
        elif type == 'MLFBcond':
            return self.create_update_list_MLFBcond()
        elif type == 'cyclic':
            return self.create_update_list_cyclic()
        else:
            raise ValueError("Invalid type. Choose 'MLFB', 'FB', 'MLFBcond' or 'cyclic'.")

    def create_update_list_MLFB(self):
        update_list = [[(0, 'approx')]]
        updated_blocks = [(0, 'approx')]
        for i in range(self.max_levels):
            updated_blocks.append((i, 'details'))
            update_list.append(copy.deepcopy(updated_blocks))
        return update_list

    def create_update_list_FB(self):
        update_list = []
        updated_blocks = [(0, 'approx')] + [(level, 'details') for level in range(self.max_levels)]
        update_list = [copy.deepcopy(updated_blocks) for _ in range(self.max_levels)]
        return update_list

    def create_update_list_MLFBcond(self):
        update_list = [[(0, 'approx')]]
        updated_blocks = [(0, 'approx')]
        for i in range(self.max_levels):
            updated_blocks.append((i, 'details'))
            update_list.append(copy.deepcopy(updated_blocks))
            update_list.append([(i, 'details')])
        return update_list

    def create_update_list_cyclic(self):
        update_list = [[(0, 'approx')]]
        for i in range(self.max_levels):
            update_list.append([(i, 'details')])
        return update_list


class BlockCoordinateDescentOptimized(BlockCoordinateDescent):
    def __init__(self, img_size, wv_type, physics, data_fidelity, prior, max_levels, stepsize=1e-3, device='cpu'):
        super().__init__(img_size, wv_type, physics, data_fidelity, prior, max_levels, stepsize, device)
        self.stored_gradient_terms = None  # To store A^T A Pi_i^* coeff_i for each block

    def initialize_stored_terms(self, x_wavelet, y):
        """Initialize stored gradient terms: A^T A Pi_i^* coeff_i for each block"""
        self.stored_gradient_terms = {}

        # Constant term - A^T y
        self.stored_constant = - self.physics.A_adjoint(y)

        # Approximation
        approx_contribution = self.physics.A_adjoint(
            self.physics.A(self.reconstruct_image(
                self.proj.project_adjoint(x_wavelet[0], mode="approx", level=0)
            ))
        )
        self.stored_gradient_terms['approx'] = approx_contribution

        # For each detail
        for level in range(self.max_levels):
            detail_contribution = self.physics.A_adjoint(
                self.physics.A(self.reconstruct_image(
                    self.proj.project_adjoint(x_wavelet[level+1], 'details', level=level)
                ))
            )
            self.stored_gradient_terms[f'details_{level}'] = detail_contribution

    def update_stored_gradient_efficiently(self, old_coeff, new_coeff, mode, level):
        if mode == 'approx':
            key = 'approx'
            coeff_diff = new_coeff - old_coeff
            diff_img = self.reconstruct_image(
                self.proj.project_adjoint(coeff_diff, mode='approx', level=0)
            )

            diff_contribution = self.physics.A_adjoint(self.physics.A(diff_img))

            self.stored_gradient_terms[key] += diff_contribution

        elif mode == 'details':
            key = f'details_{level}'
            coeff_diff = tuple(new_coeff[i] - old_coeff[i] for i in range(3))

            diff_coeffs = self.proj.project_adjoint(coeff_diff, mode, level)
            diff_img = self.reconstruct_image(diff_coeffs)

            diff_contribution = self.physics.A_adjoint(self.physics.A(diff_img))

            self.stored_gradient_terms[key] += diff_contribution

    def get_efficient_gradient(self):
        # Total gradient : sum of stored terms + constant term
        grad_img = self.stored_constant.clone()
        for term in self.stored_gradient_terms.values():
            grad_img += term

        return self.img_to_wavelet(grad_img)

    def update_blocks_optimized(self, x_wavelet, y, n_iter_coarse, updated_blocks, reg_weight):
        if self.stored_gradient_terms is None:
            self.initialize_stored_terms(x_wavelet, y)

        for level, mode in updated_blocks:
            if mode == 'approx':
                old_coeff = x_wavelet[0].clone()
                coeff = old_coeff.clone()

                for i in range(n_iter_coarse):
                    grad_wavelet = self.get_efficient_gradient()
                    grad_proj = self.proj.project(grad_wavelet, mode, level=0)

                    # Gradient step
                    new_coeff = coeff - self.stepsize * grad_proj

                    # Update associated gradient term
                    self.update_stored_gradient_efficiently(coeff, new_coeff, mode, 0)

                    coeff = new_coeff
                    x_wavelet[0] = coeff

                    self.current_iter += 1
                    self.compute_metrics(x_wavelet)

            elif mode == 'details':
                old_coeff = tuple(x_wavelet[level + 1][c].clone() for c in range(3))
                coeff = list(old_coeff)

                for i in range(n_iter_coarse):
                    # Compute the gradient
                    grad_wavelet = self.get_efficient_gradient()
                    # Project on the current block
                    grad_proj = self.proj.project(grad_wavelet, mode, level)

                    # Update each of the 3 detail coefficients
                    new_coeff = []
                    for c in range(3):
                        new_c = coeff[c] - self.stepsize * grad_proj[c]
                        new_c = self.prior.prox(new_c, gamma=reg_weight * self.stepsize)
                        new_coeff.append(new_c)
                        x_wavelet[level + 1][c] = new_c

                    # Update associated gradient terms
                    self.update_stored_gradient_efficiently(
                        tuple(coeff), tuple(new_coeff), mode, level
                    )

                    coeff = new_coeff

                    self.current_iter += 1
                    self.compute_metrics(x_wavelet)

        return x_wavelet

    def run_optimized(self, y, x0, x_true, n_iter, n_iter_coarse, reg_weight, update_mode='MLFB', metrics=False):
        self.reg_weight = reg_weight
        self.y = y
        self.x_true = x_true

        self.stepsizeATy = self.stepsize * self.physics.A_adjoint(self.y)

        self.losses = []
        self.times = []
        self.psnrs = []
        self.current_iter = 0
        self.cycles = []

        xk_wavelet = self.img_to_wavelet(x0)

        # Update list
        update_list = UpdateList(self.max_levels).get_list(type=update_mode)

        if metrics:
            start = time.process_time()

        with torch.no_grad():
            with tqdm(range(n_iter), desc=f"BCD {update_mode}") as t:
                for it in t:
                    if metrics:
                        x_recon = self.reconstruct_image(xk_wavelet)
                        t.set_postfix_str(f"loss={self.data_fidelity.fn(x_recon, y, self.physics).item() + self.reg_weight * self.wavelet_prior.fn(x_recon).item():.2f}")

                    # Update detail coefficients from coarse to fine
                    for updated_blocks in update_list:
                        xk_wavelet = self.update_blocks_optimized(
                            xk_wavelet, y, n_iter_coarse, updated_blocks, reg_weight
                        )
                    self.cycles.append(self.current_iter)

        x_recon = self.reconstruct_image(xk_wavelet)

        if metrics:
            self.times = [t - start for t in self.times]  # Start at 0
            return x_recon, self.losses, self.times, self.cycles, self.psnrs
        return x_recon


if __name__ == "__main__":
    import os
    import json
    from datetime import datetime

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    x_true = dinv.utils.load_example("butterfly.png", device=device)

    # Wavelet parameters
    J = 3
    wv_type = 'haar'

    # Physics
    """filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
    physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")"""
    sigma = 0.01
    noise_model = dinv.physics.GaussianNoise(sigma=sigma)
    physics = dinv.physics.Inpainting(tensor_size=x_true.shape[1:], mask=0.7, device=device, noise_model=noise_model)
    seed = torch.manual_seed(0)  # Random seed for reproducibility

    sigma = 0.01
    physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

    # Observation
    y = physics(x_true)

    dinv.utils.plot([x_true, y], titles=['Original', 'Observation'], cmap='gray')

    # Objective function
    data_fidelity = dinv.optim.L2()
    #prior = dinv.optim.TVPrior(n_it_max=50)
    prior = dinv.optim.L1Prior()
    reg_weight = 1e-4

    # Parameters
    n_iter = 1000
    Anorm2 = physics.compute_norm(x_true).item()
    stepsize = 0.2/Anorm2
    update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'
    print(f"Stepsize: {stepsize}")

    cbp = True

    if not cbp:
        if platform.system() == "Darwin":
            EXPERIMENTS_ROOT = Path("/Users/edgardesainte-mareville/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/blocks")
        else:
            EXPERIMENTS_ROOT = Path("/home/edgar/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/blocks")
    else:
        EXPERIMENTS_ROOT = Path(__file__).resolve().parent / "../experiments_results/blocks"  # relatif au repo

    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = f"exp_{timestamp}_J{J}_mode{update_mode}_reg{reg_weight}_niter{n_iter}_sigma{sigma}"
    exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    params = {
        "image": "butterfly.png",
        "physics": "Inpainting + Gaussian noise",
        "sigma": sigma,
        "reg_weight": reg_weight,
        "n_iter": n_iter,
        "stepsize": stepsize,
        "J": J,
        "wavelet": wv_type,
        "update_mode": update_mode,
        "comment": "With approx thresholding in BCD"
    }
    params_path = os.path.join(exp_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize, device=device)

    x0 = y.clone()

    for update_mode in ['FB', 'MLFB', 'cyclic', 'MLFBcond']:
        bcd.run(y, x0, x_true=x_true, n_iter=10, n_iter_coarse=5, reg_weight=reg_weight, update_mode=update_mode, metrics=False)

    import sys
    sys.exit()

    print("Running BCD cond...")
    x_recon_cond, loss_cond, times_cond, cycles_cond, psnr_cond = bcd.run(y, x0, x_true=x_true, n_iter=n_iter, n_iter_coarse=10, reg_weight=reg_weight, update_mode='MLFBcond', metrics=True)

    print("Running BCD MLFB...")
    x_recon_mlfb, loss_mlfb, times_mlfb, cycles_mlfb, psnr_mlfb = bcd.run(y, x0, x_true=x_true, n_iter=n_iter, n_iter_coarse=10, reg_weight=reg_weight, update_mode='MLFB', metrics=True)

    print("Running FB...")
    n_iter_fb = min(len(loss_cond), len(loss_mlfb))
    x_recon_fb, loss_fb, times_fb, psnr_fb = bcd.FB(y, num_iterations=n_iter_fb, reg_weight=reg_weight, metrics=True)

    psnrs = [PSNR(y, x_true).item(), PSNR(x_recon_fb, x_true).item(), PSNR(x_recon_mlfb, x_true).item(), PSNR(x_recon_cond, x_true).item()]
    psnrs = [f"{p:.2f}" for p in psnrs]

    # Créer une figure avec 4 sous-graphiques côte à côte
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    # Plot 1: Loss vs iterations
    axes[0].plot(loss_cond, color=colors[0], label="BCD cond", linewidth=2)
    axes[0].plot(loss_mlfb, color=colors[1], label="BCD MLFB", linewidth=2)
    axes[0].plot(loss_fb, color=colors[2], label="FB", linewidth=2)
    axes[0].scatter(cycles_cond, [loss_cond[i-1] for i in cycles_cond],
                color=colors[0], marker="o", s=80, label="cycles_cond")
    axes[0].scatter(cycles_mlfb, [loss_mlfb[i-1] for i in cycles_mlfb],
                color=colors[1], marker="x", s=80, label="cycles_mlfb")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss over Iterations\nNumber of cycles: {len(cycles_cond)}")
    axes[0].legend(frameon=True)

    # Plot 2: Loss vs time
    axes[1].plot(times_cond, loss_cond, color=colors[0], label='BCD cond', linewidth=2)
    axes[1].plot(times_mlfb, loss_mlfb, color=colors[1], label='BCD MLFB', linewidth=2)
    axes[1].plot(times_fb, loss_fb, color=colors[2], label='FB', linewidth=2)
    axes[1].set_xlabel('CPU time (s)')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss over CPU Time')
    axes[1].legend(frameon=True)

    # Plot 3: PSNR vs iterations
    axes[2].plot(psnr_cond, color=colors[0], label="BCD cond", linewidth=2)
    axes[2].plot(psnr_mlfb, color=colors[1], label="BCD MLFB", linewidth=2)
    axes[2].plot(psnr_fb, color=colors[2], label="FB", linewidth=2)
    axes[2].scatter(cycles_cond, [psnr_cond[i-1] for i in cycles_cond],
                color=colors[0], marker="o", s=80, label="cycles_cond")
    axes[2].scatter(cycles_mlfb, [psnr_mlfb[i-1] for i in cycles_mlfb],
                color=colors[1], marker="x", s=80, label="cycles_mlfb")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("PSNR")
    axes[2].set_title(f"PSNR over Iterations\nNumber of cycles: {len(cycles_cond)}")
    axes[2].legend(frameon=True)

    # Plot 4: PSNR vs time
    axes[3].plot(times_cond, psnr_cond, color=colors[0], label='BCD cond', linewidth=2)
    axes[3].plot(times_mlfb, psnr_mlfb, color=colors[1], label='BCD MLFB', linewidth=2)
    axes[3].plot(times_fb, psnr_fb, color=colors[2], label='FB', linewidth=2)
    axes[3].set_xlabel('CPU time (s)')
    axes[3].set_ylabel('PSNR')
    axes[3].set_title('PSNR over CPU Time')
    axes[3].legend(frameon=True)

    # Ajuster l'espacement et sauvegarder
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "all_plots_combined.pdf"), bbox_inches='tight')
    plt.show()

    dinv.utils.plot([x_true, y, x_recon_fb, x_recon_mlfb, x_recon_cond], titles=['Original', f'Observation \nPSNR: {psnrs[0]}', f'Reconstructed (FB) \nPSNR: {psnrs[1]}', f'Reconstructed (BCD MLFB) \nPSNR: {psnrs[2]}', f'Reconstructed (BCD cond) \nPSNR: {psnrs[3]}'], cmap='gray', save_fn=os.path.join(exp_dir, "reconstructions.pdf"))