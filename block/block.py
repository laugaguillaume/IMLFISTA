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

PSNR = dinv.metric.PSNR()

from block.utils import wavelet_numpy_to_torch, wavelet_torch_to_numpy

class Projection():

    def __init__(self, image_size, wv_type, num_levels):
        self.image_size = image_size
        self.wv_type = wv_type
        self.num_levels = num_levels

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
        coeffs_zero = pywt.wavedec2(zero, self.wv_type, level=self.num_levels)
        coeffs_zero = wavelet_numpy_to_torch(coeffs_zero)

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
        else:
            raise ValueError("Invalid type. Choose 'MLFB', 'FB' or 'MLFBcond'.")

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

class BlockCoordinateDescent():
    def __init__(self, img_size, wv_type, physics, data_fidelity, prior, max_levels, stepsize=1e-3):
        self.img_size = img_size
        self.wv_type = wv_type
        self.physics = physics
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.max_levels = max_levels
        self.stepsize = stepsize

        self.proj = Projection(self.img_size, self.wv_type, self.max_levels)

        self.n_iter_tot = 0

    def reconstruct_image(self, coeffs):
        """Reconstruct image from wavelet coefficients"""
        coeffs_np = wavelet_torch_to_numpy(coeffs)
        x = pywt.waverec2(coeffs_np, self.wv_type)
        return torch.tensor(x, dtype=torch.float32)

    def img_to_wavelet(self, x):
        """Convert image to wavelet coefficients"""
        coeffs = pywt.wavedec2(x.cpu().numpy(), self.wv_type, level=self.max_levels)
        return wavelet_numpy_to_torch(coeffs)

    def update_blocks(self, x_wavelet, y, updated_blocks, reg_weight):
        # Update list is a list of tuples (level, mode) where mode is 'approx' or 'details'

        x_img = self.reconstruct_image(x_wavelet)
        y_wavelet = self.img_to_wavelet(y)

        grad = self.data_fidelity.grad(x_img, y, self.physics)
        grad_wavelet = self.img_to_wavelet(grad)

        approx = x_wavelet[0]
        APiVTa = self.physics.A(self.reconstruct_image(self.proj.project_adjoint(approx, mode="approx", level=0)))

        PiVTPiVy = self.proj.project_adjoint(  # - Π_V^* Π_V y
                    self.proj.project(y_wavelet, mode="details", level=0),
                    mode="details", level=0
                )
        PiVTPiVy = self.reconstruct_image(PiVTPiVy)

        adj_img = self.physics.A_adjoint(APiVTa - PiVTPiVy)         # image tensor
        adj_wavelet = self.img_to_wavelet(adj_img)                 # list of coeffs (torch)
        grad_proj = self.proj.project(adj_wavelet, mode="details", level=0)

        #dinv.utils.plot(grad_proj[0])
        #print("Coherence norm :", torch.norm(grad_proj))

        for level, mode in updated_blocks:
            self.n_iter_tot += 1
            if mode == 'approx':
                coeff = self.proj.project(x_wavelet, mode, level=0)

                coeff = coeff - self.stepsize * self.proj.project(grad_wavelet, mode, level=0)
                coeff = self.prior.prox(coeff, gamma=reg_weight*self.stepsize)
                x_wavelet[0] = coeff

            elif mode == 'details':
                coeff = self.proj.project(x_wavelet, mode, level=level)

                for c in range(3):
                    coeff[c] = coeff[c] - self.stepsize * self.proj.project(grad_wavelet, mode, level=level)[c]
                    coeff[c] = self.prior.prox(coeff[c], gamma=reg_weight * self.stepsize)

                    x_wavelet[level + 1][c] = coeff[c]

            else:
                raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

        return x_wavelet

    def run(self, y, x0, x_true, n_iter, reg_weight, update_mode='MLFB', metrics=False):
        # Wavelet prior to compute the objective function values
        wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device='cpu')
        xk_wavelet = self.img_to_wavelet(x0)

        # Update list
        update_list = UpdateList(self.max_levels).get_list(type=update_mode)
        #print(update_list)

        if metrics:
            loss, times = [], []
            start = time.process_time()

        for it in range(n_iter):
            if metrics:
                x_recon = self.reconstruct_image(xk_wavelet)
                crit = self.data_fidelity.fn(x_recon, y, self.physics).item() + reg_weight * wavelet_prior.fn(x_recon).item()
                print(f"Iteration {it}/{n_iter}, Crit: {crit:.2f}")
                loss.append(crit)
                times.append(time.process_time() - start)

            # Update detail coefficients from coarse to fine
            for updated_blocks in update_list:
                xk_wavelet = self.update_blocks(xk_wavelet, y, updated_blocks=updated_blocks, reg_weight=reg_weight)

        x_recon = self.reconstruct_image(xk_wavelet)
        if metrics:
            return x_recon, loss, times
        return x_recon


    def FB(self, y, num_iterations, reg_weight, metrics=False):
        # Wavelet prior to compute the objective function values AND to use in the proximal step
        wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device='cpu')
        xk = copy.deepcopy(y)
        if metrics:
            loss, times = [], []
            start = time.process_time()

        for it in range(num_iterations):
            if metrics:
                crit = self.data_fidelity.fn(xk, y, self.physics).item() + reg_weight * wavelet_prior.fn(xk).item()
                print(f"Iteration {it}/{num_iterations}, Crit: {crit:.2f}")
                loss.append(crit)
                times.append(time.process_time() - start)
            xk = xk - self.stepsize * self.data_fidelity.grad(xk, y, self.physics)
            xk = wavelet_prior.prox(xk, gamma=reg_weight*self.stepsize)

        if metrics:
            return xk, loss, times
        return xk


if __name__ == "__main__":
    import os
    import json
    from datetime import datetime

    device = torch.device('cpu')
    x_true = dinv.utils.load_example("butterfly.png", device=device)

    # Wavelet parameters
    J = 3
    wv_type = 'haar'

    # Physics
    filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
    physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")
    seed = torch.manual_seed(0)  # Random seed for reproducibility

    sigma = 0.01
    physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

    # Observation
    y = physics(x_true)

    # Objective function
    data_fidelity = dinv.optim.L2()
    #prior = dinv.optim.TVPrior(n_it_max=50)
    prior = dinv.optim.L1Prior()
    reg_weight = 1e-2

    # Parameters
    n_iter = 1000
    Anorm2 = physics.compute_norm(x_true).item()
    stepsize = 0.1/Anorm2
    update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'
    print(f"Stepsize: {stepsize}")

    if platform.system() == "Darwin":  # macOS
        EXPERIMENTS_ROOT = "/Users/edgardesainte-mareville/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/blocks"
    else:  # Linux ou autre
        EXPERIMENTS_ROOT = "/home/edgar/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/blocks"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = f"exp_{timestamp}_J{J}_mode{update_mode}_reg{reg_weight}_niter{n_iter}_sigma{sigma}"
    exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    params = {
        "image": "butterfly.png",
        "physics": "blur + Gaussian noise",
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

    bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize)

    x0 = y.clone()

    x_recon, loss, times = bcd.run(y, x0, x_true=x_true, n_iter=n_iter, reg_weight=reg_weight, update_mode=update_mode, metrics=True)
    x_recon_fb, loss_fb, times_fb = bcd.FB(y, num_iterations=n_iter, reg_weight=reg_weight, metrics=True)

    # To have roughly the same number of iterations for FB and BCD
    '''n_iter_tot_bcd = int(bcd.n_iter_tot/4)
    x_recon_fb, loss_fb = bcd.FB(y, num_iterations=n_iter_tot_bcd, reg_weight=reg_weight, metrics=True)
    print(f"Total number of iterations BCD: {n_iter_tot_bcd}")
    loss =  np.array(loss).repeat(int(n_iter_tot_bcd/n_iter))'''

    psnrs = [PSNR(y, x_true).item(), PSNR(x_recon_fb, x_true).item(), PSNR(x_recon, x_true).item()]
    psnrs = [f"{p:.2f}" for p in psnrs]

    # Plot loss vs iterationns
    plt.figure()
    plt.plot(loss, label=f'BCD {update_mode}')
    plt.plot(loss_fb, label='FB')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "loss_iterations.pdf"))
    plt.show()

    # Plot loss vs time
    plt.figure()
    plt.plot(times, loss, label=f'BCD {update_mode}')
    plt.plot(times_fb, loss_fb, label='FB')
    plt.xlabel('CPU time (s)')
    plt.ylabel('Loss')
    plt.title('Loss over CPU Time')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "loss_time.pdf"))
    plt.show()

    dinv.utils.plot([x_true, y, x_recon_fb, x_recon], titles=['Original', f'Observation \nPSNR: {psnrs[0]}', f'Reconstructed (FB) \nPSNR: {psnrs[1]}', f'Reconstructed (BCD) \nPSNR: {psnrs[2]}'], cmap='gray', save_fn=os.path.join(exp_dir, "reconstructions.pdf"))