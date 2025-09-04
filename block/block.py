import torch
import numpy as np
import pywt
import deepinv as dinv
import matplotlib.pyplot as plt
import copy

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

        return coeffs_zero

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

    def reconstruct_image(self, coeffs):
        """Reconstruct image from wavelet coefficients"""
        coeffs_np = wavelet_torch_to_numpy(coeffs)
        x = pywt.waverec2(coeffs_np, self.wv_type)
        return torch.tensor(x, dtype=torch.float32)

    def img_to_wavelet(self, x):
        """Convert image to wavelet coefficients"""
        coeffs = pywt.wavedec2(x.cpu().numpy(), self.wv_type, level=self.max_levels)
        return wavelet_numpy_to_torch(coeffs)

    def update_blocks(self, x_wavelet, y, update_list, reg_weight):
        # Update list is a list of tuples (level, mode) where mode is 'approx' or 'details'

        x_img = self.reconstruct_image(x_wavelet)

        grad = self.data_fidelity.grad(x_img, y, self.physics)
        grad_wavelet = self.img_to_wavelet(grad)

        for level, mode in update_list:
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

        return x_wavelet

    def run(self, y, x0, x_true, num_iterations, reg_weight, metrics=False):
        wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device='cpu')

        xk_wavelet = self.img_to_wavelet(x0)
        update_list = [(0, 'approx')]
        #update_list = [(0, 'approx')] + [(level, 'details') for level in range(self.max_levels)]
        if metrics:
            loss = []

        for it in range(num_iterations):
            if metrics:
                x_recon = self.reconstruct_image(xk_wavelet)
                crit = self.data_fidelity.fn(x_recon, y, self.physics).item() + reg_weight * wavelet_prior.fn(x_recon).item()
            print(f"Iteration {it}/{num_iterations}, Crit: {crit:.2f}")
            loss.append(crit)

            # Update detail coefficients from coarse to fine
            for level in range(self.max_levels):
                xk_wavelet = self.update_blocks(xk_wavelet, y, update_list=update_list, reg_weight=reg_weight)
                update_list.append((level, 'details'))


        x_recon = self.reconstruct_image(xk_wavelet)
        return x_recon, loss


    def FB(self, y, num_iterations, reg_weight, metrics=False):
        wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device='cpu')
        xk = copy.deepcopy(y)
        if metrics:
            loss = []

        for it in range(num_iterations):
            if metrics:
                crit = self.data_fidelity.fn(xk, y, self.physics).item() + reg_weight * wavelet_prior.fn(xk).item()
                print(f"Iteration {it}/{num_iterations}, Crit: {crit:.2f}")
                loss.append(crit)
            xk = xk - self.stepsize * self.data_fidelity.grad(xk, y, self.physics)
            xk = wavelet_prior.prox(xk, gamma=reg_weight*self.stepsize)

        return xk, loss


if __name__ == "__main__":
    device = torch.device('cpu')
    x_true = dinv.utils.load_example("butterfly.png", device=device)

    J = 3

    # Test BCD
    filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
    physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")
    seed = torch.manual_seed(0)  # Random seed for reproducibility

    sigma = 0.01
    physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

    # Construct observation and display original image
    y = physics(x_true)

    data_fidelity = dinv.optim.L2()
    prior = dinv.optim.L1Prior()

    Anorm2 = physics.compute_norm(x_true)
    stepsize = 0.01/Anorm2
    print(f"Stepsize: {stepsize}")

    bcd = BlockCoordinateDescent(x_true.shape, 'haar', physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize)

    y = physics(x_true)
    x0 = y.clone()

    n_iter = 200
    x_recon, loss = bcd.run(y, x0, x_true=x_true, num_iterations=n_iter, reg_weight=0, metrics=True)
    x_recon_fb, loss_fb = bcd.FB(y, num_iterations=n_iter, reg_weight=0, metrics=True)

    # Plot loss
    plt.figure()
    plt.plot(loss, label='BCD')
    plt.plot(loss_fb, label='FB')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.legend()
    plt.show()

    #x_recon = x_recon_fb
    dinv.utils.plot([x_true, y, x_recon_fb, x_recon], titles=['Original', 'Blurred', 'Reconstructed (FB)', 'Reconstructed (BCD)'], cmap='gray')