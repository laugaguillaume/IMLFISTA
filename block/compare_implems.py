import deepinv as dinv
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import cProfile
import time

from block.block import BlockCoordinateDescent, BlockCoordinateDescentOptimized

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = torch.manual_seed(0)

x_true = dinv.utils.load_example('butterfly.png', device=device)

noise_model = dinv.physics.GaussianNoise(sigma=0.01)
physics = dinv.physics.Inpainting(img_size=x_true.shape[1:], mask=0.8, device=device, noise_model=noise_model)
Anorm2 = physics.compute_norm(x_true)

y = physics(x_true)

data_fidelity = dinv.optim.L2()
prior = dinv.optim.L1Prior() # It's the prior we apply to the wavelet coefficients (therefore the problem is equivalent to a Wavelet L1 prior on the image)

reg_weight = 0.1  # Regularization weight for the prior
stepsize = 0.1/Anorm2  # Stepsize for the gradient descent update
n_iter = 15
n_iter_coarse = 10

bcd = BlockCoordinateDescent(x_true.shape, wv_type='db8', physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=3, stepsize=stepsize)
bcd_opt = BlockCoordinateDescentOptimized(x_true.shape, wv_type='db8', physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=3, stepsize=stepsize)

print("=== BCD STANDARD ===")
pr1 = cProfile.Profile()
pr1.enable()
start = time.time()
recon, loss, times, cycles, psnr = bcd.run(y, y.clone(), x_true=x_true, n_iter=n_iter,
                                          n_iter_coarse=n_iter_coarse, reg_weight=reg_weight,
                                          update_mode='cyclic', metrics=True)
end = time.time()
pr1.disable()
print(f"Temps total: {end - start:.3f}s")
pr1.print_stats(sort='cumulative')

print("\n" + "="*60)
print("=== BCD OPTIMISÉ ===")
pr2 = cProfile.Profile()
pr2.enable()
start = time.time()
recon_opt, loss_opt, times_opt, cycles_opt, psnr_opt = bcd_opt.run_optimized(y, y.clone(), x_true=x_true,
                                                                                   n_iter=n_iter, n_iter_coarse=n_iter_coarse,
                                                                                   reg_weight=reg_weight, update_mode='cyclic', metrics=True)
end = time.time()
pr2.disable()
print(f"Temps total: {end - start:.3f}s")
pr2.print_stats(sort='cumulative')

dinv.utils.plot([x_true, y, recon, recon_opt], titles=['True', 'Measurement', 'BCD', 'BCD Optimized'], suptitle='Reconstruction results', cmap='gray')

# --- LOSS vs ITERATION ---
plt.figure(figsize=(12, 5))
plt.plot(loss, label='BCD', color='blue')
plt.plot(loss_opt, label='BCD Optimized', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.show()

# --- LOSS vs TIME ---
plt.figure(figsize=(12, 5))
plt.plot(times, loss, label='BCD', color='blue')
plt.plot(times_opt, loss_opt, label='BCD Optimized', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Loss')
plt.title('Loss vs Time')
plt.legend()
plt.show()

# --- PSNR vs ITERATION ---
plt.figure(figsize=(12, 5))
plt.plot(psnr, label='BCD', color='blue')
plt.plot(psnr_opt, label='BCD Optimized', color='orange')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Iteration')
plt.legend()
plt.show()

# --- PSNR vs TIME ---
plt.figure(figsize=(12, 5))
plt.plot(times, psnr, label='BCD', color='blue')
plt.plot(times_opt, psnr_opt, label='BCD Optimized', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Time')
plt.legend()
plt.show()