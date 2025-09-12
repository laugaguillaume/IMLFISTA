import os
import platform
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepinv as dinv

from block.block import BlockCoordinateDescent
from multilevel.multilevel import ParametersMultilevel, MultiLevel, MultiLevelWavelets
from multilevel.multilevel_initialization import ml_init_pnp

PSNR = dinv.metric.PSNR()

device = torch.device('cpu')
x_true = dinv.utils.load_example("butterfly.png", device=device)

# Wavelet parameters
J = 3
wv_type = 'haar'

# Physics
blur_variance = (2, 2)
filter_0 = dinv.physics.blur.gaussian_blur(sigma=blur_variance, angle=0.0)
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
n_iter = 100
Anorm2 = physics.compute_norm(x_true).item()
stepsize = 0.1/Anorm2
update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'
print(f"Stepsize: {stepsize}")

#%%%--- Experiment directory setup %%%---
if platform.system() == "Darwin":  # macOS
    EXPERIMENTS_ROOT = "/Users/edgardesainte-mareville/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/compare_methods"
else:  # Linux ou autre
    EXPERIMENTS_ROOT = "/home/edgar/kDrive/Documents/Thèse/Experiments/multilevel_conditional_reconstruction/compare_methods"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
exp_name = f"exp_{timestamp}_J{J}_mode{update_mode}_reg{reg_weight}_niter{n_iter}_sigma{sigma}"
exp_dir = os.path.join(EXPERIMENTS_ROOT, exp_name)
os.makedirs(exp_dir, exist_ok=True)

params = {
    "image": "butterfly.png",
    "physics": f'Blur (variance {blur_variance}) + Gaussian noise (sigma={sigma})',
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

#%%%--- Reconstruction %%%---

def run_FB(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_MLFB(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_BCD(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_PnP(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_MLPnP(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_MLFBcond(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

def run_BCDcond(params):
    recon = None
    loss = None
    psnr = None
    times = None
    return recon, loss, psnr, times

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