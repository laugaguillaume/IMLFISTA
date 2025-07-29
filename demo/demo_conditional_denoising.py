import torch
import deepinv as dinv
from deepinv.loss.metric import PSNR
from multilevel.multilevel import WaveletDenoiserConditional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import an image
file_name = "butterfly.png"

url = f"https://huggingface.co/datasets/deepinv/images/resolve/main/{file_name}?download=true"
x_true = dinv.utils.load_url_image(url=url).to(device)

# Add noise to the image
sigma = 0.1 # Noise level
y = x_true + sigma * torch.randn_like(x_true)

denoiser = WaveletDenoiserConditional(
    level=3,
    wv="db8",
    non_linearity="soft",
    device=device
)

x_est = denoiser.forward(y, gamma=0.1)

psnrs = [PSNR()(x_true, y).item(), PSNR()(x_true, x_est).item()]
psnrs = [f"{psnr:.2f}" for psnr in psnrs]
dinv.utils.plot(
    [x_true, y, x_est],
    titles=["Original", f"Noisy ({psnrs[0]})", f"Reconstructed ({psnrs[1]})"],
    cmap="gray",
    figsize=[6, 6],
)