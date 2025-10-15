import torch
import torch.nn.functional as F
import math

# ---------------------------
# utilitaires : convolution (blur) et son adjoint
# ---------------------------
def make_gaussian_kernel(k=9, sigma=2.0, device='cpu'):
    """Crée un kernel gaussien 2D (torch.Tensor) de taille k x k"""
    ax = torch.arange(-(k//2), k//2 + 1, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def conv2d_batch(x, kernel):
    """
    x: tensor (B,1,H,W)
    kernel: (kh,kw)
    retourne: (B,1,H,W)
    """
    B = x.shape[0]
    kh, kw = kernel.shape
    k = kernel.view(1, 1, kh, kw).to(x.device)
    pad_h = kh // 2
    pad_w = kw // 2
    return F.conv2d(x, k.repeat(1,1,1,1), padding=(pad_h, pad_w))

def convT2d_batch(x, kernel):
    """
    Adjoint de conv par kernel (ici conv transpose) :
    x: (B,1,H,W)
    kernel: (kh,kw)
    """
    kh, kw = kernel.shape
    k = kernel.view(1, 1, kh, kw).to(x.device)
    pad_h = kh // 2
    pad_w = kw // 2
    # conv_transpose2d expects (out_channels, in_channels, kh, kw) kernel
    return F.conv_transpose2d(x, k.repeat(1,1,1,1), padding=(pad_h, pad_w))

# ---------------------------
# Proximal operator de TV (Chambolle) — version torch
# ---------------------------
def prox_tv_chambolle(f, weight, n_iter=50):
    """
    prox_{weight * TV}(f) via l'algorithme de Chambolle (ROF denoising prox).
    f: (B,1,H,W), torch tensor
    weight: scalaire >=0
    renvoie: prox (B,1,H,W)
    Référence (idée) : Chambolle, 2004.
    """
    # si weight==0, prox identique
    if weight == 0:
        return f.clone()

    device = f.device
    B, C, H, W = f.shape
    # dual variable p = (px, py) shape (B,2,H,W)
    p = torch.zeros((B, 2, H, W), device=device, dtype=f.dtype)
    # paramètre temporel (stabilité)
    tau = 0.25

    # we operate on u = f (rename)
    u = f

    # Precompute f/weight
    f_over_w = f / weight

    for _ in range(n_iter):
        # div p
        px = p[:, 0]
        py = p[:, 1]
        # backward differences for divergence
        div_p = torch.zeros_like(px)
        # px difference: px(x,y) - px(x-1,y)
        div_p[:, :] += px - torch.roll(px, shifts=1, dims=3)
        # py difference: py(x,y) - py(x,y-1)
        div_p[:, :] += py - torch.roll(py, shifts=1, dims=2)

        # compute gradient of (div_p - f/weight)
        q = div_p - f_over_w[:,0]
        # gradient: forward differences
        qx = torch.roll(q, shifts=-1, dims=3) - q  # d/dx forward
        qy = torch.roll(q, shifts=-1, dims=2) - q  # d/dy forward

        # update p
        px_new = p[:,0] + tau * qx
        py_new = p[:,1] + tau * qy
        norm = torch.sqrt(px_new**2 + py_new**2).clamp(min=1.0)
        p[:,0] = px_new / norm
        p[:,1] = py_new / norm

    # final prox: u - weight * div p
    px = p[:,0]
    py = p[:,1]
    div_p = torch.zeros_like(px)
    div_p[:, :] += px - torch.roll(px, shifts=1, dims=3)
    div_p[:, :] += py - torch.roll(py, shifts=1, dims=2)
    # prox result
    result = f - weight * div_p.unsqueeze(1)
    return result

# ---------------------------
# PSNR util
# ---------------------------
def psnr(x, ref, data_range=1.0):
    mse = F.mse_loss(x, ref)
    return 10 * torch.log10((data_range**2) / mse)

# ---------------------------
# Simulation : image, blur, bruit
# ---------------------------
def simulate_blur_and_noise(img, kernel, sigma_noise=0.01):
    """img: (H,W) float [0,1] torch tensor"""
    device = img.device
    x = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    b = conv2d_batch(x, kernel)
    noise = torch.randn_like(b) * sigma_noise
    return (b + noise), x

# ---------------------------
# Algorithm Forward-Backward (proximal gradient)
# ---------------------------
def deblur_fb(b, kernel, lam=0.1, tau=0.5, n_iter=200, prox_iters=50, init=None):
    """
    b: observed blurred noisy image (1,1,H,W)
    kernel: (kh,kw) tensor
    lam: regularization weight (lambda)
    tau: step size (choisir < 1 / Lip(A^T A) ; empirique)
    n_iter: nombre d'itérations FB
    prox_iters: nombre d'itérations pour la prox TV (Chambolle)
    """
    device = b.device
    if init is None:
        x = b.clone()
    else:
        x = init.clone()

    for k in range(n_iter):
        # gradient step: grad = A^T(Ax - b)
        Ax = conv2d_batch(x, kernel)
        resid = Ax - b
        grad = convT2d_batch(resid, kernel)
        y = x - tau * grad

        # proximal step for tau * lambda * TV
        x = prox_tv_chambolle(y, weight=tau * lam, n_iter=prox_iters)

        if (k % 50 == 0) or (k == n_iter-1):
            with torch.no_grad():
                cur_psnr = psnr(x, true_img_tensor).item() if 'true_img_tensor' in globals() else float('nan')
            print(f"Iter {k:4d} | (approx) ||res||_2={resid.norm().item():.4f} | PSNR={cur_psnr:.2f}")

    return x

# ---------------------------
# Exemple d'utilisation
# ---------------------------
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Charger une image de test : ici juste un pattern synthétique
    H, W = 128, 128
    true_img = torch.zeros((H, W), dtype=torch.float32)
    true_img[30:100, 30:100] = 1.0  # carré --> edges pour TV voir effet
    true_img = true_img.to(device)

    # kernel et observation
    kernel = make_gaussian_kernel(k=9, sigma=3.0, device=device)
    b, true_img_tensor = simulate_blur_and_noise(true_img, kernel, sigma_noise=0.02)
    b = b.to(device)

    # paramètres FB
    lam = 0.12         # poids TV
    tau = 0.6          # step size (empirique) -> surveiller stabilité
    n_iter = 400
    prox_iters = 25

    x_rec = deblur_fb(b, kernel, lam=lam, tau=tau, n_iter=n_iter, prox_iters=prox_iters)

    # affichage (si tu es dans un notebook)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1); plt.imshow(true_img.cpu(), cmap='gray'); plt.title('Original'); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(b[0,0].cpu(), cmap='gray'); plt.title('Blur + bruit'); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(x_rec[0,0].cpu().detach(), cmap='gray'); plt.title('Reconstruction FB+TV'); plt.axis('off')
        plt.show()
    except Exception:
        pass
