import deepinv as dinv
import torch
import torch.nn.functional as F
import pywt


PSNR = dinv.metric.PSNR()


def nabla(I):
    '''
    Computes the nabla operator (finite differences) on the input image I.
    Comes from deepinv : https://deepinv.github.io/deepinv/auto_examples/unfolded/demo_custom_prior_unfolded.html

    Args:
        I (torch.Tensor) of shape (b, c, h, w)

    Returns:
        G (torch.Tensor) of shape (b, c, h, w, 2), where
        G[..., 0] is the gradient in the x direction
        G[..., 1] is the gradient in the y direction
    '''
    b, c, h, w = I.shape
    G = torch.zeros((b, c, h, w, 2))
    G[:, :, :-1, :, 0] = G[:, :, :-1, :, 0] - I[:, :, :-1]
    G[:, :, :-1, :, 0] = G[:, :, :-1, :, 0] + I[:, :, 1:]
    G[:, :, :, :-1, 1] = G[:, :, :, :-1, 1] - I[..., :-1]
    G[:, :, :, :-1, 1] = G[:, :, :, :-1, 1] + I[..., 1:]

    return G

def local_average(I, kernel_size=3):
    '''
    Computes the average local gradient of the input image I using a kernel of size kernel_size.

    Args:
        I (torch.Tensor) of shape (b, c, h, w)
        kernel_size (int): Size of the kernel for average pooling

    Returns:
        G_avg (torch.Tensor) of shape (b, c, h, w, 2), where
        G_avg[..., 0] is the average gradient in the x direction
        G_avg[..., 1] is the average gradient in the y direction
    '''
    G = nabla(I)  # (b, c, h, w, 2)
    b, c, h, w, _ = G.shape
    G = G.permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w)                                 # Rearrange: (b, c, h, w, 2) -> (b, 2c, h, w)
    G_avg = F.avg_pool2d(G, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) # Average pooling (Padding with zeros)
    G_avg = G_avg.view(b, c, 2, h, w).permute(0, 1, 3, 4, 2)                             # Reshape back: (b, 2c, h, w) -> (b, c, h, w, 2)

    return G_avg

def min_max_normalize(im):
    '''
    Min-max normalization of the input image im.
    '''
    img = im.clone()
    shape = img.shape
    img = img.reshape(shape[0], -1)
    mini = img.min(1)[0]
    maxi = img.max(1)[0]
    idx = mini < maxi
    mini = mini[idx].unsqueeze(1)
    maxi = maxi[idx].unsqueeze(1)
    img[idx, :] = (img[idx, :] - mini) / (maxi - mini)
    img = img.reshape(shape)
    return img

def apply_prox(wavelet_coeffs, prior, gamma):
    '''
    Applies the proximal operator to the wavelet coefficients.

    Args:
        wavelet_coeffs = (cAn, dn, dn-1, ..., d1)
        prior = Prior object
        gamma = regularization parameter

    Returns:
        wavelet_coeffs = (cAn, dn, dn-1, ..., d1) with the proximal operator applied to each detail coefficient
    '''
    cAn, *details = wavelet_coeffs
    levels = len(details)

    for level in range(levels):
        for c in range(3):
            details[level][c] = prior.prox(details[level][c], gamma = gamma)

    return (cAn, *details)

def comparative_plot_wavelets(coeffs_list, level, labels=None, save_fn=None, grayscale=True):
    '''
    Plot wavelet coefficients from an arbitrary number of images for comparison.

    Args:
        coeffs_list: List of coefficient sets from multiple images
        level: The wavelet decomposition level to plot
        labels: List of labels for each image (defaults to Image1, Image2, etc.)
        save_fn: Optional filename to save the plot
    '''
    # Extract details from each set of coefficients
    num_images = len(coeffs_list)
    if labels is None:
        labels = [f'Image{i+1}' for i in range(num_images)]

    # Extract d_list (details list) from each coefficient set
    approx_list = [coeffs[0] for coeffs in coeffs_list]
    d_list = [coeffs[1:] for coeffs in coeffs_list]

    levels = len(d_list[0])


    # Stack coefficients from all images
    if grayscale:
        img_list = [
        torch.cat([a.squeeze(0) for a in approx_list], dim=-2),  # approx
        torch.cat([d[level][0].squeeze(0) for d in d_list], dim=-2),  # H details
        torch.cat([d[level][1].squeeze(0) for d in d_list], dim=-2),  # V details
        torch.cat([d[level][2].squeeze(0) for d in d_list], dim=-2),  # D details
        ]

        print(img_list[0].shape)

    else:
        img_list = [
            torch.stack(approx_list, dim=1).squeeze(),  # approximation coefficients
            torch.stack([d[level][0] for d in d_list], dim=1).squeeze(),  # H details
            torch.stack([d[level][1] for d in d_list], dim=1).squeeze(),  # V details
            torch.stack([d[level][2] for d in d_list], dim=1).squeeze(),  # D details
        ]

    titles = ['approx', 'H', 'V', 'D']
    label_str = ", ".join([f"row {i+1}: {label}" for i, label in enumerate(labels)])

    if save_fn:
        dinv.utils.plot(
            img_list,
            titles=titles,
            suptitle=f'Wavelet coefficients at scale {levels - level} \n({label_str})',
            tight=True,
            save_fn=save_fn,
            rescale_mode='min_max',
            cmap='gray'
        )
    else:
        dinv.utils.plot(
            img_list,
            titles=titles,
            suptitle=f'Wavelet coefficients at scale {levels - level} \n({label_str})',
            tight=True,
            rescale_mode='min_max',
            cmap='gray'
        )

def get_approximation_previous_scale(coeffs, wavelet_type='haar'):
    '''
    Computes the approximation coefficients of the previous scale (scale j-1) from the current coefficients (scale j).

    Args:
        coeffs = (a_J, d_J, ..., d_1)

    Returns:
        approx_fine = a_{J-1} = W^T (a_J, d_J)
    '''
    if coeffs[0].shape != coeffs[1][0].shape:
        raise ValueError("Approximation and detail coefficients must have the same shape.")

    approx_fine = pywt.idwt2((coeffs[0], coeffs[1]), wavelet=wavelet_type, mode='periodization')
    approx_fine = torch.tensor(approx_fine, device=coeffs[0].device, dtype=coeffs[0].dtype)
    return approx_fine

def idwt_torch_pywt(coeffs, L_adjoint):
    approx_torch = coeffs[0]
    details_torch = coeffs[1:]
    approx_reconstructed_np = approx_torch.numpy()
    details_reconstructed_np = [(detail[0].numpy(), detail[1].numpy(), detail[2].numpy()) for detail in details_torch]

    return torch.tensor(L_adjoint([approx_reconstructed_np] + details_reconstructed_np))

def wavelet_numpy_to_torch(coeffs):
    approx_np = coeffs[0]
    details_np = coeffs[1:]
    approx_torch = torch.tensor(approx_np, device='cpu', dtype=torch.float32)
    details_torch = [torch.tensor(detail, device='cpu', dtype=torch.float32) for detail in details_np]
    return (approx_torch, *details_torch)

def psnr_details(coefficients_y, coefficients_x):
    """
    Computes the sum of PSNRs for each detail coefficient of the wavelet transform.

    Args:
        coefficients_y: Wavelet coefficients of the noisy image.
        coefficients_x: Wavelet coefficients of the true image.

    Returns:
        psnr: sum of PSNRs for each detail coefficients
    """
    psnr_values = []
    for i in range(1, len(coefficients_y)):
        psnr_value = PSNR()(coefficients_y[i], coefficients_x[i])
        psnr_values.append(psnr_value.item())
    return sum(psnr_values)