import torch
import numpy as np

def wavelet_torch_to_numpy(coeffs):
    """
    Convert wavelet coefficients from PyTorch tensors to NumPy arrays.

    Args:
        coeffs: Tuple/List containing (approx_tensor, detail_tensor1, detail_tensor2, ...)
                where approx_tensor is the approximation coefficients as a torch.Tensor
                and detail_tensors are the detail coefficients as torch.Tensors

    Returns:
        Tuple containing (approx_array, detail_tuple1, detail_tuple2, ...)
        where detail_tuples are (horizontal, vertical, diagonal) tuples as expected by PyWavelets
    """
    approx_torch = coeffs[0]
    details_torch = coeffs[1:]

    approx_np = approx_torch.detach().cpu().numpy()

    # Convert detail coefficients to the proper PyWavelets format
    details_np = []
    for detail in details_torch:
        if isinstance(detail, torch.Tensor) and len(detail.shape) >= 1 and detail.shape[0] == 3:
            # If detail has shape [3, ...], convert to tuple of 3 arrays
            detail_np = detail.detach().cpu().numpy()
            detail_tuple = (detail_np[0], detail_np[1], detail_np[2])
            details_np.append(detail_tuple)
        else:
            # If detail is already in the right format or different structure
            detail_np = detail.detach().cpu().numpy()
            details_np.append(detail_np)

    return (approx_np, *details_np)

def wavelet_numpy_to_torch(coeffs):
    """
    Convert wavelet coefficients from NumPy arrays to PyTorch tensors.

    Args:
        coeffs: Tuple containing (approx_array, detail_array1, detail_array2, ...)
                where approx_array is the approximation coefficients as a numpy.ndarray
                and detail_arrays are the detail coefficients as numpy.ndarrays

    Returns:
        Tuple containing (approx_tensor, detail_tensor1, detail_tensor2, ...)
        where tensors are torch.Tensors
    """
    approx_np = coeffs[0]
    details_np = coeffs[1:]
    approx_torch = torch.tensor(approx_np, device='cpu', dtype=torch.float32)
    details_torch = [torch.tensor(detail, device='cpu', dtype=torch.float32) for detail in details_np]
    return list((approx_torch, *details_torch))