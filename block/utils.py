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
        # If detail has shape [3, ...], convert to tuple of 3 arrays
        detail_tuple = (detail[0].detach().cpu().numpy(), detail[1].detach().cpu().numpy(), detail[2].detach().cpu().numpy())
        details_np.append(detail_tuple)
    return (approx_np, *details_np)

def wavelet_numpy_to_torch(coeffs, device='cpu'):
    """
    Convert wavelet coefficients from NumPy arrays to PyTorch tensors.

    Args:
        coeffs: Tuple containing (approx_array, detail_array1, detail_array2, ...)
                where approx_array is the approximation coefficients as a numpy.ndarray
                and detail_arrays are the detail coefficients as numpy.ndarrays
        device: Device to place the tensors on ('cpu' or 'cuda')

    Returns:
        Tuple containing (approx_tensor, detail_tensor1, detail_tensor2, ...)
        where tensors are torch.Tensors
    """
    approx_np = coeffs[0]
    details_np = coeffs[1:]

    # Convert approximation coefficients
    approx_torch = torch.from_numpy(approx_np.astype(np.float32)).to(device)

    # Convert detail coefficients efficiently
    details_torch = []
    for detail in details_np:
        if isinstance(detail, tuple) and len(detail) == 3:
            # detail is a tuple of (horizontal, vertical, diagonal) arrays
            h_arr, v_arr, d_arr = detail

            # Ensure arrays are contiguous and float32
            h_arr = np.ascontiguousarray(h_arr.astype(np.float32))
            v_arr = np.ascontiguousarray(v_arr.astype(np.float32))
            d_arr = np.ascontiguousarray(d_arr.astype(np.float32))

            # Stack the three arrays into a single array with shape [3, ...]
            detail_stacked = np.stack([h_arr, v_arr, d_arr], axis=0)

            # Convert to torch tensor
            detail_torch = torch.from_numpy(detail_stacked).to(device)
            details_torch.append(detail_torch)
        else:
            # Fallback for other formats
            if not detail.flags.c_contiguous:
                detail = np.ascontiguousarray(detail)
            if detail.dtype != np.float32:
                detail = detail.astype(np.float32)

            detail_torch = torch.from_numpy(detail).to(device)
            details_torch.append(detail_torch)

    return [approx_torch, *details_torch]