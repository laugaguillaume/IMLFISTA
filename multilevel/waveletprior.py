import pywt
import torch
import numpy as np

from deepinv.models import WaveletDenoiser
from deepinv.optim.prior import WaveletPrior

"""
This is an attempt to re-implement DeepInv'w WaveletDenoiser so it is able to handle the padding mode 'periodization'.
This class does not work yet.
"""

class MyWaveletDenoiser(WaveletDenoiser):
    def __init__(
            self,
            level: int = 3,
            wv: str = "db8",
            device: torch.device = "cpu",
            non_linearity: str = "soft",
            mode: str = "zero",
            wvdim: int = 2,
    ):
        super().__init__(level=level, wv=wv, device=device, non_linearity=non_linearity, mode=mode, wvdim=wvdim)

    def _to_numpy(self, coeffs):
        if self.dimension == 2:
            return [coeffs[0].cpu().numpy()] + [
                tuple(d[c].cpu().numpy() for c in range(3)) for d in coeffs[1:]
            ]
        else:
            raise NotImplementedError("Only 2D wavelet transforms are supported in this function.")

    def _to_tensor(self, coeffs):
        if self.dimension == 2:
            return [torch.from_numpy(c) if isinstance(c, np.ndarray) else [torch.from_numpy(d) for d in c] for c in coeffs]
        else:
            raise NotImplementedError("Only 2D wavelet transforms are supported in this function.")

    def dwt(self, x):
        r"""
        Applies the wavelet decomposition.
        """
        if self.dimension == 2:
            dec = self._to_tensor(pywt.wavedec2(
                x.cpu().numpy(), pywt.Wavelet(self.wv), mode=self.mode, level=self.level
            ))
        else:
            raise NotImplementedError("Only 2D wavelet transforms are supported in this function.")
        dec = [list(t) if isinstance(t, tuple) else t for t in dec]
        return dec

    def psi(self, x, wavelet="db2", level=2, dimension=2, mode="zero"):
        r"""
        Returns a flattened list containing the wavelet coefficients.

        :param torch.Tensor x: input image.
        :param str wavelet: mother wavelet.
        :param int level: decomposition level.
        :param int dimension: dimension of the wavelet transform (either 2 or 3).
        """
        if dimension == 2:
            dec = self._to_tensor(pywt.wavedec2(x.cpu().numpy(), pywt.Wavelet(wavelet), mode=mode, level=level))
            dec = list(dec)
            vec = [decl.flatten(1, -1) for l in range(1, len(dec)) for decl in dec[l]]
        else:
            raise NotImplementedError("Only 2D wavelet transforms are supported in this function.")
        return vec

    def iwt(self, coeffs):
        r"""
        Applies the wavelet recomposition.
        """

        coeffs = self._list_to_tuple(coeffs)
        if self.dimension == 2:
            rec = self._to_tensor(pywt.waverec2(self._to_numpy(coeffs), pywt.Wavelet(self.wv)))
        else:
            raise NotImplementedError("Only 2D wavelet transforms are supported in this function.")
        return rec

if __name__ == "__main__":
    import torch, deepinv
    from deepinv.optim.prior import WaveletPrior

    # Example usage
    denoiser = MyWaveletDenoiser(level=3, wv='db8', mode='periodization', device=torch.device('cpu'))

    # Create a random tensor
    x = deepinv.utils.load_url_image(deepinv.utils.get_image_url('cameraman.png'), grayscale=True)

    deepinv.utils.plot(x)

    # Apply the wavelet prior
    coeffs = denoiser.dwt(x)
    print("Wavelet coefficients:", coeffs)

    deepinv.utils.plot([coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]], titles=['Approximation', 'Horizontal', 'Vertical', 'Diagonal'])

    coeffs_flat = denoiser.psi(x)
    print("Flattened wavelet coefficients:", coeffs_flat)

    deepinv.utils.plot(coeffs_flat)