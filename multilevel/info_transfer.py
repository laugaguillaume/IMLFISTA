import numpy
import torch.nn
import torch.nn.functional as F
import deepinv
import pywt
"""
Created on Oct 18 2024
@author: Nils Laurent
"""

class DownsamplingTransfer:
    def __init__(self, def_filter, padding="circular", factor=2):
        self.filter_object = def_filter
        self.padding = padding
        self.factor = factor
        self.op = None  # will be created on first use
        self.wavelet_type = self.filter_object.wavelet_type() if hasattr(self.filter_object, 'wavelet_type') else None

    def _initialize_operator(self, x, target_shape):
        device = x.device
        dtype = x.dtype

        filt_2d = self._getfilter().type(dtype)

        padding = "valid" if isinstance(self.filter_object, Dirac) else self.padding

        self.op = deepinv.physics.Downsampling(
            target_shape, filter=filt_2d, factor=self.factor, device=device, padding=padding
        )
    def _getfilter(self):
        return self.filter_object.get_2d_filter().unsqueeze(0).unsqueeze(0)

    def to_coarse(self, x, target_shape):
        if self.op is None:
            self._initialize_operator(x, target_shape)

        if x.dim() == 3:
            return self.op.A(x.unsqueeze(0)).squeeze(0)
        return self.op.A(x)

    def to_fine(self, x, target_shape):
        if self.op is None:
            self._initialize_operator(x, target_shape)

        if isinstance(self.filter_object, Dirac):
            upsample = torch.nn.Upsample(scale_factor=self.factor, mode='nearest')
            return upsample(x)
        return self.op.A_adjoint(x) * self.factor ** 2

    def to_coarse_wavelet(self, x):
        """
        Applies a wavelet decomposition to x and returns a dictionary of wavelet components.
        """
        components = {}
        if not hasattr(self.filter_object, 'wavelet_type'):
            raise ValueError("This filter does not support wavelet decomposition.")

        wavelet_type = self.filter_object.wavelet_type()
        coeffs = pywt.wavedec2(
            x.cpu().numpy(), wavelet=wavelet_type, mode='periodization', level=1
        )

        components['LL'] = torch.from_numpy(coeffs[0]).to(x.device)
        components['LH'] = torch.from_numpy(coeffs[1][0]).to(x.device)
        components['HL'] = torch.from_numpy(coeffs[1][1]).to(x.device)
        components['HH'] = torch.from_numpy(coeffs[1][2]).to(x.device)

        return components

    def to_fine_wavelet(self, components):
        """
        Applies the inverse wavelet transform and returns the reconstructed tensor.
        """
        LL = components['LL'].cpu().numpy()
        LH = components['LH'].cpu().numpy()
        HL = components['HL'].cpu().numpy()
        HH = components['HH'].cpu().numpy()

        coeffs = (LL, (LH, HL, HH))

        reconstructed = pywt.idwt2(coeffs, self.wavelet_type, mode='periodization')
        reconstructed = torch.from_numpy(reconstructed)

        return reconstructed

    def to(self, device):
        # Just in case someone tries to move it like a model
        if self.op is not None:
            self.op.to(device)
        return self

    # Tentative de réimplémentation des ondelettes avec des convolutions. Mais on n'a pas W^T(Wx) = x... On utilise donc la méthode de pywt pour l'instant.
    '''def to_coarse_wavelet_conv(self, x, target_shape):
        if not hasattr(self.filter_object, 'get_2d_filter_H'):
            raise ValueError("This filter does not support wavelet decomposition.")
        device = x.device
        dtype = x.dtype

        filters = {
            'LL': self.filter_object.get_2d_filter().type(dtype),
            'LH': self.filter_object.get_2d_filter_H().type(dtype),
            'HL': self.filter_object.get_2d_filter_V().type(dtype),
            'HH': self.filter_object.get_2d_filter_D().type(dtype)
        }

        components = {}
        for name, filt in filters.items():
            filt = filt.to(dtype).unsqueeze(0).unsqueeze(0).to(device)
            op = deepinv.physics.Downsampling(
                target_shape, filter=filt, factor=self.factor, device=device, padding=self.padding
            )
            result = op.A(x.unsqueeze(0)) if x.dim() == 3 else op.A(x)
            components[name] = result.squeeze(0) if x.dim() == 3 else result

        return components'''


# ==========================
#       filter list
# ==========================


class Kaiser:
    def __str__(self):
        return 'kaiser'

    def get_2d_filter(self):
        # N = 10
        # beta = 10.0
        k0 = torch.tensor([
            0.0004, 0.0310, 0.2039, 0.5818, 0.9430,
            0.9430, 0.5818, 0.2039, 0.0310, 0.0004
        ])
        return torch.outer(k0, k0)

class SincFilter:
    def __str__(self):
        return 'sinc'

    def get_2d_filter(self):
        sinc_dinv = deepinv.physics.blur.sinc_filter(factor=2, length=11, windowed=True)
        return sinc_dinv[0, 0, :, :]

class CFir:  # custom FIR filter
    def __str__(self):
        return 'cfir'

    def get_2d_filter(self):
        # order + 1 coefficients
        k0 = torch.tensor([
            -0.015938026, 0.000019591, 0.013033937, -0.000004666, -0.018657837, 0.000020187, 0.026570831, 0.000002218,
            -0.038348155, 0.000018390, 0.058441238, 0.000007421, -0.102893218, 0.000011707, 0.317258819, 0.500004593,
            0.317258819, 0.000011707, -0.102893218, 0.000007421, 0.058441238, 0.000018390, -0.038348155, 0.000002218,
            0.026570831, 0.000020187, -0.018657837, -0.000004666, 0.013033937, 0.000019591, -0.015938026
        ])
        return torch.outer(k0, k0)


class BlackmannHarris:
    def __str__(self):
        return 'blackmannharris'

    def get_2d_filter(self):
        # 8 coefficients
        k0 = torch.tensor([
            3.9818e-05, 1.3299e-02, 1.3252e-01, 3.5415e-01, 3.5415e-01, 1.3252e-01, 1.3299e-02, 3.9818e-05]
        )
        return torch.outer(k0, k0)

class Dirac:
    def __str__(self):
        return 'dirac'

    def get_2d_filter(self):
        k0 = torch.tensor([1.0])
        return k0.unsqueeze(0)

class Daubechies8:
    def __str__(self):
        return 'daubechies8'

    def get_2d_filter(self):
        k0 = torch.tensor([0.2304,0.7148,0.6309,-0.0280,-0.1870,0.0308,0.0329,-0.0106])
        return torch.outer(k0, k0)

    def wavelet_type(self):
        return 'db8'

class Gaussian:
    def __str__(self):
        return 'gaussian'

    def get_2d_filter(self):
        k0 = torch.tensor([0.0001, 0.0334, 0.3328, 0.8894,
                           0.8894, 0.3328, 0.0334, 0.0001])
        return torch.outer(k0, k0)

class Haar:
    def __str__(self):
        return 'haar'

    def get_2d_filter(self):
        k0 = torch.tensor([1.0, 1.0])
        return torch.outer(k0, k0)

    def wavelet_type(self):
        return('haar')

    '''def get_2d_filter_H(self):
        # Haar filter for horizontal direction
        k0 = torch.tensor([1.0, 1.0])
        k1 = torch.tensor([-1.0, 1.0])
        return torch.outer(k0, k1)

    def get_2d_filter_V(self):
        # Haar filter for vertical direction
        k0 = torch.tensor([1.0, 1.0])
        k1 = torch.tensor([-1.0, 1.0])
        return torch.outer(k1, k0)

    def get_2d_filter_D(self):
        # Haar filter for diagonal direction
        k1 = torch.tensor([-1.0, 1.0])
        return torch.outer(k1, k1)'''

class Symlet8:
    def __str__(self):
        return 'symlet8'

    def get_2d_filter(self):
        None

    def wavelet_type(self):
        return 'sym8'

filter_classes = {
    "dirac": Dirac,
    "blackmannharris": BlackmannHarris,
    "cfir": CFir,
    "kaiser": Kaiser,
    "sinc": SincFilter,
    "gaussian": Gaussian,
    "daubechies8": Daubechies8,
    "haar": Haar,
    "symlet8": Symlet8
}

def create_filter(name):
    if name in filter_classes:
        return filter_classes[name]()
    else:
        raise ValueError(f"Unknown filter type: {name}")

if __name__ == "__main__":
    import deepinv as dinv
    device = 'cpu'
    x = deepinv.utils.load_url_image(
        url=deepinv.utils.get_image_url("butterfly.png"), img_size=256).to(device)

    filter = create_filter("haar")
    downsampler = DownsamplingTransfer(filter).to(device)
    components = downsampler.to_coarse_wavelet(x)

    # Visualize the wavelet components
    '''deepinv.utils.plot(
        [components['LL'], components['LH'], components['HL'], components['HH']],
        titles=['Approximation (LL)', 'Horizontal (LH)', 'Vertical (HL)', 'Diagonal (HH)'],
        cmap='gray'
    )'''

    # Reconstruct the image from wavelet components
    reconstructed = downsampler.to_fine_wavelet(components)
    #deepinv.utils.plot([x, reconstructed], titles=['Original Image', 'Reconstructed Image'], cmap='gray')

    coarse = downsampler.to_coarse(x, x.shape[-3:])

    deepinv.utils.plot([x, coarse], titles=['Original Image', 'Coarse Image'], cmap='gray')
    print(f"Original image shape: {x.shape}")

    # Test with DeepInv's Downsampling class directly
    downsampler = dinv.physics.Downsampling(img_size=(128, 128), factor=2, device=device)
    x_downsampled = downsampler.A(x)
    print(f"Original image shape: {x.shape}")

    # Test simple manual downsampling first (this should always work)
    print(f"\n--- Manual downsampling test ---")
    x_manual_coarse = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    print(f"Manual downsampled shape: {x_manual_coarse.shape}")

    x_manual_up = torch.nn.functional.interpolate(x_manual_coarse, size=(256, 256), mode='bilinear', align_corners=False)
    print(f"Manual upsampled shape: {x_manual_up.shape}")

    # Test what happens with multilevel downsampling
    print(f"\n--- Multilevel manual downsampling ---")
    current = x.clone()
    for level in range(1, 4):
        current = torch.nn.functional.avg_pool2d(current, kernel_size=2, stride=2)
        print(f"Level {level}: {current.shape}")

    # Now test your DownsamplingTransfer class
    print(f"\n--- Testing your DownsamplingTransfer class ---")
    try:
        from multilevel.info_transfer import DownsamplingTransfer, create_filter

        filter_type = 'haar'
        filter_obj = create_filter(filter_type)
        print(f"Filter created: {type(filter_obj)}")

        info_transfer = DownsamplingTransfer(filter_obj).to(device)
        print(f"DownsamplingTransfer created")

        # Initialize the operator
        info_transfer._initialize_operator(x, x.shape[-3:])
        print(f"Operator initialized")

        # Test just the to_coarse operation
        print("Testing to_coarse operation...")
        x_coarse = info_transfer.to_coarse(x, x.shape[-3:])
        print(f"Coarse image shape: {x_coarse.shape}")

        if x_coarse.shape[-2:] == x.shape[-2:]:
            print("ERROR: Coarse image has same spatial dimensions as original!")
            print("This means downsampling is not working.")
        else:
            print("SUCCESS: Downsampling worked correctly")

            # Only test upsampling if downsampling worked
            print("Testing to_fine operation...")
            try:
                x_fine_reconstructed = info_transfer.to_fine(x_coarse, x.shape[-3:])
                print(f"Fine reconstructed shape: {x_fine_reconstructed.shape}")
            except Exception as e:
                print(f"to_fine failed: {e}")

    except ImportError as e:
        print(f"Could not import DownsamplingTransfer: {e}")
    except Exception as e:
        print(f"Error with DownsamplingTransfer: {e}")
        import traceback
        traceback.print_exc()

    # Test creating masks at different resolutions (this is what's failing in your main code)
    print(f"\n--- Testing mask creation for inpainting ---")
    mask_256 = torch.ones(1, 1, 256, 256, device=device)
    mask_256[:, :, 64:192, 64:192] = 0  # Create a hole
    print(f"Original mask shape: {mask_256.shape}")

    # Downsample mask manually
    mask_128 = torch.nn.functional.avg_pool2d(mask_256, kernel_size=2, stride=2)
    mask_64 = torch.nn.functional.avg_pool2d(mask_128, kernel_size=2, stride=2)
    print(f"Downsampled mask shapes: {mask_128.shape}, {mask_64.shape}")

    # Test creating inpainting physics at different scales
    print(f"Testing inpainting physics creation...")
    try:
        physics_256 = dinv.physics.Inpainting(img_size=(256, 256), mask=mask_256, device=device)
        print(f"256x256 physics created successfully")

        physics_128 = dinv.physics.Inpainting(img_size=(128, 128), mask=mask_128, device=device)
        print(f"128x128 physics created successfully")

        physics_64 = dinv.physics.Inpainting(img_size=(64, 64), mask=mask_64, device=device)
        print(f"64x64 physics created successfully")

        # Test applying physics
        y_256 = physics_256(x)
        print(f"Original observation shape: {y_256.shape}")

        x_128 = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        y_128 = physics_128(x_128)
        print(f"128 observation shape: {y_128.shape}")

    except Exception as e:
        print(f"Error creating inpainting physics: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n--- Summary ---")
    print("If manual downsampling works but DownsamplingTransfer doesn't,")
    print("then the issue is in your custom implementation.")
    print("You could use simple avg_pool2d as a workaround.")

    # Now let's see what happens with a mask (like in inpainting)
    # Create a mask at the original resolution
    mask_original = torch.ones(1, 1, 128, 128, device=device)  # Full mask
    mask_original[:, :, 32:96, 32:96] = 0  # Create a hole in the middle
    print(f"Original mask shape: {mask_original.shape}")

    # Downsample the mask
    mask_coarse = info_transfer.to_coarse(mask_original, mask_original.shape[-3:])
    print(f"Coarse mask shape: {mask_coarse.shape}")

    # This shows the expected behavior - each level should be half the resolution
    print("\nExpected multilevel sizes:")
    current_size = 128
    for level in range(1, 4):  # 3 levels
        current_size //= 2
        print(f"Level {level}: {current_size}x{current_size}")

    # Test what happens when we create inpainting physics at different scales
    print("\nTesting inpainting physics creation:")
    try:
        # Original scale
        physics_orig = dinv.physics.Inpainting(
            img_size=(128, 128),
            mask=mask_original,
            device=device
        )
        print(f"Original physics created successfully with mask shape: {physics_orig.mask.shape}")

        # Coarse scale
        physics_coarse = dinv.physics.Inpainting(
            img_size=mask_coarse.shape[-2:],  # Use the actual coarse mask dimensions
            mask=mask_coarse,
            device=device
        )
        print(f"Coarse physics created successfully with mask shape: {physics_coarse.mask.shape}")

    except Exception as e:
        print(f"Error creating physics: {e}")

    # Test the actual problematic scenario from your code
    print("\nTesting the problematic scenario:")
    print("This simulates what happens in your multilevel setup...")

    # Simulate your original code's approach
    try:
        # This is what your code was doing (problematic)
        data = mask_original  # physics.mask.data
        coarse_data = info_transfer.to_coarse(data, data.shape)  # Wrong: should be data.shape[-3:]
        print(f"Problematic approach - coarse_data shape: {coarse_data.shape}")

    except Exception as e:
        print(f"Error with problematic approach: {e}")

    try:
        # This is the correct approach
        data = mask_original
        coarse_data = info_transfer.to_coarse(data, data.shape[-3:])  # Correct: use last 3 dimensions
        print(f"Correct approach - coarse_data shape: {coarse_data.shape}")

        # Create physics with correct parameters
        physics_fixed = dinv.physics.Inpainting(
            img_size=coarse_data.shape[-2:],  # Use height, width only
            mask=coarse_data,
            device=device
        )
        print(f"Fixed physics created successfully")

    except Exception as e:
        print(f"Error with fixed approach: {e}")