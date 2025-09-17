import deepinv as dinv
from multilevel.multilevel import MultilevelPhysics

"""
Under construction......
"""

class MultilevelInpainting(MultilevelPhysics):
    def __init__(self, physics, img_shape, scale, information_transfer, filter, device='cpu', **kwargs):
        super().__init__(physics, img_shape, scale, filter, device=device, **kwargs)
        self.information_transfer = information_transfer

        if not physics.isInstance(dinv.physics.Inpainting):
            raise ValueError("Physics model is not an instance of Inpainting")

        coarse_physics = {f'level{self.levels}': self.physics}
        data = self.physics.mask.data
        for i in range(self.levels-1, 0, -1):
            coarse_data = self.information_transfer.to_coarse_wavelet(data, data.shape)
            coarse_physics[f'level{i}'] = dinv.physics.Inpainting(
                tensor_size=coarse_data.shape[1:], mask=coarse_data, device=self.device
            )

        return coarse_physics

if __name__ == "__main__":
    import torch
    from multilevel.info_transfer import DownsamplingTransfer, create_filter

    device = torch.device('cpu')
    x_true = dinv.utils.load_example("butterfly.png", device=device)

    # Wavelet parameters
    J = 3
    wv_type = 'haar'

    # Physics
    filter_0 = dinv.physics.blur.gaussian_blur(sigma=(2, 2), angle=0.0)
    physics = dinv.physics.Blur(filter_0, device=device, padding="reflect")
    #physics = dinv.physics.Inpainting(mask=0.7, img_size=x_true.shape[1:])
    seed = torch.manual_seed(0)  # Random seed for reproducibility

    sigma = 0.01
    physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

    # Observation
    y = physics(x_true)

    filter = create_filter("haar")
    information_transfer = DownsamplingTransfer(filter)
    coarse_physics = MultilevelInpainting(physics, x_true.shape, J, information_transfer, filter)