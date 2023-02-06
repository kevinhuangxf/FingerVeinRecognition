import torch
import os
import sys
sys.path.append("/home/kevinhuang/github/FingerVeinRecognition/src/networks/mraa")

from region_predictor import RegionPredictor
from bg_motion_predictor import BGMotionPredictor
from generator import Generator
from model import ReconstructionModel

def main():
    # region_predictor = RegionPredictor(32, 10, 3, 1024, 5, 0.1, pca_based=True)
    region_predictor = RegionPredictor(
        block_expansion = 32, 
        num_regions = 10, 
        num_channels = 3, 
        max_features = 1024,
        num_blocks = 5, 
        temperature = 0.1, 
        estimate_affine=False, 
        scale_factor=1,
        pca_based=False, 
        fast_svd=False, 
        pad=3
    )
    x = torch.randn(1,3,256,256)
    region_out = region_predictor(x)

    bg_predictor = BGMotionPredictor(
        block_expansion=32, 
        num_channels=3, 
        max_features=1024, 
        num_blocks=5, 
        bg_type='zero'
    )
    bg_out = bg_predictor(x, x)

    generator = Generator(
        num_channels=3, 
        num_regions=10, 
        block_expansion=64, 
        max_features=512, 
        num_down_blocks=2,
        num_bottleneck_blocks=6, 
        pixelwise_flow_predictor_params=None, 
        skips=True, 
        revert_axis_swap=True
    )
    gen_out= generator(x, region_out, region_out)

    train_params = dict(
        num_epochs=50, # 100
        num_repeats=25, # 50
        epoch_milestones=[60, 90],
        lr=2.0e-4,
        batch_size=32,
        dataloader_workers=6,
        checkpoint_freq=5, # 50
        scales=[1, 0.5, 0.25, 0.125],
        transform_params=dict(
            sigma_affine=0.05,
            sigma_tps=0.005,
            points_tps=5
        ),
        loss_weights=dict(
            perceptual=[10, 10, 10, 10, 10],
            equivariance_shift=10,
            equivariance_affine=10
        )
    )

    model = ReconstructionModel(region_predictor, bg_predictor, generator, train_params)

    pass

if __name__ == "__main__":
    main()