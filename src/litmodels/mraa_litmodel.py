import numpy as np
from collections import OrderedDict
from scipy.spatial import ConvexHull

import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from src.core.solvers.helper import get_lr_scheduler, get_optimizer
from src.core.losses import PerceptualLoss
from src.networks.mraa.model import Transform, ImagePyramide, Vgg19
from src.networks.mraa.avd_network import AVDNetwork


class MRAALitModel(LightningModule):

    def __init__(self, region_predictor, bg_predictor, generator, train_params, avd_params=None, pretrained_weights=None, losses=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.region_predictor = region_predictor
        self.bg_predictor = bg_predictor
        self.generator = generator
        self.train_params = train_params
        self.perceptual_loss = PerceptualLoss()

        self.loss_weights = train_params['loss_weights']

        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, 3)
        self.vgg = Vgg19()

        self.save_hyperparameters(ignore=['region_predictor', 'bg_predictor', 'generator'])

        if self.hparams.avd_params is not None:
            self.avd_network = AVDNetwork(num_regions=self.hparams.avd_params.num_regions)

        if pretrained_weights:
            state_dict = torch.load(pretrained_weights)['state_dict']
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        bg_params = self.bg_predictor(x['source'], x['driving'])
        generated = self.generator(x['source'], source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)
        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})

        return generated

    def standard_step(self, x):
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        bg_params = self.bg_predictor(x['source'], x['driving'])
        generated = self.generator(x['source'], source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)
        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})

        loss_values = {}

        # perceptual loss
        if sum(self.loss_weights['perceptual']) != 0:
            pyramide_real = self.pyramid(x['driving'])
            pyramide_generated = self.pyramid(generated['prediction'])

            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        # equivariance losses
        if (self.loss_weights['equivariance_shift'] + self.loss_weights['equivariance_affine']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_region_params = self.region_predictor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_region_params'] = transformed_region_params

            if self.loss_weights['equivariance_shift'] != 0:
                value = torch.abs(driving_region_params['shift'] -
                                  transform.warp_coordinates(transformed_region_params['shift'])).mean()
                loss_values['equivariance_shift'] = self.loss_weights['equivariance_shift'] * value

            if self.loss_weights['equivariance_affine'] != 0:
                affine_transformed = torch.matmul(transform.jacobian(transformed_region_params['shift']),
                                                  transformed_region_params['affine'])

                normed_driving = torch.inverse(driving_region_params['affine'])
                normed_transformed = affine_transformed
                value = torch.matmul(normed_driving, normed_transformed)
                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                if self.generator.pixelwise_flow_predictor.revert_axis_swap:
                    value = value * torch.sign(value[:, :, 0:1, 0:1])

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_affine'] = self.loss_weights['equivariance_affine'] * value

        loss_values['loss'] = sum([v for k, v in loss_values.items()])

        return generated, loss_values
    
    def avd_step(self, x):

        def random_scale(region_params, scale):
            theta = torch.rand(region_params['shift'].shape[0], 2) * (2 * scale) + (1 - scale)
            theta = torch.diag_embed(theta).unsqueeze(1).type(region_params['shift'].type())
            new_region_params = {'shift': torch.matmul(theta, region_params['shift'].unsqueeze(-1)).squeeze(-1),
                                'affine': torch.matmul(theta, region_params['affine'])}
            return new_region_params

        lambda_shift = self.hparams.avd_params.lambda_shift
        lambda_affine = self.hparams.avd_params.lambda_affine
        random_scale_factor = self.hparams.avd_params.random_scale_factor

        with torch.no_grad():
            regions_params_id = self.region_predictor(x['source'])
            regions_params_pose_gt = self.region_predictor(x['driving'])
            regions_params_pose = random_scale(regions_params_pose_gt, scale=random_scale_factor)

        rec = self.avd_network(regions_params_id, regions_params_pose)

        reconstruction_shift = lambda_shift * \
                                torch.abs(regions_params_pose_gt['shift'] - rec['shift']).mean()
        reconstruction_affine = lambda_affine * \
                                torch.abs(regions_params_pose_gt['affine'] - rec['affine']).mean()

        generated = self.generator(x['source'], source_region_params=regions_params_id,
                                driving_region_params=rec)

        rec = x['driving'] - generated['prediction']

        loss = reconstruction_shift + reconstruction_affine
        loss_dict = {
            'rec': rec,
            'rec_shift': reconstruction_shift, 
            'rec_affine': reconstruction_affine,
            'loss': loss
        }

        return loss_dict

    def training_step(self, batch, batch_idx):
        if self.hparams.avd_params is not None:
            loss_values = self.avd_step(batch)
        else:
            generated, loss_values = self.standard_step(batch)

        # logging
        self.log_dict({'(train)' + k: v for k, v in loss_values.items()},
                      on_step=True,
                      on_epoch=False,
                      prog_bar=False)
        
        return loss_values

    def validation_step(self, batch, batch_idx):
        # return self.animate_same_video(batch, batch_idx)
        print("Animation Callback!")

    def evaluation_step(self, outputs):
        results = OrderedDict()

        if 'loss' in outputs[0]:
            results['loss'] = torch.mean(torch.stack(
                [output['loss'] for output in outputs]),
                                         dim=0)
        if 'PSNR' in outputs[0]:
            results['PSNR'] = torch.mean(torch.stack(
                [output['PSNR'] for output in outputs]),
                                         dim=0)
        if 'SSIM' in outputs[0]:
            results['SSIM'] = torch.mean(torch.stack(
                [output['SSIM'] for output in outputs]),
                                         dim=0)

        return results
    
    def validation_epoch_end(self, outputs):
        results = self.evaluation_step(outputs)
        self.log_dict({'(val)' + k: v for k, v in results.items()})

    def test_step(self, batch, batch_idx):
        print("Animation Callback!")

    def configure_optimizers(self):
        if self.hparams.avd_params:
            optimizer = get_optimizer([
                dict(name='avd_network', params=self.avd_network.parameters()),
            ], self.hparams.optimizer)
        else:
            optimizer = get_optimizer([
                dict(name='region_predictor', params=self.region_predictor.parameters()),
                dict(name='bg_predictor', params=self.bg_predictor.parameters()),
                dict(name='generator', params=self.generator.parameters())
            ], self.hparams.optimizer)

        lr_scheduler = get_lr_scheduler(optimizer, self.hparams.lr_scheduler)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def update_by_relative_motion(self, source_region_params, driving_region_params_initial, driving_region_params):
        new_region_params = {k: v for k, v in driving_region_params.items()}
        source_area = ConvexHull(source_region_params['shift'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(driving_region_params_initial['shift'][0].data.cpu().numpy()).volume
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

        shift_diff = (driving_region_params['shift'] - driving_region_params_initial['shift'])
        shift_diff *= movement_scale
        new_region_params['shift'] = shift_diff + source_region_params['shift']

        affine_diff = torch.matmul(driving_region_params['affine'],
                                   torch.inverse(driving_region_params_initial['affine']))
        new_region_params['affine'] = torch.matmul(affine_diff, source_region_params['affine'])
        return new_region_params
