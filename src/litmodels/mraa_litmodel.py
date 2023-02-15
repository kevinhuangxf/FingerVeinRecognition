import torch
import lpips
from pytorch_lightning import LightningModule
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from src.core.solvers.helper import get_lr_scheduler, get_optimizer
from src.networks.mraa.model import Transform


class MRAALitModel(LightningModule):

    def __init__(self, region_predictor, bg_predictor, generator, train_params, losses=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.region_predictor = region_predictor
        self.bg_predictor = bg_predictor
        self.generator = generator
        self.train_params = train_params
        # self.losses = losses

        self.loss_weights = train_params['loss_weights']
        self.loss_lpips = lpips.LPIPS(net='vgg')
        self.save_hyperparameters(ignore=['region_predictor', 'bg_predictor', 'generator'])

    def forward(self, x):
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        bg_params = self.bg_predictor(x['source'], x['driving'])
        generated = self.generator(x['source'], source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)
        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})

        return generated

    def training_step(self, batch, batch_idx):
        x = batch

        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        bg_params = self.bg_predictor(x['source'], x['driving'])
        generated = self.generator(x['source'], source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)
        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})

        loss_values = {}

        # perceptual loss
        # loss_values['perceptual'] = self.loss_lpips(x['driving'], generated['prediction']).mean()
        # loss_values['perceptual'] = self.losses.perceptual_loss(x['driving'], generated['prediction'])[0]

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

        # logging
        self.log_dict(loss_values)
        # self.trainer.logger.experiment.log(loss_values)
        # if self.global_step % 100 == 0:
        #     self.trainer.logger.log_image(
        #         key="Images", 
        #         images=[x['driving'], generated['prediction']], 
        #         caption=["driving", "prediction"]
        #     )

        return loss_values

    def validation_step(self, batch, batch_idx):
        x = batch
        generated = self.forward(x)

        loss_dict = {}
        loss_dict['PSNR'] = psnr(x['driving'], generated['prediction'])
        loss_dict['SSIM'] = ssim(x['driving'], generated['prediction'])

        self.log_dict(loss_dict)

        return loss_dict

    def test_step(self, batch, batch_idx):
        generated = self.forward(batch)
        return generated

    def configure_optimizers(self):
        optimizer = get_optimizer([
            dict(name='region_predictor', params=self.region_predictor.parameters()),
            dict(name='bg_predictor', params=self.bg_predictor.parameters()),
            dict(name='generator', params=self.generator.parameters())
        ], self.hparams.optimizer)

        lr_scheduler = get_lr_scheduler(optimizer, self.hparams.lr_scheduler)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
