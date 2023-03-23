import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

# from src.utils.common import tensor2img
from src.core.solvers.helper import get_lr_scheduler, get_optimizer
from src.core.losses import PerceptualLoss
from src.networks.mraa.model import Transform, ImagePyramide, Vgg19

import wandb

class MRAALitModel(LightningModule):

    def __init__(self, region_predictor, bg_predictor, generator, train_params, losses=None, optimizer=None, lr_scheduler=None):
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


    def forward(self, x):
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        bg_params = self.bg_predictor(x['source'], x['driving'])
        generated = self.generator(x['source'], source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)
        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})

        return generated

    def shared_step(self, x):
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

    def training_step(self, batch, batch_idx):
        generated, loss_values = self.shared_step(batch)

        # logging
        self.log_dict({'(train)' + k: v for k, v in loss_values.items()},
                      on_step=True,
                      on_epoch=False,
                      prog_bar=False)
        
        return loss_values

    def validation_step(self, batch, batch_idx):
        x = batch
        loss_dict = {}

        drivings = []
        predictions = []
        source_region_params = self.region_predictor(x['video'][:, 0])
        for frame_idx in range(x['video'].shape[1]):
            source = x['video'][:, 0]
            driving = x['video'][:, frame_idx]
            driving_region_params = self.region_predictor(driving)
            drivings.append(driving)

            bg_params = self.bg_predictor(source, driving)
            out = self.generator(source, source_region_params=source_region_params,
                            driving_region_params=driving_region_params, bg_params=bg_params)

            out['source_region_params'] = source_region_params
            out['driving_region_params'] = driving_region_params
            predictions.append(out['prediction'])

        t_driv = torch.cat(drivings, 0).unsqueeze(0)
        t_pred = torch.cat(predictions, 0).unsqueeze(0)
        vis = torch.cat((t_driv, t_pred), dim=3)


        each_val_epoch_steps = len(self.trainer.datamodule.data_val) \
                               // self.trainer.datamodule.hparams.val_batch_size + 1
        val_global_step = self.current_epoch * each_val_epoch_steps + batch_idx

        if isinstance(self.trainer.logger, TensorBoardLogger):
            # self.trainer.logger.experiment.add_images(
            #     "driving",
            #     x['driving'], 
            #     self.global_step
            # )
            self.trainer.logger.experiment.add_video(
                'drivings / predictions', vis, val_global_step
            )
        elif isinstance(self.trainer.logger, WandbLogger):
            # self.trainer.logger.log_image(
            #     key="Images", 
            #     images=[x['driving'], generated['prediction'], transformed_frame], 
            #     caption=["driving", "prediction", "transformed_frame"]
            # )
            self.trainer.logger.experiment.log(
                {"video": wandb.Video((vis[0].cpu().numpy() * 255).astype(np.uint8), fps=4, format="gif")}
            )

        # transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
        # transformed_frame = transform.transform_frame(x['driving'])
        # transformed_region_params = self.region_predictor(transformed_frame)

        # generated['transformed_frame'] = transformed_frame
        # generated['transformed_region_params'] = transformed_region_params

        # loss_dict['PSNR'] = psnr(x['driving'], generated['prediction'])
        # loss_dict['SSIM'] = ssim(x['driving'], generated['prediction'])

        return loss_dict

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
        x = batch

        source = x['source']
        source_region_params = self.region_predictor(source)

        for i, video in enumerate(x['videos']):

            predictions = []    
            for frame_idx in range(video.shape[1]):
                driving = video[:, frame_idx]
                driving_region_params = self.region_predictor(driving)

                bg_params = self.bg_predictor(source, driving)
                out = self.generator(source, source_region_params=source_region_params,
                                driving_region_params=driving_region_params, bg_params=bg_params)

                out['source_region_params'] = source_region_params
                out['driving_region_params'] = driving_region_params
                predictions.append(out['prediction'])

            t_pred = torch.cat(predictions, 0).unsqueeze(0)
            T = len(predictions)
            sources = torch.tile(source.unsqueeze(1), (1, T, 1, 1, 1))
            vis = torch.cat((sources, video, t_pred), dim=4)

            # save video to local storage
            

            if isinstance(self.trainer.logger, TensorBoardLogger):
                # self.trainer.logger.experiment.add_images(
                #     "driving",
                #     x['driving'], 
                #     self.global_step
                # )
                self.trainer.logger.experiment.add_video(
                    'drivings / predictions', vis, i # batch_idx
                )
            elif isinstance(self.trainer.logger, WandbLogger):
                # self.trainer.logger.log_image(
                #     key="Images", 
                #     images=[x['driving'], generated['prediction'], transformed_frame], 
                #     caption=["driving", "prediction", "transformed_frame"]
                # )
                self.trainer.logger.experiment.log(
                    {"video": wandb.Video((vis[0].cpu().numpy() * 255).astype(np.uint8), fps=4, format="gif")}
                )

        return None

    def configure_optimizers(self):
        optimizer = get_optimizer([
            dict(name='region_predictor', params=self.region_predictor.parameters()),
            dict(name='bg_predictor', params=self.bg_predictor.parameters()),
            dict(name='generator', params=self.generator.parameters())
        ], self.hparams.optimizer)

        lr_scheduler = get_lr_scheduler(optimizer, self.hparams.lr_scheduler)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
