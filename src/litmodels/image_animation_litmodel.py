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

from src.networks.image_animation.spynet import SPyNet, flow_warp

import wandb

class ImageAnimationLitModel(LightningModule):

    def __init__(self, losses=None, optimizer=None, lr_scheduler=None):
        super().__init__()

        self.flow_predictor = SPyNet(pretrained=None)

        self.loss_weights = dict(
            perceptual=[10, 10, 10, 10, 10]
        )
        self.scales = [1, 0.5, 0.25, 0.125]

        self.perceptual_loss = PerceptualLoss()
        self.pyramid = ImagePyramide(self.scales, 3)
        self.vgg = Vgg19()

        self.save_hyperparameters(ignore=[])


    def forward(self, source, driving):
        optical_flow = self.flow_predictor(source, driving)
        optical_flow = optical_flow.permute(0, 2, 3, 1)
        warped = flow_warp(source, optical_flow)
        return warped

    def shared_step(self, batch):
        source, driving = batch
        generated = self.forward(source, driving)

        loss_values = {}

        # perceptual loss
        if sum(self.loss_weights['perceptual']) != 0:
            pyramide_real = self.pyramid(driving)
            pyramide_generated = self.pyramid(generated)

            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

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
        loss_dict = {}

        source, driving = batch
        generated = self.forward(source, driving)

        vis = torch.cat((driving, generated), dim=3)

        each_val_epoch_steps = len(self.trainer.datamodule.data_val) \
                               // self.trainer.datamodule.hparams.val_batch_size + 1
        val_global_step = self.current_epoch * each_val_epoch_steps + batch_idx

        if isinstance(self.trainer.logger, TensorBoardLogger):
            self.trainer.logger.experiment.add_images(
                "driving / warpped",
                vis, 
                val_global_step
            )
            # self.trainer.logger.experiment.add_video(
            #     'drivings / predictions', vis, val_global_step
            # )
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
            dict(name='flow_predictor', params=self.flow_predictor.parameters()),
        ], self.hparams.optimizer)

        lr_scheduler = get_lr_scheduler(optimizer, self.hparams.lr_scheduler)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
