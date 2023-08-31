import numpy as np
from collections import OrderedDict

from sklearn.decomposition import PCA
from copy import deepcopy
import random
from src.networks.mraa.logger import Visualizer

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from src.utils.common import get_model_params
from src.core.solvers.helper import get_lr_scheduler, get_optimizer
from src.core.losses import PerceptualLoss
from src.networks.mraa.model import Transform, ImagePyramide, Vgg19

from src.networks.image_animation.spynet import SPyNet, flow_warp


class ImageAnimationLitModel(LightningModule):

    def __init__(self, animation_model, region_predictor=None, generator=None, train_params=None, losses=None, optimizer=None, lr_scheduler=None):
        super().__init__()

        self.animation_model = animation_model
        self.region_predictor = animation_model.region_predictor
        self.generator = animation_model.generator
        self.train_params = train_params
        self.perceptual_loss = PerceptualLoss()

        self.loss_weights = train_params['loss_weights']

        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, 1)
        self.vgg = Vgg19()

        print('animation_model params: ')
        get_model_params(self.animation_model.region_predictor.parameters())
        get_model_params(self.animation_model.generator.parameters())

        self.driving_dict = {}

        # # load pre-trained model
        # model_state_dict = torch.load('/media/user/Toshiba4T/Kevin/Development/FingerVeinRecognition/lightning_logs/experiment/version_680/checkpoints/epoch_249.ckpt')['state_dict']
        # if model_state_dict is not None:
        #     self.load_state_dict(model_state_dict, strict=True)

        self.save_hyperparameters(ignore=[])

    def standard_step(self, x):
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        generated = self.generator(x['source'], source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=None)
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

        # mse loss
        # loss_values['mse'] = F.mse_loss(generated['prediction'], x['driving'])

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
        generated, loss_values = self.standard_step(batch)

        # logging
        self.log_dict({'(train)' + k: v for k, v in loss_values.items()},
                      on_step=True,
                      on_epoch=False,
                      prog_bar=False)
        
        return loss_values

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
        print('callback')

    def configure_optimizers(self):

        optimizer = get_optimizer([
            # dict(name='region_predictor', params=self.region_predictor.parameters()),
            # dict(name='generator', params=self.generator.parameters())
            dict(name='animation_model', params=self.animation_model.parameters())
        ], self.hparams.optimizer)

        lr_scheduler = get_lr_scheduler(optimizer, self.hparams.lr_scheduler)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


    def get_motion_list(self, videos):
        for v_id, video in enumerate(videos):
            self.driving_dict[f'{v_id}'] = []
            for f_id in range(video.shape[1]):
                driving = video[:, f_id]
                driving_region_params = self.region_predictor(driving)
                self.driving_dict[f'{v_id}'].append(driving_region_params)

        shift_diff_list = []
        for k, v in self.driving_dict.items():
            for i in range(1, len(v)):
                shift_diff_list.append((v[i]['shift'] - v[i-1]['shift']))
        
        return shift_diff_list

    def get_principle_motion_vectors(self, shift_diff_list, n_components):
        shift_data = torch.concat(shift_diff_list, dim=0)
        shift_data = shift_data.view(shift_data.shape[0], -1)
        shift_data = shift_data.cpu().numpy()

        # PCA
        pca = PCA(n_components=n_components)
        shift_pca_data = pca.fit_transform(shift_data)

        origin_shape = shift_diff_list[0].shape
        motion_vectors = [torch.from_numpy(
            motion_vector.reshape(origin_shape[-2:])[np.newaxis, ...].astype(np.float32)).cuda()
            for motion_vector in pca.components_]

        k_list = pca.explained_variance_ratio_ / sum(pca.explained_variance_ratio_)

        return motion_vectors, k_list
    
    def get_random_motion_vector(self, k_list, motion_vectors):
        # Generate random weights and make sure they sum to 1
        weights = [random.random() for _ in range(10)]
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]

        random_vector = 0
        for w, k, vector in zip(weights, k_list, motion_vectors):
            random_vector += w * k * vector

        return random_vector

    def infer_new_shift_param(self, source, shift_diff):

        source_region_params = self.region_predictor(source)
        driving_region_params = deepcopy(source_region_params)

        visualizer = Visualizer(kp_size=source_region_params['shift'].shape[-2])
        visualize_list = []

        for i in range(-10, 11):

            driving_region_params['shift'] = source_region_params['shift'] - shift_diff * i * 0.1

            out = self.generator(source, source_region_params=source_region_params,
                            driving_region_params=driving_region_params, bg_params=None)
            out['source_region_params'] = source_region_params
            out['driving_region_params'] = driving_region_params
            images = visualizer.visualize(source, out)
            visualize_list.append(images)
        
        return visualize_list

    def infer_motion_vector(self, source, motion_vector):

        source_region_params = self.region_predictor(source)
        driving_region_params = deepcopy(source_region_params)
        driving_region_params['shift'] = source_region_params['shift'] + motion_vector

        out = self.generator(source, source_region_params=source_region_params,
                        driving_region_params=driving_region_params, bg_params=None)
        out['source_region_params'] = source_region_params
        out['driving_region_params'] = driving_region_params

        return out
