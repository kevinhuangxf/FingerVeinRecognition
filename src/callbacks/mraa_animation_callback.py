import cv2
import random
import wandb
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.utils.common import tensor2img


class MRAAAnimationCallback(Callback):
    def __init__(self, method="standard", save_path=None):
        super().__init__()
        self.method = method
        self.save_path = save_path

        if self.save_path is not None:
            self.save_path = Path(save_path)
            self.save_path.mkdir(exist_ok=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print("Animate Video:")
        self.animate_same_video(trainer)
    
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print("Animate Video:")
        self.animate_random_video(trainer, pl_module, batch, batch_idx)

    def animate_same_video(self, trainer, pl_module, batch, batch_idx):
        x = batch
        loss_dict = {}

        drivings = []
        predictions = []
        source_region_params = pl_module.region_predictor(x['video'][:, 0])
        for frame_idx in range(x['video'].shape[1]):
            source = x['video'][:, 0]
            driving = x['video'][:, frame_idx]
            driving_region_params = pl_module.region_predictor(driving)
            drivings.append(driving)

            bg_params = pl_module.bg_predictor(source, driving)
            out = pl_module.generator(source, source_region_params=source_region_params,
                            driving_region_params=driving_region_params, bg_params=bg_params)

            out['source_region_params'] = source_region_params
            out['driving_region_params'] = driving_region_params
            predictions.append(out['prediction'])

        t_driv = torch.cat(drivings, 0).unsqueeze(0)
        t_pred = torch.cat(predictions, 0).unsqueeze(0)
        vis = torch.cat((t_driv, t_pred), dim=3)

        each_val_epoch_steps = len(trainer.datamodule.data_val) \
                               // trainer.datamodule.hparams.val_batch_size + 1
        val_global_step = pl_module.current_epoch * each_val_epoch_steps + batch_idx

        if isinstance(trainer.logger, TensorBoardLogger):
            # self.trainer.logger.experiment.add_images(
            #     "driving",
            #     x['driving'], 
            #     self.global_step
            # )
            trainer.logger.experiment.add_video(
                'drivings / predictions', vis, val_global_step
            )
        elif isinstance(trainer.logger, WandbLogger):
            # self.trainer.logger.log_image(
            #     key="Images", 
            #     images=[x['driving'], generated['prediction'], transformed_frame], 
            #     caption=["driving", "prediction", "transformed_frame"]
            # )
            trainer.logger.experiment.log(
                {"video": wandb.Video((vis[0].cpu().numpy() * 255).astype(np.uint8), fps=4, format="gif")}
            )

        return loss_dict

    def animate_random_video(self, trainer, pl_module, batch, batch_idx):
        x = batch
        num_video = len(x['video'])

        # if has source sequence, the x['video'] will be the driving samples
        if len(x['source']) != 0:
            for i, source in enumerate(tqdm(x['source'])):
                video_idx = random.choice(range(num_video))
                video = x['video'][video_idx]

                sources = []
                predictions = []
                driving_region_params_initial = None
                for frame_idx in range(video.shape[1]):
                    # source = video[:, frame_idx]
                    source_region_params = pl_module.region_predictor(source)
                    driving = video[:, frame_idx]
                    driving_region_params = pl_module.region_predictor(driving)
                    if driving_region_params_initial is None:
                        driving_region_params_initial = driving_region_params

                    if self.method == 'relative':
                        driving_region_params = pl_module.update_by_relative_motion(source_region_params, driving_region_params_initial, driving_region_params)

                    if pl_module.hparams.avd_params:
                        driving_region_params = pl_module.avd_network(source_region_params, driving_region_params)

                    bg_params = pl_module.bg_predictor(source, driving)
                    out = pl_module.generator(source, source_region_params=source_region_params,
                                    driving_region_params=driving_region_params, bg_params=bg_params)

                    out['source_region_params'] = source_region_params
                    out['driving_region_params'] = driving_region_params
                    sources.append(source)
                    predictions.append(out['prediction'])

                t_pred = torch.cat(predictions, 0).unsqueeze(0)
                T = len(predictions)
                # sources = torch.tile(source.unsqueeze(1), (1, T, 1, 1, 1))
                sources = torch.cat(sources, 0).unsqueeze(0)
                vis = torch.cat((sources, video, t_pred), dim=4)

                # save video to local storage
                if self.save_path:
                    for j, pred in enumerate(predictions):
                        pred_img = tensor2img(pred)
                        # pred_img = cv2.resize(pred_img, (144, 64))
                        
                        pred_img = cv2.resize(pred_img, (64, 144))
                        pred_img = cv2.rotate(pred_img, cv2.ROTATE_90_CLOCKWISE)

                        # current_idx = i * 12 + j
                        # cv2.imwrite(f"results/mraa_fvusm_processed/{(current_idx):0{8}d}.png", pred_img)
                        # cv2.imwrite(f"{self.save_path}/{(i//4+1):0{3}d}_{i%12+1}_{(j+1):0{2}d}.png", pred_img)
                        cv2.imwrite((self.save_path / f"{(i//4+1):0{3}d}_{i%12+1}_{(j+1):0{2}d}.png").as_posix(), pred_img)

                if isinstance(trainer.logger, TensorBoardLogger):
                    # self.trainer.logger.experiment.add_images(
                    #     "driving",
                    #     x['driving'], 
                    #     self.global_step
                    # )
                    trainer.logger.experiment.add_video(
                        'drivings / predictions', vis, i # batch_idx
                    )
                elif isinstance(trainer.logger, WandbLogger):
                    # self.trainer.logger.log_image(
                    #     key="Images", 
                    #     images=[x['driving'], generated['prediction'], transformed_frame], 
                    #     caption=["driving", "prediction", "transformed_frame"]
                    # )
                    trainer.logger.experiment.log(
                        {"video": wandb.Video((vis[0].cpu().numpy() * 255).astype(np.uint8), fps=4, format="gif")}
                    )
            
            return None
        # if no source sequence, random select samples from x['video'] as driving samples
        else:
            num_video = len(x['video'])
            for i, video in enumerate(tqdm(x['video'])):
                driving_idx = random.choice(range(num_video))
                while True:
                    if i == driving_idx:
                        driving_idx =  random.choice(range(num_video))
                    else:
                        break
                sources = []
                predictions = []
                driving_region_params_initial = None
                for frame_idx in range(video.shape[1]):
                    source = video[:, frame_idx]
                    source_region_params = pl_module.region_predictor(source)
                    driving = x['video'][driving_idx][:, frame_idx]
                    driving_region_params = pl_module.region_predictor(driving)
                    if driving_region_params_initial is None:
                        driving_region_params_initial = driving_region_params

                    if self.method == 'relative':
                        driving_region_params = pl_module.update_by_relative_motion(source_region_params, driving_region_params_initial, driving_region_params)

                    if pl_module.hparams.avd_params:
                        driving_region_params = pl_module.avd_network(source_region_params, driving_region_params)

                    bg_params = pl_module.bg_predictor(source, driving)
                    out = pl_module.generator(source, source_region_params=source_region_params,
                                    driving_region_params=driving_region_params, bg_params=bg_params)

                    out['source_region_params'] = source_region_params
                    out['driving_region_params'] = driving_region_params
                    sources.append(source)
                    predictions.append(out['prediction'])

                t_pred = torch.cat(predictions, 0).unsqueeze(0)
                T = len(predictions)
                # sources = torch.tile(source.unsqueeze(1), (1, T, 1, 1, 1))
                sources = torch.cat(sources, 0).unsqueeze(0)
                vis = torch.cat((sources, x['video'][driving_idx], t_pred), dim=4)

                # save video to local storage
                if self.save_path:
                    for j, pred in enumerate(predictions):
                        pred_img = tensor2img(pred)
                        pred_img = cv2.resize(pred_img, (144, 64))
                        # current_idx = i * 12 + j
                        # cv2.imwrite(f"results/mraa_fvusm_processed/{(current_idx):0{8}d}.png", pred_img)
                        # cv2.imwrite(f"{self.save_path}/{(i//4+1):0{3}d}_{i%12+1}_{(j+1):0{2}d}.png", pred_img)
                        cv2.imwrite((self.save_path / f"{(i//4+1):0{3}d}_{i%12+1}_{(j+1):0{2}d}.png").as_posix(), pred_img)

                if isinstance(trainer.logger, TensorBoardLogger):
                    # self.trainer.logger.experiment.add_images(
                    #     "driving",
                    #     x['driving'], 
                    #     self.global_step
                    # )
                    trainer.logger.experiment.add_video(
                        'drivings / predictions', vis, i # batch_idx
                    )
                elif isinstance(trainer.logger, WandbLogger):
                    # self.trainer.logger.log_image(
                    #     key="Images", 
                    #     images=[x['driving'], generated['prediction'], transformed_frame], 
                    #     caption=["driving", "prediction", "transformed_frame"]
                    # )
                    trainer.logger.experiment.log(
                        {"video": wandb.Video((vis[0].cpu().numpy() * 255).astype(np.uint8), fps=4, format="gif")}
                    )
