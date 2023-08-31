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
from src.networks.mraa.logger import Visualizer


class MRAAAnimationCallback(Callback):
    def __init__(self, method="standard", save_path=None):
        super().__init__()
        self.method = method
        self.save_path = save_path

        if self.save_path is not None:
            self.save_path = Path(save_path)
            self.save_path.mkdir(exist_ok=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.animate_same_video(trainer, pl_module, batch, batch_idx)
    
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print("Animate Video:")
        # self.animate_random_video(trainer, pl_module, batch, batch_idx)
        # self.infer_motion_vectors(trainer, pl_module, batch)
        self.infer_predefined_motion_vectors(pl_module, batch)

    def infer_motion_vectors(self, trainer, pl_module, batch):
        motion_list = pl_module.get_motion_list(batch['video'])

        motion_list_np = np.concatenate([m.cpu().numpy() for m in motion_list], axis=0)
        
        torch.save(pl_module.animation_model.state_dict(), self.save_path / 'animation_model.pth')
        np.save(self.save_path / "motion_vectors.npy", motion_list_np)

        # motion_vectors, k_list = pl_module.get_principle_motion_vectors(motion_list, 10)
        
        sample_per_class = trainer.datamodule.hparams.datasets.test.sample_per_class
        for i, source in enumerate(tqdm(batch['source'])):
            for j in range(sample_per_class):
                # use random principle motion vector
                # random_vector = pl_module.get_random_motion_vector(motion_vectors, k_list)

                # use random vector from motion list
                random_vector = motion_list[random.choice(range(len(motion_list)))]

                out = pl_module.infer_motion_vector(source, random_vector)
                # images = visualizer.visualize(source, out)

                pred_img = tensor2img(out['prediction'])
                # pred_img = cv2.resize(pred_img, (64, 144))
                # pred_img = cv2.rotate(pred_img, cv2.ROTATE_90_CLOCKWISE)

                cv2.imwrite((self.save_path / f"{(i//4+1):0{3}d}_{i%12+1}_{(j+1):0{2}d}.png").as_posix(), pred_img)

    def infer_predefined_motion_vectors(self, pl_module, batch):
        motion_vectors_path = '/media/user/Toshiba4T/Kevin/Development/FingerVeinRecognition/rolling_horizontal.npy'
        motion_vectors=torch.from_numpy(np.load(motion_vectors_path)).cuda()
        visualized_list = pl_module.infer_new_shift_param(batch['source'][0], motion_vectors[0])
        for i, viz in enumerate(visualized_list):
            cv2.imwrite((f"viz_{i}.png"), viz)

    def animate_same_video(self, trainer, pl_module, batch, batch_idx):
        x = batch
        loss_dict = {}

        sources = []
        drivings = []
        predictions = []
        source_region_params = pl_module.region_predictor(x['video'][:, 0])
        visualizer = Visualizer(kp_size=source_region_params['shift'].shape[-2])
        visualize_list = []
        for frame_idx in range(x['video'].shape[1]):
            source = x['video'][:, 0]
            sources.append(source)
            driving = x['video'][:, frame_idx]
            driving_region_params = pl_module.region_predictor(driving)
            drivings.append(driving)

            out = pl_module.generator(source, source_region_params=source_region_params,
                            driving_region_params=driving_region_params, bg_params=None)

            out['source_region_params'] = source_region_params
            out['driving_region_params'] = driving_region_params
            images = visualizer.visualize(source, out, driving)
            visualize_list.append(torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0))

            predictions.append(out['prediction'])
        if visualize_list is not None:
            vis = torch.cat(visualize_list, 0).unsqueeze(0)
        else:
            t_sour = torch.cat(sources, 0).unsqueeze(0)
            t_driv = torch.cat(drivings, 0).unsqueeze(0)
            t_pred = torch.cat(predictions, 0).unsqueeze(0)
            if t_sour.shape[-1] <= t_sour.shape[-1]:
                vis = torch.cat((t_sour, t_driv, t_pred), dim=3)
            else:
                vis = torch.cat((t_sour, t_driv, t_pred), dim=4)

        # each_val_epoch_steps = len(trainer.datamodule.data_val) \
        #                        // trainer.datamodule.hparams.val_batch_size + 1
        # val_global_step = pl_module.current_epoch * each_val_epoch_steps + batch_idx

        val_global_step = batch_idx

        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_video(
                'drivings / predictions', vis, val_global_step)

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
