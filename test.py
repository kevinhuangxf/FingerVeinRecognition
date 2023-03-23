import logging

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import get_pl_logger

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='conf')
def main(cfg: DictConfig) -> None:
    log.info(f'Conf: \n{OmegaConf.to_yaml(cfg)}')

    log.info(f'Instantiating datamodule <{cfg.datamodule._target_}>')
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    log.info(f'Instantiated datamodule: {datamodule}')

    log.info(f'Instantiating litmodel <{cfg.litmodel._target_}>')
    litmodel = hydra.utils.instantiate(cfg.litmodel)
    log.info(f'Instantiated litmodel: {litmodel}')

    log.info('Instantiating callbacks...')
    callbacks = [v for k, v in hydra.utils.instantiate(cfg.callbacks).items()]
    log.info(f'Instantiated callbacks: {callbacks}')

    log.info('Instantiating logger...')
    logger = get_pl_logger(cfg)
    log.info(f'Instantiated logger: {logger}')

    log.info(f'Instantiating trainer <{cfg.trainer._target_}>')
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    log.info(f'Instantiated trainer: {trainer}')

    # load pre-trained model
    model_state_dict = None
    if cfg.trainer.resume_from_checkpoint.endswith('.ckpt'):
        model_state_dict = torch.load(
            cfg.trainer.resume_from_checkpoint)['state_dict']
    elif cfg.trainer.resume_from_checkpoint.endswith('.pth'):
        model_state_dict = torch.load(cfg.trainer.resume_from_checkpoint,
                                      map_location='cuda:0')
    if model_state_dict is not None:
        litmodel.load_state_dict(model_state_dict, strict=True)

    trainer.test(litmodel, datamodule)


if __name__ == '__main__':
    main()
