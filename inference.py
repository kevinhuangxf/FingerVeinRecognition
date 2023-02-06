import torch
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

# from src.litmodels.fvr_litmodel import FVRLitModel

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

