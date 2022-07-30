from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def get_pl_logger(exp_conf, sub_dir=None):
    """pl_logger"""
    logger = True

    if 'logger' in exp_conf:
        _save_dir = exp_conf.logger.save_dir
        _name = exp_conf.logger.experiment_name
        _version = exp_conf.logger.version
        if exp_conf.logger.type == 'tensorboard':
            logger = TensorBoardLogger(
                save_dir=_save_dir, name=_name, version=_version, sub_dir=sub_dir)
        elif exp_conf.logger.type == 'wandb':
            logger = WandbLogger(project=_name, save_dir=_save_dir, log_model='all')
    else:
        print('No logger in experiment conf, use default logger!')

    return logger
