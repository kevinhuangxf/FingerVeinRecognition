import hydra
import pytest
from omegaconf import OmegaConf as oc


@pytest.mark.parametrize('conf_path', ['configs/datamodule/fvr.yaml'])
def test_datamodule(conf_path):
    conf = oc.load(conf_path)
    datamodule = hydra.utils.instantiate(conf)
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)
