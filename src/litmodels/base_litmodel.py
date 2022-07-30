from pytorch_lightning import LightningModule


class BaseLitmodel(LightningModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
