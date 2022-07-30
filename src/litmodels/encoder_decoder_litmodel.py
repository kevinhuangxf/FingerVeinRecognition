from pytorch_lightning import LightningModule


class EncoderDecoderLitmodel(LightningModule):

    def __init__(self, backbone, head, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.head = head
