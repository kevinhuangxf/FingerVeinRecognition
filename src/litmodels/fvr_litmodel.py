from pytorch_lightning import LightningModule


class FVRLitModel(LightningModule):

    def __init__(self, backbone, head, losses, optimizer):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.save_hyperparameters(ignore=['backbone', 'head'])

    def forward(self, x):
        x = self.backbone(x)
        out = self.head(x)
        return out

    def training_step(self, batch, batch_idx):
        data, labels = batch
        features = self.backbone(data)
        head_features, outputs = self.head(features)

        loss_dict = dict(
            loss_triplet=self.hparams.losses.tripletloss(head_features, labels)[0],
            loss_cosface=self.hparams.losses.cosface(outputs, labels),
        )
        loss_dict['loss'] = sum([v for k, v in loss_dict.items()])

        return loss_dict

    def validation_step(self, batch, batch_idx):
        # data, labels = batch
        # features = self.backbone(data)
        # outputs = self.head(features)

        # loss_dict = dict(
        #     # loss_triplet=self.hparams.losses.(features, labels),
        #     loss_cosface=self.hparams.losses.cosface(outputs, labels),
        # )
        # loss_dict['loss'] = sum([v for k, v in loss_dict.items()])

        # return loss_dict
        return None

    def test_step(self, batch, batch_idx):
        # data, labels = batch
        # features = self.backbone(data)
        # outputs = self.head(features)

        # loss_dict = dict(
        #     # loss_triplet=self.hparams.losses.(features, labels),
        #     loss_cosface = self.hparams.losses.cosface(outputs, labels)
        # )

        # return loss_dict
        return None

    def configure_optimizers(self):
        return {
            'optimizer':
            self.hparams.optimizer([
                dict(name='backbone', params=self.backbone.parameters()),
                dict(name='head', params=self.head.parameters()),
            ]),
        }
