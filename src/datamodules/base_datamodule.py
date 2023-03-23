import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BaseDatamodule(pl.LightningDataModule):

    def __init__(self,
                 datasets,
                 transforms=None,
                 train_batch_size: int = 2,
                 val_batch_size: int = 1,
                 test_batch_size: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if transforms is not None:
            datasets['train'].transforms = transforms['train']
            datasets['val'].transforms = transforms['val']
            datasets['test'].transforms = transforms['test']

        self.data_train = datasets['train']
        self.data_val = datasets['val']
        self.data_test = datasets['test']

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
