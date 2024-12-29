import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class BaseDatamodule(pl.LightningDataModule):

    def __init__(self,
                 datasets,
                 transforms=None,
                 num_repeats=1,
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

        if num_repeats > 1:
            self.data_train = DatasetRepeater(datasets['train'], num_repeats)
        else:
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
