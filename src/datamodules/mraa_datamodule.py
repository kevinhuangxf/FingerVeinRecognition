import os.path as osp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fvusm_frames_dataset import FVUSMFramesDataset


class MRAADatamodule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str = 'data/',
                 infer_dir: str = '',
                 sample_per_class = '',
                 train_batch_size: int = 2,
                 val_batch_size: int = 1,
                 test_batch_size: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        tfrms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.PILToTensor(),
            # transforms.ConvertImageDtype(torch.float),
        ])

        train_dataset = FVUSMFramesDataset(
            osp.join(data_dir, 'train'),
            tfrms,
            sample_per_class, 
            mode='train'
        )

        val_dataset = FVUSMFramesDataset(
            osp.join(data_dir, 'test'), 
            tfrms,
            sample_per_class, 
            mode='val'
        )

        test_dataset = FVUSMFramesDataset(
            osp.join(data_dir, 'train'), # use train-set to animate 
            tfrms,
            sample_per_class, 
            mode='test', 
            infer_dir=infer_dir
        )

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset

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
