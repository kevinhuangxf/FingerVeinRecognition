import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fvusm_dataset import BalancedBatchSampler, FVUSMDataset


class FVRDatamodule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str = 'data/',
                 train_batch_size: int = 2,
                 val_batch_size: int = 1,
                 test_batch_size: int = 1,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # trfms
        normalize = transforms.Normalize(
            mean=[
                0.5,
            ], std=[
                0.5,
            ])
        transform_train = []
        transform_train.append(
            transforms.RandomResizedCrop(size=(64, 144), scale=(0.5, 1.0), ratio=(2.25, 2.25)))
        transform_train.append(transforms.RandomRotation(degrees=3))
        transform_train.append(transforms.RandomPerspective(distortion_scale=0.3, p=0.9))
        transform_train.append(transforms.ColorJitter(brightness=0.7, contrast=0.7))
        transform_train.append(transforms.ToTensor())
        transform_train.append(normalize)
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        self.data_train = FVUSMDataset(
            root=self.hparams.data_dir,
            sample_per_class=12,
            transforms=transform_train,
            mode='train',
            inter_aug='')
        self.data_val = FVUSMDataset(
            root=self.hparams.data_dir,
            sample_per_class=12,
            transforms=transform_test,
            mode='test',
            inter_aug='')
        self.data_test = self.data_val

        self.train_batch_sampler = BalancedBatchSampler(self.data_train, n_classes=8, n_samples=4)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_sampler=self.train_batch_sampler,
            # batch_size=self.hparams.train_batch_size,
            # num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            # shuffle=True,
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
