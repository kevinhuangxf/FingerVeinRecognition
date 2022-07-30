from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fvusm_dataset import BalancedBatchSampler, FVUSMDataset


def test_train_dataloader():

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

    trainset = FVUSMDataset(
        root='/mnt/disk_d/Data/FVUSM/FV-USM-processed',
        sample_per_class=12,
        transforms=transform_train,
        mode='train',
        inter_aug='')

    train_batch_sampler = BalancedBatchSampler(trainset, n_classes=8, n_samples=4)
    trainloader = DataLoader(
        trainset, batch_sampler=train_batch_sampler, num_workers=4, pin_memory=True)
    batch = next(iter(trainloader))
    print(batch)
