import random
import numpy as np
from PIL import Image
from pathlib import Path

from torch.utils.data.dataset import Dataset


class MultiFramesDataset(Dataset):
    """ FVUSMDataset for FV-USM-processed
        root: Root dir for the FVUSM dataset.
        transform: Trsansforms for images.
        sample_per_class: How many samples in each class.
        mode: Train set or test set.
        inter_aug: Inter augmentation, 'LF' or 'TB'

    """

    def __init__(self, root, transforms=[], nframes=6, mode='train', preload=False, infer_dir=''):
        self.data_dir = Path(root)
        self.transforms = transforms
        self.frame_paths = sorted(list(self.data_dir.rglob("*.png")))
        self.nframes = nframes
        self.videos = len(self.frame_paths) // self.nframes
        self.mode = mode

        # pre-load all frames
        # self.latent_data = []
        # for file_name in self.files:
        #     latent = np.load(file_name.as_posix())
        #     self.latent_data.append(latent)

    def get_source_driving_ids(self, index):
        source_idx = None
        driving_idx = None

        if self.mode == 'train' or self.mode == 'test':
            sample_id_list = list(range(self.nframes))
            source_idx = random.choice(sample_id_list)
            driving_idx = random.choice(sample_id_list)
            while True:
                if source_idx == driving_idx:
                    driving_idx = random.choice(sample_id_list)
                else:
                    break
        
        source_idx = index * self.nframes + source_idx
        driving_idx = index * self.nframes + driving_idx

        return source_idx, driving_idx

    def __getitem__(self, index):
        source_idx, driving_idx = self.get_source_driving_ids(index)
        source_img = Image.open(self.frame_paths[source_idx]).convert('RGB')
        driving_img = Image.open(self.frame_paths[driving_idx]).convert('RGB')
        source = self.transforms(source_img)
        driving = self.transforms(driving_img)
        return source, driving

    def __len__(self):
        return self.videos
