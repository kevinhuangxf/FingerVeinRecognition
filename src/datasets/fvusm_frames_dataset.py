import cv2
import random
import numpy as np
from pathlib import Path

from torch.utils.data.dataset import Dataset


class FVUSMFramesDataset(Dataset):
    """ FVUSMDataset for FV-USM-processed
        root: Root dir for the FVUSM dataset.
        transform: Trsansforms for images.
        sample_per_class: How many samples in each class.
        mode: Train set or test set.
        inter_aug: Inter augmentation, 'LF' or 'TB'

    """

    def __init__(self, root, transforms=[], sample_per_class=6, mode='train', infer_dir=''):
        self.transform = transforms
        self.files = sorted(list(Path(root).rglob("*.png")))
        self.sample_per_class = sample_per_class
        self.class_num = len(self.files) // sample_per_class
        self.mode = mode

        self.img_data = []
        for file_name in self.files:
            img = cv2.imread(file_name.as_posix())
            img = cv2.resize(img, (256, 256))
            self.img_data.append(img)
        
        # infer data
        self.infer_data = []
        self.infer_files = []
        if infer_dir:
            self.infer_files = sorted(list(Path(infer_dir).rglob("*.png")))
            for infer_file in self.infer_files:
                img = cv2.imread(infer_file.as_posix())
                img = cv2.resize(img, (256, 256))
                self.infer_data.append(img)

    def __getitem__(self, index):
        out = {}
        if self.mode == 'train':
            sample_id_list = list(range(self.sample_per_class))
            source_idx = random.choice(sample_id_list)
            driving_idx = random.choice(sample_id_list)
            while True:
                if source_idx == driving_idx:
                    driving_idx = random.choice(sample_id_list)
                else:
                    break

            source = self.img_data[index * self.sample_per_class + source_idx]
            driving = self.img_data[index * self.sample_per_class + driving_idx]

            source = (source.transpose(2, 0, 1) / 255.).astype(np.float32)
            driving = (driving.transpose(2, 0, 1) / 255.).astype(np.float32)
            
            out['source'] = source
            out['driving'] = driving
        elif self.mode == 'val':
            if len(self.infer_data) != 0:
                # set source
                source = self.infer_data[index]
                out['source'] = (source.transpose(2, 0, 1) / 255.).astype(np.float32)
                # random select id from img_data
                index = random.choice(range(self.class_num))

            frames = []
            for i in range(self.sample_per_class):
                fid = index * self.sample_per_class + i
                image = self.img_data[fid]
                frames.append(image)

            frames = np.array(frames)
            frames = frames.transpose(0, 3, 1, 2) / 255.
            out['video'] = frames.astype(np.float32)
        elif self.mode == 'test':
            if len(self.infer_data) != 0:
                # set source
                source = self.infer_data[index]
                out['source'] = (source.transpose(2, 0, 1) / 255.).astype(np.float32)
                # random select id from img_data
                index = random.choice(range(self.class_num))

            num_videos = len(self.img_data) // self.sample_per_class

            videos = []
            for video_start_idx in [self.sample_per_class * i for i in list(range(num_videos))]:
                frames = []
                for img_idx in range(video_start_idx, video_start_idx + self.sample_per_class):
                    image = self.img_data[img_idx]
                    frames.append(image)
                frames = np.array(frames)
                frames = frames.transpose(0, 3, 1, 2) / 255.
                videos.append(frames.astype(np.float32))

            out['videos'] = videos

        return out

    def __len__(self):
        if len(self.infer_files) != 0:
            return len(self.infer_files)
        else:
            return self.class_num
