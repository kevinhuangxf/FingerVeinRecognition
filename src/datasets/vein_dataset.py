import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data.dataset import Dataset


class VeinDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 transforms,
                 samples_per_class):
        self.data_dir = Path(data_root)
        self.transforms = transforms
        self.img_paths = sorted(list(self.data_dir.rglob("*.*")))
        self.img_data = [Image.open(img_path).convert('RGB') for img_path in self.img_paths]

        if not isinstance(self.transforms, list):
            self.transforms = list(self.transforms)

        self.class_num = len(self.img_data) // samples_per_class
        self.labels = np.arange(self.class_num).repeat(samples_per_class)

    def __getitem__(self, index):
        img = self.img_data[index]

        out = [transform(img) for transform in self.transforms]

        if len(out) == 1:
            out = out[0]

        return out, self.labels[index]
    
    def __len__(self):
        return len(self.img_data)
