from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pathlib import Path
import config.config as cfg
import os
import torchvision

class CatAndDogDataModule(VisionDataModule):
    def __init__(self, data_dir: str,
                 transform = None,
                 batch_size: int = 32,
                 pic_size: int = 224,
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.pic_size = pic_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def data_transforms(self, batch):
        """Apply _train_transforms across a batch."""
        transform = self._default_transform() if self.transform is None else self.transform
        batch['image'] = [transform(pil_img.convert("RGB")) for pil_img in batch['image']]
        return batch

    def setup(self, stage=None):
        self.train_dataset = load_from_disk(Path.joinpath(Path(self.data_dir), 'train'))
        self.train_dataset.set_transform(self.data_transforms)

        self.val_dataset = load_from_disk(Path.joinpath(Path(self.data_dir), 'val'))
        self.val_dataset.set_transform(self.data_transforms)

        self.test_dataset = load_from_disk(Path.joinpath(Path(self.data_dir), 'test'))
        self.test_dataset.set_transform(self.data_transforms)

    def _default_transform(self):
        return transforms.Compose([torchvision.transforms.Resize((self.pic_size, self.pic_size)),
                                    torchvision.transforms.ToTensor()])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)
