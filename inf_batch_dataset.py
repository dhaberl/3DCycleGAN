import os

from torch.utils.data import Dataset


class InferenceBatchDataset(Dataset):
    def __init__(self, image_files, transforms):
        self.image_files = image_files
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return (
            self.transforms(self.image_files[index]),
            os.path.basename(self.image_files[index]).split(".")[0],
            self.image_files[index],
        )
