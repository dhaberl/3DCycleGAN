from torch.utils.data import Dataset
from glob import glob
from os.path import join, basename
from random import randint
from natsort import natsorted
from pprint import pprint


class BatchDataset(Dataset):
    def __init__(self, root, transform=None, unpaired=False, mode="train"):
        self.transform = transform
        self.unpaired = unpaired

        self.files_A = natsorted(glob(join(root, f"{mode}/images") + "/*SUV*.nii.gz"))
        self.files_B = natsorted(glob(join(root, f"{mode}/labels") + "/*SUV*.nii.gz"))

    def __getitem__(self, index):
        selected_file_A = self.files_A[index % len(self.files_A)]
        item_A = self.transform(selected_file_A)
        id_A = basename(selected_file_A)

        if self.unpaired:
            random_index = randint(0, len(self.files_B) - 1)
            selected_file_B = self.files_B[random_index]
            item_B = self.transform(selected_file_B)
            id_B = basename(selected_file_B)
        else:
            selected_file_B = self.files_B[index % len(self.files_B)]
            item_B = self.transform(selected_file_B)
            id_B = basename(selected_file_B)

        return {"A": item_A, "B": item_B, "ID_A": id_A, "ID_B": id_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def print_keys(self):
        keys_A = [basename(i) for i in self.files_A]
        keys_B = [basename(i) for i in self.files_B]
        print(f"IMAGE KEYS (A): {len(keys_A)}")
        pprint(keys_A)
        print()
        print(f"LABEL KEYS (B): {len(keys_B)}")
        pprint(keys_B)
        print()
