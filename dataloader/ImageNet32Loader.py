from PIL import Image
import os
import os.path
import numpy as np
import pickle

# from torchvision.datasets import VisionDataset
from dataloader.VisionDataset import VisionDataset

class ImageNet32(VisionDataset):
    """

    imagenet32


    """

    data_start='train_data_batch_'
    data_end='data.npy'
    label_end='labels.npy'

    def __init__(self, root, train=True, transform=None, target_transform=None,download=False):

        super(ImageNet32, self).__init__(root, transform=transform,target_transform=target_transform)

        self.train = train  # training set or test set
        self.data = []
        self.targets = []
        if train:
            # now load the picked numpy arrays
            # load_data
            for num in range(1, 11):
                data_path = root + '/' + self.data_start + str(num) + self.data_end
                tmp_data = np.load(data_path)
                self.data.append(tmp_data)
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

            # load labels
            for num in range(1, 11):
                data_path = root + '/' + self.data_start + str(num) + self.label_end
                tmp_data = np.load(data_path)
                self.targets.extend((tmp_data-1).tolist())
        else:
            val_data_path=root+'/'+'data.npy'
            val_label_path=root+'/'+'labels.npy'

            # load_data
            tmp_data = np.load(val_data_path)
            self.data.append(tmp_data)
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

            # load labels
            tmp_data = np.load(val_label_path)
            self.targets.extend((tmp_data-1).tolist())



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


