import numpy as np

from torch.utils.data import Dataset, DataLoader


class DigitDataset(Dataset):
    def __init__(self, images, classes, transform=None):
        self.transform = transform

        self.images = images
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(self.images[idx])

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.classes[idx]


def prepare_loader(images, classes, transform, batch_size, num_workers=8, pin_memory=True):
    tmp = DigitDataset(images=images, classes=classes, transform=transform)
    loader = DataLoader(tmp, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    return loader