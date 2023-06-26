import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import albumentations as A


def apply_transforms(dataset_mean):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    aug_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=1,
                        fill_value=[0.4914, 0.4822, 0.4471], mask_fill_value=None),
    ])

    return transform, test_transform, aug_transform


class CIFAR10Albumentations(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, aug_transform=None):
        super(CIFAR10Albumentations, self).__init__(root, train=train, transform=transform,
                                                    target_transform=target_transform, download=download)
        self.aug_transform = aug_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.array(img)

        if self.aug_transform:
            augmented = self.aug_transform(image=img)
            img = augmented["image"]

        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def create_data_loaders(batch_size, num_workers, dataset_mean):
    transform, test_transform, aug_transform = apply_transforms(dataset_mean)

    trainset = CIFAR10Albumentations(root='./data', train=True, download=True, transform=transform,
                                     aug_transform=aug_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
