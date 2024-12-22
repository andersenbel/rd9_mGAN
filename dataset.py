import torch
from torchvision import datasets, transforms


def get_cifar10_dataloader(batch_size=16, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
