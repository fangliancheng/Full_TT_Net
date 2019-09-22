import os
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets

def imagenet_loader(settings, dir, batch_size=None, shuffle=False, data_augment=False):
    if batch_size is None:
        batch_size = settings.BATCH_SIZE
    d = os.path.join(settings.DATASET_PATH, dir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if data_augment:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomSizedCrop(settings.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(settings.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(d, transform),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=settings.WORKERS, pin_memory=True)
    return loader


def minist_loader(settings, dir, batch_size=None, shuffle=False):
    if batch_size is None:
        batch_size = settings.BATCH_SIZE

    #for minist dataset
    kwargs = {'num_workers': settings.WORKERS, 'pin_memory': True} if settings.GPU else {}
    if dir == 'train':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                    batch_size=settings.BATCH_SIZE, shuffle=True, **kwargs)
        return train_loader
    if dir == 'val':
        val_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=settings.BATCH_SIZE, shuffle=True, **kwargs)
        return val_loader

    else:
        print('Wrong arg!')


def cifar_loader(settings, dir, batch_size=None, shuffle=False, data_augment=False):
    if batch_size is None:
        batch_size = settings.BATCH_SIZE

    #for minist dataset
    kwargs = {'num_workers': settings.WORKERS, 'pin_memory': True} if settings.GPU else {}
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    if dir == 'train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size,
                                                  shuffle=True, num_workers=2)
        return trainloader
    if dir == 'val':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size,
                                                  shuffle=False, num_workers=2)
        return testloader
    else:
        print('Wrong arg!')
