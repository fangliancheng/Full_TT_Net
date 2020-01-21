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
                    batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return train_loader
    if dir == 'val':
        val_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return val_loader

    else:
        print('Wrong arg!')


def cifar_loader(settings, dir, batch_size=None, shuffle=False, data_augment=False):
    if batch_size is None:
        batch_size = settings.BATCH_SIZE

    kwargs = {'num_workers': settings.WORKERS, 'pin_memory': True} if settings.GPU else {}
    if settings.FEATURE_EXTRACT == 'tt':
        transform_train = transforms.Compose(
            [#transforms.Resize(size=(64,64)),
             #settings.IMG_SIZE = 224
             #transforms.RandomResizedCrop(settings.IMG_SIZE),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        transform_val = transforms.Compose(
            [#transforms.Resize(settings.IMG_SIZE),
             #settings.IMG_SIZE = 224
             #transforms.RandomResizedCrop(settings.IMG_SIZE),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    elif settings.FEATURE_EXTRACT == 'cnn':
            transform_train = transforms.Compose(
                [#transforms.Resize(size=(64,64)),
                 #settings.IMG_SIZE = 224
                 transforms.RandomResizedCrop(settings.IMG_SIZE),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ])
            transform_val = transforms.Compose(
                [transforms.Resize(settings.IMG_SIZE),
                 #settings.IMG_SIZE = 224
                 #transforms.RandomResizedCrop(settings.IMG_SIZE),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ])
    else:
        transform_train = transforms.Compose(
            [#transforms.Resize(size=(64,64)),
             #settings.IMG_SIZE = 224
             #transforms.RandomResizedCrop(settings.IMG_SIZE),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        transform_val = transforms.Compose(
            [#transforms.Resize(settings.IMG_SIZE),
             #settings.IMG_SIZE = 224
             #transforms.RandomResizedCrop(settings.IMG_SIZE),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    if dir == 'train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size,
                                                  shuffle=True, num_workers=2,drop_last=True)
        return trainloader
    if dir == 'val':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_val)
        testloader = torch.utils.data.DataLoader(testset, batch_size,
                                                  shuffle=False, num_workers=2,drop_last=True)
        return testloader
    else:
        print('Wrong arg!')

def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import pdb
    import argparse
    from torchvision import datasets, transforms
    import my_models
    import easydict as edict
    import t3nsor as t3
    from dataloader import imagenet_loader
    from dataloader import cifar_loader
    from dataloader import minist_loader
    from t3nsor import TensorTrainBatch
    from t3nsor import TensorTrain
    from torch import autograd

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    settings = edict.EasyDict({
        "GPU" : True,
        "IMG_SIZE" : 224,
        #"MODEL_FILE" : "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format(args.arch, args.mark, args.dataset, 62),
        "MODEL_FILE" : None,
        "FINETUNE": False,
        "WORKERS" : 12,
        "BATCH_SIZE" : 4,
        "PRINT_FEQ" : 1,
        "LR" : 0.1,
        "EPOCHS" : 45,
        "CLIP_GRAD": 5,
        "ITERATE_NUM":6,
        "TT_SHAPE":[4,8,4,8],
        "TT_RANK": 100,
    })

    input=images
    tt_cores_1 = []
    tt_cores_2 = []
    tt_cores_3 = []
    tt_batch_cores_1 = []
    tt_batch_cores_2 = []
    tt_batch_cores_3 = []
    for i in range(0,settings.BATCH_SIZE):
        tt_cores_1 += t3.to_tt_tensor(input[i, 0, :, :].view(settings.TT_SHAPE), max_tt_rank=settings.TT_RANK).tt_cores
        tt_cores_2 += t3.to_tt_tensor(input[i, 1, :, :].view(settings.TT_SHAPE), max_tt_rank=settings.TT_RANK).tt_cores
        tt_cores_3 += t3.to_tt_tensor(input[i, 2, :, :].view(settings.TT_SHAPE), max_tt_rank=settings.TT_RANK).tt_cores
    pdb.set_trace()
     #unsqueeze
    tt_core_1_unsq = [torch.unsqueeze(i, 0) for i in tt_cores_1]
    tt_core_2_unsq = [torch.unsqueeze(i, 0) for i in tt_cores_2]
    tt_core_3_unsq = [torch.unsqueeze(i, 0) for i in tt_cores_3]
    #print('tt_core_1_unsq:',tt_core_1_unsq[0:(settings.BATCH_SIZE-1)*4+1:4])
    #print('shape:',[i.shape for i in tt_core_1_unsq[1:(settings.BATCH_SIZE-1)*4+2:4]])
    for shift in range(1, 5):
        tt_batch_cores_1.append(torch.cat(tt_core_1_unsq[shift-1:(settings.BATCH_SIZE-1)*4+shift:4], dim=0))
        tt_batch_cores_2.append(torch.cat(tt_core_2_unsq[shift-1:(settings.BATCH_SIZE-1)*4+shift:4], dim=0))
        tt_batch_cores_3.append(torch.cat(tt_core_3_unsq[shift-1:(settings.BATCH_SIZE-1)*4+shift:4], dim=0))

    input_tt = [TensorTrainBatch(tt_batch_cores_1), TensorTrainBatch(tt_batch_cores_2), TensorTrainBatch(tt_batch_cores_3)]
    dense_1 = input_tt[0].full().view(4, 32, 32)
    dense_2 = input_tt[1].full().view(4, 32, 32)
    dense_3 = input_tt[2].full().view(4, 32, 32)
    #pdb.set_trace()
    dense_cat = torch.stack([dense_1, dense_2, dense_3], dim=1)
    #pdb.set_trace()
    # show images
    imshow(torchvision.utils.make_grid(images))
    imshow(torchvision.utils.make_grid(dense_cat))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ =='__main__':
    main()
