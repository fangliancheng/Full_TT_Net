import argparse
import torch
from torchvision import datasets, transforms, models
from my_models import *
from model_wideresnet import WideResNet
import easydict as edict
import t3nsor as t3
from common import *
from dataloader import imagenet_loader
from dataloader import cifar_loader
from dataloader import minist_loader
from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain
from torch import autograd
import torch.nn as nn
import pdb
import math

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--arch', default='LeNet', type=str, help='arch')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
parser.add_argument('--mark', default='t5', type=str, help='mark')

args = parser.parse_args()

settings = edict.EasyDict({
    "GPU": True,
    "DATASET": args.dataset,
    "DATASET_PATH": DATASET_PATH[args.dataset],
    "NUM_CLASSES": NUM_CLASSES[args.dataset],
    "FEATURE_EXTRACT": None,
    "MODEL_FILE": None,
    "FINETUNE": False,
    "WORKERS": 12,
    "BATCH_SIZE": 256,
    "PRINT_FEQ": 10,
    "LR": 0.1,
    "EPOCHS": 90,
    "TT_RANK": 4,
    "IS_OUTPUT_FORM": 'dense',
})

if settings.GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(model, train_loader, dir=None, sample_covariance_tt_core_list=None, pre_model=None):
    model.train()

    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), settings.LR, momentum=0.9, weight_decay=1e-4)

    if settings.MODEL_FILE is not None:
        check_point = torch.load(settings.MODEL_FILE)
        state_dict = check_point['state_dict']
        model.load_state_dict(state_dict)
        epoch_cur = check_point['epoch']
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
    else:
        epoch_cur = -1

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = settings.LR * (0.1 ** (epoch // (settings.EPOCHS // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(epoch_cur+1, settings.EPOCHS):
        adjust_learning_rate(optimizer, epoch)

        print('Epoch[%d/%d]' % (epoch, settings.EPOCHS))
        # train
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            target = target.to(device=device)
            input = input.to(device=device)
            if pre_model is not None:
                #print('input shape:', input.shape)
                input = pre_model(input)
                reduced_cov = t3.important_sketching(input, sample_covariance_tt_core_list, settings)
                # Then use target, reduced_cov as response and new dimension-reduced feature to do supervised learning
                input = reduced_cov

            # compute output
            output = model(input)
            loss = criterion(output, target)
            if loss > 10:
                print('loss explosion!!')

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % settings.PRINT_FEQ == 0:
                print('Train: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    top1=top1, top5=top5))
        print(' * TIMESTAMP {}'.format(time.strftime("%Y-%m-%d-%H:%M")))
        print(' * TRAIN Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        torch.save({
            'epoch': epoch,
            'best_prec1': top1.avg,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(dir, 'epoch_%d.pth' % epoch))


def val(model, val_loader, epoch, sample_covariance_tt_core_list=None, pre_model=None):
    if settings.MODEL_FILE is not None:
        check_point = torch.load(settings.MODEL_FILE)
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in check_point[
        #     'state_dict'].items()}
        state_dict = check_point['state_dict']
        model.load_state_dict(state_dict)
    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # val
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device=device)
        input = input.to(device=device)
        if pre_model is not None:
            input = pre_model(input)
            reduced_cov = t3.important_sketching(input, sample_covariance_tt_core_list, settings)
            input = reduced_cov
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            fc_output = model(input_var)

            loss = criterion(fc_output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(fc_output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        if i % settings.PRINT_FEQ == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    print(' * VAL Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


def main():
    val_loader = cifar_loader(settings, 'val')
    train_loader = cifar_loader(settings, 'train', shuffle=True, data_augment=True)
    #
    # pre_processing_model = LeNet()
    # print('pre_processing model:', pre_processing_model)
    # if settings.GPU:
    #     pre_processing_model = nn.DataParallel(pre_processing_model).cuda()
    #
    # #set location to save pre_processing model weights
    # settings.OUTPUT_FOLDER = "result/pytorch_{}_{}_{}".format('LeNet', 'channel_15', args.dataset)
    # snapshot_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'snapshot'))
    # train(pre_processing_model, train_loader, snapshot_dir)
    #
    # #observe the performance of this weak network
    # #Note: 67.288 top1 test accuracy, 94.922 top5
    # for epoch in range(settings.EPOCHS - 10, settings.EPOCHS):
    #     settings.MODEL_FILE = "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format('LeNet', 'channel_15, args.dataset, epoch)
    #     val(pre_processing_model, val_loader, epoch)

    ###########################################################################################################################
    lenet_model_file = "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format('LeNet', 'channel_15', args.dataset, 29)
    check_point = torch.load(lenet_model_file)
    state_dict = check_point['state_dict']
    #for name in state_dict:
    #    print(name)
    pre_processing_model = nn.DataParallel(LeNet_partial()).cuda()

    #for name, param in pre_processing_model.named_parameters():
    #    print(name, param)
    #only conv layer weights will be loaded
    #print('before weight load:', pre_processing_model.module.conv1.weight)
    pre_processing_model.load_state_dict(state_dict, strict=False)
    #print('after weight load:', pre_processing_model.module.conv1.weight)
    #print('pre_model:', pre_processing_model)
    #set the conv layer to be untrainable, not necessary
    for name, param in pre_processing_model.named_parameters():
        param.requires_grad = False

    # Use full dataset to compute sample covariance tensor
    sample_covariance_tt_core_list = t3.construct_sample_covariance_tensor('full', settings, pre_model=pre_processing_model)

    #trainable network, should be TT-Net once backward issue is fixed
    model = important_sketching_input_wideresnet(settings)
    print('classifier network:', model)

    if settings.GPU:
        model = nn.DataParallel(model).cuda()

    settings.OUTPUT_FOLDER = "result/pytorch_{}_{}_{}".format('LeNet_Cov_WideResNet', args.mark, args.dataset)
    snapshot_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'snapshot'))
    train(model, train_loader, snapshot_dir, sample_covariance_tt_core_list, pre_model=pre_processing_model)
    for epoch in range(settings.EPOCHS - 10, settings.EPOCHS):
        settings.MODEL_FILE = "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format('LeNet_Cov_WideResNet', args.mark, args.dataset, epoch)
        val(model, val_loader, epoch, sample_covariance_tt_core_list, pre_processing_model)


if __name__ == '__main__':
    main()
