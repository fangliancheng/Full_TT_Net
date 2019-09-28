import argparse
from torchvision import datasets, transforms, models
import my_models
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
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
import pdb

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--arch', default='pre_cnn', type=str, help='arch')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
parser.add_argument('--mark', default='t5', type=str, help='mark')
#parser.add_argument('--gpu', default='1', type=int, help='GPU id to use.')
args = parser.parse_args()

settings = edict.EasyDict({
    "GPU" : True,
    "IMG_SIZE" : 224,
    "CNN_MODEL" : MODEL_DICT[args.arch],
    "DATASET" : args.dataset,
    "DATASET_PATH" : DATASET_PATH[args.dataset],
    "NUM_CLASSES" : NUM_CLASSES[args.dataset],
    #"MODEL_FILE" : "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format(args.arch, args.mark, args.dataset, 62),
    "MODEL_FILE" : None,
    "FINETUNE": False,
    "WORKERS" : 12,
    "BATCH_SIZE" : 64,
    "PRINT_FEQ" : 10,
    "LR" : 0.1,
    "EPOCHS" : 45,
    "CLIP_GRAD": 0,
    "ITERATE_NUM":6,
    "TT_SHAPE":[4,8,4,4],
    "TT_RANK": 4,
    "FEATURE_EXTRACT":'cnn',
    "OTT": False,
    "BENCHMARK": True,
})

if settings.GPU:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# import os
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

settings.OUTPUT_FOLDER = "result/pytorch_{}_{}_{}".format(args.arch, args.mark, args.dataset)
snapshot_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'snapshot'))

def train(model, train_loader, val_loader, dir=None):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #Loading a checkpoint for resuming training
    if settings.MODEL_FILE is not None:
        check_point = torch.load(settings.MODEL_FILE)
        state_dict = check_point['state_dict']
        model.load_state_dict(state_dict)
        epoch_cur = check_point['epoch']
    else:
        epoch_cur = -1

    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    params_to_update = model.parameters()
    if settings.FEATURE_EXTRACT=='cnn':
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    #weight_decay set to a big value: 1e-1
    optimizer = torch.optim.SGD(params_to_update, settings.LR, momentum=0.9, weight_decay=1e-3)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = settings.LR * (0.1 ** (epoch // (settings.EPOCHS // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # def move_buffer_to_gpu(optimizer):
    #     for p in optimizer.state.keys():
    #         print('move working!!!')
    #         param_state = optimizer.state[p]
    #         buf = param_state["momentum_buffer"]
    #         param_state["momentum_buffer"] = buf.cuda()  # move buf to device

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

        # if epoch == 2:
        #     p()
        for batch_idx, (input, target) in enumerate(train_loader):
            if settings.GPU:
                target = target.to(device=device)
                input = input.to(device=device)

            if settings.FEATURE_EXTRACT == 'tt':
                input_tt = t3.input_to_tt(input,settings)

            data_time.update(time.time() - end)

            # compute output
            with autograd.detect_anomaly():
                if settings.FEATURE_EXTRACT == 'tt':
                    output = model(input_tt)
                else:
                    output = model(input)
                loss = criterion(output, target)
                #pdb.set_trace()
                if loss > 10:
                    print('loss explosion!!')
                    #break
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # # compute gradient and do SGD step
            # optimizer.zero_grad()
            # loss.backward()

            #gradient clipping
            if settings.CLIP_GRAD != 0:
                clip_value = settings.CLIP_GRAD
                nn.utils.clip_grad_norm(model.parameters(),clip_value)

            #print(optimizer.state_dict())
            #move_buffer_to_gpu(optimizer)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % settings.PRINT_FEQ == 0:
                print('Train: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    top1=top1, top5=top5))
        print(' * TIMESTAMP {}'.format(time.strftime("%Y-%m-%d-%H:%M")))
        print(' * TRAIN Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        torch.save({
            'epoch': epoch,
            'best_prec1': top1.avg,
            'state_dict': model.state_dict(),
            #optimizer_state_dict
            #loss
        }, os.path.join(dir, 'epoch_%d.pth' % epoch))


def validate(model, val_loader):
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
        target = target.to(device)
        input = input.to(device)

        #if settings.GPU:
            #target = target.cuda(non_blocking=True)
            #input = input.cuda(non_blocking=True)
        with torch.no_grad():
            #input_var = torch.autograd.Variable(input)
            #target_var = torch.autograd.Variable(target)

            # compute output
            fc_output = model(input)

            loss = criterion(fc_output, target)

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
    # val_loader = imagenet_loader(settings, 'val')
    # train_loader = imagenet_loader(settings, 'train', shuffle=True, data_augment=True)
    val_loader = cifar_loader(settings, 'val')
    train_loader = cifar_loader(settings, 'train', shuffle=True, data_augment=True)
    #val_loader = minist_loader(settings, 'val')
    #train_loader = minist_loader(settings, 'train', shuffle=True)
    # model = finetune_model
    #model = settings.CNN_MODEL(type='ptt_solver', pretrained=settings.FINETUNE, num_classes=settings.NUM_CLASSES)

    if settings.FEATURE_EXTRACT == 'cnn' and not settings.BENCHMARK:
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False

        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=True)
        num_ftrs = model.fc.in_features #512
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model.fc = settings.CNN_MODEL(settings)
    if settings.BENCHMARK == True:
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False

        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=True)
        num_ftrs = model.fc.in_features #512
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model.fc = nn.Linear(512,10)
    else:
        model = settings.CNN_MODEL(settings)

    if settings.GPU:
        #model = nn.DataParallel(model).cuda()
        model = model.cuda()
        #print('model parameter:',dict(model.named_parameters()))

    print(model)
    model.train()

    train(model, train_loader, val_loader, snapshot_dir)
    for epoch in range(settings.EPOCHS - 10, settings.EPOCHS):
        settings.MODEL_FILE = "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format(args.arch, args.mark, args.dataset, epoch)
        validate(model, val_loader)


if __name__ == '__main__':
    main()
