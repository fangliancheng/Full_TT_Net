import argparse
from torchvision import datasets, transforms, models
import my_models
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
parser.add_argument('--arch', default='important_sketching_ftt_1hidden_relu_net', type=str, help='arch')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
parser.add_argument('--mark', default='t5_5tt_relu', type=str, help='mark')
#parser.add_argument('--gpu', default='1', type=int, help='GPU id to use.')
args = parser.parse_args()

settings = edict.EasyDict({
    "GPU": True,
    "IMG_SIZE": 224,
    "CNN_MODEL": MODEL_DICT[args.arch],
    "DATASET": args.dataset,
    "DATASET_PATH": DATASET_PATH[args.dataset],
    "NUM_CLASSES": NUM_CLASSES[args.dataset],
    #"MODEL_FILE" : "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format(args.arch, args.mark, args.dataset, 62),
    "MODEL_FILE": None,
    "FINETUNE": False,
    "WORKERS": 12,
    "BATCH_SIZE": 64,
    "PRINT_FEQ": 10,
    "LR": 0.01,
    "EPOCHS": 45,
    "CLIP_GRAD": 0,
    "ITERATE_NUM": 6,
    #"TT_SHAPE": SHAPE_DICT[args.arch],
    #"TT_SHAPE": [4,8,4,8], #Shpae of Cifar 10 data is 32*32
    #"TT_SHAPE_512":[4,8,4,4],   #Output shape of ResNet 18 pretrained network(drop the last classification layer) is dimension 512
    #"TT_MATRIX_SHAPE": [None, [4, 8, 4, 4]], #Linear transformation in TT
    "INPUT_TT_MATRIX_IS_SHAPE": [None, [3, 4, 8, 4, 8]],
    "TT_RANK": 3,
    "FEATURE_EXTRACT": EXTRACT_DICT[args.arch],
    #"FEATURE_EXTRACT": None,
    #"FEATURE_EXTRACT":'tt', #or cnn
    "OTT": False,
    "BENCHMARK": False, #put a trainable fc layer under a untrainable pretrained ResNet18
    "LEARNABLE": True,
    #Use curr_minibatch to construct sample covariance, Alternative: full: use all training data to construct sample covariance
    #"SAMPLE_MODE": 'curr_minibatch',
    "SAMPLE_MODE": 'full',
    "IS_OUTPUT_FORM": FEATURE_FORM[args.arch],
    "PGD": True,
})

device = torch.device('cuda:0')
torch.cuda.set_device(device)

if settings.GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

settings.OUTPUT_FOLDER = "result/pytorch_{}_{}_{}".format(args.arch, args.mark, args.dataset)
snapshot_dir = dir(os.path.join(settings.OUTPUT_FOLDER, 'snapshot'))


def train(model, train_loader, val_loader, dir=None, sample_covariance_tt_core_list=None):
    #switch to train mode
    model.train()
    print('setting.MODULE_FILE:', settings.MODEL_FILE)
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

    #criterion = nn.MSELoss().cuda()

    params_to_update = model.parameters()
    if settings.FEATURE_EXTRACT == 'cnn':
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

    #weight_decay set to a big value: 1e-1
    optimizer = torch.optim.SGD(params_to_update, settings.LR, momentum=0.9, weight_decay=1e-3)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = settings.LR * (0.1 ** (epoch // (settings.EPOCHS // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # #Use full dataset to compute sample covariance tensor
    # sample_covariance_tt_core_list = t3.construct_sample_covariance_tensor('full', settings)

    for epoch in range(epoch_cur+1, settings.EPOCHS):
        adjust_learning_rate(optimizer, epoch)

        print('Epoch[%d/%d]' % (epoch, settings.EPOCHS))
        # train
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter() #batch_time = data_time + forward_time + backward_time
        data_time = AverageMeter()
        forward_time = AverageMeter()
        end = time.time()

        for batch_idx, (input, target) in enumerate(train_loader):

            if settings.GPU:
                target = target.to(device=device)
                input = input.to(device=device)

            if settings.FEATURE_EXTRACT == 'tt':
                input_tt = t3.input_to_tt_tensor(input, settings)

            if settings.FEATURE_EXTRACT == 'important_sketching':
                if settings.SAMPLE_MODE == 'curr_minibatch':
                    sample_covariance_tt_core_list = t3.construct_sample_covariance_tensor(settings.SAMPLE_MODE, settings, input, target)

                reduced_cov = t3.important_sketching(input, sample_covariance_tt_core_list, settings)
                #Then use target, reduced_cov as response and new dimension-reduced feature to do supervised learning
                input = reduced_cov

            data_time.update(time.time() - end)
            #for forward time
            end_new = time.time()
            # compute output
            with autograd.detect_anomaly():
                if settings.FEATURE_EXTRACT == 'tt':
                    output = model(input_tt)
                else:
                    output, svd_matrix_list = model(input)
                forward_time.update(time.time() - end_new)

                def my_loss(matrix_list, cri, output, target, alpha=0.00000001):
                    loss = cri(output, target)
                    for svd_matrix in matrix_list:
                        for curr_core in svd_matrix:
                            #pdb.set_trace()
                            assert(len(curr_core.shape) == 2)
                            for row_idx in range(min(curr_core.shape[0], curr_core.shape[1])):
                                if alpha * torch.sum(torch.abs(curr_core[row_idx, :]/curr_core[row_idx, row_idx])) > 10:
                                    print("adding a huge loss!")
                                loss = loss + alpha * torch.sum(torch.abs(curr_core[row_idx, :]/curr_core[row_idx, row_idx])) ** 2
                    return loss

                loss = my_loss(svd_matrix_list, criterion, output, target, 0.00000001)
                #loss = criterion(output, target)

                if loss > 10:
                    print('loss explosion!!')
                    #break
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()

            #observe SVD layer gradient
            #for name, param in model.named_parameters():
            #    if param.requires_grad:
            #            print(name, param.data, param.grad)
            # print('1', torch.max(model.learnable_conversion.weight_v_mm.grad), torch.min(model.learnable_conversion.weight_v_mm.grad))
            # print('2', torch.max(model.learnable_conversion.weight_u_mm.grad), torch.min(model.learnable_conversion.weight_u_mm.grad))
            # print('3', torch.max(model.learnable_conversion.weight_s_hadmard.grad), torch.min(model.learnable_conversion.weight_s_hadmard.grad))

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            #replace input.size(0) with settings.BATCH_SIZE
            losses.update(loss.item(), settings.BATCH_SIZE)
            top1.update(prec1.item(), settings.BATCH_SIZE)
            top5.update(prec5.item(), settings.BATCH_SIZE)

            # # compute gradient and do SGD step
            # optimizer.zero_grad()
            # loss.backward()

            #gradient clipping
            if settings.CLIP_GRAD != 0:
                clip_value = settings.CLIP_GRAD
                nn.utils.clip_grad_norm(model.parameters(), clip_value)

            #print(optimizer.state_dict())
            #move_buffer_to_gpu(optimizer)
            optimizer.step()

            def PGD(m):
                if t3.frobenius_norm_squared(m.module.ip1.weight) > 1:
                    for core_idx in range(5):
                        m.module.ip1.weight.tt_cores[core_idx].data = m.module.ip1.weight.tt_cores[core_idx].data/torch.sqrt(t3.frobenius_norm_squared(m.module.ip1.weight))
                if t3.frobenius_norm_squared(m.module.ip2.weight) > 1:
                    for core_idx in range(5):
                        m.module.ip2.weight.tt_cores[core_idx].data = m.module.ip2.weight.tt_cores[core_idx].data/torch.sqrt(t3.frobenius_norm_squared(m.module.ip1.weight))
                if t3.frobenius_norm_squared(m.module.ip3.weight) > 1:
                    for core_idx in range(5):
                        m.module.ip3.weight.tt_cores[core_idx].data = m.module.ip3.weight.tt_cores[core_idx].data/torch.sqrt(t3.frobenius_norm_squared(m.module.ip1.weight))
                if t3.frobenius_norm_squared(m.module.ip4.weight) > 1:
                    for core_idx in range(5):
                        m.module.ip4.weight.tt_cores[core_idx].data = m.module.ip4.weight.tt_cores[core_idx].data/torch.sqrt(t3.frobenius_norm_squared(m.module.ip1.weight))
                if t3.frobenius_norm_squared(m.module.ip5.weight) > 1:
                    for core_idx in range(5):
                        m.module.ip5.weight.tt_cores[core_idx].data = m.module.ip5.weight.tt_cores[core_idx].data/torch.sqrt(t3.frobenius_norm_squared(m.module.ip1.weight))
                if t3.frobenius_norm_squared(m.module.ip6.weight) > 1:
                    for core_idx in range(5):
                        m.module.ip6.weight.tt_cores[core_idx].data = m.module.ip6.weight.tt_cores[core_idx].data/torch.sqrt(t3.frobenius_norm_squared(m.module.ip1.weight))

            if settings.PGD:
                PGD(model)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % settings.PRINT_FEQ == 0:
                print('Train: [{0}/{1}]\t'
                      'Fwd {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(train_loader), forward_time=forward_time, batch_time=batch_time, data_time=data_time, loss=losses,
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


def validate(model, val_loader, sample_covariance_tt_core_list=None):
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
        if settings.FEATURE_EXTRACT == 'important_sketching':
            reduced_cov = t3.important_sketching(input, sample_covariance_tt_core_list, settings)
            input  = reduced_cov
        #input_tt = t3.input_to_tt_tensor(input, settings)
        #if settings.GPU:
            #target = target.cuda(non_blocking=True)
            #input = input.cuda(non_blocking=True)
        with torch.no_grad():
            #input_var = torch.autograd.Variable(input)
            #target_var = torch.autograd.Variable(target)

            # compute output
            #fc_output = model(input_tt)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), settings.BATCH_SIZE)
        top1.update(prec1.item(), settings.BATCH_SIZE)
        top5.update(prec5.item(), settings.BATCH_SIZE)

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
    #print('current GPU:', torch.cuda.current_device())
    # val_loader = imagenet_loader(settings, 'val')
    # train_loader = imagenet_loader(settings, 'train', shuffle=True, data_augment=True)
    val_loader = cifar_loader(settings, 'val')
    train_loader = cifar_loader(settings, 'train', shuffle=True, data_augment=True)

    if settings.FEATURE_EXTRACT == 'cnn' and not settings.BENCHMARK:
        #pdb.set_trace()
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False

        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=True)
        num_ftrs = model.fc.in_features #512
        model.fc = settings.CNN_MODEL(settings)
        #pdb.set_trace()
    elif settings.BENCHMARK:
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False

        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=True)
        num_ftrs = model.fc.in_features #512
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model.fc = nn.Linear(512, 10)
    else:
        model = settings.CNN_MODEL(settings)
        print(model)

    if settings.GPU:
        #model = nn.DataParallel(model).cuda()
        #model = model.cuda()
        model = nn.DataParallel(model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        assert(torch.cuda.device_count() > 1)
        #print('model parameter:', dict(model.named_parameters()))
        #.set_trace()

    #print(model)
    #model.train()
    #model = my_models.tt_input_resnet18(settings).cuda()
    # model = my_models.learnable_tt_wide_resnet(settings).cuda()
    # print(model)
    #test cifar10 ---> TT ---> Resnet
    #model = models.__dict__['resnet18']().cuda()
    #sample_test(settings,train_loader,val_loader,snapshot_dir)

    # Use full dataset to compute sample covariance tensor
    sample_covariance_tt_core_list = t3.construct_sample_covariance_tensor('full', settings)

    train(model, train_loader, val_loader, snapshot_dir, sample_covariance_tt_core_list)
    #only test the last 10 epoch
    for epoch in range(settings.EPOCHS - 10, settings.EPOCHS):
        settings.MODEL_FILE = "result/pytorch_{}_{}_{}/snapshot/epoch_{}.pth".format(args.arch, args.mark, args.dataset, epoch)
        validate(model, val_loader, sample_covariance_tt_core_list)


if __name__ == '__main__':
    main()
