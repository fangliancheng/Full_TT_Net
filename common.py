import time
import torch
import os
import torch.nn as nn
#from models.resnet import *
import sys
from ptt_vgg import *
from my_models import *
from model_wideresnet import WideResNet

DATASET_PATH = {
    'MINIST': '',
    'imagenet': '/home/liancheng/Documents/imagenet',
    'CIFAR10': '',
}

NUM_CLASSES = {
    'MINIST': 10,
    'imagenet': 1000,
    'CIFAR10': 10,
}

MODEL_DICT = {
     #'LeNet_300_100': lenet_300_100,
     'ptt_vgg11': ptt_vgg11,
     'ptt_vgg11_bn': ptt_vgg11_bn,
     'ptt_vgg13': ptt_vgg13,
     'ptt_vgg13_bn': ptt_vgg13_bn,
     'ptt_vgg16': ptt_vgg16,
     'ptt_vgg16_bn': ptt_vgg16_bn,
     'ptt_vgg19_bn': ptt_vgg19_bn,
     'ptt_vgg19': ptt_vgg19,
     #'resnet18': resnet18,
     'manifold_net': manifold_Net,
     #'pre_cnn': pre_mani_net,
     #'linear': pre_FTT_Net,
     'cw_fc_net': cw_fc_net,
     'slp': slp,
     'important_sketching_wideResnet': important_sketching_input_wideresnet,
     'important_sketching_ptt_net': 'important_sketching',
     'important_sketching_ftt_1hidden_relu_net': IS_FTT_1_layer_relu,
     'important_sketching_ftt_multi_relu': IS_FTT_multi_layer_relu,
     'important_sketching_logistic': Logistic,
     'important_sketching_ftt_logistic': IS_FTT_Logistic,
     'normal_logistic_cifar': normal_logistic,
     'normal_wide_resnet_cifar': WideResNet,
 }

SHAPE_DICT = {
    'pre_cnn': [4, 8, 4, 4],
    'manifold_net': [4, 8, 4, 8],
    'cw_fc_net': [4, 8, 4, 8],
    'slp': None,
 }

EXTRACT_DICT = {
    'pre_cnn': 'cnn',
    'manifold_net': 'tt',
    'linear': 'cnn',
    'cw_fc_net': 'tt',
    'slp': None,
    'important_sketching_wideResnet': 'important_sketching',
    'important_sketching_ptt_1hidden_relu_net': 'important_sketching',
    'important_sketching_ftt_1hidden_relu_net': 'important_sketching',
    'important_sketching_ftt_multi_relu': 'important_sketching',
    'important_sketching_logistic': 'important_sketching',
    'important_sketching_ftt_logistic': 'important_sketching',
    'normal_logistic_cifar': 'normal_logistic',
    'normal_wide_resnet_cifar': 'normal_wide_resnet',
}

FEATURE_FORM = {
    'important_sketching_wideResnet': 'dense',
    'important_sketching_ftt_1hidden_relu_net': 'tt_matrix',
    'important_sketching_ftt_multi_relu': 'dense', #for DataParallel
    'important_sketching_logistic': 'dense',
    'important_sketching_ftt_logistic': 'tt_matrix',
    'normal_logistic_cifar': None,
    'normal_wide_resnet_cifar': None,
}

def skew_sym_part(m):
    return 1/2 * (m.t() - m)


def to_list(tuple):
    return [i for i in tuple]


def log_f(f, console=True):
    def log(msg):
        f.write(msg)
        f.write('\n')
        if console:
            p(msg)
    return log


def dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except Exception:
    term_width = 100

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def val(settings, loader, model):
    if settings.GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    # val
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.eval()
    for i, (input, target) in enumerate(loader):
        start_idx = i * settings.BATCH_SIZE
        end_idx = min((i + 1) * settings.BATCH_SIZE, len(loader.dataset))
        # input = torch.FloatTensor(input)
        if settings.GPU:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        fc_output = model(input_var)

        loss = criterion(fc_output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(fc_output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if i % settings.PRINT_FEQ == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1, top5=top5))


    print(' * VAL Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
